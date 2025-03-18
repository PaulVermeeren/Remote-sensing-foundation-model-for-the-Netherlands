from functools import partial

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from swin_unet import PatchEmbedding, BasicBlock, PatchExpanding, BasicBlockUp
from utils.pos_embed import get_2d_sincos_pos_embed

import torch.nn.functional as F
from torchvision import models
from timm.models.layers import trunc_normal_

import matplotlib.pyplot as plt

class SwinMAE(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 512, patch_size: int = 4, mask_ratio: float = 0.8, in_chans: int = 6, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 8, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, high_pass: bool = False, patch_norm: bool = True, temporal_size: int = 6, pimask: bool=False, temporal_bool: bool=True, mask_size: int=4, include_decoder: bool = True):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = 96 # (patch_size ** 2) * temporal_size
        self.decoder_embed_dim = self.embed_dim * (2 ** (self.num_layers - 1))
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.in_chans = in_chans
        self.temporal_size = temporal_size
        self.temporal_bool = temporal_bool
        self.pimask = pimask
        self.include_decoder = include_decoder
        
        
        
        self.mask_radius = mask_size
        self.high_pass = high_pass
        self.high_pass_window_size = self.patch_size * self.window_size
        self.high_pass_prob = 0.25

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=self.embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.layers = self.build_layers()
        
        if self.include_decoder:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
            self.first_patch_expanding = PatchExpanding(dim=self.decoder_embed_dim, norm_layer=norm_layer)
            
            #self.deconv1 = nn.ConvTranspose2d(in_channels=384, out_channels=256, kernel_size=4, stride=2, padding=1)
            #self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
            #self.conv1 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
            
            self.deconv1 = nn.ConvTranspose2d(in_channels=self.decoder_embed_dim // (2 ** 1), out_channels=self.decoder_embed_dim // 3, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(in_channels=self.decoder_embed_dim // 3, out_channels=self.decoder_embed_dim // (2 ** 2), kernel_size=4, stride=2, padding=1)
            self.conv1 = nn.Conv2d(in_channels=self.decoder_embed_dim // (2 ** 2), out_channels=self.decoder_embed_dim //  (2 ** 3), kernel_size=3, stride=1, padding=1)
       
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
           
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        

    def patchify(self, imgs):
        """
        imgs: (N, T, C, H, W)
        x: (N, T, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[3] == imgs.shape[4] and imgs.shape[3] % p == 0
    
        h = w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], self.in_chans, h, p, w, p))
        x = torch.einsum('ntchpwq->nthwpqc', x)
        x = x.reshape(imgs.shape[0], imgs.shape[1], h * w, p ** 2 * self.in_chans)
        return x

    def unpatchify(self, x):
        """
        x: (N, T, L, patch_size**2 * C)
        imgs: (N, T, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[2] ** .5)
        assert h * w == x.shape[2]
    
        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, self.in_chans))
        x = torch.einsum('nthwpqc->ntchpwq', x)
        imgs = x.reshape(x.shape[0], x.shape[1], self.in_chans, h * p, h * p)
        return imgs

    def window_masking(self, x: torch.Tensor, r: int = 4, masking=0.8,
                   remove: bool = False, mask_len_sparse: bool = False):
        """
        The new masking method, masking the adjacent r*r number of patches together
    
        Optional whether to remove the mask patch,
        if so, the return value returns one more sparse_restore for restoring the order to x
    
        Optionally, the returned mask index is sparse length or original length,
        which corresponds to the different size choices of the decoder when restoring the image
    
        x: [B, T, L, D]
        r: There are r*r patches in a window
        remove: Whether to remove the mask patch
        mask_len_sparse: Whether the returned mask length is a sparse short length
        """
        
        x = rearrange(x, 'B T H W C -> B T (H W) C')
        B, T, L, D = x.shape
        assert int(L ** 0.5 / r) == L ** 0.5 / r
        d = int(L ** 0.5 // r)
    
        # Generate the same mask for each temporal slice
        noise = torch.rand(B, d ** 2, device=x.device)
        sparse_shuffle = torch.argsort(noise, dim=1)
        sparse_restore = torch.argsort(sparse_shuffle, dim=1)
        sparse_keep = sparse_shuffle[:, :int(d ** 2 * (1 - masking))]
    
        index_keep_part = torch.div(sparse_keep, d, rounding_mode='floor') * d * r ** 2 + sparse_keep % d * r
        index_keep = index_keep_part
        for i in range(r):
            for j in range(r):
                if i == 0 and j == 0:
                    continue
                index_keep = torch.cat([index_keep, index_keep_part + int(L ** 0.5) * i + j], dim=1)
    
        index_all = np.expand_dims(range(L), axis=0).repeat(B, axis=0)
        index_mask = np.zeros([B, int(L - index_keep.shape[-1])], dtype=int)
        for i in range(B):
            index_mask[i] = np.setdiff1d(index_all[i], index_keep.cpu().numpy()[i], assume_unique=True)
        index_mask = torch.tensor(index_mask, device=x.device)
    
        index_shuffle = torch.cat([index_keep, index_mask], dim=1)
        index_restore = torch.argsort(index_shuffle, dim=1)
    
        if mask_len_sparse:
            mask = torch.ones([B, d ** 2], device=x.device)
            mask[:, :sparse_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=sparse_restore)
        else:
            mask = torch.ones([B, L], device=x.device)
            mask[:, :index_keep.shape[-1]] = 0
            mask = torch.gather(mask, dim=1, index=index_restore)
    
        x_masked_all = []
        for t in range(T):
            x_t = x[:, t, :, :]  # [B, L, D]
            if remove:
                x_masked_t = torch.gather(x_t, dim=1, index=index_keep.unsqueeze(-1).repeat(1, 1, D))
                x_masked_t = rearrange(x_masked_t, 'B (H W) C -> B H W C', H=int(x_masked_t.shape[1] ** 0.5))
                x_masked_all.append(x_masked_t)
            else:
                x_masked_t = torch.clone(x_t)
                for i in range(B):
                    x_masked_t[i, index_mask.cpu().numpy()[i, :], :] = self.mask_token
                x_masked_t = rearrange(x_masked_t, 'B (H W) C -> B H W C', H=int(x_masked_t.shape[1] ** 0.5))
                x_masked_all.append(x_masked_t)
    
        x_masked_all = torch.stack(x_masked_all, dim=1)  # Combine the masked temporal slices
        
        if remove:
            return x_masked_all, mask, sparse_restore
        else:
            return x_masked_all, mask
            

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True,
                temporal_bool=self.temporal_bool,
                temporal_size=self.temporal_size)
            layers.append(layer)
        return layers

        
    def high_pass_filter(self, x):
        # Assuming x is [B*T, C, H, W]
        _, C, H, W = x.shape
        
        # Define the high-pass filter kernel
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=x.dtype, device=x.device)
        kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
        
        # Apply padding
        x_padded = F.pad(x, (1, 1, 1, 1), mode='reflect')
        
        # Apply the filter
        x_filtered = F.conv2d(x_padded, kernel, groups=C)
        
        return x_filtered
    
    def apply_random_high_pass(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        
        # Calculate number of windows
        n_windows_h = H // self.high_pass_window_size
        n_windows_w = W // self.high_pass_window_size
        
        # Create a mask for selected windows
        window_mask = torch.zeros((B*T, 1, n_windows_h, n_windows_w), device=x.device)
        window_mask = window_mask.bernoulli_(self.high_pass_prob)
        window_mask = window_mask.repeat_interleave(self.high_pass_window_size, dim=2)
        window_mask = window_mask.repeat_interleave(self.high_pass_window_size, dim=3)
        
        # Expand window mask to all channels
        window_mask = window_mask.expand(-1, C, -1, -1)
        
        # Apply high-pass filter to the entire input
        x_filtered = self.high_pass_filter(x)
        
        # Combine filtered and original based on window mask
        x = torch.where(window_mask.bool(), x_filtered, x)
        
        return x.view(B, T, C, H, W)
        
    def forward_encoder(self, x):  
    
        if self.high_pass:
            x = self.apply_random_high_pass(x)
        
    
        x = self.patch_embed(x)
        
        B, T, H, W, _ = x.shape  # x is [B, T, H, W, C]
        x, mask = self.window_masking(x, r=self.mask_radius, masking=self.mask_ratio, remove=False, mask_len_sparse=False)

        if self.pimask:
            _, small_mask = self.window_masking(x, r=1, masking=0.8, remove=False, mask_len_sparse=False)
            mask[mask == 1] = small_mask[mask == 1]
        
        for layer in self.layers:
            x = layer(x)
                        
        return x, mask

    def forward_decoder(self, x):
        device = x.device    
        x = self.first_patch_expanding(x)
        
        batch_size, seq_len, height, width, channels = x.shape
        x = x.view(batch_size * seq_len, channels, height, width)  # Shape: [12, 384, 32, 32]
        
        x = self.deconv1(x)  # Shape: [12, 256, 64, 64]
        x = self.deconv2(x)  # Shape: [12, 128, 128, 128]
        x = self.conv1(x)    # Shape: [12, 96, 128, 128] 
        
        # Reshape to desired output shape
        x = x.view(batch_size, seq_len, x.shape[2] ** 2, self.embed_dim)
        
        return x  

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [B, T, C, H, W]
        pred: [B, T, L, p*p*C]
        mask: [B, L], 0 is keep, 1 is remove,
        """
        
        zero_mask = (imgs == 0).all(dim=(3,4))
        
        non_zero_mask = ~zero_mask
        average_channels = torch.sum(non_zero_mask) / (imgs.shape[0] * imgs.shape[1]) # som van non padded kanelen gedeeld door batch en temporal size
                   
        zero_mask = zero_mask.unsqueeze(-1).unsqueeze(-1)
        zero_mask = zero_mask.expand_as(imgs)
        pred = self.unpatchify(pred)
        pred[zero_mask] = 0
        pred = self.patchify(pred)
                
        # Patchify each frame
        target = self.patchify(imgs)  # [B, T, L, p*p*C]
        
        # Compute the loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, T, L]
        
        mask = mask.unsqueeze(1) # add Temporal dimension
    
        # Apply the mask and average the loss
        loss = (loss * mask).sum() / mask.sum()
    
        ###### calculate loss only over masked patches
        
        mask_expanded = mask.unsqueeze(-1)  # [B, T, L, 1]
    
        #target = self.patchify(imgs)  # [B, T, L, p*p*C]
        target = target * mask_expanded  # Apply mask to target patches
        target = self.unpatchify(target)  # [B, T, C, H, W]
        
        pred = pred * mask_expanded  # Apply mask to pred patches
        pred = self.unpatchify(pred)  # [B, T, C, H, W]
        
        ######
        
        # Calculate spectral loss
        pred_spectral = pred.mean(dim=2)  # [B, T, H, W] average over channels
        target_spectral = target.mean(dim=2)  # [B, T, H, W] average over channels
        
        loss_spectral = (pred_spectral - target_spectral) ** 2  # (x_p - x_t) ^ 2 after averaging over spectral dimension
        loss_spectral = loss_spectral.sum() / (mask.sum() * average_channels)  # Normalize by mask sum and number of channels
      
        ######
        
        loss = (loss + loss_spectral) / imgs.shape[1]
        
        return loss

    def forward(self, x):
        # gradient masking for padded channels                        
        non_zero_mask = ~(x == 0).all(dim=(3,4))
        padded_mask = non_zero_mask.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        
        latent, mask = self.forward_encoder(x)
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask, padded_mask


def swin_mae(**kwargs):
    model = SwinMAE( 
        img_size=kwargs.get('img_size', 512),
        patch_size=kwargs.get('patch_size', 4),
        in_chans=kwargs.get('in_chans', 6),
        depths=kwargs.get('depths', (2, 2, 6, 2)),
        num_heads=kwargs.get('num_heads', (3, 6, 12, 24)),
        window_size=kwargs.get('window_size', 8),
        mask_ratio=kwargs.get('mask_ratio', 0.75),
        mask_size=kwargs.get('mask_size', 4),
        pimask=kwargs.get('pimask', False),
        high_pass=kwargs.get('high_pass', False),
        temporal_size=kwargs.get('temporal_size', 6),
        temporal_bool=kwargs.get('temporal_bool', True),
        include_decoder=kwargs.get('include_decoder', True),
        qkv_bias=True, mlp_ratio=4, drop_path_rate=0.1, drop_rate=0, attn_drop_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    return model