import torch
import torch.nn as nn
import torch.nn.functional as func

from einops import rearrange
from typing import Optional, Tuple


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 4, in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size,) * 2, stride=(patch_size,) * 2)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    #def padding(self, x: torch.Tensor) -> torch.Tensor:
    #    _, _, H, W = x.shape
    #    if H % self.patch_size != 0 or W % self.patch_size != 0:
    #        x = func.pad(x, (0, self.patch_size - W % self.patch_size,
    #                         0, self.patch_size - H % self.patch_size,
    #                         0, 0))
    #    return x

    def forward(self, x):
        #x = self.padding(x) 
        B, T = x.shape[:2]
        x = rearrange(x, 'B T C H W -> (B T) C H W') # merge B and T to apply Conv to each image seperately.
        x = self.proj(x)
        x = rearrange(x, '(B T) C H W -> B T H W C', B=B, T=T)
        x = self.norm(x)
        return x



class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        # x shape: [B, T, H, W, C]
        x = self.padding(x)
        x = self.merging(x)
        B, T, H, W, C = x.shape
        x = x.view(B * T, H, W, C)  # Flatten B and T for LayerNorm
        x = self.norm(x)
        x = x.view(B, T, H, W, C)  # Reshape back to original dimensions
        x = self.reduction(x)
        return x



class PatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        # x shape: [B, T, H, W, C]
        B, T, H, W, C = x.shape
        
        # Expand the patches
        x = self.expand(x)  # [B, T, H, W, 2*C]

        # Rearrange the expanded patches
        x = rearrange(x, 'B T H W (P1 P2 C) -> B T (H P1) (W P2) C', P1=2, P2=2)
        
        # Normalize
        B, T, new_H, new_W, new_C = x.shape
        x = x.view(B * T, new_H, new_W, new_C)  # Flatten B and T for LayerNorm
        x = self.norm(x)
        x = x.view(B, T, new_H, new_W, new_C)  # Reshape back to original dimensions

        return x


class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4, P2=4)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: Optional[bool] = True,
                 attn_drop: Optional[float] = 0., proj_drop: Optional[float] = 0., shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        if shift:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_size = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_size, coords_size], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        _, T, H, W, _ = x.shape

        x = rearrange(x, 'B T (Nh Mh) (Nw Mw) C -> (B T Nh Nw) Mh Mw C', Nh=H // self.window_size, Nw=W // self.window_size)
        return x

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        _, T, H, W, _ = x.shape

        assert H % self.window_size == 0 and W % self.window_size == 0, "H or W is not divisible by window_size"

        img_mask = torch.zeros((1, T, H, W, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, T, H, W, _ = x.shape
    
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            mask = self.create_mask(x)
        else:
            mask = None

        x = self.window_partition(x)
        Bn, Mh, Mw, _ = x.shape
        x = rearrange(x, 'Bn Mh Mw C -> Bn (Mh Mw) C')
        qkv = rearrange(self.qkv(x), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads, Mh * Mw, Mh * Mw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'Bn Nh (Mh Mw) C -> Bn Mh Mw (Nh C)', Mh=Mh)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(B T Nh Nw) Mh Mw C -> B T (Nh Mh) (Nw Mw) C', T=T, Nh=H // Mh, Nw=W // Mw)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x



class WindowAttentionTemporal(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int, qkv_bias: Optional[bool] = True,
                 attn_drop: Optional[float] = 0., proj_drop: Optional[float] = 0., shift: bool = False):
        super().__init__()
        self.window_size = window_size  # (T, H, W)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        if shift:
            self.shift_size = window_size[1] // 2  # Spatial shift size (same for H and W)
        else:
            self.shift_size = 0

        # Update relative position bias table to include temporal dimension
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # Generate and store relative position index
        coords = torch.stack(torch.meshgrid(
            [torch.arange(window_size[0]), torch.arange(window_size[1]), torch.arange(window_size[2])], 
            indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        for i in range(3):
            relative_coords[:, :, i] += self.window_size[i] - 1
            relative_coords[:, :, i] *= 2 * self.window_size[i] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = x.view(B, 
                   T // self.window_size[0], self.window_size[0],
                   H // self.window_size[1], self.window_size[1],
                   W // self.window_size[2], self.window_size[2], 
                   C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        return windows
        
    def window_reverse(self, windows: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (T * H * W / (self.window_size[0] * self.window_size[1] * self.window_size[2])))
        x = windows.view(B, 
                         T // self.window_size[0], 
                         H // self.window_size[1], 
                         W // self.window_size[2], 
                         self.window_size[0], 
                         self.window_size[1], 
                         self.window_size[2], 
                         -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
        return x

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        img_mask = torch.zeros((1, T, H, W, 1), device=x.device)
        h_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, T, H, W, C = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            mask = self.create_mask(x)
        else:
            mask = None
    
        x_windows = self.window_partition(x)
        B_, N, _ = x_windows.shape
        qkv = self.qkv(x_windows).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
    
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
    
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
    
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
    
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
    
        x = self.window_reverse(x, T, H, W)
    
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return x

                


class HighFrequencyBranch(nn.Module):
    def __init__(self, embedding_dim):
        super(HighFrequencyBranch, self).__init__()
        self.in_out_dim = embedding_dim // 2
        
        # top branch
        self.top_conv1x1_1 = nn.Conv2d(self.in_out_dim, self.in_out_dim, kernel_size=1, padding=0, bias=False)
        self.top_conv3x3 = nn.Conv2d(self.in_out_dim, self.in_out_dim, kernel_size=3, padding=1, bias=False, groups=self.in_out_dim)
        self.top_gelu = nn.GELU()

        # bottom branch
        self.bot_max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.bot_conv1x1_2 = nn.Conv2d(self.in_out_dim, self.in_out_dim, kernel_size=1)
        self.bot_gelu = nn.GELU()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        # top branch
        x1 = self.top_conv1x1_1(x1)
        x1 = self.top_conv3x3(x1)
        x1 = self.top_gelu(x1)
        
        # bottom branch
        x2 = self.bot_max_pool(x2)
        x2 = self.bot_conv1x1_2(x2)
        x2 = self.bot_gelu(x2)
        
        return torch.cat((x1, x2), axis=1)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, temporal_bool=True, temporal_size=6):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if temporal_bool and temporal_size > 1:
            self.attn = WindowAttentionTemporal(dim, window_size=(temporal_size,window_size,window_size), num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=drop, shift=shift)
        else:
            self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=drop, shift=shift)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.HighFrequencyBranch = HighFrequencyBranch(dim) 

    def forward(self, x):
        b, t, h, w, c = x.shape
        hx = x.permute(0, 1, 4, 2, 3).contiguous().view(b * t, c, h, w)
        hx = self.HighFrequencyBranch(hx)
        hx = hx.view(b, t, c, h, w).permute(0, 1, 3, 4, 2).contiguous()
        
        x_copy = x
        x = self.norm1(x)

        x = self.attn(x)
        x = self.drop_path(x)
        x = x + x_copy

        x_copy = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x_copy
        
        return x + hx


class BasicBlock(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 8, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm, patch_merging: bool = True, temporal_bool: bool =True, temporal_size: int=6):
        super(BasicBlock, self).__init__()
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer,
                temporal_bool=temporal_bool,
                temporal_size=temporal_size)
            for i in range(depth)])

        if patch_merging:
            self.downsample = PatchMerging(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicBlockUp(nn.Module):
    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 8, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer=nn.LayerNorm):
        super(BasicBlockUp, self).__init__()
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_expanding:
            self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.upsample(x)
        return x


class SwinUnet(nn.Module):
    def __init__(self, patch_size: int = 4, in_chans: int = 3, num_classes: int = 1000, embed_dim: int = 96,
                 window_size: int = 8, depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer=nn.LayerNorm, patch_norm: bool = True):
        super().__init__()

        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = self.build_layers()
        self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.skip_connection_layers = self.skip_connection()
        self.norm_up = norm_layer(embed_dim)
        self.final_patch_expanding = FinalPatchExpanding(dim=embed_dim, norm_layer=norm_layer)
        self.head = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=(1, 1), bias=False)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
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
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer)
            layers_up.append(layer)
        return layers_up

    def skip_connection(self):
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            layer = nn.Linear(dim * 2, dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)

        x = self.first_patch_expanding(x)

        for i, layer in enumerate(self.layers_up):
            x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            x = layer(x)

        x = self.norm_up(x)
        x = self.final_patch_expanding(x)

        x = rearrange(x, 'B H W C -> B C H W')
        x = self.head(x)
        return x
