import torch
import torch.nn as nn
import torch.nn.functional as F

class FDAF(nn.Module):
    """Flow Dual-Alignment Fusion Module adapted for merged B*T and two image embeddings.

    Args:
        in_channels (int): Input channels of features for each image.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='IN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='GELU')
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        kernel_size = 5
        # We concatenate two features so input is `in_channels * 2`
        self.flow_make = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=True, groups=in_channels),
            nn.InstanceNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input feature maps of shape [B*T, C, H, W].
                              Where T=2 (two input images).
        """
        # Step 1: Reshape to [B, 2, C, H, W] -> Separate two images per batch.
        B_T, C, H, W = x.size()
        B = B_T // 2  # Since T=2, divide B*T by 2 to get original batch size
        x = x.view(B, 2, C, H, W)  # Reshape to [B, 2, C, H, W]

        # Step 2: Split into x1 (image 1 features) and x2 (image 2 features)
        x1 = x[:, 0, :, :, :]  # Features from the first image
        x2 = x[:, 1, :, :, :]  # Features from the second image

        # Step 3: Concatenate x1 and x2 along the channel dimension
        output = torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension -> [B, 2*C, H, W]

        # Step 4: Generate flow field and apply flow-based alignment
        flow = self.flow_make(output)  # Generate flow
        f1, f2 = torch.chunk(flow, 2, dim=1)  # Split flow into two parts (for x1 and x2)

        # Step 5: Warp x1 and x2 based on the generated flow
        x1_feat = self.warp(x1, f1)  # Warp x1 using flow f1
        x2_feat = self.warp(x2, f2)  # Warp x2 using flow f2

        # Step 6: Return the aligned feature maps, or apply further fusion as needed
        return x1_feat, x2_feat

    @staticmethod
    def warp(x, flow):
        """Warp feature map `x` based on the flow field."""
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output


        
class ChangeHead(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(ChangeHead, self).__init__()
        
        # Instantiate the FDAF module for each scale
        self.fdaf1 = FDAF(in_channels[0])  # For first layer features
        self.fdaf2 = FDAF(in_channels[1])  # For second layer features
        self.fdaf3 = FDAF(in_channels[2])  # For third layer features
        self.fdaf4 = FDAF(in_channels[3])  # For fourth layer features

        # Upsampling layers to progressively upsample from 32x32 to 1024x1024
        self.upsample1 = nn.ConvTranspose2d(in_channels[3], in_channels[2], kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(in_channels[2], in_channels[1], kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(in_channels[1], in_channels[0], kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(in_channels[0], in_channels[0] // 2, kernel_size=2, stride=2)

        # Output segmentation map for change detection
        self.final_conv = nn.Conv2d(in_channels[0] // 2, num_classes, kernel_size=1)

    def forward(self, features):
        """
        features: list of feature maps from 4 layers.
        Assumes that the input list 'features' contains 4 tensors,
        each from different stages of the encoder, with decreasing resolutions.
        """
        # Step 1: Apply FDAF at each feature scale
        f1_1, f1_2 = self.fdaf1(features[0])  # Resolution: (B, in_channels[0], 256, 256)
        f2_1, f2_2 = self.fdaf2(features[1])  # Resolution: (B, in_channels[1], 128, 128)
        f3_1, f3_2 = self.fdaf3(features[2])  # Resolution: (B, in_channels[2], 64, 64)
        f4_1, f4_2 = self.fdaf4(features[3])  # Resolution: (B, in_channels[3], 32, 32)

        # Step 2: Gradual upsampling with skip connections

        # Upsample f4 (32x32 -> 64x64) and add f3 (upsampled to match the size of f4)
        x = self.upsample1(f4_1 + f4_2)  # (B, in_channels[2], 64, 64)
        f3_1_upsampled = F.interpolate(f3_1, size=x.shape[2:], mode='bilinear', align_corners=False)
        f3_2_upsampled = F.interpolate(f3_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + (f3_1_upsampled + f3_2_upsampled)  # Add upsampled skip connection from f3

        # Upsample x (64x64 -> 128x128) and add f2 (upsampled to match the size of x)
        x = self.upsample2(x)  # (B, in_channels[1], 128, 128)
        f2_1_upsampled = F.interpolate(f2_1, size=x.shape[2:], mode='bilinear', align_corners=False)
        f2_2_upsampled = F.interpolate(f2_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + (f2_1_upsampled + f2_2_upsampled)  # Add upsampled skip connection from f2

        # Upsample x (128x128 -> 256x256) and add f1 (upsampled to match the size of x)
        x = self.upsample3(x)  # (B, in_channels[0], 256, 256)
        f1_1_upsampled = F.interpolate(f1_1, size=x.shape[2:], mode='bilinear', align_corners=False)
        f1_2_upsampled = F.interpolate(f1_2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = x + (f1_1_upsampled + f1_2_upsampled)  # Add upsampled skip connection from f1

        # Upsample x (256x256 -> 512x512)
        x = self.upsample4(x)  # (B, in_channels[0] // 2, 512, 512)

        # Final upsampling to 1024x1024 using interpolation
        x = F.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # Step 3: Final segmentation map
        output = self.final_conv(x)  # (B, num_classes, 1024, 1024)
        
        return output