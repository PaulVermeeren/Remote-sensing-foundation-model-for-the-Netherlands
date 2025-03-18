import torch
import torch.nn as nn
import torch.nn.functional as F

class ChangeHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_classes=2):
        super().__init__()
        self.in_channels = in_channels

        # Reduce channel dimensions
        self.conv_reduce = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_channels
        ])

        # Difference and attention modules
        self.diff_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Sigmoid()
            ) for _ in in_channels
        ])

        # Feature pyramid network with residual connections
        self.fpn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            ) for _ in range(len(in_channels) - 1)
        ])

        # Edge-preserving convolutions
        self.edge_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ) for _ in range(len(in_channels))
        ])

        # Additional upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ) for _ in range(3)  # Upsample 3 times to get from 128x128 to 1024x1024
        ])

        # Final prediction layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

    def forward(self, features):
        multi_scale_features = []

        for i, feature in enumerate(features):
            B, T, C, H, W = feature.shape
            assert T == 2, "Input should contain exactly two time steps"

            # Split and reduce channels
            f1, f2 = feature[:, 0], feature[:, 1]
            f1 = self.conv_reduce[i](f1)
            f2 = self.conv_reduce[i](f2)

            # Compute difference and attention
            diff = torch.abs(f1 - f2)
            attention = self.diff_attention[i](diff)
            attended_diff = diff * attention

            # Apply edge-preserving convolution
            attended_diff = self.edge_conv[i](attended_diff)

            multi_scale_features.append(attended_diff)

        # Feature Pyramid Network with residual connections
        for i in range(len(multi_scale_features) - 1, 0, -1):
            upsampled = F.interpolate(multi_scale_features[i], size=multi_scale_features[i-1].shape[-2:], mode='nearest')
            upsampled = self.fpn[i-1](upsampled)
            multi_scale_features[i-1] = multi_scale_features[i-1] + upsampled

        # Use the highest resolution feature map
        x = multi_scale_features[0]

        # Additional upsampling
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)

        # Final prediction
        output = self.final_conv(x)

        return output