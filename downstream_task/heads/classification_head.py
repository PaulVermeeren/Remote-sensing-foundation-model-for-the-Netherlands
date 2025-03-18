import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, num_classes=0):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for c in in_channels
        ])

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * len(in_channels), hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        processed_features = []
        target_size = features[0].size()[2:]  # Reference spatial size

        for i, feature in enumerate(features):
            f = self.conv_blocks[i](feature)
            f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            processed_features.append(f)

        fused_features = torch.cat(processed_features, dim=1)
        fused_features = self.fusion_conv(fused_features)

        pooled = F.adaptive_avg_pool2d(fused_features, (1, 1)).view(fused_features.size(0), -1)
        output = self.classifier(pooled)
        
        return output
