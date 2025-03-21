import torch.nn.functional as F
import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, img_size):
        super().__init__()
        self.proj = nn.Linear(input_dim, img_size)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class SegFormerHead(nn.Module):
    def __init__(self, feature_strides, in_channels, img_size, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.feature_strides = feature_strides
        self.img_size = img_size

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, img_size=img_size)
        self.linear_c3 = MLP(input_dim=c3_in_channels, img_size=img_size)
        self.linear_c2 = MLP(input_dim=c2_in_channels, img_size=img_size)
        self.linear_c1 = MLP(input_dim=c1_in_channels, img_size=img_size)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(img_size * 4, img_size, 1, bias=False),
            nn.BatchNorm2d(img_size),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(img_size, num_classes, kernel_size=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        n, _, h, w = c1.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.linear_pred(_c)
        
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return x