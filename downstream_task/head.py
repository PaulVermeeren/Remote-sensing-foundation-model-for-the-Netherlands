import torch.nn.functional as F
import torch.nn as nn
import torch

from heads.segformer_head import SegFormerHead
from heads.classification_head import ClassificationHead
#from heads.change_head import ChangeHead
from heads.fdaf_head import ChangeHead

class FineTuneDecoder(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.task = args.task

        img_size = args.img_size
        num_classes=args.num_classes
        unfreeze_layers=args.unfreeze_layers

        feature_dimensions = [192, 384, 768, 768]

        if self.task == 'segmentation':
            self.unet = True
            self.head = SegFormerHead(
                feature_strides=[4, 8, 16, 32],
                in_channels=feature_dimensions,
                img_size=img_size,
                num_classes=num_classes
            )
        elif self.task == 'classification':
            self.unet = True
            self.head = ClassificationHead(
                in_channels=feature_dimensions,
                num_classes=num_classes
            )
        elif self.task == 'change':
            self.unet = True
            self.head = ChangeHead(
                in_channels=feature_dimensions,
                #num_classes=num_classes
            )

        # freeze layers from the start, if same as length layers, then freeze nothing
        if unfreeze_layers < len(self.model.layers): # 4
            layers_from_start = len(self.model.layers) - unfreeze_layers
            for param in self.model.layers[:layers_from_start].parameters():
                param.requires_grad = False

    def forward(self, x):
        # Use only the encoder part of the base model
        B, T, _, H, W = x.shape

        x = self.model.patch_embed(x)

        x_unet = []
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            x_unet.append(x)

        # Reshape features to [B*T, C, H, W] for the segmentation head
        if self.unet:
            features = [f.permute(0, 1, 4, 2, 3).contiguous().view(-1, f.shape[-1], f.shape[2], f.shape[3]) for f in x_unet]

            # remove T=1 dimension for classification
            if self.task == "classification":
                features = [f.squeeze(1) for f in features]
        else:
            features = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, x.shape[-1], x.shape[2], x.shape[3])

        # call head
        x = self.head(features)


        # Reshape to [B, T, num_classes, H, W]
        if self.task == "segmentation":
            x = x.view(B, T, *x.shape[1:])

        return x