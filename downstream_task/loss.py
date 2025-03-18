import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import math

from torchvision.ops import nms
from einops import rearrange

def calculate_loss(args, output, target, validation=False):
    if args.task == "segmentation":
        B, T, C, H, W = output.shape
        output_loss = output.reshape(B*T, C, H, W)
        target_loss = target.reshape(B*T, H, W).long()

        if "vegetatie" in args.dataset:
            target_loss = target_loss - 1
            ignore_index = -1
        else:
            ignore_index = None

        if args.cross:
            loss = F.cross_entropy(output_loss, target_loss, ignore_index=ignore_index, reduction='mean')
        else:
            loss_fn = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True, ignore_index=ignore_index)
            loss = loss_fn(output_loss, target_loss)

    elif args.task == "classification":
        target = target.view(-1).long()  # labels: [B]
        loss = F.cross_entropy(output, target)

    elif args.task == "change":
        target = target.squeeze(1).long()
        loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        loss = loss_fn(output, target)

    return loss