import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
import torch.nn as nn

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.append(module_path)
import swin_mae

import head


def get_args_parser():
    parser = argparse.ArgumentParser('SwinMAE fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=3, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--gpu', default="", type=str,
                        help='')
    parser.add_argument('--task', default="", type=str,
                        help='classification, segmentation')
    parser.add_argument('--dataset', default="", type=str,
                        help='dataset name')
    parser.add_argument('--interpolate', default=0, type=int,
                        help='')
    parser.add_argument('--maxdata', default=1, type=float,
                        help='')
    parser.add_argument('--unfreeze_layers', default=4, type=int,
                        help='')
    parser.add_argument('--num_augmentations', default=1, type=int,
                        help='')

    parser.add_argument('--continue_params', action='store_false',
                        help='')
    parser.add_argument('--transfer_learning', action='store_false',
                        help='')
    parser.add_argument('--earlystop', action='store_true',
                        help='')
    # Model parameters
    parser.add_argument('--patch_size', default=4, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--img_size', default=512, type=int,
                        help='')
    parser.add_argument('--in_chans', default=6, type=int,
                        help='')
    parser.add_argument('--temporal_size', default=6, type=int,
                        help='')
    parser.add_argument('--window_size', default=8, type=int,
                        help='')
    parser.add_argument('--depths', default=(2, 2, 6, 2), type=tuple,
                        help='')
    parser.add_argument('--num_heads', default=(3, 6, 12, 24), type=tuple,
                        help='')
    parser.add_argument('--mask_size', default=4, type=int,
                        help='')
    parser.add_argument('--temporal_bool', action='store_false',
                        help='')
    parser.add_argument('--pimask', action='store_false',
                        help='')
    parser.add_argument('--high_pass', action='store_false',
                        help='')
    parser.add_argument('--cross', action='store_true',
                        help='')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.001, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_path', default='', type=str,
                        help='Train.csv path')
    parser.add_argument('--val_path', default='', type=str,
                        help='Train.csv path')
    parser.add_argument('--tag', default='', type=str,
                        help='tag')
    parser.add_argument('--tag_long', default='', type=str,
                        help='tag long')

    parser.add_argument('--output_dir', default='output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--resume_head', default='',
                        help='resume from checkpoint')
    parser.add_argument('--checkpoint', default='',
                        help='For performance script')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--disable_logging', action='store_true',
                        help='Disable all logging for testing purposes')

    return parser


def adapt_channels(state_dict, new_in_channels):
    if new_in_channels != 6:  # pretrained on 6
        # Get the current patch_embed.proj.weight
        old_weight = state_dict['patch_embed.proj.weight']
        old_in_channels = old_weight.shape[1]
        out_channels, _, kernel_size, _ = old_weight.shape

        # Create a new weight tensor with the desired number of input channels
        new_weight = torch.zeros(out_channels, new_in_channels, kernel_size, kernel_size)

        # Copy existing channels
        if new_in_channels <= old_in_channels:
            # If reducing channels, just copy the first 'new_in_channels' channels
            new_weight[:, :new_in_channels, :, :] = old_weight[:, :new_in_channels, :, :]
        else:
            # If increasing channels, repeat the original channels until we fill new_in_channels
            repeat_factor = (new_in_channels + old_in_channels - 1) // old_in_channels  # Number of repetitions needed
            repeated_weight = old_weight.repeat(1, repeat_factor, 1, 1)  # Repeat old channels
            new_weight[:, :new_in_channels, :, :] = repeated_weight[:, :new_in_channels, :, :]

        # Update the state dict with the new weight
        state_dict['patch_embed.proj.weight'] = new_weight

    return state_dict



def adapt_img_size(checkpoint, new_img_size):
    adapted_checkpoint = checkpoint.copy()
    old_size = 1024  # Assuming the original size was 1024x1024
    new_size = new_img_size

    for key in checkpoint:
        if 'segmentation_head' in key:
            old_shape = checkpoint[key].shape
            if len(old_shape) == 2:
                if old_shape[0] == old_size:
                    # For 2D tensors (weights)
                    adapted_checkpoint[key] = F.adaptive_avg_pool1d(checkpoint[key].unsqueeze(0), new_size).squeeze(0)
                elif old_shape[1] == old_size:
                    # For 2D tensors (weights) where the second dimension is the one to be adapted
                    adapted_checkpoint[key] = F.adaptive_avg_pool1d(checkpoint[key].t().unsqueeze(0), new_size).squeeze(0).t()
            elif len(old_shape) == 1:
                # For 1D tensors (biases, running_mean, running_var)
                adapted_checkpoint[key] = F.adaptive_avg_pool1d(checkpoint[key].unsqueeze(0).unsqueeze(0), new_size).squeeze(0).squeeze(0)
            elif len(old_shape) == 4:
                # For 4D tensors (convolution weights)
                if old_shape[1] == old_size:
                    adapted_checkpoint[key] = F.adaptive_avg_pool2d(checkpoint[key], (new_size, checkpoint[key].shape[-1]))
                elif old_shape[2] == old_size and old_shape[3] == old_size:
                    adapted_checkpoint[key] = F.adaptive_avg_pool2d(checkpoint[key], (new_size, new_size))

    return adapted_checkpoint




def adapt_relative_position(checkpoint, original_window_size, new_window_size):
    """
    Adapts the relative position bias table and relative position index in the checkpoint
    to match the new window size.

    Args:
        checkpoint (dict): The pretrained model's state dictionary.
        original_window_size (tuple): The original window size (T, H, W).
        new_window_size (tuple): The new window size (T, H, W).

    Returns:
        dict: The updated checkpoint.
    """
    keys_to_update = [key for key in checkpoint.keys() if "relative_position_bias_table" in key]

    for key in keys_to_update:
        # Adapt relative_position_bias_table
        original_bias_table = checkpoint[key]
        num_heads = original_bias_table.shape[1]

        # Calculate original and new relative position dimensions
        original_entries = (
            (2 * original_window_size[0] - 1) *
            (2 * original_window_size[1] - 1) *
            (2 * original_window_size[2] - 1)
        )
        new_entries = (
            (2 * new_window_size[0] - 1) *
            (2 * new_window_size[1] - 1) *
            (2 * new_window_size[2] - 1)
        )

        print(f"Original bias table shape for key {key}: {original_bias_table.shape}")
        print(f"Original entries: {original_entries}, New entries: {new_entries}")

        # Reshape to 3D grid (T, H, W)
        original_bias_table = original_bias_table.view(
            2 * original_window_size[0] - 1,
            2 * original_window_size[1] - 1,
            2 * original_window_size[2] - 1,
            num_heads
        )
        print(f"Reshaped bias table to grid: {original_bias_table.shape}")

        # Trim temporal dimension
        new_bias_table = original_bias_table[
            : (2 * new_window_size[0] - 1),  # Trim T
            : (2 * new_window_size[1] - 1),  # Trim H
            : (2 * new_window_size[2] - 1),  # Trim W
            :
        ]
        print(f"Trimmed bias table to new size: {new_bias_table.shape}")

        # Flatten back to 2D
        new_bias_table = new_bias_table.view(-1, num_heads)
        print(f"Flattened bias table back to 2D: {new_bias_table.shape}")

        checkpoint[key] = new_bias_table

    # Adapt relative_position_index
    keys_to_remove = [key for key in checkpoint.keys() if 'relative_position_index' in key]

    # Remove those keys from the checkpoint
    for key in keys_to_remove:
        del checkpoint[key]
        #print(f"Removed {key}")

    return checkpoint

#def adapt_temporal(checkpoint, old_temporal_size=6, new_temporal_size=2):
    # Check for relative position bias key in the checkpoint
#    for key in checkpoint.keys():
#        if 'relative_position_bias_table' in key:
#            relative_bias = checkpoint[key]

            # Calculate dimensions for the new temporal size
            # We assume the spatial window size (H, W) remains the same, only T changes.
#            spatial_size = int((relative_bias.shape[0] // (2 * old_temporal_size - 1)) ** (1 / 2))

            # Reshape to separate temporal, height, and width dimensions
 #           relative_bias = relative_bias.view(
 #               2 * old_temporal_size - 1,
 ##               2 * spatial_size - 1,
 #               2 * spatial_size - 1,
 #               -1  # number of heads
 #           )

            # Slice the temporal dimension to keep only the center around T=2
 #           new_bias = relative_bias[old_temporal_size - new_temporal_size : old_temporal_size + new_temporal_size - 1, :, :, :]
 #
 #           # Flatten back to the original shape
 #           checkpoint[key] = new_bias.view(-1, relative_bias.shape[-1])
 #           print(f"Adapted {key} to new temporal size {new_temporal_size}")

 #   return checkpoint


def create_model(args):
    model = swin_mae.__dict__['swin_mae'](img_size=args.img_size, patch_size=args.patch_size, in_chans=args.in_chans, depths=args.depths, num_heads=args.num_heads, window_size=args.window_size, mask_ratio=args.mask_ratio, mask_size=args.mask_size, temporal_bool=args.temporal_bool, pimask=args.pimask, high_pass=args.high_pass,include_decoder=False, temporal_size=args.temporal_size)

    if args.resume and not args.resume_head:
        checkpoint = torch.load(args.resume, map_location='cpu')
        adapted_checkpoint = adapt_channels(checkpoint['model'], args.in_chans)
        adapted_checkpoint = adapt_relative_position(checkpoint['model'], (6,8,8), (args.temporal_size,8,8))
        model.load_state_dict(checkpoint['model'], strict=False)
        if dist.get_rank() == 0:
            print(f"Loaded pre-trained model from {args.resume}")

    model = head.FineTuneDecoder(model, args)

    # resume_head code in main loop

    return args, model