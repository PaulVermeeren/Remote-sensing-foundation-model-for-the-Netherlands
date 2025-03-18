import sys
import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import json
import tifffile as tiff
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loader import CustomDataLoader
from config import config
import loss
from misc import get_args_parser, create_model

def visualize_vegetatie(images, masks, predictions, index, checkpoint):
    CLASS_NAMES = [
        "Background", 'Water', 'Hard surface', 'Grassland', 'Reed', 'Forest', 'Thicket'
    ]

    ground_truth_cmap = ListedColormap([
        '#FFFFFF',  # 0: Background (Transparent)
        '#3498DB',  # 1: Water (Bright Blue)
        '#95A5A6',  # 2: Verhard oppervlakte (Cool Gray)
        '#2ECC71',  # 3: Gras & Akker (Bright Green)
        '#F1C40F',  # 4: Riet & Ruigte (Bright Yellow)
        '#16A085',  # 5: Bos (Emerald Green)
        '#9B59B6',  # 6: Struweel (Amethyst Purple)
    ])

    custom_cmap = ListedColormap([
        '#3498DB',  # 1: Water (Bright Blue)
        '#95A5A6',  # 2: Verhard oppervlakte (Cool Gray)
        '#2ECC71',  # 3: Gras & Akker (Bright Green)
        '#F1C40F',  # 4: Riet & Ruigte (Bright Yellow)
        '#16A085',  # 5: Bos (Emerald Green)
        '#9B59B6',  # 6: Struweel (Amethyst Purple)
    ])

    fig = plt.figure(figsize=(18, 8))  # Increased height for legend
    gs = GridSpec(2, 3, height_ratios=[4, 1])  # 2 rows, 3 columns, with space for legend

    # Determine the unique classes in masks and predictions
    unique_classes = torch.unique(torch.cat((masks.view(-1), predictions.view(-1))))
    num_classes = len(unique_classes)


    # Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    input_image = images[0, :3].permute(1, 2, 0).numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize to [0, 1]
    ax1.imshow(input_image)
    ax1.set_title("Input Image", fontsize=26)
    ax1.axis('off')

    # Predicted Mask
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(predictions[0].float(), cmap=custom_cmap, vmin=0, vmax=num_classes-1)
    ax2.set_title("Predicted Mask", fontsize=26)
    ax2.axis('off')

    # Ground Truth Mask
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(masks[0].squeeze().float(), cmap=ground_truth_cmap, vmin=0, vmax=num_classes-1)
    ax3.set_title("Ground Truth Mask", fontsize=26)
    ax3.axis('off')

    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=ground_truth_cmap(i / (num_classes - 1)), edgecolor='none') for i in range(len(CLASS_NAMES))]
    fig.legend(legend_elements, CLASS_NAMES, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.05), fontsize=26)

    plt.tight_layout()
    plt.savefig(f'{index}_{checkpoint}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def visualize_vegetatie_temp(images, masks, predictions, original_output, index, n):
    CLASS_NAMES = [
        "Background", 'Water', 'Hard surface', 'Grassland', 'Reed', 'Forest', 'Thicket'
    ]

    ground_truth_cmap = ListedColormap([
        '#FFFFFF',  # 0: Background (Transparent)
        '#3498DB',  # 1: Water (Bright Blue)
        '#95A5A6',  # 2: Verhard oppervlakte (Cool Gray)
        '#2ECC71',  # 3: Gras & Akker (Bright Green)
        '#F1C40F',  # 4: Riet & Ruigte (Bright Yellow)
        '#16A085',  # 5: Bos (Emerald Green)
        '#9B59B6',  # 6: Struweel (Amethyst Purple)
    ])

    custom_cmap = ListedColormap([
        '#3498DB',  # 1: Water (Bright Blue)
        '#95A5A6',  # 2: Verhard oppervlakte (Cool Gray)
        '#2ECC71',  # 3: Gras & Akker (Bright Green)
        '#F1C40F',  # 4: Riet & Ruigte (Bright Yellow)
        '#16A085',  # 5: Bos (Emerald Green)
        '#9B59B6',  # 6: Struweel (Amethyst Purple)
    ])

    plt.rcParams['figure.figsize'] = [20, 28]  # Adjusted for 7 rows
    plt.rcParams['font.size'] = 18  # Set global font size to 18

    for i in range(6):  # Loop through all 6 images
        # Input Image
        plt.subplot(7, 3, 1 + i * 3)
        show_image(images[i, :3].permute(1, 2, 0), title=f"Input Image {i+1}", cmap=custom_cmap)

        # Predicted Mask
        plt.subplot(7, 3, 2 + i * 3)
        show_image(predictions[i].float(), title=f"Predicted Mask {i+1}", is_mask=True, cmap=custom_cmap)
        cbar = plt.colorbar(ticks=range(6))
        cbar.set_ticklabels(CLASS_NAMES[1:])
        cbar.ax.tick_params(labelsize=18)

        # Ground Truth Mask
        plt.subplot(7, 3, 3 + i * 3)
        show_image(masks[i].squeeze().float(), title=f"Ground Truth Mask {i+1}", is_mask=True, cmap=ground_truth_cmap)
        cbar = plt.colorbar(ticks=range(7), label='Class' if i == 5 else '')
        cbar.set_ticklabels(CLASS_NAMES)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label('Class', fontsize=18)

    # Aggregate predictions
    aggregate_pred = aggregate_temporal(original_output)[0].argmax(dim=0)

    # Plot aggregate prediction
    plt.subplot(7, 3, 20)
    show_image(aggregate_pred.float(), title="Aggregate Prediction (Smoothed)", is_mask=True, cmap=custom_cmap)
    cbar = plt.colorbar(ticks=range(6))
    cbar.set_ticklabels(CLASS_NAMES[1:])
    cbar.ax.tick_params(labelsize=18)

    # Plot aggregate ground truth
    plt.subplot(7, 3, 21)
    show_image(masks.float()[0][0], title="Aggregate Ground Truth", is_mask=True, cmap=ground_truth_cmap)
    cbar = plt.colorbar(ticks=range(7), label='Class')
    cbar.set_ticklabels(CLASS_NAMES)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Class', fontsize=18)

    plt.tight_layout()
    plt.savefig(f'{index}_{n}_segmentation_plot.png', bbox_inches='tight')
    plt.show()


def visualize_potsdam(images, masks, predictions, index, checkpoint):
    CLASS_NAMES = ["Impervious surfaces", "Buildings", "Low vegetation", "Trees", "Cars","Clutter/background"]

    custom_cmap = ListedColormap([
        '#3498DB',  # 1: Water (Bright Blue)
        '#95A5A6',  # 2: Verhard oppervlakte (Cool Gray)
        '#2ECC71',  # 3: Gras & Akker (Bright Green)
        '#F1C40F',  # 4: Riet & Ruigte (Bright Yellow)
        '#16A085',  # 5: Bos (Emerald Green)
        '#9B59B6',  # 6: Struweel (Amethyst Purple)
    ])

    fig = plt.figure(figsize=(18, 8))  # Increased height for legend
    gs = GridSpec(2, 3, height_ratios=[4, 1])  # 2 rows, 3 columns, with space for legend

    # Determine the unique classes in masks and predictions
    unique_classes = torch.unique(torch.cat((masks.view(-1), predictions.view(-1))))
    num_classes = len(unique_classes)


    # Input Image
    ax1 = fig.add_subplot(gs[0, 0])
    input_image = images[0, :3].permute(1, 2, 0).numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize to [0, 1]
    ax1.imshow(input_image)
    ax1.set_title("Input Image", fontsize=26)
    ax1.axis('off')

    # Predicted Mask
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(predictions[0].float(), cmap=custom_cmap, vmin=0, vmax=num_classes)
    ax2.set_title("Predicted Mask", fontsize=26)
    ax2.axis('off')

    # Ground Truth Mask
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(masks[0].squeeze().float(), cmap=custom_cmap, vmin=0, vmax=num_classes)
    ax3.set_title("Ground Truth Mask", fontsize=26)
    ax3.axis('off')

    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=custom_cmap(i / (num_classes)), edgecolor='none') for i in range(len(CLASS_NAMES))]
    fig.legend(legend_elements, CLASS_NAMES, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.05), fontsize=26)

    plt.tight_layout()
    plt.savefig(f'{index}_{checkpoint}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def visualize_levircd(images, masks, predictions, index, checkpoint):
    CLASS_NAMES = [
        "Background", 'Change'
    ]

    custom_cmap = ListedColormap([
        '#000000',  # 0: Background (Transparent)
        '#D4E2EC',  # 1: Change (White)
    ])

    fig = plt.figure(figsize=(24, 10))  # Adjusted for more images
    gs = GridSpec(2, 4, height_ratios=[4, 1])  # 2 rows, 4 columns for A, B, masks, and predictions

    # Input Images
    ax1 = fig.add_subplot(gs[0, 0])
    input_image_A = images[0].permute(1, 2, 0).numpy()
    ax1.imshow(input_image_A)
    ax1.set_title("Input Image A (Before Change)", fontsize=26)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    input_image_B = images[1].permute(1, 2, 0).numpy()
    ax2.imshow(input_image_B)
    ax2.set_title("Input Image B (After Change)", fontsize=26)
    ax2.axis('off')

    # Predicted Mask
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(predictions.float(), cmap=custom_cmap, vmin=0, vmax=len(CLASS_NAMES)-1)
    ax3.set_title("Predicted Mask", fontsize=26)
    ax3.axis('off')

    # Ground Truth Mask
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(masks.squeeze().float(), cmap=custom_cmap, vmin=0, vmax=len(CLASS_NAMES)-1)
    ax4.set_title("Ground Truth Mask", fontsize=26)
    ax4.axis('off')

    # Create legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=custom_cmap(i / (len(CLASS_NAMES) - 1)), edgecolor='none') for i in range(len(CLASS_NAMES))]
    fig.legend(legend_elements, CLASS_NAMES, loc='lower center', ncol=len(CLASS_NAMES), bbox_to_anchor=(0.5, 0.05), fontsize=26)

    plt.tight_layout()
    plt.savefig(f'{index}_{checkpoint}.png', bbox_inches='tight') #, dpi=300)
    plt.close(fig)



def check_required_args(args):
    missing_args = []

    if args.checkpoint == "":
        missing_args.append("--checkpoint argument is required. [filename in ../output_dir/]")
    if args.dataset == "":
        missing_args.append("--dataset argument is required. [resisc45, ucmerced, ..., potsdam]")

    if missing_args:
        for error in missing_args:
            print(f"Error: {error}")
        sys.exit(1)



if __name__ == '__main__':
    args = get_args_parser().parse_args()
    check_required_args(args)
    args = config(args)
    args.resume = "" # remove pretrained model
    file_path = "../output_dir/"+args.checkpoint

    # Dataset
    dataset_test = CustomDataLoader(args, path="../"+args.test_path, num_augmentations=0, prepend="../")

    # create model and load checkpoint
    args, model = create_model(args)
    checkpoint = torch.load(file_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f"Loaded pre-trained model from {file_path}")

    index = [7,9,12,14,17]
    for i in index:
        images, masks = dataset_test[i]

        # run one batch
        x = images.unsqueeze(0) # add batch dimension
        with torch.no_grad():
            outputs = model(x.float())


        if args.dataset == "vegetatie":
            visualize_vegetatie(images, masks, outputs.argmax(dim=2).squeeze(0), i, args.checkpoint)

        elif args.dataset == "potsdam":
            visualize_potsdam(images, masks, outputs.argmax(dim=2).squeeze(0), i, args.checkpoint)

        elif args.dataset == "levircd":
            visualize_levircd(images, masks, outputs.argmax(dim=1).squeeze(0), i, args.checkpoint)

