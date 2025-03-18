import os
import numpy as np
import torch
import torch.nn.functional as F

import json
import tifffile as tiff
from torch.utils.data import Dataset

from .augmentations import apply_augmentations

class CustomDataLoader(Dataset):
    def __init__(self, root_dir, json_path, num_augmentations=1, maxdata=1):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'labels')
        
        self.num_augmentations = num_augmentations
        self.samples = self.make_dataset(maxdata)
        self.preprocess_samples()
        
    def make_dataset(self, maxdata):
        samples = []
        quadrants = 16  # Assuming q1 to q16
    
        # Walk through the directory
        for root, dirs, files in os.walk(self.image_dir):
            # Filter for files with the pattern xxx_(0 to 5)_q(1 to 16)
            filtered_files = [f for f in files if f.endswith('.tif')]
    
            # Group files by the common prefix (the part before '_q')
            file_groups = {}
            for filen in filtered_files:
                # Split by the underscore and quadrant information
                parts = filen.split(".")[0].split('_')
                if len(parts) >= 3:
                    prefix = '_'.join(parts[:-2])  # e.g., 'xxx'
                    index = int(parts[-2])         # e.g., 0 to 5
                    quadrant = int(parts[-1][1:])  # e.g., q1 to q16
    
                    # Ensure a dictionary structure to store files by prefix and quadrant
                    if prefix not in file_groups:
                        file_groups[prefix] = {}
    
                    if quadrant not in file_groups[prefix]:
                        file_groups[prefix][quadrant] = []
    
                    # Add the file to the correct quadrant group
                    file_groups[prefix][quadrant].append((index, os.path.join(root, filen)))
    
        # Now merge the filenames for 0 to 5 for each quadrant and append to samples
        for prefix in file_groups:
            for quadrant in range(1, quadrants + 1):
                if quadrant in file_groups[prefix]:
                    # Sort by index (0 to 5) to ensure correct order
                    quadrant_files = sorted(file_groups[prefix][quadrant], key=lambda x: x[0])
                    filenames = [f[1] for f in quadrant_files]
    
                    # Only add if we have exactly 6 files for the quadrant
                    if len(filenames) == 6:
                        # Add samples for non-augmented data
                        samples.append((filenames, False, quadrant))
                        
                        # Add augmented samples
                        for _ in range(self.num_augmentations):
                            samples.append((filenames, True, quadrant))
    
        return samples

    def __getitem__(self, index):
        filenames, augment, quadrant = self.samples[index]
        imgs = []
        
        # Load and process each temporal image (with score 0 to 5)
        for filename in filenames:
            # No need to append quadrant number manually, as it's already part of the filename
            img_path = os.path.join(self.image_dir, filename)  # Directly use filename
            
            # Load the image
            img = tiff.imread(img_path)
            img = torch.tensor(img, dtype=torch.float32)
            
            imgs.append(img)
        
        imgs = torch.stack(imgs, dim=0)  # Stack along the temporal dimension
        
        # Load and process the mask
        # Use the common part of the filename without the score (e.g., `xxx_q1.npy`)
        base_filename = filenames[0].split('_')[0]  # Extract the base filename (e.g., `xxx`)
        mask_filename = f"{base_filename}_q{quadrant}.npy"  # Create the mask filename using the quadrant only
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Load the mask from .npy file
        mask = np.load(mask_path)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        # Repeat the mask to match temporal size if necessary (if you need multiple masks for each temporal image)
        mask = mask.unsqueeze(0).unsqueeze(0).repeat(6, 1, 1, 1)
        
        # Apply augmentations if needed
        if augment:
            imgs, mask = apply_augmentations(imgs, mask)
        
        return imgs, mask

    def __len__(self):
        return len(self.samples)