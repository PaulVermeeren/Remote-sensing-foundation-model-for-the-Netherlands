import os
import numpy as np
import torch
import torch.nn.functional as F

import tifffile as tiff
from torch.utils.data import Dataset

from .augmentations import apply_augmentations

class CustomDataLoader(Dataset):
    def __init__(self, root_dir, num_augmentations=1, temporal_size=1, in_chans=3):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'labels')
        self.num_augmentations = num_augmentations
        self.temporal_size = temporal_size
        self.in_chans = in_chans
        self.samples = self.make_dataset()

    def make_dataset(self):
        samples = []
        image_files = sorted(os.listdir(self.image_dir))
        
        for i, image_file in enumerate(image_files):
            if image_file.endswith('.tif'):
                image_path = os.path.join(self.image_dir, image_file)
                mask_path = os.path.join(self.mask_dir, image_file)
                
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    # Add original sample: A, B, and the mask (zeros for A, label for B)
                    samples.append((image_path, mask_path, False))
                    
                    # Add augmented samples if applicable
                    for _ in range(self.num_augmentations):
                        samples.append((image_path, mask_path, True))
        
        return samples
        
    def translate_mask(self, img):
        mapping = {
            (255, 255, 255): 0,  # Impervious surfaces
            (0, 0, 255): 1,      # Buildings
            (0, 255, 255): 2,    # Low vegetation
            (0, 255, 0): 3,      # Trees
            (255, 255, 0): 4,    # Cars
            (255, 0, 0): 5       # Clutter/background
        }
        
        height, width, _ = img.shape
        
        label_image = np.zeros((height, width), dtype=np.uint8)
        
        for rgb_value, class_label in mapping.items():
            mask = np.all(img == rgb_value, axis=-1)
            label_image[mask] = class_label
        
        return label_image

    def __getitem__(self, index):
        img_path, mask_path, augment = self.samples[index]
        
        # Load and process image
        img = tiff.imread(img_path)
        mask = tiff.imread(mask_path)
        
        # Normalize using percentiles
        min_val = np.percentile(img, 2, axis=(0, 1))
        max_val = np.percentile(img, 98, axis=(0, 1))
        img = np.clip(img, min_val, max_val)
        img = (img - min_val) / (max_val - min_val + 1e-06)
        
        # (H, W, C) to (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        
        # add T=1 dimension
        img = img.unsqueeze(0)
        
        #img = F.interpolate(img, size=(3072, 3072), mode='bilinear', align_corners=False)
        
        # translate mask to class mapping
        mask = self.translate_mask(mask)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add channel dimension and temporal
        
        if augment:
            img, mask = apply_augmentations(img, mask)
            
        mask.squeeze(0)
        
        return img, mask

    def __len__(self):
        return len(self.samples)