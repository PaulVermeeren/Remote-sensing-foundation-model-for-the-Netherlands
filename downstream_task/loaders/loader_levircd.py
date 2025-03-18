import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import apply_augmentations

class CustomDataLoader(Dataset):
    def __init__(self, root_dir, num_augmentations=1, temporal_size=2, in_chans=3):
        self.root_dir = root_dir
        self.a_dir = os.path.join(root_dir, 'A')
        self.b_dir = os.path.join(root_dir, 'B')
        self.label_dir = os.path.join(root_dir, 'label')
        self.num_augmentations = num_augmentations
        self.samples = self.make_dataset()
        self.temporal_size = temporal_size
        self.in_chans = in_chans

    def make_dataset(self):
        samples = []
        a_files = sorted(os.listdir(self.a_dir))
        
        for i, a_file in enumerate(a_files):
            if a_file.endswith('.png'):
                a_path = os.path.join(self.a_dir, a_file)
                b_file = os.path.splitext(a_file)[0] + '.png'
                b_path = os.path.join(self.b_dir, b_file)
                label_file = os.path.splitext(a_file)[0] + '.png'
                label_path = os.path.join(self.label_dir, label_file)
                
                if os.path.exists(b_path) and os.path.exists(label_path):
                    # Add original sample: A, B, and the mask (zeros for A, label for B)
                    samples.append((a_path, b_path, label_path, False))
                    
                    # Add augmented samples if applicable
                    for _ in range(self.num_augmentations):
                        samples.append((a_path, b_path, label_path, True))
        
        return samples

    def __getitem__(self, index):
        a_path, b_path, label_path, augment = self.samples[index]
        
        # Load and process image A
        a_image = Image.open(a_path).convert('RGB')
        a_image = np.array(a_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load and process image B
        b_image = Image.open(b_path).convert('RGB')
        b_image = np.array(b_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Normalize both images using percentiles
        def normalize(img):
            min_val = np.percentile(img, 2, axis=(0, 1))
            max_val = np.percentile(img, 98, axis=(0, 1))
            img = np.clip(img, min_val, max_val)
            return (img - min_val) / (max_val - min_val + 1e-06)
    
        a_image = normalize(a_image)
        b_image = normalize(b_image)
        
        # Ensure both images have the required channels (pad if necessary)
        if a_image.shape[2] < self.in_chans:
            padded_a_image = np.zeros((a_image.shape[0], a_image.shape[1], self.in_chans), dtype=a_image.dtype)
            padded_a_image[:, :, :a_image.shape[2]] = a_image
            a_image = padded_a_image
        
        if b_image.shape[2] < self.in_chans:
            padded_b_image = np.zeros((b_image.shape[0], b_image.shape[1], self.in_chans), dtype=b_image.dtype)
            padded_b_image[:, :, :b_image.shape[2]] = b_image
            b_image = padded_b_image
        
        # Convert images to (C, H, W)
        a_image = np.transpose(a_image, (2, 0, 1))
        b_image = np.transpose(b_image, (2, 0, 1))
        
        a_image = torch.tensor(a_image, dtype=torch.float32)
        b_image = torch.tensor(b_image, dtype=torch.float32)
        
        # Load and process mask (zero mask for A, label mask for B)
        label_mask = np.array(Image.open(label_path).convert('L'), dtype=np.float32) / 255.0  # Load label for image B
        label_mask = torch.tensor(label_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add Temporal and channel dimension
        
        image_stack = torch.stack((a_image, b_image), dim=0)  # Shape: (4, C, H, W)
        
        if augment:
            image_stack, label_mask = apply_augmentations(image_stack, label_mask)
        
        return image_stack, label_mask

    def __len__(self):
        return len(self.samples)