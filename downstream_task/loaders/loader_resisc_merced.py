import os
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset

from .augmentations import apply_augmentations

class CustomDataLoader(Dataset):
    def __init__(self, root_dir, interpolate=0, num_augmentations=0, maxdata=1):
        self.root_dir = root_dir
        self.interpolate = interpolate
        self.num_augmentations = num_augmentations
        
        self.classes, self.class_to_idx = self._find_classes(self.root_dir)
        self.samples = self.make_dataset(maxdata)
        
    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def make_dataset(self, maxdata):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                count = 0
                for fname in sorted(fnames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                        if count/len(fnames) > maxdata:
                            break
                        count += 1
                        
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[class_name])
                        samples.append(item)
                        
                        # Add augmented samples if applicable
                        for _ in range(self.num_augmentations):
                            samples.append((path, self.class_to_idx[class_name], True))
            
        
        return samples
    
    def __getitem__(self, index):
        if len(self.samples[index]) == 2:
            path, target = self.samples[index]
            augment = False
        else:
            path, target, augment = self.samples[index]
        
        # Load image
        image = Image.open(path).convert('RGB')
        
        min_val = np.percentile(image, 2, axis=(0, 1))
        max_val = np.percentile(image, 98, axis=(0, 1))
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val + 1e-06)
        
        # Convert image to tensor (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        target = torch.tensor(target, dtype=torch.long)
        
        # add T=1 dimension
        image = image.unsqueeze(0)
        
        # interpolate for irregular size
        if self.interpolate > 0:
            image = F.interpolate(image, size=(self.interpolate, self.interpolate), mode='bilinear', align_corners=False)
        
        # augmentation
        if augment:
            image, _ = apply_augmentations(image, image)
            
        return image, target

    def __len__(self):
        return len(self.samples)