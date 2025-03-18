import os
import numpy as np
import torch
import torch.nn.functional as F

import json
import h5py
import tifffile as tiff
from torch.utils.data import Dataset

from .augmentations import apply_augmentations


class CustomDataLoader(Dataset):
    def __init__(self, root_dir, num_augmentations=1, maxdata=1, prepend=""):
        self.json_path = root_dir
        self.prepend = prepend
        self.h5_path = prepend+'../../data/vegetatie/linkerkant_classified_grids.h5'
        self.num_augmentations = num_augmentations
        self.samples = self.make_dataset(self.json_path, maxdata)
        self.preprocess_samples()

    def make_dataset(self, filepath, maxdata):
        samples = []
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for idx, key in enumerate(data):
            if (len(samples)*6/(self.num_augmentations+1)) > maxdata and maxdata != 1:
                break
            temporal_list = data[key]
            paths = []
            for datapoint in temporal_list:
                filename = os.path.basename(datapoint.get('uri'))
                path = os.path.join(self.prepend+'../../data/vegetatie_temp/', filename)
                paths.append(path)

            if len(paths) == 6:
                samples.append((key, paths, False))

                # Add augmented samples
                for _ in range(self.num_augmentations):
                    samples.append((key, paths, True))

        return samples

    def preprocess_samples(self):
        valid_samples = []

        with h5py.File(self.h5_path, 'r') as h5_file:
            for idx, sample in enumerate(self.samples):
                key, paths, augment = sample

                # Check if all image files exist
                if all(os.path.exists(path) for path in paths):
                    # Check if corresponding h5 entry exists
                    if key in h5_file:
                        valid_samples.append(sample)
                    else:
                        print(f"Warning: No h5 entry found for key {key}")
                else:
                    print(f"Warning: Missing image file(s) for key {key}")

        self.samples = valid_samples
        print(f"Preprocessing complete. Unique {len(self.samples)/(self.num_augmentations+1)} samples before augmenting {self.num_augmentations}.")


    def __getitem__(self, index):
        key, paths, augment = self.samples[index]
        imgs = []
        augment=True
        for path in paths:
            img = tiff.imread(path)

            # pad for 4 channels
            if img.shape[2] == 4:
                padded_image = np.zeros((img.shape[0], img.shape[1], 6), dtype=img.dtype)
                padded_image[:, :, :4] = img
                img = padded_image
            #img = img[:,:,:4]

            # Normalize using percentiles
            min_val = np.percentile(img, 2, axis=(0, 1))
            max_val = np.percentile(img, 98, axis=(0, 1))
            img = np.clip(img, min_val, max_val)
            img = (img - min_val) / (max_val - min_val + 1e-06)

            # (H, W, C) to (C, H, W)
            img = np.transpose(img, (2, 0, 1))

            img = torch.tensor(img, dtype=torch.float32)

            #img = img.unsqueeze(0)  # Add batch dimension temporarily
            #img = F.interpolate(img, size=(1024, 1024), mode='bilinear', align_corners=False)
            #img = img.squeeze(0)


            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)

        # Load mask from H5 file
        with h5py.File(self.h5_path, 'r') as h5_file:
            mask = np.array(h5_file[key])


        mask = torch.tensor(mask, dtype=torch.float32)

        #mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions temporarily
        #mask = F.interpolate(mask, size=(1024, 1024), mode='nearest')
        #mask = mask.squeeze(0).squeeze(0)  # Remove temporary batch and channel dimensions

        # Repeat mask to match the number of channels in the imagemask = F.interpolate(mask.float(), size=(2048, 2048), mode='nearest')
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(6, 1, 1, 1)


        if augment:
            imgs, mask = apply_augmentations(imgs, mask)

        return imgs, mask

    def __len__(self):
        return len(self.samples)