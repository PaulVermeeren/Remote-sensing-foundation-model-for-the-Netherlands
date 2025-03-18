import random
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

def random_crop(img, mask):
    _, _, h, w = img.shape
    crop_size = random.randint(w // 2, w)  # Random crop size
    
    top = torch.randint(0, h - crop_size + 1, (1,)).item()
    left = torch.randint(0, w - crop_size + 1, (1,)).item()
    
    img_cropped = img[:, :, top:top + crop_size, left:left + crop_size]
    mask_cropped = mask[:, :, top:top + crop_size, left:left + crop_size]
    
    img_resized = F.interpolate(img_cropped, size=(w, w), mode='bilinear', align_corners=False)
    mask_resized = F.interpolate(mask_cropped, size=(w, w), mode='nearest')
    
    return img_resized, mask_resized

def random_rotation(img, mask):
    rotation = random.choice([0, 90, 180, 270])
    img_rotated = TF.rotate(img, rotation)
    mask_rotated = TF.rotate(mask, rotation)
    return img_rotated, mask_rotated

def random_flip(img, mask):
    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    else:
        img = TF.vflip(img)
        mask = TF.vflip(mask)
    return img, mask

def random_color_distortion(img, mask):
    # Apply brightness adjustment
    brightness_factor = random.uniform(0.8, 1.2)
    img = img * brightness_factor
    img = torch.clamp(img, 0.0, 1.0)  # Ensure values remain valid

    # Apply contrast adjustment
    mean_per_channel = img.mean(dim=(2, 3), keepdim=True)  # Compute mean per channel
    contrast_factor = random.uniform(0.8, 1.2)
    img = (img - mean_per_channel) * contrast_factor + mean_per_channel
    img = torch.clamp(img, 0.0, 1.0)

    # Apply saturation adjustment
    # Convert to grayscale-like by averaging over spatial dimensions, and adjust intensity
    saturation_factor = random.uniform(0.8, 1.2)
    img_mean = img.mean(dim=1, keepdim=True)  # Compute the mean intensity across channels
    img = (img - img_mean) * saturation_factor + img_mean
    img = torch.clamp(img, 0.0, 1.0)
    
    return img, mask  # Mask is not affected

def random_blur(img, mask):
    kernel_size = random.choice([3, 5])
    img = TF.gaussian_blur(img, kernel_size=kernel_size)
    return img, mask  # Mask is not affected

def random_noise(img, mask):
    noise = torch.randn_like(img) * 0.05
    img = img + noise
    img = torch.clamp(img, 0.0, 1.0)
    return img, mask  # Mask is not affected

def apply_augmentations(img, mask):
    """
    Apply the same augmentations to both image and mask.
    img: torch.Tensor of shape (temp, C, H, W)
    mask: torch.Tensor of shape (temp, 1, H, W)
    """
    # Group 1: Spatial augmentations
    spatial_augmentations = [random_crop, random_rotation, random_flip]
    chosen_spatial_aug = random.choice(spatial_augmentations)
    img, mask = chosen_spatial_aug(img, mask)
    
    # Group 2:  augmentations
    intensity_augmentations = [random_color_distortion, random_blur, random_noise]
    chosen_intensity_aug = random.choice(intensity_augmentations)
    img, mask = chosen_intensity_aug(img, mask)
    
    return img, mask