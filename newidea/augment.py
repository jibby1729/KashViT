"""
Augmentation transforms for robust OCR training using Albumentations.
Handles aspect-ratio preserving resize and robust photometric/geometric augmentations.
"""

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Image dimensions
IMG_H = 64  # Height (increased from 32 for better vertical resolution)
MAX_W = 320  # Maximum width (padded)


def keep_ratio_resize_pad(image, **kwargs):
    """
    Resize image to target height while preserving aspect ratio,
    then pad width to MAX_W with white.
    
    Args:
        image: numpy array of shape (H, W) or (H, W, C)
    
    Returns:
        numpy array of shape (IMG_H, MAX_W) or (IMG_H, MAX_W, C)
    """
    h, w = image.shape[:2]
    
    # Calculate scale to fit height
    scale = IMG_H / h
    
    # Calculate new width (don't exceed MAX_W)
    new_w = min(int(w * scale), MAX_W)
    
    # Resize image
    resized = cv2.resize(image, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    
    # Create white padding
    if len(image.shape) == 2:
        # Grayscale
        pad = np.full((IMG_H, MAX_W), 255, dtype=resized.dtype)
    else:
        # Color (though we'll convert to grayscale anyway)
        pad = np.full((IMG_H, MAX_W, image.shape[2]), 255, dtype=resized.dtype)
    
    # Place resized image in padded canvas
    pad[:, :new_w] = resized
    
    return pad


def get_train_transforms():
    """
    Returns Albumentations compose for training with robust augmentations.
    The pipeline is structured to apply augmentations BEFORE resizing and padding.
    """
    # Step 1: Augmentations applied to the original image content
    pre_resize_transforms = A.Compose([
        # Photometric augmentations
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
        ], p=0.5),
        
        # Blur augmentations
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        
        # Noise
        A.GaussNoise(variance_limit=(20.0, 50.0), p=0.5),
        
        # Artifacts
        A.ImageCompression(quality_min=30, quality_max=60, p=0.5),
        
        # Geometric (Perspective only, Affine removed)
        A.Perspective(scale=(0.04, 0.08), p=0.2),
        
        # Ink breaks
        A.CoarseDropout(min_holes=2, max_holes=4, min_height=2, max_height=5, min_width=2, max_width=20, p=0.1),
    ])

    # Step 2: Final pipeline that includes resizing and tensor conversion
    return A.Compose([
        # Always convert to grayscale first
        A.Lambda(image=lambda x, **kwargs: x if x.ndim == 2 else cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)),
        
        # Apply the pre-resize augmentations
        pre_resize_transforms,
        
        # Apply resize and padding as the last step before tensor conversion
        A.Lambda(image=keep_ratio_resize_pad),
        
        # Convert to tensor, but DO NOT normalize yet
        ToTensorV2(p=1.0),
    ])


def get_val_transforms():
    """
    Returns Albumentations compose for validation/testing.
    Minimal transforms: grayscale conversion, resize+pad, tensor conversion.
    No augmentation applied.
    """
    return A.Compose([
        # Convert to grayscale if needed
        A.Lambda(image=lambda x, **kwargs: x if x.ndim == 2 else cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)),
        
        # Resize with aspect ratio preservation and padding
        A.Lambda(image=keep_ratio_resize_pad),
        
        # Convert to tensor, but DO NOT normalize
        ToTensorV2(p=1.0),
    ])


def pil_to_numpy(pil_image):
    """Convert PIL Image to numpy array."""
    return np.array(pil_image)


def numpy_to_pil(numpy_array):
    """Convert numpy array to PIL Image."""
    return Image.fromarray(numpy_array)

