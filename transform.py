"""
Augmentation transforms for robust OCR training, including an emboss/bevel effect.
Handles aspect-ratio preserving resize and robust photometric/geometric augmentations.
"""

import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Image dimensions (consistent with trainnew.py)
IMG_H = 64
MAX_W = 320

def keep_ratio_resize_pad(image, **kwargs):
    """
    Resize image to target height while preserving aspect ratio,
    then pad width to MAX_W with white.
    """
    h, w = image.shape[:2]
    scale = IMG_H / h
    new_w = min(int(w * scale), MAX_W)
    resized = cv2.resize(image, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    
    if len(image.shape) == 2: # Grayscale
        pad = np.full((IMG_H, MAX_W), 255, dtype=resized.dtype)
    else: # Color
        pad = np.full((IMG_H, MAX_W, image.shape[2]), 255, dtype=resized.dtype)
    
    pad[:, :new_w] = resized
    return pad

def rand_gray_bg(image, **kw):
    """
    Recolors a near-white background to a random mid-gray color.
    This helps the model generalize to text on non-white backgrounds.
    """
    bg = np.random.randint(180, 230)  # Lighter gray, closer to white (255)
    mask = (image > 245)  # treat near-white as background
    out = image.copy()
    out[mask] = bg
    return out

def get_train_transforms():
    """
    Returns Albumentations compose for training with the new robust augmentations.
    """
    return A.Compose([
        # Always convert to grayscale first
        A.Lambda(image=lambda x, **kwargs: x if x.ndim == 2 else cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)),

        # Photometric augmentations
        A.Lambda(image=rand_gray_bg, p=0.05),
        A.Emboss(alpha=(0.3, 0.7), strength=(0.2, 0.4), p=0.3),
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.9, 1.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),

        # Blur augmentations
        A.OneOf([
            A.MotionBlur(blur_limit=2, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 20.0), p=0.35),
        
        # Artifacts
        A.ImageCompression(quality_min=60, quality_max=90, p=0.4),
        
        # Other
        A.CoarseDropout(min_holes=2, max_holes=4, min_height=2, max_height=10, min_width=2, max_width=20, fill_value=0, p=0.2), # Black dropout

        # Crucial step: resize and pad, applied after other augmentations
        A.Lambda(image=keep_ratio_resize_pad),
        
        # Normalize and convert to tensor. Normalizes to [-1, 1] range.
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Returns Albumentations compose for validation/testing.
    Minimal transforms: grayscale, resize+pad, normalize, tensor conversion.
    """
    return A.Compose([
        A.Lambda(image=lambda x, **kwargs: x if x.ndim == 2 else cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)),
        A.Lambda(image=keep_ratio_resize_pad),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

def pil_to_numpy(pil_image):
    """Convert PIL Image to numpy array."""
    return np.array(pil_image)

def visualize_transforms(image_path, output_dir):
    """
    Loads an image, applies each augmentation transform individually,
    and saves the output to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Base transforms that are always applied after augmentation
    base_transform = A.Compose([
        A.Lambda(image=lambda x, **kwargs: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.ndim == 3 else x),
        A.Lambda(image=keep_ratio_resize_pad),
    ])

    transforms_to_visualize = [
        # Photometric
        (A.Lambda(image=rand_gray_bg, p=1.0), "01_rand_gray_bg"),
        (A.Emboss(alpha=(0.7, 0.7), strength=(0.8, 0.8), p=1.0), "02_emboss"),
        (A.Sharpen(alpha=(0.15, 0.15), lightness=(1.1, 1.1), p=1.0), "03_sharpen"),
        (A.RandomBrightnessContrast(brightness_limit=(0.6, 0.6), contrast_limit=(0.6, 0.6), p=1.0), "04_brightness_contrast"),
        # Blur
        (A.MotionBlur(blur_limit=(7, 7), p=1.0), "05_blur_motion"),
        (A.GaussianBlur(blur_limit=(7, 7), p=1.0), "06_blur_gaussian"),
        # Noise
        (A.GaussNoise(var_limit=(50.0, 50.0), p=1.0), "07_noise_gauss"),
        # Artifacts
        (A.ImageCompression(quality_lower=45, quality_upper=50, p=1.0), "08_compression"),
        # Other
        (A.CoarseDropout(max_holes=8, max_height=10, max_width=20, fill_value=0, p=1.0), "09_coarse_dropout_black"),
    ]

    print(f"Saving augmented images to '{output_dir}'...")
    
    # Save base transformed image
    base_image = base_transform(image=image)['image']
    cv2.imwrite(os.path.join(output_dir, "00_base_resized_padded.png"), base_image)

    for transform, name in transforms_to_visualize:
        pipeline = A.Compose([transform, base_transform])
        augmented_image = pipeline(image=image)['image']
        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), augmented_image)
        
    print("Visualization complete.")

if __name__ == '__main__':
    # This block will run when you execute `python transform.py`
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    INPUT_IMAGE = os.path.join(PROJECT_ROOT, "dataset/word_images/452.png")
    OUTPUT_DIRECTORY = "zebra"
    
    visualize_transforms(INPUT_IMAGE, OUTPUT_DIRECTORY)
