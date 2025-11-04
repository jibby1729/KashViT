"""
Script to visualize the effect of individual augmentation transforms.

Loads a sample image and applies each augmentation from the training pipeline
one by one, saving the result to the 'outputs' directory. This helps in
understanding and debugging the augmentation process.
"""

import os
import cv2
import albumentations as A
from augment import keep_ratio_resize_pad

def visualize_transforms(image_path, output_dir):
    """
    Loads an image, applies a series of transforms individually,
    and saves the output of each.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the image using OpenCV
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        # Albumentations expects RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"Original image shape: {image.shape}")

    # --- 1. Base (deterministic) transforms ---
    # These are always applied before any random augmentation.
    base_transform = A.Compose([
        A.Lambda(image=lambda x, **kwargs: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) if x.ndim == 3 else x),
        A.Lambda(image=keep_ratio_resize_pad),
    ])
    
    base_image = base_transform(image=image)['image']
    base_image_path = os.path.join(output_dir, "00_base_resized_padded.png")
    cv2.imwrite(base_image_path, base_image)
    print(f"Saved base transformed image to {base_image_path}")

    # --- 2. Individual augmentations for visualization ---
    # We define each transform with stronger, non-random settings to see its effect clearly.
    transforms_to_visualize = [
        # Photometric
        (A.CLAHE(clip_limit=8.0, tile_grid_size=(4, 4), p=1.0), "01_photometric_clahe_strong"),
        (A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=1.0), "02_photometric_brightness_contrast_strong"),
        (A.RandomGamma(gamma_limit=(40, 160), p=1.0), "03_photometric_gamma_strong"),
        # Blur
        (A.MotionBlur(blur_limit=7, p=1.0), "04_blur_motion"),
        (A.GaussianBlur(blur_limit=(3, 7), p=1.0), "05_blur_gaussian"),
        # Noise
        (A.GaussNoise(var_limit=(20.0, 50.0), p=1.0), "06_noise_gauss"),
        # Artifacts
        (A.ImageCompression(quality_lower=30, quality_upper=50, p=1.0), "07_artifact_compression"),
        # Geometric (Affine removed)
        (A.Perspective(scale=0.08, p=1.0), "08_geometric_perspective_strong"),
        # Other
        (A.CoarseDropout(max_holes=8, max_height=10, max_width=20, fill_value=255, p=1.0), "09_other_coarse_dropout_white"),
        (A.CoarseDropout(max_holes=8, max_height=10, max_width=20, fill_value=0, p=1.0), "10_other_coarse_dropout_black"),
    ]

    count = 0
    for transform, name in transforms_to_visualize:
        # Pipeline: Augment the original image first, THEN apply base transforms (resize/pad)
        pipeline = A.Compose([
            transform,
            base_transform
        ])
        
        # Apply the transform
        augmented_image = pipeline(image=image)['image']
        
        # Construct filename
        filename = f"{name}.png"
        save_path = os.path.join(output_dir, filename)
        
        # Save the resulting image
        cv2.imwrite(save_path, augmented_image)
        count += 1

    print(f"\nSuccessfully saved {count} individually augmented images to '{output_dir}'")

if __name__ == "__main__":
    # Get the absolute path to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the input image, now an absolute path
    INPUT_IMAGE = os.path.join(PROJECT_ROOT, "unseen_tests/59.png")
    # Directory to save the output images, relative to the script's location
    OUTPUT_DIRECTORY = "outputs"
    
    visualize_transforms(INPUT_IMAGE, OUTPUT_DIRECTORY)
