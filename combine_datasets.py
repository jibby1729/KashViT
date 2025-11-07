"""
Script to combine extraset and original datasets into clean_dataset folder.
Processes first 35k valid extraset images (inverting if needed) and all original dataset images.
Creates train.txt, val.txt, test.txt files for fast loading during training.
"""

import os
import csv
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm


def load_dictionary(dict_file):
    """
    Load character dictionary from koashurkhat_dict.txt into a set.
    
    Args:
        dict_file: Path to dictionary file
    
    Returns:
        set: Set of valid characters
    """
    char_set = set()
    try:
        with open(dict_file, 'r', encoding='utf-8') as f:
            for line in f:
                char = line.strip()
                if char:
                    char_set.add(char)
    except FileNotFoundError:
        print(f"Error: Dictionary file not found at {dict_file}")
    return char_set


def is_valid_label(text, valid_chars):
    """
    Check if all characters in text are in valid_chars set.
    
    Args:
        text: Label text to validate
        valid_chars: Set of valid characters
    
    Returns:
        bool: True if all characters are valid
    """
    if not text:
        return False
    return all(char in valid_chars for char in text)


def detect_if_needs_inversion(image):
    """
    Detects if an image has inverted colors (white text on black background)
    and needs to be inverted to match training distribution (black text on white background).
    
    Uses histogram analysis: if significantly more pixels are dark (< 128) than bright (>= 128),
    it's likely inverted.
    
    Args:
        image: numpy array image (grayscale or color)
    
    Returns:
        tuple: (needs_inversion: bool, mean_val: float, dark_ratio: float)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate mean pixel value
    mean_val = np.mean(gray)
    
    # Calculate histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    
    # Count dark pixels (0-127) vs bright pixels (128-255)
    dark_pixels = np.sum(hist[:128])
    bright_pixels = np.sum(hist[128:])
    total_pixels = dark_pixels + bright_pixels
    
    if total_pixels > 0:
        dark_ratio = dark_pixels / total_pixels
    else:
        dark_ratio = 0.0
    
    # Heuristic: if mean is low (< 100) AND dark pixels dominate (> 60%),
    # it's likely inverted (dark background with bright text)
    needs_inversion = mean_val < 100 and dark_ratio > 0.6
    
    return needs_inversion, mean_val, dark_ratio


def invert_image(image):
    """
    Invert an image: white becomes black, black becomes white.
    
    Args:
        image: numpy array image
    
    Returns:
        numpy array: Inverted image
    """
    return 255 - image


def process_extraset(extraset_csv, extraset_images_dir, output_images_dir, valid_chars, limit=35000):
    """
    Process extraset dataset: filter to first 35k valid labels, invert images if needed.
    
    Args:
        extraset_csv: Path to extraset labels.csv file
        extraset_images_dir: Directory containing extraset images
        output_images_dir: Directory to save processed images
        valid_chars: Set of valid characters
        limit: Maximum number of rows to process (default: 35000)
    
    Returns:
        list: List of tuples (image_path, label_text) for valid images
    """
    processed_data = []
    valid_count = 0
    invalid_count = 0
    inverted_count = 0
    current_idx = 0
    
    print(f"\nProcessing extraset dataset (first {limit} rows)...")
    print(f"CSV file: {extraset_csv}")
    print(f"Images directory: {extraset_images_dir}")
    
    os.makedirs(output_images_dir, exist_ok=True)
    
    try:
        with open(extraset_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            
            # Find the index of the 'text' column
            try:
                text_index = header.index('text')
                image_id_index = header.index('image_id')
            except ValueError as e:
                print(f"Error: Required column not found in CSV header: {header}")
                return processed_data, current_idx
            
            for row in tqdm(reader, desc="Processing extraset", total=limit):
                if valid_count >= limit:
                    break  # Stop after reaching limit
                
                if len(row) <= max(text_index, image_id_index):
                    continue
                
                image_id = row[image_id_index].strip()
                label_text = row[text_index].strip()
                
                # Check if label is valid
                if not is_valid_label(label_text, valid_chars):
                    invalid_count += 1
                    continue
                
                # Valid label found
                image_filename = f"{image_id}.png"
                image_path = os.path.join(extraset_images_dir, image_filename)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    invalid_count += 1
                    continue
                
                # Load image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not load image: {image_path}")
                    invalid_count += 1
                    continue
                
                # Detect if inversion is needed
                needs_inv, mean_val, dark_ratio = detect_if_needs_inversion(img)
                
                # Apply inversion if needed
                if needs_inv:
                    img = invert_image(img)
                    inverted_count += 1
                
                # Save processed image with sequential naming
                output_filename = f"{current_idx:06d}.png"
                output_path = os.path.join(output_images_dir, output_filename)
                cv2.imwrite(output_path, img)
                
                # Store data: relative path and label
                relative_path = f"clean_dataset/images/{output_filename}"
                processed_data.append((relative_path, label_text))
                
                valid_count += 1
                current_idx += 1
            
    except FileNotFoundError:
        print(f"Error: CSV file not found: {extraset_csv}")
    except Exception as e:
        print(f"Error processing extraset: {e}")
    
    print(f"\nExtraset processing complete:")
    print(f"  - Valid images processed: {valid_count}")
    print(f"  - Invalid images skipped: {invalid_count}")
    print(f"  - Images inverted: {inverted_count}")
    
    return processed_data, current_idx


def process_original_dataset(original_images_dir, original_labels_dir, output_images_dir, start_idx):
    """
    Process original dataset: copy all images to output directory.
    
    Args:
        original_images_dir: Directory containing original images (word_images)
        original_labels_dir: Directory containing original labels
        output_images_dir: Directory to save processed images
        start_idx: Starting index for sequential naming
    
    Returns:
        list: List of tuples (image_path, label_text) for all images
    """
    processed_data = []
    current_idx = start_idx
    
    print(f"\nProcessing original dataset...")
    print(f"Images directory: {original_images_dir}")
    print(f"Labels directory: {original_labels_dir}")
    
    os.makedirs(output_images_dir, exist_ok=True)
    
    # Get all PNG files
    all_images = [f for f in os.listdir(original_images_dir) if f.endswith('.png')]
    all_images.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else float('inf'))
    
    print(f"Found {len(all_images)} images to process")
    
    for image_filename in tqdm(all_images, desc="Processing original dataset"):
        image_path = os.path.join(original_images_dir, image_filename)
        
        # Get corresponding label file
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(original_labels_dir, label_filename)
        
        if not os.path.exists(label_path):
            print(f"Warning: Label file not found: {label_path}")
            continue
        
        # Read label text
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_text = f.read().strip()
        except Exception as e:
            print(f"Warning: Could not read label file {label_path}: {e}")
            continue
        
        # Copy image to output directory with sequential naming
        output_filename = f"{current_idx:06d}.png"
        output_path = os.path.join(output_images_dir, output_filename)
        
        # Copy image file
        shutil.copy2(image_path, output_path)
        
        # Store data: relative path and label
        relative_path = f"clean_dataset/images/{output_filename}"
        processed_data.append((relative_path, label_text))
        
        current_idx += 1
    
    print(f"\nOriginal dataset processing complete:")
    print(f"  - Images processed: {len(processed_data)}")
    
    return processed_data


def create_train_val_test_files(all_data, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Create train.txt, val.txt, test.txt files from combined dataset.
    
    Args:
        all_data: List of tuples (image_path, label_text)
        output_dir: Directory to save the txt files
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
    """
    print(f"\nCreating train/val/test splits...")
    
    # Shuffle data
    random.shuffle(all_data)
    
    # Calculate splits
    total = len(all_data)
    train_split = int(total * train_ratio)
    val_split = train_split + int(total * val_ratio)
    
    train_data = all_data[:train_split]
    val_data = all_data[train_split:val_split]
    test_data = all_data[val_split:]
    
    # Write files
    train_file = os.path.join(output_dir, 'train.txt')
    val_file = os.path.join(output_dir, 'val.txt')
    test_file = os.path.join(output_dir, 'test.txt')
    
    print(f"Writing train.txt ({len(train_data)} samples)...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for img_path, label in train_data:
            f.write(f"{img_path}\t{label}\n")
    
    print(f"Writing val.txt ({len(val_data)} samples)...")
    with open(val_file, 'w', encoding='utf-8') as f:
        for img_path, label in val_data:
            f.write(f"{img_path}\t{label}\n")
    
    print(f"Writing test.txt ({len(test_data)} samples)...")
    with open(test_file, 'w', encoding='utf-8') as f:
        for img_path, label in test_data:
            f.write(f"{img_path}\t{label}\n")
    
    print(f"\nSplit statistics:")
    print(f"  - Training set: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"  - Validation set: {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
    print(f"  - Test set: {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")


def main():
    """Main function to combine datasets."""
    # Setup paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Input paths
    dict_file = os.path.join(project_root, 'dict', 'koashurkhat_dict.txt')
    extraset_csv = os.path.join(project_root, 'dataset', 'extraset', 'content', 'extraset', 'labels.csv')
    extraset_images_dir = os.path.join(project_root, 'dataset', 'extraset', 'content', 'extraset', 'images')
    original_images_dir = os.path.join(project_root, 'dataset', 'word_images')
    original_labels_dir = os.path.join(project_root, 'dataset', 'labels')
    
    # Output paths
    clean_dataset_dir = os.path.join(project_root, 'clean_dataset')
    output_images_dir = os.path.join(clean_dataset_dir, 'images')
    
    print("="*70)
    print("COMBINING DATASETS")
    print("="*70)
    
    # Load dictionary
    print(f"\nLoading dictionary from: {dict_file}")
    valid_chars = load_dictionary(dict_file)
    print(f"Loaded {len(valid_chars)} valid characters")
    
    # Process extraset (first 35k valid images)
    extraset_data, next_idx = process_extraset(
        extraset_csv, 
        extraset_images_dir, 
        output_images_dir, 
        valid_chars, 
        limit=35000
    )
    
    # Process original dataset (all images)
    original_data = process_original_dataset(
        original_images_dir,
        original_labels_dir,
        output_images_dir,
        start_idx=next_idx
    )
    
    # Combine data
    all_data = extraset_data + original_data
    print(f"\nTotal combined samples: {len(all_data)}")
    
    # Create a master CSV of all labels
    write_master_csv(all_data, clean_dataset_dir)
    
    # Create train/val/test splits
    create_train_val_test_files(all_data, clean_dataset_dir)
    
    print("\n" + "="*70)
    print("DATASET COMBINATION COMPLETE!")
    print("="*70)
    print(f"Output directory: {clean_dataset_dir}")
    print(f"Images directory: {output_images_dir}")
    print(f"Train/val/test files: {clean_dataset_dir}/train.txt, val.txt, test.txt")
    print(f"Master labels file: {clean_dataset_dir}/labels.csv")


def write_master_csv(all_data, output_dir):
    """
    Writes all image paths and labels to a single master CSV file.
    
    Args:
        all_data: List of tuples (image_path, label_text)
        output_dir: Directory to save the CSV file
    """
    csv_path = os.path.join(output_dir, 'labels.csv')
    print(f"\nWriting master labels file to: {csv_path}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'text'])  # Write header
        
        # Sort data by image path for consistency before writing
        sorted_data = sorted(all_data)
        
        for img_path, label in sorted_data:
            writer.writerow([img_path, label])

if __name__ == '__main__':
    main()

