"""
Testing script for the robust OCR model.
Uses the new architecture with stripe tokenization and grayscale input.
"""

import torch
from trainnew import SimpleViT, OCRDataset, load_char_dict, collate_fn, decode_output
from augment import get_val_transforms
from torch.utils.data import DataLoader
import editdistance
from tqdm import tqdm
import os

# Get the absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
TEST_FILE = os.path.join(PROJECT_ROOT, "dataset/test.txt")
DICT_FILE = os.path.join(PROJECT_ROOT, "dict/koashurkhat_dict.txt")

# IMPORTANT: Set this to the path of your saved model checkpoint
MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, "newidea/model_checkpoints_new/best_model_epoch_150.pth")  # Update X


def test(model, dataloader, char_list, device):
    """
    Test the model on test set and compute word accuracy and character error rate.
    """
    model.eval()
    total_correct_words = 0
    total_words = 0
    total_char_dist = 0
    total_char_len = 0
    wrong_predictions = []
    
    file_list = dataloader.dataset.data
    
    with torch.no_grad():
        for i, (images, labels, lengths) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = outputs.cpu()
            
            # Decode predictions and labels
            decoded_preds = decode_output(preds, char_list)
            
            start = 0
            for j, length in enumerate(lengths):
                label_text = "".join([char_list[c-1] for c in labels[start:start+length]])
                pred_text = decoded_preds[j]
                
                # Look up the image path using the index
                original_index = i * dataloader.batch_size + j
                img_path, _ = file_list[original_index]
                img_name = os.path.basename(img_path)

                # Print some examples
                if i == 0 and j < 20:  # Print first 20 examples of the first batch
                    print(f"  - Image: {img_name:<15} | Label: {label_text}, Predicted: {pred_text}")
                
                # Word Accuracy
                if label_text == pred_text:
                    total_correct_words += 1
                else:
                    # Store the first 20 wrong predictions
                    if len(wrong_predictions) < 20:
                        wrong_predictions.append((img_name, label_text, pred_text))
                
                # Character Error Rate
                total_char_dist += editdistance.eval(pred_text, label_text)
                total_char_len += len(label_text)
                
                start += length
            
            total_words += len(lengths)
            
    word_accuracy = (total_correct_words / total_words) * 100
    char_error_rate = (total_char_dist / total_char_len) * 100
    
    return word_accuracy, char_error_rate, wrong_predictions


def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_CHECKPOINT}")
    
    # Load character dictionary
    char_dict, char_list, num_classes, unk_id = load_char_dict(DICT_FILE)
    print(f"Loaded {num_classes-1} characters (plus blank token)")
    print(f"UNK token ID: {unk_id}")
    
    # Load model checkpoint
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    
    # Initialize model
    model = SimpleViT(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset and loader
    val_transform = get_val_transforms()
    test_dataset = OCRDataset(TEST_FILE, char_dict, unk_id, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=8)
                            
    print("\nStarting evaluation on the test set...")
    word_acc, cer, wrong_preds = test(model, test_loader, char_list, DEVICE)
    
    print("\n" + "="*80)
    print(" " * 25 + "First 20 Incorrect Predictions")
    print("="*80)
    print(f"{'Image':<25} | {'Ground Truth':<25} | {'Predicted':<25}")
    print("-" * 80)
    for img_name, label, pred in wrong_preds:
        print(f"{img_name:<25} | {label:<25} | {pred:<25}")
    
    print("\n" + "="*50)
    print("                Test Set Results")
    print("="*50)
    print(f"  - Word Accuracy: {word_acc:.2f}%")
    print(f"  - Character Error Rate (CER): {cer:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()

