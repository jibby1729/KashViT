"""
Testing script for the robust OCR model on the combined clean_dataset.
"""

import torch
from torch.utils.data import DataLoader
import editdistance
from tqdm import tqdm
import os

# Import from our new training and transform scripts
from train import SimpleViT, OCRDataset, load_char_dict, collate_fn, decode_output
from transform import get_val_transforms

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Updated Paths ---
TEST_FILE = os.path.join(PROJECT_ROOT, "clean_dataset/test.txt")
DICT_FILE = os.path.join(PROJECT_ROOT, "dict/koashurkhat_dict.txt")

# --- IMPORTANT: Update this to the path of your newly trained model ---
MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, "model_checkpoints/epoch_382.pth") # Update X

def test(model, dataloader, char_list, device):
    model.eval()
    total_correct_words, total_words, total_char_dist, total_char_len = 0, 0, 0, 0
    wrong_predictions = []
    file_list = dataloader.dataset.data
    
    with torch.no_grad():
        for i, (images, labels, lengths) in enumerate(tqdm(dataloader, desc="Testing")):
            images = images.to(device)
            outputs = model(images).cpu()
            decoded_preds = decode_output(outputs, char_list)
            
            start = 0
            for j, length in enumerate(lengths):
                label_text = "".join([char_list[c-1] for c in labels[start:start+length]])
                pred_text = decoded_preds[j]
                
                original_index = i * dataloader.batch_size + j
                img_path, _ = file_list[original_index]
                img_name = os.path.basename(img_path)

                if label_text == pred_text:
                    total_correct_words += 1
                elif len(wrong_predictions) < 20:
                    wrong_predictions.append((img_name, label_text, pred_text))
                
                total_char_dist += editdistance.eval(pred_text, label_text)
                total_char_len += len(label_text)
                start += length
            total_words += len(lengths)
            
    word_accuracy = (total_correct_words / total_words) * 100
    char_error_rate = (total_char_dist / total_char_len) * 100
    return word_accuracy, char_error_rate, wrong_predictions

def main():
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Checkpoint not found at '{MODEL_CHECKPOINT}'")
        print("Please update the MODEL_CHECKPOINT variable in test.py to point to a valid trained model.")
        return

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_CHECKPOINT}")
    
    char_dict, char_list, num_classes, unk_id = load_char_dict(DICT_FILE)
    print(f"Loaded {num_classes-1} characters (plus blank token)")
    
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model = SimpleViT(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_dataset = OCRDataset(TEST_FILE, char_dict, unk_id, get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
                            
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
