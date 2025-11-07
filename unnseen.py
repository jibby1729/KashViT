import torch
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# Import necessary components from our existing scripts
from train import SimpleViT, load_char_dict, decode_output
from transform import get_val_transforms, pil_to_numpy

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
# Directory with images to predict
UNSEEN_DIR = os.path.join(PROJECT_ROOT, "unseen_tests")
# Path to the character dictionary
DICT_FILE = os.path.join(PROJECT_ROOT, "dict/koashurkhat_dict.txt")

# --- IMPORTANT: Update this to the path of your best trained model ---
# For example: model_checkpoints/epoch_21.pth
MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, "model_checkpoints/epoch_382.pth") 

def predict_unseen(model, image_dir, char_list, transform, device):
    """
    Predicts text for all images in a given directory.
    """
    model.eval()
    
    # Find all common image file types
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in '{image_dir}'")
        return

    print(f"\nFound {len(image_files)} images to predict in '{image_dir}'")
    print("-" * 50)
    
    with torch.no_grad():
        for filename in image_files:
            image_path = os.path.join(image_dir, filename)
            
            try:
                # Load image using PIL (robust) and convert to grayscale numpy array
                img = Image.open(image_path).convert('L')
                img_np = pil_to_numpy(img)
                
                # Apply the same validation transforms used during training
                transformed = transform(image=img_np)
                image_tensor = transformed['image']
                
                # Add batch dimension (B, C, H, W) and move to the correct device
                image_tensor = image_tensor.unsqueeze(0).to(device)
                
                # Get model prediction
                outputs = model(image_tensor).cpu()
                
                # Decode the output sequence to text
                decoded_preds = decode_output(outputs, char_list)
                pred_text = decoded_preds[0] if decoded_preds else ""
                
                print(f"File: {filename:<25} | Prediction: \"{pred_text}\"")

            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")

def main():
    if 'epoch_X.pth' in MODEL_CHECKPOINT or not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Checkpoint not found at '{MODEL_CHECKPOINT}'")
        print("Please update the MODEL_CHECKPOINT variable in unseen.py to point to a valid trained model (e.g., model_checkpoints/epoch_21.pth).")
        return
        
    if not os.path.isdir(UNSEEN_DIR):
        print(f"Error: Directory with unseen images not found at '{UNSEEN_DIR}'")
        return

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_CHECKPOINT}")
    
    # Load character dictionary
    char_dict, char_list, num_classes, unk_id = load_char_dict(DICT_FILE)
    print(f"Loaded {num_classes-1} characters (plus blank token)")
    
    # Load model from checkpoint
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model = SimpleViT(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get validation transforms (no augmentation)
    val_transform = get_val_transforms()
                            
    # Run prediction on the unseen images
    predict_unseen(model, UNSEEN_DIR, char_list, val_transform, DEVICE)

if __name__ == "__main__":
    main()
