import torch
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# Change 1: Import from the 'newidea' training and augmentation scripts
from newidea.trainnew import SimpleViT, load_char_dict, decode_output
from newidea.augment import get_val_transforms, pil_to_numpy

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Paths ---
UNSEEN_DIR = os.path.join(PROJECT_ROOT, "unseen_tests")
DICT_FILE = os.path.join(PROJECT_ROOT, "dict/koashurkhat_dict.txt")

# Change 2: Point to the model you trained with newidea
MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, "best_models/kaggletrained.pth") 

def predict_unseen(model, image_dir, char_list, transform, device):
    model.eval()
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
                img = Image.open(image_path).convert('L')
                img_np = pil_to_numpy(img)
                transformed = transform(image=img_np)
                image_tensor = transformed['image']

                # Change 3: Manually divide by 255.0 to match the 'newidea' pipeline
                image_tensor = image_tensor.float() / 255.0
                
                image_tensor = image_tensor.unsqueeze(0).to(device)
                outputs = model(image_tensor).cpu()
                decoded_preds = decode_output(outputs, char_list)
                pred_text = decoded_preds[0] if decoded_preds else ""
                
                print(f"File: {filename:<25} | Prediction: \"{pred_text}\"")

            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")

def main():
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"Error: Checkpoint not found at '{MODEL_CHECKPOINT}'")
        return
        
    if not os.path.isdir(UNSEEN_DIR):
        print(f"Error: Directory with unseen images not found at '{UNSEEN_DIR}'")
        return

    print(f"Using device: {DEVICE}")
    print(f"Loading model from: {MODEL_CHECKPOINT}")
    
    char_dict, char_list, num_classes, unk_id = load_char_dict(DICT_FILE)
    print(f"Loaded {num_classes-1} characters (plus blank token)")
    
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model = SimpleViT(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_transform = get_val_transforms()
                            
    predict_unseen(model, UNSEEN_DIR, char_list, val_transform, DEVICE)

if __name__ == "__main__":
    main()
