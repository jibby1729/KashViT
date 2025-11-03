import torch
from train import SimpleViT, load_char_dict, get_transforms, decode_output
from PIL import Image
import os

# cfg
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DICT_FILE = "dict/koashurkhat_dict.txt"
MODEL_CHECKPOINT = "model_checkpoints/best_model_epoch_58.pth"
IMAGE_DIR = "unseen_tests"

_, char_list, num_classes = load_char_dict(DICT_FILE)
model = SimpleViT(num_classes=num_classes).to(DEVICE)
checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#Get image transforms
transform = get_transforms()

print(f"--- Transcribing images in '{IMAGE_DIR}' ---")

#Process each image
for image_file in os.listdir(IMAGE_DIR):
    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(IMAGE_DIR, image_file)
        
        # Open and transform the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(image)
        
        # Decode and print
        decoded_text = decode_output(outputs.cpu(), char_list)[0]
        print(f"{image_file:<20} | {decoded_text}")

print("--- Done ---")
