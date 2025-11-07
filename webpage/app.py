"""
Flask web application for Kashmiri OCR model inference
"""

import os
import sys
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
import secrets
from werkzeug.utils import secure_filename
import numpy as np

# --- Change 1: Add imports for BOTH pipelines ---
# Root directory for new model components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 'newidea' directory for old model components
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'newidea'))

# Imports for Model 1 (New)
from train import SimpleViT as NewSimpleViT, load_char_dict as new_load_char_dict, decode_output as new_decode_output
from transform import get_val_transforms as new_get_val_transforms

# Imports for Model 2 (Old/Kaggle)
from trainnew import SimpleViT as OldSimpleViT, load_char_dict as old_load_char_dict, decode_output as old_decode_output
from augment import get_val_transforms as old_get_val_transforms

# --- Change 2: Update Configuration ---
MODEL_1_CHECKPOINT = "best_models/largertrained.pth"  # New Model
MODEL_2_CHECKPOINT = "best_models/kaggletrained.pth"   # Old Model
DICT_FILE = "dict/koashurkhat_dict.txt"
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Add the image inversion helper functions ---
def detect_if_needs_inversion(image_np):
    """
    Detects if a grayscale numpy image is likely white text on a black background.
    """
    # A simple heuristic: if the average pixel value is very dark, it's likely inverted.
    mean_val = np.mean(image_np)
    return mean_val < 100

def invert_image(image_np):
    """Inverts a grayscale numpy image."""
    return 255 - image_np

# --- Global variables for BOTH models ---
model_1, char_list_1, transform_1 = None, None, None
model_2, char_list_2, transform_2 = None, None, None
device = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    """Load both OCR models and their related components"""
    global model_1, char_list_1, transform_1
    global model_2, char_list_2, transform_2
    global device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dict_path = os.path.join(base_dir, DICT_FILE)
        
        # --- Load Model 1 (New Pipeline) ---
        print("\n--- Loading Model 1 (New Pipeline) ---")
        checkpoint_path_1 = os.path.join(base_dir, MODEL_1_CHECKPOINT)
        _, char_list_1, num_classes_1, _ = new_load_char_dict(dict_path)
        print(f"Loading from {checkpoint_path_1}")
        model_1 = NewSimpleViT(num_classes=num_classes_1).to(device)
        checkpoint_1 = torch.load(checkpoint_path_1, map_location=device)
        model_1.load_state_dict(checkpoint_1['model_state_dict'])
        model_1.eval()
        transform_1 = new_get_val_transforms()
        print("Model 1 loaded successfully.")
        
        # --- Load Model 2 (Old Pipeline / Kaggle) ---
        print("\n--- Loading Model 2 (Old Pipeline) ---")
        checkpoint_path_2 = os.path.join(base_dir, MODEL_2_CHECKPOINT)
        _, char_list_2, num_classes_2, _ = old_load_char_dict(dict_path)
        print(f"Loading from {checkpoint_path_2}")
        model_2 = OldSimpleViT(num_classes=num_classes_2).to(device)

        # --- FIX IS HERE ---
        # 1. Load the entire checkpoint dictionary
        checkpoint_2 = torch.load(checkpoint_path_2, map_location=device)
        # 2. Load the 'model_state_dict' from within the dictionary
        model_2.load_state_dict(checkpoint_2['model_state_dict'])
        
        model_2.eval()
        transform_2 = old_get_val_transforms()
        print("Model 2 loaded successfully.")

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions from both models"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PNG or JPEG image.'}), 400
        
        # Generate secure filename
        filename = secure_filename(file.filename)
        # Add random prefix to avoid collisions
        unique_filename = secrets.token_hex(8) + '_' + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file temporarily
        file.save(filepath)
        
        try:
            # --- Process for both models ---
            image = Image.open(filepath).convert('L')
            image_np = np.array(image)
            
            # --- Inversion Check for Model 1 ONLY ---
            image_for_model_1 = image_np
            if detect_if_needs_inversion(image_np):
                print("Inverting detected white-on-black image for Model 1.")
                image_for_model_1 = invert_image(image_np)

            # --- Prediction 1 (New Model with potentially inverted image) ---
            transformed_1 = transform_1(image=image_for_model_1)
            image_tensor_1 = transformed_1['image'].unsqueeze(0).to(device)
            # Normalization is part of transform_1, no extra steps needed.
            with torch.no_grad():
                outputs_1 = model_1(image_tensor_1)
            decoded_texts_1 = new_decode_output(outputs_1.cpu(), char_list_1)
            prediction_1 = decoded_texts_1[0] if decoded_texts_1 else ''

            # --- Prediction 2 (Old Model with ORIGINAL image) ---
            transformed_2 = transform_2(image=image_np) # Use the original image_np
            image_tensor_2 = transformed_2['image'].unsqueeze(0).to(device)
            # Old pipeline requires manual scaling
            image_tensor_2 = image_tensor_2.float() / 255.0
            with torch.no_grad():
                outputs_2 = model_2(image_tensor_2)
            decoded_texts_2 = old_decode_output(outputs_2.cpu(), char_list_2)
            prediction_2 = decoded_texts_2[0] if decoded_texts_2 else ''
            
            # --- Return both results ---
            return jsonify({'prediction1': prediction_1, 'prediction2': prediction_2})
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        error_msg = str(e)
        print(f"Error during prediction: {error_msg}")
        return jsonify({'error': f'Error processing image: {error_msg}'}), 500


if __name__ == '__main__':
    print("Initializing Kashmiri OCR web application...")
    load_models()
    print("\nStarting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)

