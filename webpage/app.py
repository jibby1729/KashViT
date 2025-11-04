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

# Add parent directory to path to import from train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import SimpleViT, load_char_dict, get_transforms, decode_output

# Configuration
MODEL_CHECKPOINT = "model_checkpoints/best_model_epoch_58.pth"
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

# Global variables for model
model = None
char_list = None
transform = None
device = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the OCR model and related components"""
    global model, char_list, transform, device
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(base_dir, MODEL_CHECKPOINT)
        dict_path = os.path.join(base_dir, DICT_FILE)
        
        # Load character dictionary
        print(f"Loading dictionary from {dict_path}")
        _, char_list, num_classes = load_char_dict(dict_path)
        print(f"Loaded {num_classes - 1} characters (plus blank token)")
        
        # Load model
        print(f"Loading model from {checkpoint_path}")
        model = SimpleViT(num_classes=num_classes).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
        
        # Get transforms
        transform = get_transforms()
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction"""
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
            # Process image
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(image_tensor)
            
            # Decode prediction
            decoded_texts = decode_output(outputs.cpu(), char_list)
            prediction = decoded_texts[0] if decoded_texts else ''
            
            # Return result
            return jsonify({'prediction': prediction})
            
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
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, host='127.0.0.1', port=5000)

