# Kashmiri OCR Web Interface

A minimalistic web interface for the Kashmiri OCR model, built with Flask.

## Setup

1. **Install Flask** (if not already installed):
   ```bash
   uv pip install Flask
   ```

2. **Ensure model files are available**:
   - Model checkpoint: `model_checkpoints/best_model_epoch_58.pth`
   - Dictionary file: `dict/koashurkhat_dict.txt`

## Running the Application

From the project root directory, run:

```bash
uv run webpage/app.py
```

Or using Python directly:

```bash
python webpage/app.py
```

The Flask server will start on `http://127.0.0.1:5000`.

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000`
2. Click "Select Image" to choose an image file (PNG or JPEG)
3. Click "Read Image" to process the image
4. The prediction will be displayed below
5. You can upload another image immediately after

## Features

- Minimalistic dark theme interface
- Monospace font for clean, block-like appearance
- Asynchronous image processing (no page reload)
- Automatic file cleanup after processing
- Error handling for invalid files and processing errors

## File Structure

```
webpage/
├── app.py              # Flask application
├── static/
│   └── style.css       # Minimalistic stylesheet
├── templates/
│   └── index.html      # Main HTML page
└── uploads/            # Temporary image storage (auto-cleaned)
```

## Notes

- The model is loaded once at startup for faster predictions
- Uploaded images are temporarily saved and automatically deleted after processing
- Maximum file size is 10MB
- Supported formats: PNG, JPEG, JPG

