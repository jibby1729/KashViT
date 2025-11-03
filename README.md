# Kashmiri OCR with Vision Transformer

A lightweight Vision Transformer (ViT) model for Optical Character Recognition (OCR) on single words of Kashmiri text, written in PyTorch. This project provides scripts for training, evaluating, and running inference with the model.

## Features
- **Simple ViT Implementation**: Basic implementation of a ViT with MHA and Dropout. GeLU for activation.
- **Lightweight Architecture**: The model has approximately 900k parameters, enough to do well with the dataset so far. 
- **Specialized Loss Function**: Uses `CTCLoss`, which is ideal for sequence-to-sequence tasks like OCR where input and output lengths may vary.
- **Complete Workflow**: Includes scripts for training, evaluation, and inference on new images.

## Pre-trained Model
A pre-trained model is available in the `model_checkpoints/` directory. The best performing checkpoint is `best_model_epoch_58.pth`. You can use this directly for evaluation or inference.

## Model Performance and Limitations
The current best model (`best_model_epoch_58.pth`) achieves the following performance on the provided test set:
- **Word Accuracy**: ~80%
- **Character Error Rate (CER)**: ~4%

### Performance on "In-the-Wild" Images
While the model performs well on the test set, it struggles with new images taken from different sources, such as screenshots of words from e-books. Examples of these challenging images can be found in the `unseen_tests` folder.

To be honest, I am not entirely sure why the model is struggling with the new images. It might be due to the training data being highly uniform (clean background, specific sizing) or the dataset simply being too small to accomodate different fonts, anti-aliasing, or other digital artifacts found in real-world images. Interestingly, it can often recognize individual characters but fails to predict the entire word, suggesting the sequence processing is sensitive to these variations.

To add test images to the `unseen_tests` folder, make sure the screenshots are of single words and are in the same format as the images in the `dataset/word_images` folder.

## Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/KashViT.git
    cd KashViT
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment. First, install `uv` if you haven't already:
    ```bash
    # Install uv (if not already installed)
    # On Windows (PowerShell):
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # On macOS/Linux:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    
    Then create a virtual environment and install dependencies:
    ```bash
    uv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    uv pip install -r requirements.txt
    ```

## Dataset

The dataset is not included in this repository due to its size. You can download it from the following link:

 **https://drive.google.com/drive/folders/1dxvsapqJIuGWPm1nGIHiwBxzVWgOBwcO?usp=sharing**


1.  Download the dataset `.zip` file from the link above.
2.  Unzip the file.
3.  Rename the resulting folder to `dataset`.
4.  Place the `dataset` folder in the root of this project, at the same level as the `train.py` and `test.py` scripts.

The final directory structure should look like this:
```
KashViT/
├── dataset/
│   ├── word_images/
│   ├── labels/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── model_checkpoints/
├── train.py
├── test.py
└── ...
```

## Usage

All scripts are configured using variables at the top of each file. You can modify these to change batch sizes, learning rates, file paths, etc.

### 1. Training

To train the model from scratch or resume training:
1.  Ensure your dataset is correctly formatted.
2.  Update the paths and hyperparameters in `train.py` as needed. To resume from a checkpoint, set the `RESUME_CHECKPOINT` variable.
3.  Run the training script:
    ```bash
    uv run train.py
    ```
Model checkpoints will be saved in the `model_checkpoints/` directory.

### 2. Evaluation

To evaluate the model's performance (Word Accuracy and Character Error Rate) on the test set:
1.  Update the `MODEL_CHECKPOINT` path in `test.py` to point to your desired model (e.g., `model_checkpoints/best_model_epoch_58.pth`).
2.  Run the evaluation script:
    ```bash
    uv run test.py
    ```

### 3. Inference on New Images

To transcribe your own single-word images:
1.  Place your image files in the `unseen_tests/` folder.
2.  For best results, take screenshots that are wide and short (single word screenshots) to minimize distortion.
3.  Make sure the `MODEL_CHECKPOINT` path in `unseen.py` is set correctly.
4.  Run the inference script:
    ```bash
    uv run unseen.py
    ```
The script will print the transcribed text for each image to the console.
