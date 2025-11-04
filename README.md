# Kashmiri OCR with a Robust Vision Transformer

A Vision Transformer (ViT) model for Optical Character Recognition (OCR) on single words of Kashmiri text, written in PyTorch. This project features a robust training pipeline with data augmentation and an improved model architecture to enhance generalization to real-world images. The ViT is roughly 850k parameters, trained on a dataset of 55k images. We use MHA attention blocks and a cosine annealing scheduler for training (with AdamW optimizer), and mixed precision training (fp16) for faster training. Our activation function is GeLU. 

## Features
- **Robust ViT Architecture**: Utilizes vertical "stripe" embeddings to preserve fine details in Perso-Arabic scripts.
- **Advanced Data Augmentation**: Comprehensive on-the-fly augmentation pipeline using `Albumentations` to simulate real-world conditions (noise, blur, lighting changes, compression).
- **Aspect-Ratio Preserving Resize**: Prevents character distortion by resizing images to a fixed height while padding the width.
- **Web Interface**: Simple Flask-based web app to easily test the model with your own images via file upload or clipboard paste.
- **Complete Workflow**: Scripts for training, evaluation, and inference.

## Web Application Quickstart

The easiest way to use the model is through the included web interface.

1.  **Install Dependencies**
    ```bash
    # Install uv (if you haven't already)
    pip install uv

    # Create a virtual environment and install packages
    uv venv
    uv pip install -r requirements.txt
    ```

2.  **Run the Web App**
    ```bash
    uv run webpage/app.py
    ```

3.  **Use the Interface**
    - Open your browser and navigate to `http://127.0.0.1:5000`.
    - Click **"Select Image"** to upload an image file or press **Ctrl+V** (Cmd+V) to paste an image from your clipboard.
    - Click **"Read Image"** to see the prediction.

## Model Performance and Limitations

The model trained with the robust pipeline achieves the following performance on the **test set**:

-   **Word Accuracy**: ~90.4%
-   **Character Error Rate (CER)**: ~2.2%

The best performing checkpoint from the training run is located at `newidea/model_checkpoints_new/best_model_epoch_150.pth`.

The model performs very well on out-of-distribution digital text (e.g., screenshots from e-books, websites). I've been testing on screenshots from instagram pages like @kashmiridictionary and it performs extremely well there (due to them using a font that is present in the training data). However, it can still struggle with screenshots from physical textbooks, likely due to the unique fonts and lower quality printing which are not yet sufficiently represented in the training data.

## Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/KashViT.git
    cd KashViT
    ```

2.  **Install Dependencies**
    It is recommended to use `uv` for fast dependency management.
    ```bash
    # Install uv (if needed)
    pip install uv
    
    # Create virtual environment and install packages
    uv venv
    uv pip install -r requirements.txt
    ```

## Dataset

The dataset is not included in this repository due to its size. You can download it from the following link:

 **https://drive.google.com/drive/folders/1dxvsapqJIuGWPm1nGIHiwBxzVWgOBwcO?usp=sharing**

1.  Download the dataset `.zip` file from the link above.
2.  Unzip the file.
3.  Rename the resulting folder to `dataset`.
4.  Place the `dataset` folder in the root of this project.

The final directory structure should look like this:
```
KashViT/
├── dataset/
│   ├── word_images/
│   ├── labels/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── newidea/
├── webpage/
└── ...
```

## Advanced Usage: Training and Testing

The primary scripts for the new, robust model are located in the `newidea/` folder.

### 1. Training

To train the model from scratch or resume training:

1.  Navigate to the `newidea` directory.
    ```bash
    cd newidea
    ```
2.  To train from scratch, ensure `RESUME_CHECKPOINT_FILENAME` in `trainnew.py` is set to `None`.
3.  To resume from a checkpoint, set `RESUME_CHECKPOINT_FILENAME` to the name of your checkpoint file (e.g., `"best_model_epoch_95.pth"`).
4.  Run the training script:
    ```bash
    python trainnew.py
    ```
    Model checkpoints will be saved in the `newidea/model_checkpoints_new/` directory.

### 2. Evaluation

To evaluate a model's performance on the test set:

1.  Navigate to the `newidea` directory.
    ```bash
    cd newidea
    ```
2.  Update the `MODEL_CHECKPOINT` path in `testnew.py` to point to your desired model.
3.  Run the evaluation script:
    ```bash
    python testnew.py
    ```

---
*The original, less robust model and its corresponding `train.py` and `test.py` scripts are preserved in the project root for archival purposes.*
