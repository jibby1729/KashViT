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

This is likely due to the training data being highly uniform (e.g., consistent font, clean background, specific sizing). The model has become specialized to this data and does not generalize well to variations in fonts, anti-aliasing, or other digital artifacts found in real-world images. Interestingly, it can often recognize individual characters but fails to predict the entire word, suggesting the sequence processing is sensitive to these variations.

## Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/KashViT.git
    cd KashViT
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Dataset

The dataset is not included in this repository due to its size. You can download it from the following link:

**[LINK TO YOUR GOOGLE DRIVE HERE]**

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
    python train.py
    ```
Model checkpoints will be saved in the `model_checkpoints/` directory.

### 2. Evaluation

To evaluate the model's performance (Word Accuracy and Character Error Rate) on the test set:
1.  Update the `MODEL_CHECKPOINT` path in `test.py` to point to your desired model (e.g., `model_checkpoints/best_model_epoch_58.pth`).
2.  Run the evaluation script:
    ```bash
    python test.py
    ```

### 3. Inference on New Images

To transcribe your own single-word images:
1.  Place your image files in the `unseen_tests/` folder.
2.  For best results, take screenshots that are wide and short (approx. 10:1 width-to-height ratio) to minimize distortion.
3.  Make sure the `MODEL_CHECKPOINT` path in `unseen.py` is set correctly.
4.  Run the inference script:
    ```bash
    python unseen.py
    ```
The script will print the transcribed text for each image to the console.
