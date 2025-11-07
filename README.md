# Kashmiri OCR with Vision Transformers

This project provides a complete workflow for training and deploying robust Vision Transformer (ViT) models for Optical Character Recognition (OCR) on single words of Kashmiri text. It includes scripts for data processing, training, evaluation, and a simple web interface for live predictions.

## Features
- **Two Pre-trained Models**: Includes two distinct ViT models with different training data and augmentation strategies.
- **Advanced Data Augmentation**: Comprehensive on-the-fly augmentation pipelines using `Albumentations` to simulate real-world conditions.
- **Robust Data Cleaning**: A full pipeline to combine, clean, and prepare disparate datasets for training.
- **Web Interface**: A Flask-based web app to compare the two models side-by-side with your own images.
- **Complete Workflow**: Scripts for training, evaluation, and inference for both model pipelines.

## Quickstart & Demo

The easiest way to test and compare the models is through the included web interface.

1.  **Install Dependencies**
    It is recommended to use `uv` for fast dependency management.
    ```bash
    # Install uv (if you haven't already)
    pip install uv

    # Create a virtual environment and install packages
    uv venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    uv pip install -r requirements.txt
    ```

2.  **Run the Web App**
    ```bash
    python webpage/app.py
    ```

3.  **Use the Interface**
    - Open your browser and navigate to `http://127.0.0.1:5000`.
    - The interface allows you to either click **"Select Image"** to upload a file or simply press **Ctrl+V** (Cmd+V on Mac) to paste an image directly from your clipboard.
    - Click **"Read Image"** to see the predictions from both models appear side-by-side.

## Data Setup

This project uses two distinct datasets. The web app works out-of-the-box, but to run the training or testing scripts, you will need to download the relevant data.

### 1. `clean_dataset` (For `largertrained.pth`)

This is the combined and cleaned dataset of ~100,000 images used to train the primary model (`best_models/largertrained.pth`). The data was sourced from:
- [Omarrran/40K Kashmiri Dataset (Hugging Face)](https://huggingface.co/datasets/Omarrran/40K_kashmiri_text_and_image_dataset)
- [Kashmiri-OCR (Kaggle)](https://www.kaggle.com/datasets/nawabhussaen/kashmiri-ocr)

The `train.txt`, `val.txt`, and `test.txt` split files are included in the repository. To use them, you must download the raw images and place them in the `clean_dataset/images/` folder. The final structure should be:
```
clean_dataset/
├── images/
│   ├── 0.png, 1.png, ...
├── labels.csv
├── train.txt
├── val.txt
└── test.txt
```

### 2. `dataset` (For `kaggletrained.pth`)

This is the original ~70k image Kaggle dataset used to train the secondary model (`best_models/kaggletrained.pth`). This is required to run the scripts in the `newidea/` folder. A pre-packaged version is available at the link below.

1.  **Download from Google Drive**: [https://drive.google.com/drive/folders/1dxvsapqJIuGWPm1nGIHiwBxzVWgOBwcO?usp=sharing](https://drive.google.com/drive/folders/1dxvsapqJIuGWPm1nGIHiwBxzVWgOBwcO?usp=sharing)
2.  Unzip the file and place the resulting `dataset` folder in the root of this project.

The final project structure should look like this:
```
KashViT/
├── dataset/
│   ├── word_images/
│   ├── labels/
│   ├── train.txt, val.txt, test.txt
├── clean_dataset/
│   ├── images/
│   ├── labels.csv, train.txt, val.txt, test.txt
├── newidea/
└── ...
```

### Data Cleaning and Preparation

Before training the primary model, the datasets were combined and cleaned to create a unified, high-quality dataset of approximately 100,000 images. The process included:
- **Inverting Images**: All images with white text on a black background were inverted to a standard black-text-on-white format to ensure consistency.
- **Filtering Labels**: Any labels containing non-Kashmiri characters were removed. Specifically, only labels composed entirely of characters found in the `dict/koashurkhat_dict.txt` file were kept. This dictionary contains 64 unique Kashmiri characters. The model is trained on these 64 characters plus a special "blank" token required by the CTC loss function.
- **Removing Corrupt Images**: Any corrupted or unreadable image files were discarded.

This cleaned data was then used to create the `clean_dataset` with an 80/10/10 split for training, validation, and testing.

## The Models

Both models are Vision Transformers of ~840k parameters, using 4 transformer layers, 4 attention heads, and an embedding dimension of 128. They were trained with a batch size of 32 using the AdamW optimizer and a GELU activation function.

### Model 1: `best_models/largertrained.pth` (Primary)

- **Training Data**: Trained on the combined and cleaned `clean_dataset` of ~100k images.
- **Augmentation**: Uses a more extensive augmentation pipeline (`transform.py`).
- **Input Transform**: Expects a grayscale image. The pipeline automatically inverts white-on-black images and normalizes the image to a `[-1, 1]` range.
- **Performance**: **88.06%** Word Accuracy and **2.87%** CER on `clean_dataset/test.txt`.
- **Usage**: To train this model, use `train.py`. To evaluate it, use `test.py`. To resume training from this checkpoint, update the `RESUME_CHECKPOINT` variable in `train.py`.

### Model 2: `best_models/kaggletrained.pth` (Secondary)

- **Training Data**: Trained *only* on the original ~70k Kaggle `dataset`.
- **Augmentation**: Uses a simpler augmentation pipeline (`newidea/augment.py`).
- **Input Transform**: Expects a grayscale image, resized and scaled to a `[0, 1]` range.
- **Performance**: **93.33%** Word Accuracy and **1.50%** CER on `dataset/test.txt`.
- **Usage**: To train this model, use `newidea/trainnew.py`. To evaluate it, use `newidea/testnew.py`. To resume training from this checkpoint, update the `RESUME_CHECKPOINT` variable in `newidea/trainnew.py`.

## Inference on Unseen Images

To run inference on a folder of your own images (e.g., the images in `unseen_tests/`), you can use the provided scripts.

-   **For the Primary Model (`largertrained.pth`)**:
    ```bash
    python unnseen.py
    ```
    *Note: You may need to update the `MODEL_CHECKPOINT` path inside the script.*

-   **For the Secondary Model (`kaggletrained.pth`)**:
    ```bash
    python challenge_newidea.py
    ```
    *Note: You may need to update the `MODEL_CHECKPOINT` path inside the script.*

## Limitations

The models show strong performance on digitally generated text. However, they are still not able to generalize well to non-digitized text, such as handwriting or photographs of text printed on paper.

## Citation
If you use the 40K dataset component, please cite the original author:
```bibtex
@misc{dataset2024,
  title        = {Omarrran/40K_kashmiri_text_and_image_dataset},
  author       = {HAQ NAWAZ MALIK},
  year         = {2024},
  url          = {https://huggingface.co/datasets/Omarrran/40K_kashmiri_text_and_image_dataset/},
  note         = {Contains 40,799 images with labels }
}
```

## Reproducibility Note

The `clean_dataset` was generated using the `combine_datasets.py` script. This script handles the downloading, cleaning, filtering, and splitting of the raw datasets mentioned above. Users interested in the exact data preparation workflow can refer to this script.
