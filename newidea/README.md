# Robust OCR Implementation

This folder contains an improved implementation of the Kashmiri OCR model that addresses critical generalization issues identified in the original architecture.

## Key Improvements

### 1. **Aspect-Ratio Preserving Resize**
- **Problem**: Original code forced all images into `32x320` rectangles, causing severe distortion (stretching/squashing).
- **Solution**: Images are resized to height `64` while preserving aspect ratio, then padded with white to width `320`.
- **Impact**: Model learns from undistorted character shapes, significantly improving generalization.

### 2. **Stripe-Based Tokenization**
- **Problem**: Original `16x16` patches gave only 2 rows of tokens (very low vertical resolution), losing crucial diacritics and dots.
- **Solution**: Vertical stripes (`64` height × `4` width) that span the full height. Each token covers the entire vertical dimension.
- **Impact**: Model can now "see" fine vertical details like dots and diacritics, critical for Perso-Arabic scripts.

### 3. **Grayscale Input**
- **Problem**: RGB input with ImageNet normalization wasn't appropriate for high-contrast text images.
- **Solution**: Convert images to grayscale (1 channel) before processing.
- **Impact**: Simplifies the learning problem and removes color-dependent overfitting.

### 4. **Robust Data Augmentation**
- **Problem**: No augmentation meant the model overfit to perfect, uniform training images.
- **Solution**: Comprehensive augmentation pipeline using Albumentations:
  - Photometric: CLAHE, brightness/contrast, gamma correction
  - Blur: Motion blur, Gaussian blur
  - Noise: Gaussian noise
  - Compression artifacts: JPEG compression simulation
  - Geometric: Affine transforms (rotation ±3°, scale, shear), perspective warp
  - CoarseDropout: Simulates ink breaks
- **Impact**: Model learns to handle real-world noise, blur, and variations.

### 5. **Fixed UNK Token Handling**
- **Problem**: Unknown characters were mapped to blank token (0), causing label poisoning.
- **Solution**: Dedicated `[UNK]` token with its own ID, separate from CTC blank (0).
- **Impact**: Proper handling of out-of-vocabulary characters without corrupting training.

## Files

- **`augment.py`**: Augmentation transforms and helper functions
  - `keep_ratio_resize_pad()`: Aspect-ratio preserving resize
  - `get_train_transforms()`: Training augmentation pipeline
  - `get_val_transforms()`: Validation transforms (no augmentation)

- **`trainnew.py`**: Training script with new architecture
  - `StripeEmbedding`: Vertical stripe tokenization
  - `SimpleViT`: Updated ViT with 1-channel input
  - `OCRDataset`: Grayscale loading with Albumentations transforms
  - Proper UNK token handling

- **`testnew.py`**: Testing script for the new model
  - Uses validation transforms
  - Compatible with checkpoints from `trainnew.py`

## Dependencies

Install required packages:

```bash
uv pip install albumentations>=1.4.4
uv pip install opencv-python
```

## Usage

### Training

```bash
cd newidea
python trainnew.py
```

Model checkpoints will be saved to `model_checkpoints_new/`.

To resume training, set `RESUME_CHECKPOINT` in `trainnew.py`:

```python
RESUME_CHECKPOINT = "model_checkpoints_new/best_model_epoch_X.pth"
```

### Testing

```bash
cd newidea
python testnew.py
```

**Important**: Update `MODEL_CHECKPOINT` in `testnew.py` to point to your trained model:

```python
MODEL_CHECKPOINT = "model_checkpoints_new/best_model_epoch_X.pth"
```

## Architecture Details

### Image Dimensions
- **Height**: 64 pixels (increased from 32)
- **Width**: 320 pixels (padded)
- **Channels**: 1 (grayscale)

### Tokenization
- **Stripe width**: 4 pixels
- **Sequence length**: 80 tokens (320 / 4)
- **Vertical resolution**: Full height (64 pixels) per token

### Model Configuration
- **Embedding dimension**: 128
- **Number of heads**: 4
- **Number of layers**: 4
- **Dropout**: 0.1

## Expected Improvements

With these changes, you should see:

1. **Better generalization** to real-world images (scans, photos, screenshots)
2. **Improved handling of diacritics** due to higher vertical resolution
3. **More robust predictions** under noise, blur, and compression
4. **Correct aspect ratios** preventing character distortion
5. **Proper handling of unknown characters**

## Comparison with Original

| Feature | Original | New Implementation |
|---------|----------|-------------------|
| Image size | 32×320 (stretched) | 64×320 (aspect-ratio preserved) |
| Tokenization | 16×16 patches (2 rows) | 64×4 stripes (full height) |
| Input channels | 3 (RGB) | 1 (Grayscale) |
| Augmentation | None | Comprehensive |
| UNK handling | Maps to blank (0) | Dedicated [UNK] token |
| Vertical resolution | 2 tokens | 64 pixels per token |

## Notes

- The model checkpoint directory is `model_checkpoints_new/` to avoid conflicts with the original implementation.
- All transforms use Albumentations, which requires numpy arrays. PIL images are converted automatically.
- The UNK token is automatically added to the dictionary if not present in `dict/koashurkhat_dict.txt`.
- Training uses mixed precision (FP16) on CUDA for faster training.

## Troubleshooting

**Import errors**: Make sure you're running scripts from the `newidea/` directory or have it in your Python path.

**CUDA errors**: The code will fall back to CPU if CUDA is not available, but training will be slower.

**Checkpoint loading**: Ensure checkpoint paths are correct and relative to the project root.

