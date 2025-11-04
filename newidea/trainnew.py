"""
Robust Vision Transformer for Kashmiri OCR
Fixed architecture with aspect-ratio preserving resize, stripe tokenization,
grayscale input, and robust augmentation for better generalization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import CTCLoss
from PIL import Image
from tqdm import tqdm
import numpy as np
import editdistance
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import augmentation transforms
from augment import get_train_transforms, get_val_transforms, pil_to_numpy, IMG_H, MAX_W

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
PATCH_W = 4  # Width of each vertical stripe (64 height x 4 width patches)
NUM_HEADS = 4
NUM_LAYERS = 4
EMBED_DIM = 128
DROPOUT = 0.1

# Get the absolute path to the project root (one level up from 'newidea')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths (now absolute)
TRAIN_FILE = os.path.join(PROJECT_ROOT, "dataset/train.txt")
VAL_FILE = os.path.join(PROJECT_ROOT, "dataset/val.txt")
DICT_FILE = os.path.join(PROJECT_ROOT, "dict/koashurkhat_dict.txt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "newidea/model_checkpoints_new")

# Set this path to resume training from a saved model
# Set to None to train from scratch
# IMPORTANT: Only set the filename here. The full path is constructed below.
RESUME_CHECKPOINT_FILENAME = "best_model_epoch_125.pth" # e.g., "best_model_epoch_10.pth" or None

RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, RESUME_CHECKPOINT_FILENAME) if RESUME_CHECKPOINT_FILENAME else None

# UNK token symbol
UNK_TOKEN = "[UNK]"


def load_char_dict(dict_file):
    """
    Load character dictionary with [UNK] token for unknown characters.
    
    Returns:
        char_dict: mapping from character to index (1-indexed, blank=0)
        char_list: list of characters (indexed by dict index - 1)
        num_classes: total number of classes (blank + characters + UNK)
        unk_id: index of UNK token
    """
    char_dict = {}
    char_list = []
    
    with open(dict_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            if char and char != UNK_TOKEN:  # Skip empty lines and UNK if already present
                char_dict[char] = idx + 1  # Start from 1 (0 is blank)
                char_list.append(char)
    
    # Add UNK token if not already present
    if UNK_TOKEN not in char_dict:
        unk_id = len(char_list) + 1
        char_dict[UNK_TOKEN] = unk_id
        char_list.append(UNK_TOKEN)
    else:
        unk_id = char_dict[UNK_TOKEN]
    
    num_classes = len(char_list) + 1  # +1 for blank token (index 0)
    
    return char_dict, char_list, num_classes, unk_id


class OCRDataset(Dataset):
    """Dataset class that loads grayscale images and applies Albumentations transforms."""
    
    def __init__(self, txt_file, char_dict, unk_id, transform=None):
        self.data = []
        self.char_dict = char_dict
        self.unk_id = unk_id
        self.transform = transform
        
        # Get the project root from the path of the txt_file to build absolute paths
        project_root_from_data = os.path.dirname(os.path.dirname(os.path.abspath(txt_file)))

        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_path_relative, text = parts
                    # Construct absolute path for the image
                    img_path_absolute = os.path.join(project_root_from_data, img_path_relative)

                    if os.path.exists(img_path_absolute):
                        self.data.append((img_path_absolute, text))
        
        print(f"Loaded {len(self.data)} samples from {txt_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, text = self.data[idx]
        
        # Load as grayscale
        img = Image.open(img_path).convert('L')
        
        # Convert PIL to numpy for Albumentations
        img_np = pil_to_numpy(img)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img_np)
            img_tensor = transformed['image']
        else:
            # Fallback: convert to tensor manually
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            if len(img_tensor.shape) == 2:
                img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
        
        # CONVERT TO FLOAT and scale to [0, 1] range
        img_tensor = img_tensor.float() / 255.0

        # Ensure single channel: (1, H, W)
        if len(img_tensor.shape) == 3 and img_tensor.shape[0] != 1:
            img_tensor = img_tensor[0:1]  # Take first channel if multiple
        
        # Create label: map characters to IDs, use UNK for unknown chars
        label = []
        for char in text:
            if char in self.char_dict:
                label.append(self.char_dict[char])
            else:
                # Map unknown characters to UNK token, NOT blank (0)
                label.append(self.unk_id)
        
        return img_tensor, torch.tensor(label, dtype=torch.long), len(label)


class StripeEmbedding(nn.Module):
    """
    Vertical stripe embedding: creates tokens that span full height.
    Each token is a vertical stripe of width patch_w.
    This preserves vertical detail important for diacritics.
    """
    
    def __init__(self, img_h=IMG_H, patch_w=PATCH_W, embed_dim=EMBED_DIM, in_ch=1):
        super().__init__()
        self.img_h = img_h
        self.patch_w = patch_w
        self.embed_dim = embed_dim
        self.num_patches = MAX_W // patch_w
        
        # Convolution: kernel spans full height (img_h) and patch_w width
        self.conv = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=(img_h, patch_w),
                              stride=(img_h, patch_w), bias=False)
    
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) where H=IMG_H, W=MAX_W
        
        Returns:
            (B, T, E) where T=num_patches, E=embed_dim
        """
        # x: (B, 1, H, W)
        x = self.conv(x)  # (B, E, 1, T) where T = W // patch_w
        x = x.squeeze(2).transpose(1, 2)  # (B, T, E)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head self-attention and MLP."""
    
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleViT(nn.Module):
    """Vision Transformer with stripe-based tokenization."""
    
    def __init__(self, num_classes, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, 
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.patch_embed = StripeEmbedding(img_h=IMG_H, patch_w=PATCH_W, 
                                          embed_dim=embed_dim, in_ch=1)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) 
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.patch_embed(x) + self.pos_embed  # (B, T, E)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # (B, T, num_classes)


def collate_fn(batch):
    """Collate function for DataLoader: handles variable-length labels."""
    images, labels, lengths = [], [], []
    for img, label, length in batch:
        images.append(img)
        labels.extend(label.tolist())
        lengths.append(length)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), lengths


def decode_output(preds, char_list):
    """
    Decode CTC predictions: collapses repeated characters and removes blank token.
    Note: UNK tokens are kept in predictions (they get decoded as [UNK]).
    """
    decoded_texts = []
    for pred in preds:
        sequence = torch.argmax(pred, dim=1)
        decoded_sequence = []
        for i, idx in enumerate(sequence):
            idx_val = idx.item()
            # Skip blank token (0), but keep everything else (including UNK)
            if idx_val != 0 and (i == 0 or idx_val != sequence[i-1].item()):
                decoded_sequence.append(idx_val)
        decoded_texts.append(''.join([char_list[i-1] for i in decoded_sequence]))
    return decoded_texts


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp=True, scheduler=None):
    """Training epoch."""
    model.train()
    total_loss = 0
    for images, labels, lengths in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        
        if use_amp and scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                B, T, C = outputs.shape
                log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                input_lengths = torch.full((B,), T, dtype=torch.long)
                target_lengths = torch.tensor(lengths, dtype=torch.long)
                loss = criterion(log_probs, labels, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Clip gradients after backward pass
            scaler.unscale_(optimizer) # Unscale for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            B, T, C = outputs.shape
            log_probs = outputs.permute(1, 0, 2).log_softmax(2)
            input_lengths = torch.full((B,), T, dtype=torch.long)
            target_lengths = torch.tensor(lengths, dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients after backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
        
        # Step the scheduler at each iteration
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, char_list, device, use_amp=True):
    """Validation epoch."""
    model.eval()
    total_loss, total_correct_words, total_words, total_char_dist, total_char_len = 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels, lengths in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            if use_amp and device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    B, T, C = outputs.shape
                    log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                    input_lengths = torch.full((B,), T, dtype=torch.long)
                    target_lengths = torch.tensor(lengths, dtype=torch.long)
                    loss = criterion(log_probs, labels, input_lengths, target_lengths)
                    total_loss += loss.item()
            else:
                outputs = model(images)
                B, T, C = outputs.shape
                log_probs = outputs.permute(1, 0, 2).log_softmax(2)
                input_lengths = torch.full((B,), T, dtype=torch.long)
                target_lengths = torch.tensor(lengths, dtype=torch.long)
                loss = criterion(log_probs, labels, input_lengths, target_lengths)
                total_loss += loss.item()
            
            decoded_texts = decode_output(outputs.cpu(), char_list)
            start = 0
            for i, length in enumerate(lengths):
                label_text = "".join([char_list[c-1] for c in labels[start:start+length]])
                pred_text = decoded_texts[i]
                if label_text == pred_text:
                    total_correct_words += 1
                total_char_dist += editdistance.eval(pred_text, label_text)
                total_char_len += len(label_text)
                start += length
            total_words += len(lengths)
    
    avg_loss = total_loss / len(dataloader)
    word_accuracy = (total_correct_words / total_words) * 100
    char_error_rate = (total_char_dist / total_char_len) * 100
    return avg_loss, word_accuracy, char_error_rate


def main():
    print(f"Using device: {DEVICE}")
    print(f"Image dimensions: {IMG_H} x {MAX_W}")
    print(f"Patch width: {PATCH_W}, Sequence length: {MAX_W // PATCH_W}")
    
    char_dict, char_list, num_classes, unk_id = load_char_dict(DICT_FILE)
    print(f"Loaded {num_classes-1} characters (plus blank token)")
    print(f"UNK token ID: {unk_id}")
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Create datasets
    train_dataset = OCRDataset(TRAIN_FILE, char_dict, unk_id, train_transform)
    val_dataset = OCRDataset(VAL_FILE, char_dict, unk_id, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=collate_fn, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=12, pin_memory=True)
    
    # Initialize model
    model = SimpleViT(num_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    criterion = CTCLoss(blank=0, reduction='mean')
    
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Use Cosine Annealing scheduler
    # T_max is the total number of training steps. len(train_loader) * NUM_EPOCHS
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * NUM_EPOCHS, eta_min=1e-5)

    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
    
    # For CPU-only systems, we need to handle autocast differently
    use_amp = DEVICE.type == 'cuda'
    
    start_epoch, best_val_loss = 0, float('inf')
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {start_epoch}. Best validation loss was {best_val_loss:.4f}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, use_amp, scheduler)
        val_loss, word_acc, char_err_rate = validate(model, val_loader, criterion, char_list, DEVICE, use_amp)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Word Acc: {word_acc:.2f}% | CER: {char_err_rate:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

