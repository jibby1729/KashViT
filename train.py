"""
Simple Vision Transformer for Kashmiri OCR
A lightweight implementation using PyTorch
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import CTCLoss
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import editdistance

# ==================== Configuration ====================
DEVICE = torch.device('cpu')
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 60
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 320
NUM_HEADS = 4
NUM_LAYERS = 4
EMBED_DIM = 128
DROPOUT = 0.1

# Paths
TRAIN_FILE = "dataset/train.txt"
VAL_FILE = "dataset/val.txt"
DICT_FILE = "dict/koashurkhat_dict.txt"
CHECKPOINT_DIR = "model_checkpoints"

# Set this path to resume training from a saved model
# Example: "model_checkpoints/best_model_epoch_5.pth"
# Set to None to train from scratch
RESUME_CHECKPOINT = "model_checkpoints/best_model_epoch_40.pth"


# ==================== Dataset Class ====================
class OCRDataset(Dataset):
    def __init__(self, txt_file, char_dict, transform=None):
        self.data = []
        self.char_dict = char_dict
        self.transform = transform
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_path, text = parts
                    if os.path.exists(img_path):
                        self.data.append((img_path, text))
        
        print(f"Loaded {len(self.data)} samples from {txt_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, text = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = []
        for char in text:
            if char in self.char_dict:
                label.append(self.char_dict[char])
            else:
                label.append(0)
        
        return img, torch.tensor(label, dtype=torch.long), len(label)


# ==================== Simple Vision Transformer ====================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(IMAGE_HEIGHT, IMAGE_WIDTH), patch_size=16, embed_dim=EMBED_DIM):
        super().__init__()
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
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
    def __init__(self, num_classes, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


# ==================== Helper Functions ====================
def load_char_dict(dict_file):
    char_dict, char_list = {}, []
    with open(dict_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            if char:
                char_dict[char] = idx + 1
                char_list.append(char)
    return char_dict, char_list, len(char_list) + 1

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def collate_fn(batch):
    images, labels, lengths = [], [], []
    for img, label, length in batch:
        images.append(img)
        labels.extend(label.tolist())
        lengths.append(length)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), lengths

def decode_output(preds, char_list):
    decoded_texts = []
    for pred in preds:
        sequence = torch.argmax(pred, dim=1)
        decoded_sequence = []
        for i, idx in enumerate(sequence):
            if idx.item() != 0 and (i == 0 or idx.item() != sequence[i-1].item()):
                decoded_sequence.append(idx.item())
        decoded_texts.append(''.join([char_list[i-1] for i in decoded_sequence]))
    return decoded_texts

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for images, labels, lengths in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            B, T, C = outputs.shape
            log_probs = outputs.permute(1, 0, 2).log_softmax(2)
            input_lengths = torch.full((B,), T, dtype=torch.long)
            target_lengths = torch.tensor(lengths, dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, char_list, device):
    model.eval()
    total_loss, total_correct_words, total_words, total_char_dist, total_char_len = 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels, lengths in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
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
                if label_text == pred_text: total_correct_words += 1
                total_char_dist += editdistance.eval(pred_text, label_text)
                total_char_len += len(label_text)
                start += length
            total_words += len(lengths)
    
    avg_loss = total_loss / len(dataloader)
    word_accuracy = (total_correct_words / total_words) * 100
    char_error_rate = (total_char_dist / total_char_len) * 100
    return avg_loss, word_accuracy, char_error_rate

# ==================== Main Training Loop ====================
def main():
    print(f"Using device: {DEVICE}")
    char_dict, char_list, num_classes = load_char_dict(DICT_FILE)
    print(f"Loaded {num_classes-1} characters (plus blank token)")
    
    transform = get_transforms()
    train_dataset = OCRDataset(TRAIN_FILE, char_dict, transform)
    val_dataset = OCRDataset(VAL_FILE, char_dict, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=collate_fn, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=16, pin_memory=True)
    
    model = SimpleViT(num_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    criterion = CTCLoss(blank=0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch, best_val_loss = 0, float('inf')
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        print(f"Resumed from epoch {start_epoch}. Best validation loss was {best_val_loss:.4f}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, word_acc, char_err_rate = validate(model, val_loader, criterion, char_list, DEVICE)
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
