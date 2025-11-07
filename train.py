
#Uses the combined clean_dataset 

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import CTCLoss
from PIL import Image
from tqdm import tqdm
import numpy as np
import editdistance
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

# Import from our new transform script
from transform import get_train_transforms, get_val_transforms, pil_to_numpy, IMG_H, MAX_W

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LEARNING_RATE = 5e-5 
NUM_EPOCHS = 500
PATCH_W = 4
NUM_HEADS = 4
NUM_LAYERS = 4
EMBED_DIM = 128
DROPOUT = 0.1
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Updated Paths ---
TRAIN_FILE = os.path.join(PROJECT_ROOT, "clean_dataset/train.txt")
VAL_FILE = os.path.join(PROJECT_ROOT, "clean_dataset/val.txt")
DICT_FILE = os.path.join(PROJECT_ROOT, "dict/koashurkhat_dict.txt")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "model_checkpoints")
RESUME_CHECKPOINT = os.path.join(PROJECT_ROOT, "model_checkpoints/epoch_382.pth")

UNK_TOKEN = "[UNK]"

# --- Model and Data Loading (Copied from trainnew.py, with one key change) ---

def load_char_dict(dict_file):
    char_dict, char_list = {}, []
    with open(dict_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            char = line.strip()
            if char and char != UNK_TOKEN:
                char_dict[char] = idx + 1
                char_list.append(char)
    
    if UNK_TOKEN not in char_dict:
        unk_id = len(char_list) + 1
        char_dict[UNK_TOKEN] = unk_id
        char_list.append(UNK_TOKEN)
    else:
        unk_id = char_dict[UNK_TOKEN]
    
    num_classes = len(char_list) + 1  # +1 for blank token
    return char_dict, char_list, num_classes, unk_id

class OCRDataset(Dataset):
    def __init__(self, txt_file, char_dict, unk_id, transform=None):
        self.data, self.char_dict, self.unk_id, self.transform = [], char_dict, unk_id, transform
        project_root_from_data = os.path.dirname(os.path.dirname(os.path.abspath(txt_file)))
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_path_relative, text = parts
                    img_path_absolute = os.path.join(project_root_from_data, img_path_relative)
                    if os.path.exists(img_path_absolute):
                        self.data.append((img_path_absolute, text))
        print(f"Loaded {len(self.data)} samples from {txt_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, text = self.data[idx]
        img = Image.open(img_path).convert('L')
        img_np = pil_to_numpy(img)
        
        if self.transform:
            transformed = self.transform(image=img_np)
            img_tensor = transformed['image']
        else: # Fallback if no transforms
            img_tensor = torch.from_numpy(img_np).float().unsqueeze(0)
        
        # IMPORTANT: Manual scaling is REMOVED because A.Normalize now handles it.
        
        label = [self.char_dict.get(char, self.unk_id) for char in text]
        return img_tensor, torch.tensor(label, dtype=torch.long), len(label)

class StripeEmbedding(nn.Module):
    def __init__(self, img_h=IMG_H, patch_w=PATCH_W, embed_dim=EMBED_DIM, in_ch=1):
        super().__init__()
        self.num_patches = MAX_W // patch_w
        self.conv = nn.Conv2d(in_ch, embed_dim, kernel_size=(img_h, patch_w), stride=(img_h, patch_w), bias=False)
    def forward(self, x):
        return self.conv(x).squeeze(2).transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, num_classes, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.patch_embed = StripeEmbedding(img_h=IMG_H, patch_w=PATCH_W, embed_dim=embed_dim, in_ch=1)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks: x = block(x)
        return self.head(self.norm(x))

def collate_fn(batch):
    images, labels, lengths = [], [], []
    for img, label, length in batch:
        images.append(img); labels.extend(label.tolist()); lengths.append(length)
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

def train_epoch(model, dataloader, criterion, optimizer, scaler, device, scheduler):
    model.train()
    total_loss = 0
    for images, labels, lengths in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            B, T, C = outputs.shape
            
            # 1. Calculate log_softmax as required by CTCLoss
            log_probs = outputs.permute(1, 0, 2).log_softmax(2)
            
            input_lengths = torch.full((B,), T, dtype=torch.long)
            target_lengths = torch.tensor(lengths, dtype=torch.long)

            # 2. Cast to float32 for the stable loss calculation to avoid the "Half" error
            loss = criterion(log_probs.float(), labels, input_lengths, target_lengths)
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()
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

                # Apply the same fix here
                log_probs = outputs.permute(1, 0, 2).log_softmax(2)

                input_lengths, target_lengths = torch.full((B,), T, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)
                
                # And cast to float32 here
                loss = criterion(log_probs.float(), labels, input_lengths, target_lengths)

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

def main():
    print(f"Using device: {DEVICE}")
    char_dict, char_list, num_classes, unk_id = load_char_dict(DICT_FILE)
    print(f"Loaded {num_classes-1} characters (plus blank token)")
    
    train_dataset = OCRDataset(TRAIN_FILE, char_dict, unk_id, get_train_transforms())
    val_dataset = OCRDataset(VAL_FILE, char_dict, unk_id, get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=12, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=12, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    
    model = SimpleViT(num_classes=num_classes).to(DEVICE)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Add zero_infinity=True for stability.
    criterion = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Use Cosine Annealing with Warm Restarts scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                            T_0=20 * len(train_loader), # Cycle length of 20 epochs
                                            T_mult=1,                  # Keep cycle length constant
                                            eta_min=1e-5)              # Minimum learning rate

    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch, best_val_loss = 0, float('inf')
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        # It's often better to reset optimizer and epoch when fine-tuning on a new dataset
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('val_loss', float('inf')) # Use .get for safety
        print(f"Resumed from epoch {start_epoch}. Best validation loss was {best_val_loss:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        # Pass the scheduler back to the training function
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, scheduler)
        val_loss, word_acc, char_err_rate = validate(model, val_loader, criterion, char_list, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Word Acc: {word_acc:.2f}% | CER: {char_err_rate:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # --- Updated Checkpoint Naming ---
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
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
