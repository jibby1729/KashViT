import os
import random
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
images_folder = "dataset/word_images"
labels_folder = "dataset/labels"
dataset_folder = "dataset"
dict_folder = "dict"

# Create folders if they don't exist
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(dict_folder, exist_ok=True)

# -----------------------------
# Collect all image files
# -----------------------------
all_files = [f.split(".")[0] for f in os.listdir(images_folder) if f.endswith(".png")]
if not all_files:
    print("No images found in word_images/")
    exit()

random.shuffle(all_files)

# Split into train (80%), val (10%), and test (10%)
train_split = int(len(all_files) * 0.8)
val_split = int(len(all_files) * 0.9)

train_files = all_files[:train_split]
val_files = all_files[train_split:val_split]
test_files = all_files[val_split:]

# -----------------------------
# Function to write train.txt / val.txt
# -----------------------------
def write_list(file_list, out_file):
    all_chars = set()
    with open(out_file, "w", encoding="utf-8") as f:
        for file_id in tqdm(file_list, desc=f"Creating {os.path.basename(out_file)}"):
            img_path = f"{images_folder}/{file_id}.png"
            label_path = f"{labels_folder}/{file_id}.txt"
            if not os.path.exists(label_path):
                print(f"Warning: Label file {label_path} not found, skipping.")
                continue
            with open(label_path, "r", encoding="utf-8") as lf:
                text = lf.read().strip()
            f.write(f"{img_path}\t{text}\n")
            all_chars.update(list(text))
    return all_chars

# Generate train.txt, val.txt, and test.txt
train_chars = write_list(train_files, os.path.join(dataset_folder, "train.txt"))
val_chars = write_list(val_files, os.path.join(dataset_folder, "val.txt"))
write_list(test_files, os.path.join(dataset_folder, "test.txt")) # Test chars are not used for dict

# -----------------------------
# Create character dictionary
# -----------------------------
all_chars = sorted(list(train_chars.union(val_chars)))
dict_path = os.path.join(dict_folder, "koashurkhat_dict.txt")
with open(dict_path, "w", encoding="utf-8") as f:
    for c in all_chars:
        f.write(f"{c}\n")

print("Dataset preparation complete!")
print(f"- train.txt: {len(train_files)} images")
print(f"- val.txt: {len(val_files)} images")
print(f"- test.txt: {len(test_files)} images")
print(f"- Character dictionary saved to: {dict_path}")
