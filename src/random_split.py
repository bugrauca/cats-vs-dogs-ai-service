import os, shutil, random

# current folder: dataset/cats and dataset/dogs
base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

for split_dir in [train_dir, val_dir]:
    for cls in ["cats", "dogs"]:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

split_ratio = 0.8  # 80% train, 20% val
for cls in ["cats", "dogs"]:
    src_dir = os.path.join(base_dir, cls)
    images = os.listdir(src_dir)
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_images, val_images = images[:split_idx], images[split_idx:]

    for img in train_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dir, cls, img))
    for img in val_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_dir, cls, img))

print("Dataset split complete!")
