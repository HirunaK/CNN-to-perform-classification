import os
import tarfile
import urllib.request
import random
import cv2

# Target URL for the standard Flowers dataset
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ARCHIVE_NAME = "flower_photos.tgz"
BASE_OUTPUT_DIR = "data"

# 1. Download dataset archive
if not os.path.exists(ARCHIVE_NAME):
    print("Downloading flowers dataset archive...")
    urllib.request.urlretrieve(DATASET_URL, ARCHIVE_NAME)
    print("Download completed.")

# 2. Extract archive
RAW_DIR = "flower_photos_raw"
if not os.path.exists(RAW_DIR):
    print("Extracting files...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall(path=RAW_DIR)

RAW_DATA_PATH = os.path.join(RAW_DIR, "flower_photos")
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
target_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
class_map = dict(zip(classes, target_classes))

# 3. Create destination folders (70% train, 15% val, 15% test)
splits = ['train', 'val', 'test']
for split in splits:
    for cls in target_classes:
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, split, cls), exist_ok=True)

# 4. Resize to 64x64 and save to splits
train_ratio, val_ratio = 0.70, 0.15

for src_cls, dst_cls in class_map.items():
    cls_dir = os.path.join(RAW_DATA_PATH, src_cls)
    if not os.path.exists(cls_dir):
        continue

    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_map = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, img_list in split_map.items():
        for img_name in img_list:
            src_path = os.path.join(cls_dir, img_name)
            dst_path = os.path.join(BASE_OUTPUT_DIR, split, dst_cls, img_name)

            img = cv2.imread(src_path)
            if img is not None:
                img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst_path, img_resized)

print("Dataset prepared successfully in 'data/' folder!")