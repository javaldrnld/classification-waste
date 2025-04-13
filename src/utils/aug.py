import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm

# ---- Config ---- #
INPUT_DIR = Path("../data/toaug")
OUTPUT_DIR = Path("../data/aug/ew")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Augmentations ---- #
AUGS = {
    'hflip': A.HorizontalFlip(p=1),
    'rotate': A.Rotate(limit=[-90, 90], p=1),
    'brightness': A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.3], contrast_limit=[-0.1, 0.3],p=1),
    'gray': A.ToGray(p=0.5),
    'blur': A.GaussianBlur(blur_limit=(3, 7), p=1),
}

# ---- Create Output Folder Structure ---- #
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Loop through each class directory
for class_dir in INPUT_DIR.iterdir():
    if not class_dir.is_dir():
        continue

    class_output_dir = OUTPUT_DIR / class_dir.name
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # List all images in the current class folder
    images = list(class_dir.glob("*"))
    
    # Use tqdm to add a progress bar
    for img_path in tqdm(images, desc=f"Processing {class_dir.name}", unit="image"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        base_name = img_path.stem

        # Individual augmentations
        for key, aug in AUGS.items():
            aug_image = aug(image=image)['image']
            save_path = class_output_dir / f"{base_name}_{key}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        # Random combination of 2 augmentations
        combo = random.sample(list(AUGS.keys()), 2)
        combo_name = "_".join(combo)
        combined_aug = A.Compose([AUGS[combo[0]], AUGS[combo[1]]])
        aug_image = combined_aug(image=image)['image']
        save_path = class_output_dir / f"{base_name}_{combo_name}.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))