from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt

# --- Load a sample image ---
IMG_PATH = Path("../data/new_test/IMG_0979.jpeg").resolve()  # Change to your test image
image = cv2.imread(str(IMG_PATH))
# Handle failed load
if image is None:
    raise FileNotFoundError(f"Failed to load image: {IMG_PATH}. Please check the path or file.")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- Augmentations (modifiable) ---
AUGS = {
    'original': A.Compose([]),
    'hflip': A.HorizontalFlip(p=1),
    'rotate':A.Rotate(limit=[-90, 90], p=1),
    'brightness': A.RandomBrightnessContrast(brightness_limit=[-0.1, 0.3], contrast_limit=[-0.1, 0.3], p=1),
    'gray': A.ToGray(p=0.5),
    'blur': A.GaussianBlur(blur_limit=(3, 7), p=1),
    'combo_example': A.Compose([
        A.Rotate(limit=20, p=1),
        A.GaussianBlur(blur_limit=(3, 5), p=1)
    ])
}

# --- Show Results ---
def show_augmented_images(image, augs):
    n = len(augs)
    plt.figure(figsize=(15, 5))
    for idx, (name, aug) in enumerate(augs.items()):
        aug_img = aug(image=image)['image']
        plt.subplot(1, n, idx + 1)
        plt.imshow(aug_img)
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_augmented_images(image, AUGS)
