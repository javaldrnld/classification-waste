import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0,
    preprocess_input,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0001
results_dir = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/data/hyperparameter_result/efficientnetb0"
os.makedirs(results_dir, exist_ok=True)  # Create directory if it doesn't exist
# load training data for visualization
x_data, y_data = [], []
for category in glob(TRAIN_DIR + "/*"):
    for file in tqdm(glob(category + "/*")):
        img_array = cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split("/")[-1])

data = pd.DataFrame({"image": x_data, "label": y_data})

# Check the data shape
# It mu st return (# of row, col) which is 536 image 2 col (image and label)
data.shape

# Data Visualization -> Checking if there's imbalanced 
colors = ["#c0d6e4", "#0c482e"]
plt.pie(
    data.label.value_counts().sort_index(),
    labels=["unacceptble", "pet_bottle"],
    colors=colors,
    startangle=90,
    explode=[0.05, 0.05]
)
plt.show()

# Define Data Generators and have strong augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT
)

# Validation and Test Datagen (Only Rescale no need to augment)
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Generate iterators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Define hyperparameter test cases
hyperparameter_tests = [
    {"dense_layers": [128], "dropout": [0.5], "batch_size": 4, "lr": 0.0001, "unfreeze_layers": 10, "epochs": 20},
    {"dense_layers": [128, 64], "dropout": [0.5, 0.6], "batch_size": 4, "lr": 0.00005, "unfreeze_layers": 10, "epochs": 20},
    {"dense_layers": [256, 128], "dropout": [0.4, 0.5], "batch_size": 8, "lr": 0.0001, "unfreeze_layers": 20, "epochs": 30},
    {"dense_layers": [512, 256, 128], "dropout": [0.4, 0.5, 0.6], "batch_size": 16, "lr": 0.00005, "unfreeze_layers": 30, "epochs": 40},
    {"dense_layers": [256], "dropout": [0.4], "batch_size": 8, "lr": 0.0001, "unfreeze_layers": 15, "epochs": 25},
    {"dense_layers": [128, 64, 32], "dropout": [0.5, 0.6, 0.3], "batch_size": 4, "lr": 0.0001, "unfreeze_layers": 10, "epochs": 20},
    {"dense_layers": [64], "dropout": [0.3], "batch_size": 8, "lr": 0.0005, "unfreeze_layers": 5, "epochs": 15},
    {"dense_layers": [128, 64], "dropout": [0.5, 0.3], "batch_size": 8, "lr": 0.0001, "unfreeze_layers": 15, "epochs": 20},
    {"dense_layers": [256, 128, 64], "dropout": [0.4, 0.5, 0.6], "batch_size": 16, "lr": 0.00005, "unfreeze_layers": 20, "epochs": 30},
    {"dense_layers": [128], "dropout": [0.5], "batch_size": 4, "lr": 0.0001, "unfreeze_layers": 5, "epochs": 20},
]

# Store results
results = []

os.makedirs(results_dir, exist_ok=True)

for i, params in enumerate(hyperparameter_tests, 1):
    print(f"\n🔹 Running Test {i} with Hyperparameters: {params}")

    # Load MobileNetV2 as base model
    base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    # Freeze layers except the last `unfreeze_layers`
    for layer in base_model.layers[:-params["unfreeze_layers"]]:
        layer.trainable = False
    for layer in base_model.layers[-params["unfreeze_layers"]:]:
        layer.trainable = True

    # Build model
    model = Sequential([base_model, GlobalAveragePooling2D()])
    for units, drop in zip(params["dense_layers"], params["dropout"]):
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(drop))
    model.add(Dense(1, activation="sigmoid"))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
                  loss="binary_crossentropy", metrics=["accuracy"])

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001)

    # Train model
    history = model.fit(
        train_generator,
        epochs=params["epochs"],
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Save history as CSV
    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = history.epoch
    history_csv_path = os.path.join(results_dir, f"history_test_{i}.csv")
    history_df.to_csv(history_csv_path, index=False)

    print(f"History for Test {i} saved at {history_csv_path}")

    # Plot Training vs Validation Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.epoch, history.history["accuracy"], label="Train Accuracy", marker="o")
    plt.plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training vs Validation Accuracy (Test {i})")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Training vs Validation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.epoch, history.history["loss"], label="Train Loss", marker="o")
    plt.plot(history.epoch, history.history["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss (Test {i})")
    plt.legend()
    plt.grid()
    plt.show()

    # Evaluate on validation and test sets
    val_loss, val_acc = model.evaluate(validation_generator)
    test_loss, test_acc = model.evaluate(test_generator)

    # Store results
    results.append({
        "Test": i,
        "Dense Layers": params["dense_layers"],
        "Dropout": params["dropout"],
        "Batch Size": params["batch_size"],
        "Learning Rate": params["lr"],
        "Unfreeze Layers": params["unfreeze_layers"],
        "Epochs": params["epochs"],
        "Validation Accuracy": val_acc,
        "Test Accuracy": test_acc,
        "Validation Loss": val_loss,
        "Test Loss": test_loss,
    })

# Save results to CSV
result_path = os.path.join(results_dir, "hyperparameter_results.csv")
df = pd.DataFrame(results)
df.to_csv(result_path, index=False)
print("\n✅ All tests completed! Results saved to hyperparameter_results.csv")
