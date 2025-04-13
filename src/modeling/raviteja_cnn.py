# Import necessary library
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from cv2 import cvtColor

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Define training and test dataset
TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0001

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

# Create MobileNetV2 base model -> Freeze top layer 
# Then check the result after training and unfreeze some layer
base_model = VGG16(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)
# base_model.trainable = False
# Allow some last few layer to train
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True
    

# Build custome top layer
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# No EarlyCallbacks
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Early Callbacks in case of overfitting
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

# Calculate steps per epoch

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Train the top layers
hist = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# Create a figure with 3 rows and 1 column
fig, axes = plt.subplots(3, 1, figsize=(7, 12))  
fig.suptitle("Model Performance", fontsize=16)

# Model Accuracy Plot
axes[0].plot(hist.history['accuracy'], marker='o', label='Train Accuracy')
axes[0].plot(hist.history['val_accuracy'], marker='o', label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(loc='lower right')
axes[0].grid(True)

# Model Loss Plot
axes[1].plot(hist.history['loss'], marker='o', label='Train Loss')
axes[1].plot(hist.history['val_loss'], marker='o', label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='upper right')
axes[1].grid(True)

# Evaluate on validation and test sets
validation_evaluation = model.evaluate(validation_generator)
test_evaluation = model.evaluate(test_generator)
val_loss, val_acc = validation_evaluation[0], validation_evaluation[1]
test_loss, test_acc = test_evaluation[0], test_evaluation[1]

# Bar Plot for Validation vs Test Performance
metrics = ["Loss", "Accuracy"]
val_values = [val_loss, val_acc]
test_values = [test_loss, test_acc]
x = np.arange(len(metrics))

bars1 = axes[2].bar(x - 0.2, val_values, width=0.4, label="Validation", color="blue")
bars2 = axes[2].bar(x + 0.2, test_values, width=0.4, label="Test", color="green")

# Add labels above bars
for bar, value in zip(bars1, val_values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}", 
                 ha='center', va='bottom', fontsize=10, color="black")
for bar, value in zip(bars2, test_values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}", 
                 ha='center', va='bottom', fontsize=10, color="black")

# Labels and title
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics)
axes[2].set_ylabel("Score")
axes[2].set_title("Validation vs Test Performance")
axes[2].set_ylim(0, 1.1)
axes[2].legend()
axes[2].grid(axis="y", linestyle="--", alpha=0.7)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust to fit the suptitle
plt.show()


# ============================
# Model Saving
# ============================
models_dir = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/models_train/vgg16"
os.makedirs(models_dir, exist_ok=True)

model_name = "mobilenetv2_best_test_3"

# Save in multiple formats
model.save(os.path.join(models_dir, f"{model_name}.h5"), save_format='h5')
model.save(os.path.join(models_dir, f"{model_name}_saved_model"))
model.save_weights(os.path.join(models_dir, f"{model_name}_weights.h5"))

# Convert & Save TFLite
def convert_to_tflite(model, output_dir, model_name, quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    output_path = os.path.join(output_dir, f"{model_name}_{'quantized' if quantize else 'standard'}.tflite")

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")

convert_to_tflite(model, models_dir, model_name, quantize=False)
# convert_to_tflite(model, models_dir, model_name, quantize=True)

print(f"\n✅ Model saved in {models_dir}")