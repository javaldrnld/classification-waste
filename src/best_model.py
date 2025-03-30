import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# ============================
# Load Best Hyperparameters
# ============================
df = pd.read_csv("../data/hyperparameter_result/efficientnetb0/hyperparameter_results.csv")
best_row = df.loc[df["Test Accuracy"].idxmax()]  # Get row with highest Test Accuracy

# Extract hyperparameters
DENSE_LAYERS = eval(best_row["Dense Layers"]) if isinstance(best_row["Dense Layers"], str) else [best_row["Dense Layers"]]
DROPOUTS = eval(best_row["Dropout"]) if isinstance(best_row["Dropout"], str) else [best_row["Dropout"]]
BATCH_SIZE = int(best_row["Batch Size"])
LEARNING_RATE = float(best_row["Learning Rate"])
UNFREEZE_LAYERS = int(best_row["Unfreeze Layers"])
EPOCHS = int(best_row["Epochs"])

# Print selected hyperparameters
print("\n✅ Best Hyperparameters Selected:")
print(f"   Dense Layers: {DENSE_LAYERS}")
print(f"   Dropout Rates: {DROPOUTS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Unfreeze Layers: {UNFREEZE_LAYERS}")
print(f"   Epochs: {EPOCHS}")

# ============================
# Data Setup
# ============================
TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
IMG_HEIGHT = 224
IMG_WIDTH = 224
VALIDATION_SPLIT = 0.2

# Define Data Augmentation
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

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="binary", subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="binary", subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
)

# ============================
# Build Model with Best Hyperparameters
# ============================
base_model = EfficientNetB0(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet")

# Freeze initial layers, unfreeze last `UNFREEZE_LAYERS`
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False
for layer in base_model.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

# Build Custom Top Layers
model = Sequential([base_model, GlobalAveragePooling2D()])
for units, dropout in zip(DENSE_LAYERS, DROPOUTS):
    model.add(Dense(units, activation="relu"))
    model.add(Dropout(dropout))

# Final output layer
model.add(Dense(1, activation="sigmoid"))

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# Training the Model
# ============================
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-5)

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

hist = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# ============================
# Model Evaluation & Plotting
# ============================
validation_evaluation = model.evaluate(validation_generator)
test_evaluation = model.evaluate(test_generator)
val_loss, val_acc = validation_evaluation[0], validation_evaluation[1]
test_loss, test_acc = test_evaluation[0], test_evaluation[1]

# Create Figure for Plots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))  
fig.suptitle(f"Best Model Performance (Test {int(best_row['Test'])})", fontsize=16)

# Accuracy Plot
axes[0].plot(hist.history['accuracy'], marker='o', label='Train Accuracy')
axes[0].plot(hist.history['val_accuracy'], marker='o', label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(loc='lower right')
axes[0].grid(True)

# Loss Plot
axes[1].plot(hist.history['loss'], marker='o', label='Train Loss')
axes[1].plot(hist.history['val_loss'], marker='o', label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(loc='upper right')
axes[1].grid(True)

# Final Metrics Plot (Test vs Validation)
metrics = ['Accuracy', 'Loss']
values = [
    [val_acc, test_acc],  # Accuracies
    [val_loss, test_loss]  # Losses
]
x = np.arange(len(metrics))
width = 0.35

axes[2].bar(x - width/2, [val_acc, val_loss], width, label='Validation')
axes[2].bar(x + width/2, [test_acc, test_loss], width, label='Test')
axes[2].set_title('Final Model Performance')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics)
axes[2].legend()
axes[2].grid(True, axis='y')

# Add text labels on the bars
for i, metric in enumerate(metrics):
    axes[2].text(i - width/2, values[i][0] + 0.02, f'{values[i][0]:.4f}', 
                 ha='center', va='bottom', fontsize=9)
    axes[2].text(i + width/2, values[i][1] + 0.02, f'{values[i][1]:.4f}', 
                 ha='center', va='bottom', fontsize=9)

# Adjust y-axis limits for the third subplot based on data values
if max(val_loss, test_loss) > 1.0:
    axes[2].set_ylim(0, max(val_loss, test_loss) * 1.2)
else:
    axes[2].set_ylim(0, max(1.0, max(val_acc, test_acc) * 1.2))

# Adjust spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the Plot
plot_dir = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/references/efficientnetb0/"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, f"best_model_test_{int(best_row['Test'])}.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.show()  # Add this to display the plot
plt.close()

print(f"\n📊 Best performance plot saved: {plot_path}")

# ============================
# Model Saving
# ============================
models_dir = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/models_train/efficientnetb0"
os.makedirs(models_dir, exist_ok=True)

model_name = f"efficientnetb0_best_test_{int(best_row['Test'])}"

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
