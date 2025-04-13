import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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
df = pd.read_csv("../../results/hyperparameter_result/efficientnet_13_25/hyperparameter_results.csv")
# best_row = df.loc[df["Test Accuracy"].idxmax()]  # Get row with highest Test Accuracy
best_row = df.loc[df["Test"] == 13].squeeze()

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
TRAIN_DIR = "../../data/train"
TEST_DIR = "../../data/test"
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
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
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

## Metrics
# Compute additional test metrics
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n📌 Classification Metrics on Test Set:")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")

# Optional detailed report
print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))


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
# metrics = ['Accuracy', 'Loss']
# values = [
#     [val_acc, test_acc],  # Accuracies
#     [val_loss, test_loss]  # Losses
# ]
# x = np.arange(len(metrics))
# width = 0.35
# Final Metrics Plot (Validation vs Test including Precision, Recall, F1)
metrics = ['Accuracy', 'Loss', 'Precision', 'Recall', 'F1 Score']
validation_values = [val_acc, val_loss, np.nan, np.nan, np.nan]  # Only accuracy/loss for val
test_values = [test_acc, test_loss, precision, recall, f1]

x = np.arange(len(metrics))
width = 0.35

axes[2].bar(x - width/2, validation_values, width, label='Validation')
axes[2].bar(x + width/2, test_values, width, label='Test')
axes[2].set_title('Final Model Performance')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics)
axes[2].legend()
axes[2].grid(True, axis='y')

# Add text labels
for i, (val, test) in enumerate(zip(validation_values, test_values)):
    if not np.isnan(val):
        axes[2].text(i - width/2, val + 0.02, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    axes[2].text(i + width/2, test + 0.02, f'{test:.4f}', ha='center', va='bottom', fontsize=9)

# Adjust Y-axis
y_max = max(filter(lambda x: not np.isnan(x), validation_values + test_values))
axes[2].set_ylim(0, y_max * 1.2 if y_max > 1.0 else 1.2)


# axes[2].bar(x - width/2, [val_acc, val_loss], width, label='Validation')
# axes[2].bar(x + width/2, [test_acc, test_loss], width, label='Test')
# axes[2].set_title('Final Model Performance')
# axes[2].set_xticks(x)
# axes[2].set_xticklabels(metrics)
# axes[2].legend()
# axes[2].grid(True, axis='y')

# # Add text labels on the bars
# for i, metric in enumerate(metrics):
#     axes[2].text(i - width/2, values[i][0] + 0.02, f'{values[i][0]:.4f}', 
#                  ha='center', va='bottom', fontsize=9)
#     axes[2].text(i + width/2, values[i][1] + 0.02, f'{values[i][1]:.4f}', 
#                  ha='center', va='bottom', fontsize=9)

# # Adjust y-axis limits for the third subplot based on data values
# if max(val_loss, test_loss) > 1.0:
#     axes[2].set_ylim(0, max(val_loss, test_loss) * 1.2)
# else:
#     axes[2].set_ylim(0, max(1.0, max(val_acc, test_acc) * 1.2))

# Adjust spacing between subplots
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the Plot
plot_dir = "/home/untitled/Documents/Coding_Repository/python_journey/Capstone/waste-classification/references/efficientnet_13_25_13_gen/"
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, f"best_model_test_{int(best_row['Test'])}.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.show()  # Add this to display the plot
plt.close()

# ============================
# Confusion Matrix Plot
# ============================
cm = confusion_matrix(y_true, y_pred)
class_labels = list(test_generator.class_indices.keys())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save Confusion Matrix
cm_path = os.path.join(plot_dir, f"confusion_matrix_test_{int(best_row['Test'])}.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.show()
plt.close()

print(f"\n🧩 Confusion matrix saved: {cm_path}")

print(f"\n📊 Best performance plot saved: {plot_path}")

# ============================
# Export Metrics to CSV
# ============================
metrics_dict = {
    "Validation Accuracy": [val_acc],
    "Validation Loss": [val_loss],
    "Test Accuracy": [test_acc],
    "Test Loss": [test_loss],
    "Test Precision": [precision],
    "Test Recall": [recall],
    "Test F1 Score": [f1]
}

metrics_df = pd.DataFrame(metrics_dict)
metrics_path = os.path.join(plot_dir, f"metrics_summary_test_{int(best_row['Test'])}.csv")
metrics_df.to_csv(metrics_path, index=False)

print(f"\n📁 Metrics CSV saved: {metrics_path}")


# ============================
# Model Saving
# ============================
for attr in dir(model):
    try:
        val = getattr(model, attr)
        if isinstance(val, tf.Tensor):
            print(f"⚠️ Removing non-serializable attribute: {attr}")
            delattr(model, attr)
    except Exception as e:
        continue
models_dir = "/home/untitled/Documents/Coding_Repository/python_journey/Capstone/waste-classification/models_train/efficientnet_13_25_gen_13"
os.makedirs(models_dir, exist_ok=True)

# CHANGEaa
model_name = f"efficientnet_best_test_{int(best_row['Test'])}"

# Save in multiple formats
# model.save(os.path.join(models_dir, f"{model_name}.h5"), save_format='h5')
# model.save(os.path.join(models_dir, f"{model_name}_saved_model"))
# model.save_weights(os.path.join(models_dir, f"{model_name}_weights.h5"))

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
