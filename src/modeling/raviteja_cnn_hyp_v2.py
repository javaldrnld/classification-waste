import gc
import json
import os
import time
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.applications.efficientnet import (
# EfficientNetB0,
# preprocess_input,
# )
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

TRAIN_DIR = "../../data/train"
TEST_DIR = "../../data/test"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.0001
results_dir = "../../results/hyperparameter_result_13_25/mobilenetv2_v1"
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
# hyperparameter_tests = [
#     {"dense_layers": [128], "dropout": [0.5], "batch_size": 4, "lr": 0.0001, "unfreeze_layers": 10, "epochs": 20},
#     {"dense_layers": [128, 64], "dropout": [0.5, 0.6], "batch_size": 4, "lr": 0.00005, "unfreeze_layers": 10, "epochs": 20},
#     {"dense_layers": [256, 128], "dropout": [0.4, 0.5], "batch_size": 8, "lr": 0.0001, "unfreeze_layers": 20, "epochs": 30},
#     {"dense_layers": [512, 256, 128], "dropout": [0.4, 0.5, 0.6], "batch_size": 16, "lr": 0.00005, "unfreeze_layers": 30, "epochs": 40},
#     {"dense_layers": [256], "dropout": [0.4], "batch_size": 8, "lr": 0.0001, "unfreeze_layers": 15, "epochs": 25},
#     {"dense_layers": [128, 64, 32], "dropout": [0.5, 0.6, 0.3], "batch_size": 4, "lr": 0.0001, "unfreeze_layers": 10, "epochs": 20},
#     {"dense_layers": [64], "dropout": [0.3], "batch_size": 8, "lr": 0.0005, "unfreeze_layers": 5, "epochs": 15},
#     {"dense_layers": [128, 64], "dropout": [0.5, 0.3], "batch_size": 8, "lr": 0.0001, "unfreeze_layers": 15, "epochs": 20},
#     {"dense_layers": [256, 128, 64], "dropout": [0.4, 0.5, 0.6], "batch_size": 16, "lr": 0.00005, "unfreeze_layers": 20, "epochs": 30},
#     {"dense_layers": [128], "dropout": [0.5], "batch_size": 4, "lr": 0.0001, "unfreeze_layers": 5, "epochs": 20},
#     {"dense_layers": [128, 64], "dropout": [0.5, 0.5], "batch_size": 16, "lr": 0.0001, "unfreeze_layers": 10, "epochs": 25},
#     {"dense_layers": [256], "dropout": [0.6], "batch_size": 32, "lr": 0.00005, "unfreeze_layers": 5, "epochs": 20},
#     {"dense_layers": [128, 64, 32], "dropout": [0.4, 0.5, 0.5], "batch_size": 16, "lr": 0.0001, "unfreeze_layers": 15, "epochs": 30},
#     {"dense_layers": [64], "dropout": [0.5], "batch_size": 32, "lr": 0.00007, "unfreeze_layers": 0, "epochs": 15},
#     {"dense_layers": [128, 32], "dropout": [0.6, 0.6], "batch_size": 16, "lr": 0.0001, "unfreeze_layers": 8, "epochs": 20},
# ]


# hyperparameter_tests = [
    # Overfitting configurations
    # {"dense_layers": [1024, 512, 256], "dropout": [0.1, 0.1, 0.1], "batch_size": 2, "lr": 0.001, "unfreeze_layers": 50, "epochs": 100},
    # {"dense_layers": [512, 512], "dropout": [0, 0], "batch_size": 1, "lr": 0.0005, "unfreeze_layers": 100, "epochs": 80},
    # {"dense_layers": [1024, 512, 256, 128], "dropout": [0.05, 0.05, 0.05, 0.05], "batch_size": 4, "lr": 0.0008, "unfreeze_layers": 40, "epochs": 60},
    # {"dense_layers": [2048], "dropout": [0], "batch_size": 2, "lr": 0.001, "unfreeze_layers": 30, "epochs": 50},
    # {"dense_layers": [768, 384, 192], "dropout": [0.1, 0.1, 0.1], "batch_size": 4, "lr": 0.0005, "unfreeze_layers": 80, "epochs": 70},
    
    # Generalization configurations
# ]

hyperparameter_tests = [
    # Balanced small network - moderate dropout, batch size, and learning rate
    {"dense_layers": [128], "dropout": [0.3], "batch_size": 4, "lr": 0.0001, "weight_decay": 0.01, "unfreeze_layers": 10, "epochs": 20},
    
    # Deeper network with appropriate dropout scaling
    {"dense_layers": [128, 64], "dropout": [0.4, 0.3], "batch_size": 4, "lr": 0.00005, "weight_decay": 0.005, "unfreeze_layers": 10, "epochs": 20},
    
    # Medium network with more unfrozen layers and sufficient epochs to adapt
    {"dense_layers": [256, 128], "dropout": [0.4, 0.3], "batch_size": 8, "lr": 0.0001, "weight_decay": 0.01, "unfreeze_layers": 20, "epochs": 30},
    
    # Large network, many unfrozen layers with enough epochs to fully utilize them
    {"dense_layers": [512, 256, 128], "dropout": [0.5, 0.4, 0.3], "batch_size": 16, "lr": 0.00005, "weight_decay": 0.015, "unfreeze_layers": 30, "epochs": 45},
    
    # Single layer with moderate dropout
    {"dense_layers": [256], "dropout": [0.4], "batch_size": 8, "lr": 0.0001, "weight_decay": 0.01, "unfreeze_layers": 15, "epochs": 25},
    
    # Deep network with scaled dropout
    {"dense_layers": [128, 64, 32], "dropout": [0.4, 0.3, 0.2], "batch_size": 4, "lr": 0.0001, "weight_decay": 0.008, "unfreeze_layers": 10, "epochs": 25},
    
    # Small network with appropriate dropout
    {"dense_layers": [64], "dropout": [0.2], "batch_size": 8, "lr": 0.0002, "weight_decay": 0.01, "unfreeze_layers": 5, "epochs": 15},
    
    # Medium network with balanced parameters
    {"dense_layers": [128, 64], "dropout": [0.4, 0.2], "batch_size": 8, "lr": 0.0001, "weight_decay": 0.01, "unfreeze_layers": 15, "epochs": 25},
    
    # Deeper network with progressive dropout
    {"dense_layers": [256, 128, 64], "dropout": [0.4, 0.3, 0.2], "batch_size": 16, "lr": 0.00007, "weight_decay": 0.007, "unfreeze_layers": 20, "epochs": 35},
    
    # Simple network with few unfrozen layers
    {"dense_layers": [128], "dropout": [0.3], "batch_size": 4, "lr": 0.0001, "weight_decay": 0.01, "unfreeze_layers": 5, "epochs": 15},
    
    # Higher batch size with appropriately higher learning rate
    {"dense_layers": [128, 64], "dropout": [0.4, 0.3], "batch_size": 16, "lr": 0.0002, "weight_decay": 0.01, "unfreeze_layers": 10, "epochs": 20},
    
    # Very large batch size with properly scaled learning rate
    {"dense_layers": [256], "dropout": [0.4], "batch_size": 32, "lr": 0.0003, "weight_decay": 0.005, "unfreeze_layers": 5, "epochs": 20},
    
    # Deep network with longer training time for adaptation
    {"dense_layers": [128, 64, 32], "dropout": [0.4, 0.3, 0.2], "batch_size": 16, "lr": 0.0001, "weight_decay": 0.01, "unfreeze_layers": 15, "epochs": 30},
    
    # Minimal fine-tuning (no unfreezing) with appropriate training duration
    {"dense_layers": [64], "dropout": [0.2], "batch_size": 32, "lr": 0.0002, "weight_decay": 0.001, "unfreeze_layers": 0, "epochs": 10},
    
    # Medium complexity with moderate unfreezing
    {"dense_layers": [128, 32], "dropout": [0.4, 0.2], "batch_size": 16, "lr": 0.0001, "weight_decay": 0.01, "unfreeze_layers": 8, "epochs": 20},
    
    # Progressive learning rate with significant unfreezing
    {"dense_layers": [256, 128], "dropout": [0.4, 0.3], "batch_size": 8, "lr": 0.00005, "weight_decay": 0.015, "unfreeze_layers": 25, "epochs": 40},
    
    # Large batch with high learning rate for faster convergence
    {"dense_layers": [128, 64], "dropout": [0.3, 0.2], "batch_size": 64, "lr": 0.0005, "weight_decay": 0.02, "unfreeze_layers": 10, "epochs": 15},
]


# Store results
results = []

os.makedirs(results_dir, exist_ok=True)

for i, params in enumerate(hyperparameter_tests, 1):
    print(f"\n🔹 Running Test {i} with Hyperparameters: {params}")

    # Load MobileNetV2 as base model
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    if params["unfreeze_layers"] > 0:
        # Freeze layers except the last `unfreeze_layers`
        for layer in base_model.layers[:-params["unfreeze_layers"]]:
            layer.trainable = False
        for layer in base_model.layers[-params["unfreeze_layers"]:]:
            layer.trainable = True
    else:
        for layer in base_model.layers:
            layer.trainable = False

    # Build model
    model = Sequential([base_model, GlobalAveragePooling2D()])
    for units, drop in zip(params["dense_layers"], params["dropout"]):
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(drop))
    model.add(Dense(1, activation="sigmoid"))

    # Calculate total training steps for learning rate scheduler
    total_steps = len(train_generator) * params["epochs"]
    warmup_steps = int(total_steps * 0.1)

    # LEarning rate scheduler with warmup
    def lr_schedule(epoch, lr):
        step = epoch * len(train_generator)
        # Warmup phase
        if step < warmup_steps:
            return params["lr"] * (step / warmup_steps)
        return lr

    # Compile model with weight decay ausing AdamW
    optimizer = tfa.optimizers.AdamW(
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"]
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Define callbacks
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"model_test_{i}.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    start_time = time.time()

    # Train model
    history = model.fit(
        train_generator,
        epochs=params["epochs"],
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr, lr_scheduler, model_checkpoint],
        verbose=1
    )

    training_time = time.time() - start_time

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
    val1_loss, val_accuracy, val_auc, val_precision, val_recall = model.evaluate(validation_generator, verbose=0)
    # Store results
    result = {
        "Test": i,
        # "Dense Layers": params["dense_layers"],
        # "Dropout": params["dropout"],
        # "Batch Size": params["batch_size"],
        # "Learning Rate": params["lr"],
        # "Unfreeze Layers": params["unfreeze_layers"],
        # "Epochs": params["epochs"],
        # "Validation Accuracy": val_acc,
        # "Test Accuracy": test_acc,
        # "Validation Loss": val_loss,
        # "Test Loss": test_loss,
        "params": params,
        "val_accuracy": val_accuracy,
        "val_loss": val1_loss,
        "val_auc": val_auc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "training_time": training_time,
        "epochs_completed": len(history.history["loss"]),
        "stopped_early": len(history.history["loss"]) < params["epochs"],
        "best_epoch": np.argmin(history.history["val_loss"]) + 1
        
    }
    results.append(result)
    print(f"\n✅ Test {i} completed in {training_time:.2f} seconds")
    print(f"  • Validation Accuracy: {val_accuracy:.4f}")
    print(f"  • Validation AUC: {val_auc:.4f}")
    print(f"  • Validation Loss: {val_loss:.4f}")
    print(f"  • Best epoch: {result['best_epoch']}")
    if result["stopped_early"]:
        print(f"  • Early stopping activated at epoch {len(history.history['loss'])}")
    
    # Clear memory
    del model, history
    tf.keras.backend.clear_session()
    gc.collect()

# Save results to CSV
result_path = os.path.join(results_dir, "hyperparameter_results.csv")
df = pd.DataFrame(results)
df.to_csv(result_path, index=False)
print("\n✅ All tests completed! Results saved to hyperparameter_results.csv")

# Delete
# Print summary of all results, sorted by validation accuracy
print("\n📊 HYPERPARAMETER OPTIMIZATION RESULTS (Sorted by Validation Accuracy):")
sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)

for i, result in enumerate(sorted_results, 1):
    print(f"\n{i}. Test {result['test_id']} - Accuracy: {result['val_accuracy']:.4f}, AUC: {result['val_auc']:.4f}")
    print(f"   Parameters: {json.dumps(result['params'], indent=2)}")
    print(f"   Training time: {result['training_time'] / 60:.2f} minutes, Best epoch: {result['best_epoch']}")

# Save results to file
with open("hyperparameter_optimization_results.json", "w") as f:
    json.dump(sorted_results, f, indent=2)

print("\n💾 Results saved to hyperparameter_optimization_results.json")

# Plot top 5 configurations learning curves
plt.figure(figsize=(15, 10))

for i, result in enumerate(sorted_results[:5]):
    test_id = result["test_id"]
    # Load the history from TensorBoard logs or re-run the top models if needed
    # This is a simplified placeholder - you may need to adjust based on your TensorBoard setup
    log_dir = f"../logs/test{test_id}"
    # Placeholder for visualization code
    plt.subplot(2, 3, i+1)
    plt.title(f"Test {test_id} - Acc: {result['val_accuracy']:.4f}")
    # Add actual plotting code based on how you store/access your history

plt.tight_layout()
plt.savefig("top_models_comparison.png")
print("\n📈 Learning curves for top 5 models saved to top_models_comparison.png")