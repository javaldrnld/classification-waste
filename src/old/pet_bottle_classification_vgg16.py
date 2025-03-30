import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
CONFIG = {
    "img_height": 224,
    "img_width": 224,
    "batch_size": 16,
    "data_dir": '../data/raw',
    "train_dir": '../data/train',
    "test_dir": '../data/test',
    "models_dir": '../models_train',
    "test_size": 0.2,
    "validation_split": 0.2,
    "learning_rate": 1e-4,
    "epochs": 50,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5
}

def create_directory(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def create_train_test_split(
    data_dir, 
    train_dir, 
    test_dir, 
    test_size=0.2
):
    """Split data into training and testing sets"""
    # Identify class folders
    class_folders = [
        f for f in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, f))
    ]
    
    # Create output directories
    for directory in [train_dir, test_dir]:
        create_directory(directory)
        for cls in class_folders:
            create_directory(os.path.join(directory, cls))
    
    # Collect and split files for each class
    for cls in class_folders:
        class_path = os.path.join(data_dir, cls)
        
        # Get all image files
        files = [
            f for f in os.listdir(class_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        # Split files
        train_files, test_files = train_test_split(
            files, 
            test_size=test_size, 
            random_state=42
        )
        
        # Copy train files
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(train_dir, cls, file)
            tf.io.gfile.copy(src, dst, overwrite=True)
        
        # Copy test files
        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(test_dir, cls, file)
            tf.io.gfile.copy(src, dst, overwrite=True)
    
    return train_dir, test_dir

def create_data_preprocessing(
    train_dir, 
    test_dir, 
    img_height=224, 
    img_width=224, 
    batch_size=16,
    validation_split=0.2
):
    # Preprocessing and augmentation layers
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
    ])

    # On the fly augmentation to avoid memorizing sheesh
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset='training',
        seed=42, # Need 'to para same same lang image 
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary',  # For binary classification
        
    )

    # # 
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset='validation',
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary'  # For binary classification
    )

    # Test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary'  # For binary classification
    )

    # Apply augmentation to train dataset
    # Learned from zero to mastery python udemy course tas documentation nung keras
    augmented_train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Normalize datasets
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    augmented_train_ds = augmented_train_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_ds = val_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    test_ds = test_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return augmented_train_ds, val_ds, test_ds
def create_vgg16_model(
    img_height, 
    img_width, 
    learning_rate=1e-4, 
    freeze_base_layers=True
):
    """Create VGG16 transfer learning model"""
    # Load base VGG16 model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_height, img_width, 3)
    )
    
    # Freeze or unfreeze base layers
    base_model.trainable = not freeze_base_layers
    
    # Build transfer learning model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(
    model, 
    train_generator, 
    validation_generator, 
    epochs=50,
    early_stopping_patience=10,
    reduce_lr_patience=5
):
    """Train the model with callbacks"""
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=reduce_lr_patience,
        min_lr=1e-6
    )

    # Model checkpoint
    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )

    # Calculate steps
    # steps_per_epoch = train_generator.samples // train_generator.batch_size
    # validation_steps = validation_generator.samples // validation_generator.batch_size
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    

    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    return history

def plot_training_history(history, title="VGG16 Model Performance"):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()

def save_model(model, models_dir, model_name):
    """Save model in multiple formats"""
    create_directory(models_dir)
    
    # H5 format
    model.save(
        os.path.join(models_dir, f"{model_name}.h5"), 
        save_format='h5'
    )

    # SavedModel format
    model.save(
        os.path.join(models_dir, f"{model_name}_saved_model")
    )

    # Weights only
    model.save_weights(
        os.path.join(models_dir, f"{model_name}_weights.h5")
    )

def convert_to_tflite(model, models_dir, model_name, quantize=False):
    """Convert model to TFLite"""
    create_directory(models_dir)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    output_path = os.path.join(
        models_dir, 
        f"{model_name}_{'quantized' if quantize else 'standard'}.tflite"
    )
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")

def evaluate_model(model, validation_generator, test_generator):
    """Evaluate model performance"""
    # Validation evaluation
    val_eval = model.evaluate(validation_generator)
    print("Validation Metrics:")
    print(f"Loss: {val_eval[0]:.4f}")
    print(f"Accuracy: {val_eval[1]:.4f}")

    # Test evaluation
    test_eval = model.evaluate(test_generator)
    print("\nTest Metrics:")
    print(f"Loss: {test_eval[0]:.4f}")
    print(f"Accuracy: {test_eval[1]:.4f}")

def main():
    """Main training pipeline"""
    # Create train-test split
    train_dir, test_dir = create_train_test_split(
        CONFIG["data_dir"],
        CONFIG["train_dir"],
        CONFIG["test_dir"],
        CONFIG["test_size"]
    )

    # Create data generators
    train_generator, validation_generator, test_generator = create_data_preprocessing(
        train_dir,
        test_dir,
        CONFIG["img_height"],
        CONFIG["img_width"],
        CONFIG["batch_size"],
        CONFIG["validation_split"]
    )

    # Create VGG16 model
    model = create_vgg16_model(
        CONFIG["img_height"],
        CONFIG["img_width"],
        CONFIG["learning_rate"],
        freeze_base_layers=True
    )

    # Print model summary
    model.summary()

    # Train model
    history = train_model(
        model,
        train_generator,
        validation_generator,
        CONFIG["epochs"],
        CONFIG["early_stopping_patience"],
        CONFIG["reduce_lr_patience"]
    )

    # Plot training history
    plot_training_history(history, "VGG16 Transfer Learning")

    # Save model
    # save_model(model, CONFIG["models_dir"], "vgg16_classifier")

    # # Convert to TFLite
    # convert_to_tflite(model, CONFIG["models_dir"], "vgg16_classifier")

    # Evaluate model
    evaluate_model(model, validation_generator, test_generator)

if __name__ == "__main__":
    main()