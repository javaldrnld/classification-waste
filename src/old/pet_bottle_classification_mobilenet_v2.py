import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

# Configuration
CONFIG = {
    "img_height": 224,
    "img_width": 224,
    "batch_size": 8,
    "data_dir": '../data/raw',
    "train_dir": '../data/train',
    "test_dir": '../data/test',
    "models_dir": '../models_train',
    "test_size": 0.2,
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5
}

def create_directory(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

# Possible improvement dito is on the fly 'yong test and validation and training -> No creating offline data
def create_train_test_split(data_dir, train_dir, test_dir, test_size=0.2):
    """Split data into training and testing sets
    class_folders = []
    
    for f in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, f))
            append(f)
    """
    
    class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(class_folders)
    # Store class in dictionary
    # "pet_bottle": ["image_1", "image_2"]
    """
    for cls in class_folders:
        cls: []
    """
    all_files = {cls: [] for cls in class_folders}

    # Append lang sa list 'yong mga files under pet_bottle dir and unacceptable dir
    for cls in class_folders:
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path):
            files = [os.path.join(cls, file) for file in os.listdir(class_path) 
                     if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            all_files[cls] = files
    
    # Create train and test dir 
    for directory in [train_dir, test_dir]:
        create_directory(directory)
        
        # Create ng pet_bottle and unacceptable dir sa test and train directory
        for cls in class_folders:
            class_dir = os.path.join(directory, cls)
            create_directory(class_dir)
    
    # Here separate 'yong files into test and train using train_test_split
    for cls, files in all_files.items():
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
        
        # Copy files to train and test directories
        for file in train_files:
            src = os.path.join(data_dir, file)
            dst = os.path.join(train_dir, file)
            tf.io.gfile.copy(src, dst, overwrite=True)
            
        for file in test_files:
            src = os.path.join(data_dir, file)
            dst = os.path.join(test_dir, file)
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
        tf.keras.layers.RandomRotation(0.3),  # Increase rotation
        tf.keras.layers.RandomZoom(0.3),  # Increase zoom
        tf.keras.layers.RandomTranslation(0.3, 0.3)  # More translation
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
    class_names = train_ds.class_names

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

    return augmented_train_ds, val_ds, test_ds, class_names
def create_simple_model(img_height, img_width, learning_rate=0.001):
    """Create a simple CNN model from scratch"""
    model = Sequential([
        # First Conv Block - keep it small
        Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.1),
        
        # Second Conv Block
        Conv2D(32, (3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),
        
        # Global Pooling
        GlobalAveragePooling2D(),
        Dropout(0.5),
        
        # Output Layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transfer_model(img_height, img_width, learning_rate=0.0001):
    """Create a transfer learning model using MobileNetV2"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    # base_model.trainable = False  # Freeze the base model
    # Allow some last few layer to train
    for layer in base_model.layers[:-10]:  
        layer.trainable = False
    for layer in base_model.layers[-10:]:  
        layer.trainable = True    
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), # -> # according to google, better for smaller datasets and transfer learning ++++ avoid overfitting 
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_generator, validation_generator, epochs=50, 
                early_stopping_patience=10, reduce_lr_patience=5):
    """Train the model with callbacks"""
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=reduce_lr_patience,
        min_lr=0.00001
    )

    # Calculate steps per epoch
    # steps_per_epoch = train_generator.samples // train_generator.batch_size
    # validation_steps = validation_generator.samples // validation_generator.batch_size
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)

    # Fit the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping, reduce_lr]
    )
    
    return history

def plot_training_history(history, title="Model Performance"):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.show()

def save_model(model, models_dir, model_name):
    """Save the model in multiple formats"""
    create_directory(models_dir)
    
    # 1. Save in H5 format (weights + architecture + optimizer state)
    model.save(os.path.join(models_dir, f"{model_name}.h5"), save_format='h5')

    # 2. Save in SavedModel format for better compatibility
    model.save(os.path.join(models_dir, f"{model_name}_saved_model"))

    # 3. Save just the weights for maximum flexibility
    model.save_weights(os.path.join(models_dir, f"{model_name}_weights.h5"))


def convert_to_tflite(model, models_dir, model_name, quantize=False):
    """Convert model to TFLite format with optional quantization"""
    create_directory(models_dir)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
    
    tflite_model = converter.convert()
    output_path = os.path.join(
        models_dir,
        f"{model_name}_{'quantized' if quantize else 'standard'}.tflite"
    )

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")

def evaluate_model(model, validation_generator, test_generator):
    """Evaluate model on validation and test sets"""
    # Evaluate on the validation set
    validation_evaluation = model.evaluate(validation_generator)
    print(f"Validation Loss: {validation_evaluation[0]:.4f}")
    print(f"Validation Accuracy: {validation_evaluation[1]:.4f}")

    # Final evaluation on the test set
    test_evaluation = model.evaluate(test_generator)
    print(f"Test Loss: {test_evaluation[0]:.4f}")
    print(f"Test Accuracy: {test_evaluation[1]:.4f}")
    
    return validation_evaluation, test_evaluation

def main():
    """Main function to run the bottle classifier training pipeline"""
    # Create train-test split
    train_dir, test_dir = create_train_test_split(
        CONFIG["data_dir"], 
        CONFIG["train_dir"], 
        CONFIG["test_dir"], 
        CONFIG["test_size"]
    )
    
    # Create data generators
    train_generator, validation_generator, test_generator, class_names = create_data_preprocessing(
        train_dir, 
        test_dir, 
        CONFIG["img_height"], 
        CONFIG["img_width"], 
        CONFIG["batch_size"], 
        CONFIG["validation_split"]
    )
    
    # Create model (choose between simple and transfer learning)
    # model = create_simple_model(CONFIG["img_height"], CONFIG["img_width"], CONFIG["learning_rate"])
    model = create_transfer_model(CONFIG["img_height"], CONFIG["img_width"], learning_rate=0.0001)
    
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
    print(f"Class mapping: {class_names}")  # ✅ Correct way to get class names
    # Plot training history
    plot_training_history(history, "MobileNetV2 Transfer Learning")
    
    # Save model
    save_model(model, CONFIG["models_dir"], "mobilenet_v2")
    
    # # Convert to TFLite
    convert_to_tflite(model, CONFIG["models_dir"],"mobilenet_v2")
    
    # Optionally, create quantized model
    # convert_to_tflite(model, CONFIG["models_dir"], "bottle_classifier_v1", quantize=True)
    
    # Evaluate model
    evaluate_model(model, validation_generator, test_generator)

    # import numpy as np
    # from sklearn.metrics import classification_report, confusion_matrix

    # # ✅ Extract true labels correctly
    # y_true = tf.concat([y for x, y in test_generator], axis=0).numpy()

    # # ✅ Extract predictions correctly
    # y_pred = (tf.concat([model(x) for x, y in test_generator], axis=0).numpy() > 0.5).astype(int)

    # # Print confusion matrix and classification report
    # print(confusion_matrix(y_true, y_pred))
    # print(classification_report(y_true, y_pred, target_names=class_names))

    # # PROC
    # from sklearn.metrics import auc, roc, curve
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from sklearn.metrics import (
    #     auc,
    #     classification_report,
    #     confusion_matrix,
    #     precision_recall_curve,
    #     roc_curve,
    # )

    # # ✅ Extract true labels correctly
    # y_true = tf.concat([y for x, y in test_generator], axis=0).numpy()

    # # ✅ Extract predicted probabilities (before thresholding)
    # y_probs = tf.concat([model(x) for x, y in test_generator], axis=0).numpy().flatten()

    # # ✅ Convert probabilities to binary predictions
    # y_pred = (y_probs > 0.5).astype(int)

    # # 🔹 Print Confusion Matrix & Classification Report
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_true, y_pred))
    # print("\nClassification Report:")
    # print(classification_report(y_true, y_pred, target_names=class_names))

    # # 🔹 ROC Curve
    # fpr, tpr, _ = roc_curve(y_true, y_probs)  # Compute ROC
    # roc_auc = auc(fpr, tpr)  # Compute AUC score

    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random chance line
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend(loc="lower right")

    # # 🔹 Precision-Recall Curve
    # precision, recall, _ = precision_recall_curve(y_true, y_probs)

    # plt.subplot(1, 2, 2)
    # plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc="lower left")

    # plt.tight_layout()
    # plt.show()
 
    

if __name__ == "__main__":
    main()