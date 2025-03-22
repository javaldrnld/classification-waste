import os

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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# PALITAN NALANG 'TO
# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Paths to the directory
data_dir = '../data/raw'  # Main directory -> pet_bottle and unacceptable

# train-test split, 80/20
# Is it possible kaya na no need directory change? para random random
def create_train_test_split(data_dir, test_size=0.2):
    class_folders = os.listdir(data_dir)
    
    # Dict to store file paths by class
    # pet_bottle: for cls in class_folders -> append sa list
    all_files = {cls: [] for cls in class_folders}
    
    # Collect all file paths
    for cls in class_folders:
        class_path = os.path.join(data_dir, cls)
        if os.path.isdir(class_path):
            # ../data/raw/pet_bottle/picture.png ayon lang list comprehension by udemy course
            files = [os.path.join(cls, file) for file in os.listdir(class_path) 
                     if file.endswith(('.jpg', '.jpeg', '.png'))]
            # Append sa dict type class which is list
            all_files[cls] = files
    
    # Create train and test directories
    train_dir = '../data/train'
    test_dir = '../data/test'
    
    for directory in [train_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for cls in class_folders:
            class_dir = os.path.join(directory, cls)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
    
    # Split files for each class and move them
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

# Create train-test split
train_dir, test_dir = create_train_test_split(data_dir)

# Data generators with augmentation for training
# On the fly ang pag augment so save space also para 'di ma memorize ng model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% of training data for validation
)

# Test generator - only rescaling, no augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=8,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=8,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=8,
    class_mode='binary',
    shuffle=False
)

# Option 1: Simple CNN from scratch
def create_simple_model():
    model = Sequential([
        # First Conv Block - keep it small
        Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Option 2: Transfer learning with feature extraction
def create_transfer_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the base model
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Choose which model to use
# Simple Model -> Worst 
# Transfer Learning -> Good, just need early stopping
model = create_transfer_model()  

# Print model summary
model.summary()

# Create callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10, # 10
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.suptitle("MobileNetV2")
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

# Save the model
model.save('bottle_classifier.h5')

# Evaluate on the validation set (for hyperparameter tuning)
validation_evaluation = model.evaluate(validation_generator)
print(f"Validation Loss: {validation_evaluation[0]:.4f}")
print(f"Validation Accuracy: {validation_evaluation[1]:.4f}")

# Final evaluation on the completely separate test set
test_evaluation = model.evaluate(test_generator)
print(f"Test Loss: {test_evaluation[0]:.4f}")
print(f"Test Accuracy: {test_evaluation[1]:.4f}")

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Get predictions
y_pred = (model.predict(test_generator) > 0.5).astype(int)
y_true = test_generator.classes

# Create and plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                             display_labels=['pet_bottle', 'unacceptable'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import auc, roc_curve

# Get probability predictions
y_pred_prob = model.predict(test_generator)
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import average_precision_score, precision_recall_curve

precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
ap = average_precision_score(y_true, y_pred_prob)

plt.figure()
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, 
                          target_names=['pet_bottle', 'unacceptable']))


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image


def predict_and_display(img_path, model, class_indices=None):
    # If class_indices is not provided, use default alphabetical mapping
    if class_indices is None:
        class_indices = {'pet_bottle': 0, 'unacceptable': 1}
    
    # Create reverse mapping from index to class name
    index_to_class = {v: k for k, v in class_indices.items()}
    
    # Load the image with the correct size
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    
    # Get prediction probability
    pred_prob = model.predict(x)[0][0]
    
    # Determine predicted class
    # For binary classification with sigmoid, >0.5 is class 1, otherwise class 0
    pred_class_idx = 1 if pred_prob > 0.5 else 0
    pred_class_name = index_to_class[pred_class_idx]
    
    # Calculate confidence
    confidence = pred_prob if pred_class_idx == 1 else 1 - pred_prob
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class_name} ({confidence:.2%})\nRaw output: {pred_prob:.3f}")
    plt.axis('off')
    plt.show()
    
    return pred_class_name, confidence, pred_prob

# Test with a few examples
test_images = [
    "../data/new_test/IMG_0972.jpeg",
    "../data/new_test/IMG_0973.jpeg",
    "../data/new_test/IMG_0975.jpeg",
    "../data/new_test/IMG_0976.jpeg",
    "../data/new_test/IMG_0978.jpeg",
    "../data/new_test/IMG_0979.jpeg",
    "../data/new_test/IMG_0980.jpeg",
    "../data/new_test/IMG_0981.jpeg",
    "../data/new_test/IMG_0982.jpeg",
]

# Get class indices from your training generator
# E.g., class_indices = train_generator.class_indices
# Or define manually based on your knowledge of the dataset
class_indices = {'pet_bottle': 0, 'unacceptable': 1}  # Adjust this based on your actual mapping

for img_path in test_images:
    class_name, confidence, raw_output = predict_and_display(img_path, model, class_indices)
    print(f"File: {img_path.split('/')[-1]}")
    print(f"Class: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw model output: {raw_output:.4f}")


import matplotlib.cm as cm
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient of the predicted class with respect to the output feature map
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    # Gradient of the output wrt the last conv layer output
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of gradient over feature map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight output feature map with gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap for heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image.array_to_img(img))
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(jet_heatmap / 255)
    plt.title("Heatmap")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title("Superimposed")
    plt.axis("off")
    plt.show()


