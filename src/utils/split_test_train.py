
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

CONFIG = {
    "img_height": 224,
    "img_width": 224,
    "batch_size": 8,
    "data_dir": '../../data/aug/new_v1',
    "train_dir": '../../data/train',
    "test_dir": '../../data/test',
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

# ossible improvement dito is on the fly 'yong test and validation and training -> No creating offline data
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


train_dir, test_dir = create_train_test_split(
    CONFIG["data_dir"], 
    CONFIG["train_dir"], 
    CONFIG["test_dir"], 
    CONFIG["test_size"]
)