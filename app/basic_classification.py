import os

import cv2
import numpy as np
import tensorflow as tf


# First, let's fix the model loading issue
def load_model_with_custom_objects(model_path):
    """Load model with custom objects handling to fix compatibility issues."""
    try:
        # Try loading with standard method first
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Standard loading failed: {e}")
        
        # Try loading with custom_objects approach
        try:
            # Define a custom InputLayer that ignores batch_shape
            def custom_input_layer_deserializer(config):
                # Remove batch_shape if it exists
                if 'batch_shape' in config:
                    input_shape = config.pop('batch_shape')[1:]  # Remove batch dimension
                    config['input_shape'] = input_shape
                # Create the layer
                return tf.keras.layers.InputLayer(**config)
            
            # Register the custom deserializer
            custom_objects = {
                'InputLayer': custom_input_layer_deserializer
            }
            
            # Try loading with custom objects
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print(f"Successfully loaded model with custom objects from {model_path}")
            return model
        except Exception as detailed_e:
            print(f"Custom loading also failed: {detailed_e}")
            
            # Last resort: try to create a new model with the same architecture and load weights
            try:
                # Create the MobileNetV2 model as in your training script
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
                base_model.trainable = False
                
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                # Compile the model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Try to load just the weights
                weights_path = model_path.replace('.h5', '_weights.h5')
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                    print(f"Created new model and loaded weights from {weights_path}")
                else:
                    # If weights file doesn't exist, try loading weights from original file
                    try:
                        model.load_weights(model_path)
                        print(f"Created new model and loaded weights from {model_path}")
                    except:
                        print("Could not load weights. Using freshly initialized model.")
                
                return model
            except Exception as final_e:
                print(f"All loading methods failed: {final_e}")
                raise Exception("Could not load the model using any method.")

# Try to load the H5 model (if available)
model_path = "../models_train/transfer_learningv2.h5"  # Change to your model path
if not os.path.exists(model_path):
    model_path = "../models_train/transfer_learningv1.h5"

try:
    model = load_model_with_custom_objects(model_path)
    print("Using the full H5 model")
    using_tflite = False
except Exception as e:
    print(f"Could not load H5 model: {e}")
    # Fallback to TFLite model
    try:
        tflite_model_path = "../models_train/transfer_learningv1.tflite"
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        print("Using TFLite model")
        using_tflite = True
    except Exception as tflite_e:
        print(f"Could not load TFLite model either: {tflite_e}")
        raise Exception("No usable model found. Please check paths and model compatibility.")

# Define class labels 
labels = ["pet_bottle", "unacceptable"]  # NOTE: Reversed from your original code based on class_indices

# Get input dimensions
if using_tflite:
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
else:
    input_shape = model.input_shape
    input_height, input_width = input_shape[1], input_shape[2]

print(f"Input dimensions: {input_width}x{input_height}")

def preprocess_frame(frame):
    """Resize and preprocess frame for model input."""
    # Resize to expected input dimensions
    img = cv2.resize(frame, (input_width, input_height))
    
    # Convert BGR to RGB (TensorFlow models typically expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize according to the model's expected preprocessing
    # For MobileNetV2:
    # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    # img = preprocess_input(img)
    
    # Simple normalization (matching your training code)
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def classify_image(image):
    """Run inference on an image and return the prediction."""
    processed_image = preprocess_frame(image)

    if using_tflite:
        # Set input tensor for TFLite
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()  # Run inference
        
        # Get output tensor
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # For binary classification
        confidence = output_data[0][0]
        predicted_index = 1 if confidence > 0.5 else 0
    else:
        # Use the full model
        predictions = model.predict(processed_image, verbose=0)
        confidence = predictions[0][0]
        predicted_index = 1 if confidence > 0.5 else 0
    
    # Adjust confidence to show the confidence in the predicted class
    confidence_score = confidence if predicted_index == 1 else 1.0 - confidence
    
    return labels[predicted_index], confidence_score

# Open webcam
cap = cv2.VideoCapture(2)  # Try camera 0 first
if not cap.isOpened():
    print("Camera 0 not available, trying camera 3...")
    cap = cv2.VideoCapture(0)  # Then try camera 3

if not cap.isOpened():
    raise Exception("Could not open any camera. Please check connections.")

# Set camera properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

captured_image = None
predicted_label = None
confidence_score = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Check camera connection.")
        break

    # Always show the current camera feed
    display_frame = frame.copy()
    
    # Add instructions text to display frame
    cv2.putText(display_frame, "Press SPACE to capture and classify", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "Press Q to quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # If we have a classification result, show it on the display frame
    if predicted_label is not None:
        text = f"Prediction: {predicted_label} ({confidence_score:.2f})"
        color = (0, 255, 0) if predicted_label == "pet_bottle" else (0, 0, 255)
        cv2.putText(display_frame, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("PET Bottle Classifier", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Press space to capture and classify
        captured_image = frame.copy()
        predicted_label, confidence_score = classify_image(captured_image)
        print(f"Prediction: {predicted_label} with confidence {confidence_score:.4f}")

    elif key == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()