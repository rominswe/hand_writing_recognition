import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")  # Suppress Python warnings

# --- Configuration ---
TRAIN_DIR = 'train/'  # Folder containing training images [cite: 10]
IMG_WIDTH = 256       # Downsampled width for CPU efficiency 
IMG_HEIGHT = 64       # Downsampled height for CPU efficiency 
NUM_CLASSES = 70      # Total number of writers 
MODEL_NAME = 'model.keras' # Required filename [cite: 23, 37]

def load_data():
    images = []
    labels = []
    
    # Iterate through the training directory [cite: 10]
    for filename in os.listdir(TRAIN_DIR):
        if filename.endswith(".png"):
            # 1. Extract class label (first two characters) [cite: 11, 12]
            # Convert "01" to index 0, "70" to index 69
            label = int(filename[:2]) - 1 
            
            img_path = os.path.join(TRAIN_DIR, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # 2. Horizontal Slicing Strategy
                # Training image is 1070px high; Test image is 300px high.
                # We take 3 slices of 300px to match test feature scale.
                h, w = img.shape
                slice_height = 300
                
                for i in range(3):
                    start_row = i * slice_height
                    end_row = start_row + slice_height
                    
                    if end_row <= h:
                        slice_img = img[start_row:end_row, :]
                        
                        # Resize for CPU efficiency 
                        resized = cv2.resize(slice_img, (IMG_WIDTH, IMG_HEIGHT))
                        
                        # Normalize pixel values [0, 1] [cite: 63]
                        images.append(resized.astype('float32') / 255.0)
                        labels.append(label)

    return np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), np.array(labels)

# --- 1. Preprocessing & Data Handling [cite: 63] ---
print("Loading and preprocessing training data...")
X, y = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=67)

# --- 2. Model Development (Neural Network Only) [cite: 48, 49] ---
# A lightweight CNN suitable for a Core i5 CPU [cite: 53, 63]

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3), # To prevent overfitting
    layers.Dense(NUM_CLASSES, activation='softmax') # 70 Writer IDs
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. Training Quality [cite: 63] ---
print("Starting training...")
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val))

# --- 4. Save Model [cite: 23, 37] ---
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")