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
                # 2. Horizontal Slicing Strategy (Augmentation)
                # Training image is 1070px high; Test image is 300px high.
                # STRATEGY: Use a "Sliding Window" to generate many samples from one images.
                # Instead of 3 chunks, we slide 50px at a time.
                h, w = img.shape
                slice_height = 300
                stride = 20 # Overlap of 280px (MAXIMUM AUGMENTATION)
                
                # Check how many slices we can fit
                # If image is smaller than slice_height, resize it directly check
                if h < slice_height:
                     continue
                
                current_row = 0
                while current_row + slice_height <= h:
                    start_row = current_row
                    end_row = start_row + slice_height
                    
                    slice_img = img[start_row:end_row, :]
                    
                    # 3. Skip if slice is mostly empty (white space)
                    # Text has high contrast/variance. Blank paper has low variance.
                    if np.std(slice_img) < 20: 
                         current_row += stride
                         continue

                    # Resize for CPU efficiency 
                    resized = cv2.resize(slice_img, (IMG_WIDTH, IMG_HEIGHT))
                    
                    # Normalize pixel values [0, 1] [cite: 63]
                    images.append(resized.astype('float32') / 255.0)
                    labels.append(label)
                    
                    current_row += stride

    return np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1), np.array(labels)

# --- 1. Preprocessing & Data Handling [cite: 63] ---
print("Loading and preprocessing training data...")
X, y = load_data()
print(f"Dataset Augmented! Total samples generated: {X.shape[0]}")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=67) # Reduced val size slightly

# --- 2. Model Development (Neural Network Only) [cite: 48, 49] ---
# Improved Architecture: Deeper but lighter using GAP
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    # Block 3
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Block 4
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling2D(),
    
    layers.Dropout(0.5), # Back to 0.5
    layers.Dense(NUM_CLASSES, activation='softmax') # 70 Writer IDs
])

# Use a lower learning rate for stability
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. Training Quality [cite: 63] ---
print("Starting training...")

# CALLBACK: Save the BEST model, not the LAST model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_NAME, 
    monitor='val_loss', 
    save_best_only=True,
    mode='min',
    verbose=1
)

model.fit(X_train, y_train, 
          epochs=50, 
          batch_size=32, 
          validation_data=(X_val, y_val),
          callbacks=[checkpoint])

print("Training Complete. The best model was saved automatically during training.")