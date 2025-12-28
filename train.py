import os, json, cv2, random, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  

# --- Configuration ---
TRAIN_DIR = 'train/'  # Folder containing training images [cite: 10]
DOC_DIR = "doc"
IMG_WIDTH = 256       # Downsampled width for CPU efficiency 
IMG_HEIGHT = 96      # Downsampled height for CPU efficiency 
NUM_CLASSES = 70      # Total number of writers 
MODEL_NAME = 'model.keras' 
LABELS_JSON = "labels.json"

SEED = 67
EPOCHS = 80
BATCH_SIZE = 64
SLICE_HEIGHT = 300
STRIDE = 20
VAR_THRESHOLD = 20

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- DATA LOADING ---

datagen = ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0.02,
    zoom_range=0.02,
    fill_mode='constant',
    cval=255
)

def load_data():
    images, labels= [], []
    # Iterate through the training directory
    for filename in os.listdir(TRAIN_DIR):
        if not filename.endswith(".png"): continue
        label = int(filename[:2]) - 1 
        img = cv2.imread(os.path.join(TRAIN_DIR, filename), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        h, w = img.shape
        if h < SLICE_HEIGHT: continue
                
        current_row = 0
        while current_row + SLICE_HEIGHT <= h:
            patch = img[current_row:current_row + SLICE_HEIGHT, :]
            if np.std(patch) >= VAR_THRESHOLD: 
                patch_res = cv2.resize(patch, (IMG_WIDTH, IMG_HEIGHT))
                patch_res = patch_res.astype("float32") / 255.0
                patch_final = np.expand_dims(patch_res, axis=-1)
                # Add Original
                images.append(patch_final)
                labels.append(label)

                # Augmented patches:
                augmented_patch = datagen.random_transform(patch_final)
                images.append(augmented_patch)
                labels.append(label)
            
            current_row += STRIDE

    X = np.array(images).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    y = np.array(labels)
    
    return X, y 

# --- Preprocessing & Data Handling ---
print("Loading and preprocessing training data...")
X, y_integers = load_data()
# --- Convert to One-Hot for Label Smoothing compatibility ---
y_onehot = tf.keras.utils.to_categorical(y_integers, num_classes=NUM_CLASSES)

X_train, X_val, y_train, y_val, y_int_train, y_int_val = train_test_split(
     X, y_onehot, y_integers,
     test_size=0.2, 
     random_state=67,
     stratify=y_integers
     ) # Reduced val size slightly

# --- CLASS WEIGHTS ---
counts = Counter(y_int_train)
class_weight = {
    cls: len(y_int_train) / (NUM_CLASSES * count)
    for cls, count in counts.items()
}

# --- SAVE LABELS.JSON (Writer XX) ---
label_map = {
    str(i + 1).zfill(2): f"Writer {str(i + 1).zfill(2)}"
    for i in range(NUM_CLASSES)
}
with open(LABELS_JSON, "w") as f:
    json.dump(label_map, f, indent=2)

# --- Model Development ---
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),

    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.BatchNormalization(),
    layers.Dropout(0.5), 
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# --- Callbacks ---
early = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=12,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    MODEL_NAME,
    monitor="val_accuracy",
    save_best_only=True,
    mode='max',
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

# --- TRAIN ---
print(f"Dataset ready. Samples: {len(X_train)}. Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    shuffle=True,
    callbacks=[checkpoint, reduce_lr, early]
)

# --- VISUALIZATION ---
os.makedirs(DOC_DIR, exist_ok=True)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plot_path = os.path.join(DOC_DIR, "training_plot.png")
plt.savefig(plot_path)

print(f"Model saved as {MODEL_NAME}")
print(f"Training plot figure has been saved at: {plot_path}")