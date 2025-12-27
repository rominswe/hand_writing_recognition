import os, json, random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from collections import Counter
# from matplotlib import mat
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Configuration ---
TRAIN_DIR = "train/"
IMG_HEIGHT = 256
IMG_WIDTH = 128
NUM_CLASSES = 70
MODEL_NAME = "model.keras"
VAL_FRACTION = 0.2
SEED = 42
EPOCHS = 80
BATCH_SIZE = 32

# --- Seeds ---
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Utilities ---
def list_images(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]

def label_from_filename(fp):
    return int(os.path.basename(fp)[:2]) - 1

def to_grayscale(img):
    if img is None:
        return None
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# --- Preprocessing ---
def preprocess_image(img, target_h=IMG_HEIGHT, target_w=IMG_WIDTH):
    h, w = img.shape[0], img.shape[1]
    # Resize maintaining aspect ratio
    scale = target_h / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_h))
    # Pad or crop width
    if new_w < target_w:
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        resized = cv2.copyMakeBorder(resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
    else:
        start = (new_w - target_w) // 2
        resized = resized[:, start:start+target_w]
    return resized.astype('float32') / 255.0

# --- Data Loading & Augmentation ---
def load_data(train_dir):
    X, y = [], []
    for fp in list_images(train_dir):
        img = cv2.imread(fp)
        gray = to_grayscale(img)
        # Split horizontally into 3 slices
        slice_h = gray.shape[0] // 3
        slices = [gray[i*slice_h:(i+1)*slice_h, :] for i in range(3)]
        label = label_from_filename(fp)
        for s in slices:
            # Original + augmented
            # X.append(preprocess_image(s))
            if np.std(s) > 10: 
                # Original slice
                X.append(preprocess_image(s))
                y.append(label)
                # Augmentation using tf.keras layers
                s_tf = tf.convert_to_tensor(s[None, :, :], dtype=tf.float32) / 255.0
                aug_layer = tf.keras.Sequential([
                    layers.RandomRotation(0.1),
                    layers.RandomTranslation(0.05,0.05),
                    layers.RandomZoom(0.05,0.05),
                    layers.RandomContrast(0.1)
                    ])
                aug_img = aug_layer(s_tf)[0].numpy()
                aug_img = preprocess_image((aug_img*255).astype(np.uint8))
                X.append(aug_img)
                y.append(label)
            else:
                print(f"Skipping blank slice in {os.path.basename(fp)}")
            # # Labels
            # label = label_from_filename(fp)
            # y.extend([label, label])
    X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    y = np.array(y)
    return X, y

# --- Load dataset ---
print("Loading and preprocessing data...")
X, y = load_data(TRAIN_DIR)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_FRACTION, random_state=SEED, stratify=y)

# --- Save labels mapping to JSON ---
labels = sorted({label_from_filename(f) for f in list_images(TRAIN_DIR)})
label_to_writer = {i: f"Writer {str(l+1).zfill(2)}" for i, l in enumerate(labels)}

with open("labels.json", "w") as f:
    json.dump(label_to_writer, f, indent=2)
print("Saved labels.json")

# --- Class weights ---
counts = Counter(y_train)
class_weight = {i: (len(y_train)/(NUM_CLASSES*counts[i])) for i in counts}

# --- Build model ---
def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH,1), num_classes=NUM_CLASSES):
    inp = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32,(3,3),activation="relu",padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64,(3,3),activation="relu",padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128,(3,3),activation="relu",padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(256,(3,3),activation="relu",padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256,activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp,out)

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
early = callbacks.EarlyStopping(monitor="val_accuracy", patience=16, restore_best_weights=True)
ckpt = callbacks.ModelCheckpoint(MODEL_NAME, monitor="val_accuracy", save_best_only=True, verbose=1)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

# --- Train ---
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    shuffle=True,
    callbacks=[early, ckpt, rlr]
)

print(f"Model saved as {MODEL_NAME}")