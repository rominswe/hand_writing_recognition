import os, json, random, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# ===============================
# CONFIG 
# ===============================
TRAIN_DIR = "train/"
MODEL_OUT = "model.keras"
LABELS_OUT = "labels.json"
DOC_DIR = "doc"

IMG_SIZE = 128     
BATCH_SIZE = 32    
EPOCHS = 80        
SEED = 67
LEARNING_RATE = 5e-4 

# # ===============================
# # Seed
# # ===============================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# # # ===============================
# # # Learning rate scheduler
# # # ===============================
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=LEARNING_RATE,
#     decay_steps=5000,
#     decay_rate=0.5,
#     staircase=True
# )

# ===============================
# Data Extraction & Preprocessing
# ===============================
def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([str(p) for p in Path(folder).glob("*.png") if p.suffix.lower() in exts])

def get_label_str(filepath):
    return Path(filepath).name[:2]

def load_optimized_data():
    files = list_images(TRAIN_DIR)
    if not files:
        raise SystemExit("No training images found in " + TRAIN_DIR)

    # Prepare labels
    unique_labels = sorted({get_label_str(f) for f in files})
    label_to_index = {l: i for i, l in enumerate(unique_labels)}
    
    X, y = [], []
    
    # Progress bar for extraction
    print("Processing train images: ")
    for fp in tqdm(files, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        label_idx = label_to_index[get_label_str(fp)]
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        h, w = img.shape
        stride = 64 
        for top in range(0, h - IMG_SIZE, stride):
            for left in range(0, w - IMG_SIZE, stride):
                patch = img[top:top+IMG_SIZE, left:left+IMG_SIZE]
                
                # Ink Filter (StdDev > 15) to ignore blank paper
                if np.std(patch) > 15: 
                    X.append(patch.astype('float32') / 255.0)
                    y.append(label_idx)

    X = np.expand_dims(np.array(X), -1) 
    y = np.array(y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    
    return X_train, X_val, y_train, y_val, unique_labels

X_train, X_val, y_train, y_val, label_list = load_optimized_data()

# ===============================
# CNN Architecture
# ===============================
def build_optimized_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(), 
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5), 
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

model = build_optimized_model(len(label_list))

# PRINT MODEL SUMMARY
model.summary()

# ===============================
# Callbacks
# ===============================
ckpt = callbacks.ModelCheckpoint(
    MODEL_OUT, 
    monitor="val_accuracy", 
    save_best_only=True, 
    verbose=1
)

early = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=16,
    restore_best_weights=True,
    verbose=1
)

lr_reduce = callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5, 
    patience=5, 
    verbose=1
)

# ===============================
# Train (This creates the progress bars)
# ===============================
print("\nStarting training on CPU...")
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    verbose=1,
    callbacks=[ckpt, early, lr_reduce]
)

# ===============================
# Save JSON & Plots
# ===============================
model.save(MODEL_OUT)
label_to_writer = {
    i: f"Writer {str(int(lbl)).zfill(2)}"
    for i, lbl in enumerate(label_list)
}
with open(LABELS_OUT, "w") as f:
    json.dump(label_to_writer, f, indent=2)

# # ===============================
# # Visualization
# # ===============================
os.makedirs(DOC_DIR, exist_ok=True)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.savefig(os.path.join(DOC_DIR, "training_plot.png"))

print(f"\nModel saved as {MODEL_OUT}")
print(f"Labels mapping saved as {LABELS_OUT}")