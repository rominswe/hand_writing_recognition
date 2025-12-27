import os, json, random
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ===============================
# CONFIG (easy to adjust)
# ===============================
TRAIN_DIR = "train/"
MODEL_OUT = "model.keras"
LABELS_OUT = "labels.json"

PATCH_SIZE = 128      # square patch
PATCH_STRIDE = 64     # overlap helps style learning
PATCHES_PER_IMAGE = 40

EPOCHS = 80
BATCH_SIZE = 32
VAL_FRACTION = 0.10
SEED = 67
LEARNING_RATE = 5e-4

# ===============================
# Seed
# ===============================
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===============================
# Learning rate scheduler
# ===============================
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=LEARNING_RATE,
#     decay_steps=5000,
#     decay_rate=0.5,
#     staircase=True
# )

# ===============================
# Utilities
# ===============================
def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([
        str(p) for p in Path(folder).rglob("*")
        if p.suffix.lower() in exts
    ])

def label_from_filename(fp):
    return Path(fp).name[:2]

def extract_patches(img, size, stride, max_patches):
    h, w = img.shape[:2]
    patches = []

    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            patch = img[y:y+size, x:x+size]
            patches.append(patch)
            if len(patches) >= max_patches:
                return patches

    return patches

def preprocess_patch(patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = patch.astype(np.float32) / 255.0
    patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-7)
    return patch

# ===============================
# Load Dataset
# ===============================
files = list_images(TRAIN_DIR)
if not files:
    raise SystemExit("No training images found")

labels = sorted({label_from_filename(f) for f in files})
label_to_index = {l: i for i, l in enumerate(labels)}

X, y = [], []

for fp in tqdm(files, desc="Extracting patches"):
    img = cv2.imread(fp)
    if img is None:
        continue

    patches = extract_patches(
        img,
        PATCH_SIZE,
        PATCH_STRIDE,
        PATCHES_PER_IMAGE
    )

    for p in patches:
        X.append(preprocess_patch(p))
        y.append(label_to_index[label_from_filename(fp)])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Shuffle (CORRECT Generator usage)
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]

y_cat = to_categorical(y, num_classes=len(labels))

# ===============================
# Train / Validation Split
# ===============================
val_count = int(VAL_FRACTION * len(X))
X_val, y_val = X[:val_count], y_cat[:val_count]
X_train, y_train = X[val_count:], y_cat[val_count:]

# ===============================
# CNN Model (No Segmentation)
# ===============================
def build_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.RandomRotation(0.1)(inp)
    x = layers.RandomTranslation(0.1, 0.1)(x)
    x = layers.RandomZoom(0.1, 0.1)(x)
    x = layers.RandomFlip("horizontal")(x)

    for f in [32, 64, 128, 256]:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

model = build_model(
    (PATCH_SIZE, PATCH_SIZE, 3),
    len(labels)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

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
    patience=18,
    restore_best_weights=True,
    verbose=1
)

lr_reduce = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

# ===============================
# Train
# ===============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[ckpt, early]
)

# ===============================
# Save labels and Models
# ===============================
model.save(MODEL_OUT)
labels_int = sorted({int(f[:2]) for f in files if f[:2].isdigit() and f.endswith(".png")})
label_to_writer = {i: f"Writer {str(l).zfill(2)}" for i, l in enumerate(labels_int)}
with open(LABELS_OUT, "w") as f:
    json.dump(label_to_writer, f, indent=2)

# ===============================
# Visualization
# ===============================
DOC_DIR = "doc"
os.makedirs(DOC_DIR, exist_ok=True)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plot_path = os.path.join(DOC_DIR, "training_plot.png")
plt.savefig(plot_path)
print(f"Model saved as {MODEL_OUT}")
print(f"Training plot figure has been saved at: {plot_path}")