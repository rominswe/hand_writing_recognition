import os, json, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# --- CONFIG (MUST MATCH train.py) ---
TEST_DIR = 'test/'
DOC_DIR = "doc"
MODEL_NAME = 'model.keras'
RESULT_FILE = 'result.csv'
LABELS_JSON = "labels.json"
IMG_WIDTH = 256
IMG_HEIGHT = 96
SLICE_HEIGHT = 300 # Match training slice height
STRIDE = 25   # Step size for testing
VAR_THRESHOLD = 20

# --- LOAD MODEL --- 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if not os.path.exists(MODEL_NAME):
    raise FileNotFoundError("model.keras not found")
print("Loading model...")
model = tf.keras.models.load_model(MODEL_NAME)
print("Loaded model:", MODEL_NAME)

# --- LOAD LABEL MAP (Writer XX) ---
if not os.path.exists(LABELS_JSON):
    raise FileNotFoundError("labels.json not found")

with open(LABELS_JSON, "r") as f:
    label_map = json.load(f)

# --- Convert index → Writer XX ---
idx_to_writer = {int(k) - 1: v for k, v in label_map.items()}

# --- Testing ---
files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])
results = []
y_true, y_pred = [], []

for filename in files:
    img_path = os.path.join(TEST_DIR, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    h, w = img.shape
    slice_preds = []

    curr = 0
    while curr + SLICE_HEIGHT <= h:
        patch = img[curr:curr + SLICE_HEIGHT, :]
        if np.std(patch) >= VAR_THRESHOLD:
            patch = cv2.resize(patch, (IMG_WIDTH, IMG_HEIGHT))
            patch = patch.astype("float32") / 255.0
            patch = np.expand_dims(patch, axis=(0, -1))
            preds = model.predict(patch, verbose=0)[0]
            if np.max(preds) > 0.05:
                slice_preds.append(preds)
        curr += STRIDE

    # Backup whole-image prediction
    if not slice_preds:
        patch = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        patch = patch.astype("float32") / 255.0
        patch = np.expand_dims(patch, axis=(0, -1))
        slice_preds.append(model.predict(patch, verbose=0)[0])

    # Voting (mean probability)
    final_probs = np.mean(slice_preds, axis=0)
    pred_idx = int(np.argmax(final_probs))
    actual_idx = int(filename[:2]) - 1

    results.append([filename, idx_to_writer[actual_idx], idx_to_writer[pred_idx]])
    y_true.append(actual_idx)
    y_pred.append(pred_idx)

# --- Save Results ---
df = pd.DataFrame(results, columns=["filename", "actual", "predicted"])
df.to_csv(RESULT_FILE, index=False)

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred, labels=list(range(70)))
cm_df = pd.DataFrame(
    cm,
    index=[f"Writer {str(i+1).zfill(2)}" for i in range(70)],
    columns=[f"Writer {str(i+1).zfill(2)}" for i in range(70)]
)
cm_df.to_csv("confusion_matrix.csv")
os.makedirs(DOC_DIR, exist_ok=True)
plt.figure(figsize=(12, 8))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.title("Writer Confusion Matrix")
plt.xlabel("Predicted Writer")
plt.ylabel("Actual Writer")
plt.tight_layout()
plt.savefig(os.path.join(DOC_DIR, "writer_confusion_matrix.png"))

# --- Accuracy ---
accuracy = (np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)) * 100
print(f"Total test images: {len(y_true)}")
print(f"Average Accuracy   : {accuracy:.2f}%")
print(f"Results CSV saved at {RESULT_FILE}")
print(f"Confusion matrix saved at {DOC_DIR}/writer_confusion_matrix.png")