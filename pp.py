# import os
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import json

# # ===============================
# # Config
# # ===============================
# TEST_DIR = "test"
# IMG_HEIGHT = 300
# IMG_WIDTH = 1545  # resize test images to training width
# MODEL_PATH = "model.keras"
# LABEL_MAP = "labels.json"
# RESULT_CSV = "result.csv"

# # ===============================
# # Load model and labels
# # ===============================
# model = load_model(MODEL_PATH)
# with open(LABEL_MAP, "r") as f:
#     label_to_writer = json.load(f)

# # ===============================
# # Preprocess test images
# # ===============================
# def preprocess_image(img_path):
#     img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # ===============================
# # Inference
# # ===============================
# results = []

# for filename in os.listdir(TEST_DIR):
#     if filename.endswith(".png"):
#         img_path = os.path.join(TEST_DIR, filename)
#         x = preprocess_image(img_path)
#         pred_probs = model.predict(x)
#         pred_class = np.argmax(pred_probs, axis=1)[0]
#         actual_class = int(filename[:2]) - 1
#         results.append({
#             "filename": filename,
#             "actual_label": f"Writer {str(actual_class+1).zfill(2)}",
#             "predicted_label": label_to_writer[str(pred_class)]
#         })

# # ===============================
# # Save results
# # ===============================
# df = pd.DataFrame(results)
# df.to_csv(RESULT_CSV, index=False)
# print(f"Results saved to {RESULT_CSV}")

# # ===============================
# # Average accuracy
# # ===============================
# correct = sum([1 for r in results if r['actual_label'] == r['predicted_label']])
# accuracy = correct / len(results)
# print(f"Average accuracy: {accuracy*100:.2f}%")

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Force CPU usage as per project constraints 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TEST_DIR = "test/"
MODEL_PATH = "model.keras"
IMG_SIZE = 128

def run_inference():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    test_files = list(Path(TEST_DIR).glob("*.png"))
    
    results = []
    correct_count = 0

    print(f"Processing {len(test_files)} images...")
    for fp in test_files:
        filename = fp.name
        # Actual label from first two characters 
        actual_label = int(filename[:2])
        
        # Match training preprocessing: Grayscale + Resize
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # Center crop to square to avoid stretching handwriting
        h, w = img.shape
        side = min(h, w)
        img_crop = img[:side, :side]
        img_resized = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
        img_final = img_resized.astype('float32') / 255.0
        img_final = np.expand_dims(img_final, (0, -1)) # Batch and Channel dims

        # Predict
        prediction = model.predict(img_final, verbose=0)
        predicted_label = np.argmax(prediction) + 1 # Convert back to 1-based ID
        
        if predicted_label == actual_label:
            correct_count += 1
            
        results.append({
            "Filename": filename,
            "Actual Label": actual_label,
            "Predicted Label": predicted_label
        })

    # Calculate Average Accuracy 
    avg_acc = (correct_count / len(results)) * 100 if results else 0
    print(f"\nAverage Accuracy: {avg_acc:.2f}%")

    # Save to CSV 
    df = pd.DataFrame(results)
    df.to_csv("result.csv", index=False)
    print("Results saved to result.csv")

if __name__ == "__main__":
    run_inference()