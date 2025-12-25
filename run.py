import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Configuration ---
TEST_DIR = 'test/'             # Folder containing test images [cite: 26]
MODEL_NAME = 'model.keras'      # Saved model filename [cite: 23, 37]
RESULT_FILE = 'result.csv'      # Required output filename [cite: 30]
IMG_WIDTH = 256                # Must match training width
IMG_HEIGHT = 64                # Must match training height

def main():
    # Force TensorFlow to use CPU only (ensures compliance with project rules) [cite: 24, 53]
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 1. Load the saved model [cite: 27]
    if not os.path.exists(MODEL_NAME):
        print(f"Error: {MODEL_NAME} not found. Ensure the model is in the same directory.")
        return
    
    model = tf.keras.models.load_model(MODEL_NAME)
    print(f"Model {MODEL_NAME} loaded successfully.")

    # 2. Prepare to process test images [cite: 26]
    test_results = []
    correct_predictions = 0
    total_images = 0

    if not os.path.exists(TEST_DIR):
        print(f"Error: Folder '{TEST_DIR}' not found.")
        return

    # 3. Read the test images from the test folder [cite: 26]
    files = [f for f in os.listdir(TEST_DIR) if f.endswith('.png')]
    
    for filename in files:
        img_path = os.path.join(TEST_DIR, filename)
        
        # Preprocessing: Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Consistent Preprocessing with training (Resize and Normalize)
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            normalized = resized.astype('float32') / 255.0
            input_data = np.expand_dims(normalized, axis=(0, -1)) # Shape: (1, 64, 256, 1)

            # Predict the class
            predictions = model.predict(input_data, verbose=0)
            predicted_class_idx = np.argmax(predictions)
            predicted_label = predicted_class_idx + 1 # Convert index back to Writer ID (1-70)

            # Extract actual label from filename (first two characters) 
            actual_label = int(filename[:2])

            # Track accuracy 
            total_images += 1
            if predicted_label == actual_label:
                correct_predictions += 1

            # Store result data [cite: 29]
            test_results.append({
                'Filename': filename,
                'Actual Label': actual_label,
                'Predicted Label': predicted_label
            })

    # 4. Show the average accuracy for all test images 
    if total_images > 0:
        avg_accuracy = (correct_predictions / total_images) * 100
        print("-" * 30)
        print(f"Total Test Images: {total_images}")
        print(f"Average Accuracy: {avg_accuracy:.2f}%")
        print("-" * 30)

        # 5. Save results to CSV 
        df = pd.DataFrame(test_results)
        df.to_csv(RESULT_FILE, index=False)
        print(f"Results successfully saved to {RESULT_FILE}")
    else:
        print("No valid test images were processed.")

if __name__ == "__main__":
    main()