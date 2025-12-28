import os
import pandas as pd

ANALYZE_DIR = "analyze"
os.makedirs(ANALYZE_DIR, exist_ok=True)

# Load confusion matrix
cm = pd.read_csv("confusion_matrix.csv", index_col=0)

# Copy to avoid modifying original
errors = cm.copy()

# Set diagonal (correct predictions) to 0
for i in range(len(errors)):
    errors.iat[i, i] = 0

# Flatten and sort errors
error_pairs = (
    errors.stack()
    .reset_index()
    .rename(columns={
        "level_0": "Actual Writer",
        "level_1": "Predicted Writer",
        0: "Count"
    })
)

# Keep only real errors
error_pairs = error_pairs[error_pairs["Count"] > 0]

# Sort by most frequent mistakes
error_pairs = error_pairs.sort_values("Count", ascending=False)

print("\nTop 10 most confused writer pairs:")
print(error_pairs.head(10))

# Accuracy per writer
total_per_writer = cm.sum(axis=1)
correct_per_writer = cm.values.diagonal()
accuracy_per_writer = correct_per_writer / total_per_writer

writer_accuracy = pd.DataFrame({
    "Writer": cm.index,
    "Accuracy": accuracy_per_writer
}).sort_values("Accuracy")

print("\nBottom 10 writers by accuracy:")
print(writer_accuracy.head(10))

# SAVE RESULTS
error_pairs_path = os.path.join(ANALYZE_DIR, "writer_confusion_pairs.csv")
accuracy_path = os.path.join(ANALYZE_DIR, "writer_accuracy.csv")

error_pairs.to_csv(error_pairs_path, index=False)
writer_accuracy.to_csv(accuracy_path, index=False)

print("\nSaved analysis files:")
print(f"- {error_pairs_path}")
print(f"- {accuracy_path}")