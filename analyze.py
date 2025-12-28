import pandas as pd

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

print(error_pairs.head(10))
# Total samples per actual writer
total_per_writer = cm.sum(axis=1)

# Correct predictions per writer
correct_per_writer = cm.values.diagonal()

# Accuracy per writer
accuracy_per_writer = correct_per_writer / total_per_writer

writer_accuracy = pd.DataFrame({
    "Writer": cm.index,
    "Accuracy": accuracy_per_writer
}).sort_values("Accuracy")

print(writer_accuracy.head(10))
error_pairs.to_csv("writer_confusion_pairs.csv", index=False)
writer_accuracy.to_csv("writer_accuracy.csv", index=False)

print("Saved:")
print("- writer_confusion_pairs.csv")
print("- writer_accuracy.csv")
