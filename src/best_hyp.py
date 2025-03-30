import os

import matplotlib.pyplot as plt
import pandas as pd

# Load CSV
df = pd.read_csv("../data/hyperparameter_result/efficientnetb0/hyperparameter_results.csv")

# Find the best hyperparameter set (highest Test Accuracy)
best_row = df.loc[df["Test Accuracy"].idxmax()]

# Extract relevant values
test_id = int(best_row["Test"])
best_hyperparams = {
    "Dense Layers": best_row["Dense Layers"],
    "Dropout": best_row["Dropout"],
    "Batch Size": best_row["Batch Size"],
    "Learning Rate": best_row["Learning Rate"],
    "Unfreeze Layers": best_row["Unfreeze Layers"],
    "Epochs": best_row["Epochs"],
}
metrics = ["Validation Accuracy", "Test Accuracy", "Validation Loss", "Test Loss"]
values = [best_row["Validation Accuracy"], best_row["Test Accuracy"], best_row["Validation Loss"], best_row["Test Loss"]]

# Define save directory
save_dir = "/home/untitled/Documents/Coding Repository/python_journey/Capstone/waste-classification/references/efficientnetb0/"
os.makedirs(save_dir, exist_ok=True)

# Create figure
fig, ax = plt.subplots(figsize=(7, 5))  
fig.suptitle(f"Best Model Performance - Test Case {test_id}", fontsize=14)

# Plot Accuracy and Loss
bars = ax.bar(metrics, values, color=["blue", "green", "red", "purple"])

# Add labels above bars
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.4f}", 
            ha='center', fontsize=10, color="black")

ax.set_ylim(0, 1.1)  # Set y-axis limit
ax.set_ylabel("Score")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save the figure
save_path = os.path.join(save_dir, f"best_hyperparameter_test_{test_id}.png")
plt.savefig(save_path, bbox_inches="tight")
plt.close()  # Close the figure to free memory

# Print best hyperparameters
print(f"\n✅ Best Hyperparameter Found in Test Case {test_id}:")
for key, value in best_hyperparams.items():
    print(f"   {key}: {value}")

print(f"\n📊 Best performance plot saved: {save_path}")
