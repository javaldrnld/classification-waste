import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load CSV
df = pd.read_csv("../data/hyperparameter_result/efficientnetb0/hyperparameter_results.csv")

# Define save directory
save_dir = "../references/efficientnetb0"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Loop through each test case and create individual plots
for i, row in df.iterrows():
    test_id = int(row["Test"])  # Get test case number

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))  
    fig.suptitle(f"Model Performance - Test Case {test_id}", fontsize=14)

    # Plot Accuracy and Loss
    bars = ax.bar(["Validation Acc", "Test Acc", "Validation Loss", "Test Loss"], 
                   [row["Validation Accuracy"], row["Test Accuracy"], row["Validation Loss"], row["Test Loss"]], 
                   color=["blue", "green", "red", "purple"])

    # Add labels above bars
    for bar, value in zip(bars, [row["Validation Accuracy"], row["Test Accuracy"], row["Validation Loss"], row["Test Loss"]]):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.4f}", 
                ha='center', fontsize=10, color="black")

    ax.set_ylim(0, 1.1)  # Set y-axis limit
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the figure
    save_path = os.path.join(save_dir, f"test_case_{test_id}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print(f"✅ Saved: {save_path}")
