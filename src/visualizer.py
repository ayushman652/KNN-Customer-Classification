from pathlib import Path

import matplotlib.pyplot as plt


def plot_accuracy_vs_k(k_values, accuracies, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(k_values, accuracies, marker="o")

    plt.xlabel("K")
    plt.ylabel("Test Accuracy")
    plt.title("KNN Accuracy vs K")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_training_accuracy_vs_k(k_values, accuracies, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(k_values, accuracies, marker="o")

    plt.xlabel("K")
    plt.ylabel("Training Accuracy")
    plt.title("KNN Training Accuracy vs K")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()