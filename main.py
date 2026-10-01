from src.config import INITIAL_K, MAX_K, OUTPUTS_DIR
from src.data_loader import load_dataset, display_dataset_info
from src.preprocessing import preprocess_dataset
from src.trainer import train_knn
from src.evaluator import evaluate_model
from src.visualizer import (
    plot_accuracy_vs_k,
    plot_training_accuracy_vs_k,
)


def main():
    df = load_dataset()
    display_dataset_info(df)

    X_train, X_test, y_train, y_test, scaler = preprocess_dataset(df)

    # Initial KNN model
    model = train_knn(X_train, y_train, INITIAL_K)
    results = evaluate_model(model, X_test, y_test)

    print(f"\nKNN Results (K={INITIAL_K}):")
    print(f"Test Accuracy: {results['accuracy']:.4f}")

    # Evaluate different K values
    k_values = list(range(1, MAX_K + 1))
    test_accuracies = []
    training_accuracies = []

    for k in k_values:
        model = train_knn(X_train, y_train, k)

        test_results = evaluate_model(model, X_test, y_test)
        test_accuracies.append(test_results["accuracy"])

        training_results = evaluate_model(
            model, X_train, y_train
        )
        training_accuracies.append(training_results["accuracy"])

    best_k = k_values[test_accuracies.index(max(test_accuracies))]
    best_accuracy = max(test_accuracies)

    print(f"\nBest K: {best_k}")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")

    # Save visualizations
    plot_accuracy_vs_k(
        k_values,
        test_accuracies,
        OUTPUTS_DIR / "knn_accuracy_vs_k.png",
    )

    plot_training_accuracy_vs_k(
        k_values,
        training_accuracies,
        OUTPUTS_DIR / "knn_training_accuracy_vs_k.png",
    )


if __name__ == "__main__":
    main()