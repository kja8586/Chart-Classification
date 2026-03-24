"""
Evaluation and visualization utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion_matrix(y_true, y_pred, class_labels, save_path=None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels (array-like).
        y_pred: Predicted labels (array-like).
        class_labels: List of class name strings.
        save_path: If provided, save the figure to this path.
    """
    conf_mat = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_mat, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred, class_labels):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=class_labels))


def evaluate_model(model, test_generator, class_labels, save_dir=None):
    """
    Full model evaluation: metrics + confusion matrix + classification report.

    Args:
        model: Compiled Keras model.
        test_generator: Keras test data generator.
        class_labels: List of class name strings.
        save_dir: Directory to save confusion matrix plot.
    """
    # Evaluate metrics
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
    print(f"\nTest Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")

    # Predictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Classification report
    print_classification_report(y_true, y_pred_classes, class_labels)

    # Confusion matrix
    if save_dir:
        cm_path = os.path.join(save_dir, f"{model.name}_confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred_classes, class_labels, save_path=cm_path)
