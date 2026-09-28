"""Task-label-agnostic classification metrics for aligned model predictions."""

from collections.abc import Sequence
from typing import TypedDict

from newsbias.models.base import Prediction


class ClassMetrics(TypedDict):
    """Precision, recall, and F1 for one label plus its gold support count."""

    precision: float
    recall: float
    f1: float
    support: int


class Metrics(TypedDict):
    """Task-agnostic aggregate/class metrics and the exact label order used.

    Failed predictions reduce accuracy and class recall; their count is exposed
    separately because the confusion matrix contains only valid labels.
    """

    count: int
    accuracy: float
    macro_f1: float
    labels: list[str]
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    failed_predictions: int


def evaluate(
    gold: Sequence[str], predictions: Sequence[Prediction], *, labels: Sequence[str],
) -> Metrics:
    """Score predictions against gold labels using the caller's explicit class order.

    Predictions outside `labels` count as failed and incorrect; their gold examples
    still reduce class recall. The confusion matrix remains square over valid labels,
    while `failed_predictions` separately records outputs such as `unknown`.
    """
    if not gold or len(gold) != len(predictions):
        raise ValueError("Gold labels and predictions must be nonempty and aligned")
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Evaluation labels must be nonempty and unique")
    label_to_index = {label: index for index, label in enumerate(labels)}
    if set(gold) - set(labels):
        raise ValueError("Unknown gold labels in evaluation")
    matrix = [[0 for _ in labels] for _ in labels]
    correct = 0
    failed = 0
    for truth, prediction in zip(gold, predictions, strict=True):
        predicted_label = prediction["label"]
        if predicted_label not in label_to_index:
            failed += 1
            continue
        matrix[label_to_index[truth]][label_to_index[predicted_label]] += 1
        correct += truth == predicted_label
    per_class: dict[str, ClassMetrics] = {}
    class_f1: list[float] = []
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(labels)) if row != index)
        support = sum(1 for truth in gold if truth == label)
        predicted_count = true_positive + false_positive
        precision_value = true_positive / predicted_count if predicted_count else 0.0
        recall_value = true_positive / support if support else 0.0
        f1_value = (
            2 * precision_value * recall_value / (precision_value + recall_value)
            if precision_value + recall_value else 0.0
        )
        class_f1.append(f1_value)
        per_class[label] = {
            "precision": precision_value,
            "recall": recall_value,
            "f1": f1_value,
            "support": support,
        }
    return {
        "count": len(gold),
        "accuracy": correct / len(gold),
        "macro_f1": sum(class_f1) / len(class_f1),
        "labels": list(labels),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "failed_predictions": failed,
    }
