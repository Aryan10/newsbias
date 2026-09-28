"""Checks for generic class metrics and failed-prediction accounting."""

import pytest

from newsbias.evaluation import evaluate


def test_metrics_include_absent_classes() -> None:
    """Keep zero-support classes in macro-F1 and preserve the declared matrix order."""
    result = evaluate(["left", "left", "right"], [
        {"label": "left", "probabilities": None},
        {"label": "right", "probabilities": None},
        {"label": "right", "probabilities": None},
    ], labels=["left", "center", "right"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert result["confusion_matrix"] == [[1, 0, 1], [0, 0, 0], [0, 0, 1]]
    assert result["per_class"]["center"]["support"] == 0


def test_alignment_and_unknown_predictions() -> None:
    """Reject missing predictions and count explicit unknown outputs as failures."""
    with pytest.raises(ValueError, match="aligned"):
        evaluate(["left"], [], labels=["left"])
    failed = evaluate(
        ["left"], [{"label": "unknown", "probabilities": None, "error": "request failed"}],
        labels=["left"],
    )
    assert failed["accuracy"] == 0
    assert failed["failed_predictions"] == 1
