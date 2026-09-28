"""Tests for benchmark terminal report formatting."""

from newsbias.report import format_benchmark_report


def test_format_benchmark_report_standard() -> None:
    """Render the nested metrics schema with model, split, class table, and artifacts."""
    summary = {
        "model_identifier": "test-model",
        "split": "test",
        "sample_count": 100,
        "inference_seconds": 12.5,
        "metrics": {
            "accuracy": 0.85,
            "macro_f1": 0.83,
            "count": 100,
            "failed_predictions": 0,
            "per_class": {
                "left": {"precision": 0.88, "recall": 0.82, "f1": 0.85, "support": 35},
                "center": {"precision": 0.80, "recall": 0.83, "f1": 0.81, "support": 30},
                "right": {"precision": 0.87, "recall": 0.86, "f1": 0.86, "support": 35},
            },
        },
    }
    report = format_benchmark_report(summary, runner="local", artifacts_path="/tmp/run-1")
    assert "NEWSBIAS BENCHMARK REPORT" in report
    assert "Model:       test-model" in report
    assert "Accuracy:      0.8500" in report
    assert "Macro-F1:      0.8300" in report
    assert "left" in report
    assert "center" in report
    assert "right" in report
    assert "/tmp/run-1" in report


def test_format_benchmark_report_with_tokens_and_minutes() -> None:
    """Support legacy flat metrics while formatting long latency and token accounting."""
    summary = {
        "model_name": "gpt-5-mini",
        "dataset_name": "AllSides",
        "split": "validation",
        "total_evaluated": 50,
        "duration_seconds": 125.4,
        "accuracy": 0.76,
        "macro_f1": 0.74,
        "usage": {"prompt_tokens": 5000, "completion_tokens": 1200},
    }
    report = format_benchmark_report(summary, cached=True, runner="kaggle")
    assert "[CACHED RESULT]" in report
    assert "Latency:     2m 5.4s" in report
    assert "Tokens:      6,200 total (5,000 prompt, 1,200 completion)" in report
    assert "Accuracy:      0.7600" in report
