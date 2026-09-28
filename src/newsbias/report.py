"""Human-readable terminal summaries for recorded benchmark results."""

from typing import Any


def format_benchmark_report(
    summary: dict[str, Any],
    *,
    cached: bool = False,
    runner: str = "local",
    artifacts_path: str | None = None,
) -> str:
    """Render metrics, class breakdown, usage, provenance, and cache state as text.

    Accepts both the current nested `metrics` structure and the flatter legacy
    summary shape. Optional runner and artifact fields make the same formatter
    useful for local results, remote runs, and cache hits.
    """
    metrics = summary.get("metrics") or {
        "accuracy": summary.get("accuracy", 0.0),
        "macro_f1": summary.get("macro_f1", 0.0),
        "count": summary.get("total_evaluated", 0),
        "per_class": summary.get("per_class", {}),
        "failed_predictions": 0,
    }
    model_name = (
        summary.get("model_name")
        or summary.get("model_metadata", {}).get("parameters", {}).get("model_name")
        or summary.get("model", {}).get("parameters", {}).get("model_name")
        or summary.get("model_identifier")
        or "Unknown Model"
    )
    dataset_name = (
        summary.get("dataset_name")
        or summary.get("dataset_metadata", {}).get("parameters", {}).get("name")
        or summary.get("dataset", {}).get("parameters", {}).get("name")
        or (
            summary.get("dataset")
            if isinstance(summary.get("dataset"), str)
            else summary.get("dataset", {}).get("adapter")
        )
        or "Dataset"
    )
    split = summary.get("split", "unknown")
    sample_count = summary.get("sample_count", metrics.get("count", 0))
    inference_seconds = (
        summary.get("inference_seconds")
        or summary.get("duration_seconds")
        or 0.0
    )
    usage = summary.get("usage", {})
    failed = metrics.get("failed_predictions", 0)

    lines: list[str] = [
        "=" * 80,
        "NEWSBIAS BENCHMARK REPORT",
        "=" * 80,
        f"Model:       {model_name}",
        f"Dataset:     {dataset_name} (split: {split} | samples: {sample_count:,})",
        f"Runner:      {runner}" + (" [CACHED RESULT]" if cached else ""),
    ]
    if artifacts_path:
        lines.append(f"Artifacts:   {artifacts_path}")

    time_str = f"{inference_seconds:.2f}s"
    if inference_seconds > 60:
        mins = int(inference_seconds // 60)
        secs = inference_seconds % 60
        time_str = f"{mins}m {secs:.1f}s"
    lines.append(f"Latency:     {time_str}")

    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        comp_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + comp_tokens)
        lines.append(
            f"Tokens:      {total_tokens:,} total "
            f"({prompt_tokens:,} prompt, {comp_tokens:,} completion)"
        )

    lines.extend([
        "-" * 80,
        "Overall Metrics:",
        f"  Accuracy:      {metrics.get('accuracy', 0.0):.4f}",
        f"  Macro-F1:      {metrics.get('macro_f1', 0.0):.4f}",
        f"  Failures:      {failed}",
    ])

    per_class = metrics.get("per_class", {})
    if per_class:
        lines.extend([
            "-" * 80,
            "Class Breakdown:",
            f"  {'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}",
            "  " + "-" * 56,
        ])
        for label, stats in per_class.items():
            prec = stats.get("precision", 0.0)
            rec = stats.get("recall", 0.0)
            f1 = stats.get("f1", 0.0)
            supp = stats.get("support", 0)
            lines.append(
                f"  {str(label):<12} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {supp:<8}"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


def print_benchmark_report(
    summary: dict[str, Any],
    *,
    cached: bool = False,
    runner: str = "local",
    artifacts_path: str | None = None,
) -> None:
    """Write :func:`format_benchmark_report` output to stdout for interactive use."""
    print(
        format_benchmark_report(
            summary, cached=cached, runner=runner, artifacts_path=artifacts_path
        )
    )
