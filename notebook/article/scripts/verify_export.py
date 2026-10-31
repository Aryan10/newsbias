"""Reload an exported article model checkpoint and verify its inference contract."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

from scripts.models import ArticleModel
from scripts.training import BatchCollator, TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.model / "training_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    metadata = json.loads(manifest_path.read_text())
    torch.set_num_threads(4)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = ArticleModel.from_pretrained(args.model, local_files_only=True).eval()

    labels = metadata["labels"]
    assert labels == ["left", "center", "right"]

    cfg = TrainingConfig(**metadata["config"])
    collator = BatchCollator(tokenizer, cfg)

    smoke_articles = [
        {
            "id": "smoke_1",
            "text": "The Senate today passed a bipartisan infrastructure bill allocating 50 billion dollars for bridges and transit.",
            "label_id": 1,
        },
        {
            "id": "smoke_2",
            "text": "Progressive lawmakers demanded immediate systemic climate action, calling for an end to fossil fuel subsidies and corporate tax breaks.",
            "label_id": 0,
        },
        {
            "id": "smoke_3",
            "text": "Conservative leaders praised the free market dereglation policies and tax reductions designed to stimulate economic growth and enterprise.",
            "label_id": 2,
        },
    ]

    batch = collator(smoke_articles)
    labels_tensor = batch.pop("labels")

    start = time.monotonic()
    with torch.inference_mode():
        output = model(**batch)
        logits = output.logits
        probabilities = logits.float().softmax(-1)

    elapsed = time.monotonic() - start

    assert logits.shape == (3, 3) and torch.isfinite(logits).all()
    assert torch.allclose(probabilities.sum(-1), torch.ones(3), atol=1e-5)

    smoke_preds = []
    for art, scores in zip(smoke_articles, probabilities.tolist()):
        best_idx = int(scores.index(max(scores)))
        smoke_preds.append({
            "id": art["id"],
            "text_snippet": art["text"][:80] + "...",
            "predicted": labels[best_idx],
            "probabilities": {labels[i]: round(s, 4) for i, s in enumerate(scores)},
        })

    report = {
        "offline_reload": "passed",
        "task": "article_political_leaning",
        "authoritative_epoch": metadata["progress"]["epoch"] + 1,
        "global_step": metadata["progress"]["global_step"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "labels": labels,
        "cpu_inference_seconds": round(elapsed, 4),
        "smoke_predictions": smoke_preds,
    }

    (args.model / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
