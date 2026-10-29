"""Reload an exported sentence checkpoint offline and verify its inference contract."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.model / "training_manifest.json").read_text())
    assert metadata["config"]["mode"] == "full"
    assert metadata["progress"]["phase"] == "complete"
    assert metadata["progress"]["epoch"] == metadata["config"]["epochs"]
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True).eval()
    assert set(model.config.id2label.values()) == {"non-biased", "biased"}
    texts = [
        "The city council approved the budget on Tuesday.",
        "The corrupt administration pushed its reckless, disastrous policy.",
    ]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    start = time.monotonic()
    with torch.inference_mode():
        logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
        probabilities = logits.float().softmax(-1)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    assert torch.allclose(probabilities.sum(-1), torch.ones(2), atol=1e-6)
    report = {
        "offline_reload": "passed", "mode": metadata["config"]["mode"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "labels": model.config.id2label,
        "cpu_inference_seconds": time.monotonic() - start,
        "smoke_predictions": [
            {"text": text, "label": model.config.id2label[int(scores.argmax())],
             "probabilities": {model.config.id2label[i]: float(p) for i, p in enumerate(scores)}}
            for text, scores in zip(texts, probabilities, strict=True)
        ],
    }
    (args.model / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
