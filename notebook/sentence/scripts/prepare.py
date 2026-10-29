"""Prepare all immutable inputs locally before renting a short cloud GPU session."""

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.training import TrainingConfig, atomic_json, prepare_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["sentence", "article"], required=True)
    parser.add_argument("--mode", choices=["development", "full"], default="development")
    parser.add_argument("--data", type=Path, help="Merged sentence JSON or normalized article JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional TrainingConfig JSON overrides")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, help="Use selected development epoch count for full refit")
    parser.add_argument("--resume-preparation", action="store_true", help="Continue a interrupted initial-weight download")
    parser.add_argument("--compact", action="store_true", help="Remove unpacked inputs after creating payload.zip")
    args = parser.parse_args()
    defaults = {"task": args.task, "mode": args.mode, "seed": args.seed}
    if args.task == "article":
        defaults.update(epochs=4, batch_size=1, accumulation_steps=16, max_length=1024)
    if args.config:
        defaults.update(json.loads(args.config.read_text()))
    defaults.update(task=args.task, mode=args.mode, seed=args.seed)
    if args.epochs is not None:
        defaults["epochs"] = args.epochs
    cfg = TrainingConfig(**defaults)
    cfg.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "inputs"
    if destination.exists() and not args.resume_preparation:
        raise FileExistsError(f"Use a new output directory; {destination} already exists")
    destination.mkdir(exist_ok=True)
    if (destination / "config.json").exists():
        if json.loads((destination / "config.json").read_text()) != asdict(cfg):
            raise ValueError("Interrupted preparation belongs to a different config")
    else:
        atomic_json(destination / "config.json", asdict(cfg))
        atomic_json(destination / "data.json", prepare_records(cfg, args.data))
    scripts = Path(__file__).resolve().parent
    shutil.copytree(scripts, destination / "scripts", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    notebook = scripts.parent / f"{args.task}_training.ipynb"
    shutil.copy2(notebook, destination / "training.ipynb")
    # Materialize pretrained language weights now, not repeatedly on the GPU VM.
    from transformers import AutoModel, AutoTokenizer

    pretrained = destination / "pretrained"
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.model_revision)
    backbone = AutoModel.from_pretrained(cfg.base_model, revision=cfg.model_revision)
    backbone.save_pretrained(pretrained, safe_serialization=True)
    tokenizer.save_pretrained(pretrained)
    atomic_json(destination / "requirements.json", [
        "transformers>=4.48,<5", "datasets>=3,<5", "scikit-learn>=1.5,<2",
        "matplotlib>=3.8", "pandas>=2", "nbclient>=0.10", "ipykernel>=6",
    ])
    # Refresh support code after potentially long weight downloads.
    shutil.copytree(scripts, destination / "scripts", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    archive_path = args.output / "payload.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in destination.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(destination))
    print(f"Prepared {archive_path} ({archive_path.stat().st_size / 1e9:.2f} GB)")
    print("Keep this payload unchanged across all resume slices of this experiment.")
    if args.compact:
        shutil.rmtree(destination)


if __name__ == "__main__":
    main()
