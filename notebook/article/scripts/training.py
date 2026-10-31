"""Complete portable training loop with bounded, exactly positioned resume slices.

Checkpoints are made at optimizer boundaries and include partially completed
validation. Full-data fitting and development runs use distinct configurations.
"""

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from scripts.models import ArticleModel


def native_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()


@dataclass
class TrainingConfig:
    task: str = "sentence"
    mode: str = "development"
    base_model: str = "jhu-clsp/ettin-encoder-400m"
    model_revision: str | None = None
    seed: int = 42
    split_seed: int = 42
    epochs: int = 5
    batch_size: int = 8
    accumulation_steps: int = 4
    encoder_lr: float = 1e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06
    max_length: int = 512
    overlap: int = 128
    max_passages: int = 16
    passage_microbatch: int = 2
    classifier_dropout: float = 0.1
    source_weighted_loss: bool = False
    gradient_checkpointing: bool = True
    precision: str = "auto"
    session_seconds: int = 900
    save_reserve_seconds: int = 90
    checkpoint_every_seconds: int = 180
    early_stopping_patience: int | None = 2

    @property
    def labels(self) -> list[str]:
        return ["non-biased", "biased"] if self.task == "sentence" else ["left", "center", "right"]

    def validate(self) -> None:
        if self.task not in {"sentence", "article"} or self.mode not in {"development", "full"}:
            raise ValueError("Choose task sentence/article and mode development/full")
        for name in ("epochs", "batch_size", "accumulation_steps", "max_passages", "passage_microbatch"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_length < 4 or not 0 <= self.overlap < self.max_length - 2:
            raise ValueError("Invalid max_length/overlap")
        if self.session_seconds <= self.save_reserve_seconds or self.save_reserve_seconds < 1:
            raise ValueError("Session budget must exceed the checkpoint reserve")
        if self.checkpoint_every_seconds < 1:
            raise ValueError("checkpoint_every_seconds must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("Choose auto, fp32, fp16, or bf16 precision")
        if self.encoder_lr <= 0 or self.head_lr <= 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("Invalid learning rates or warmup fraction")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive.is_dir():
        for item in archive.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(root)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_records(config: TrainingConfig, source: str | Path | None = None) -> dict[str, Any]:
    """Normalize and deduplicate before partitioning; no outlet is selected for exclusion."""
    if source is not None:
        rows = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Local input must be a JSON list of annotated records")
        identity = {"path": str(source), "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest()}
    else:
        from datasets import load_dataset

        if config.task != "article":
            raise ValueError("Sentence training requires the project's merged_bias.json")
        revision = "4d33d17c5591ad5d4e37ffd6238d7ead8590a70e"
        dataset = "upasanachatterjee/AllSides-random-split"
        raw = load_dataset(dataset, revision=revision)
        rows = [row for partition in raw.values() for row in partition]
        identity = {"dataset": dataset, "revision": revision}

    unique: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("role") == "weak_pretrain":
            continue
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            continue
        raw_label = row.get("label")
        if raw_label is None:
            raw_label = row.get("bias_text")
        if raw_label is None:
            raw_label = row.get("label_binary")
        label = str(raw_label).strip().lower()
        if config.task == "sentence":
            label = {"0": "non-biased", "1": "biased", "non_biased": "non-biased"}.get(label, label)
        if label not in config.labels:
            raise ValueError(f"Record {index}: unsupported label {raw_label!r}")
        digest = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
        record = {
            "id": str(row.get("id") or row.get("ID") or digest), "text": text,
            "label": label, "label_id": config.labels.index(label), "text_hash": digest,
            "dataset": row.get("dataset", "allsides"),
            "sources": row.get("sources", [row.get("dataset", "allsides")]),
        }
        if digest in unique and unique[digest]["label"] != label:
            raise ValueError(f"Conflicting labels for identical normalized text: {record['id']}")
        unique.setdefault(digest, record)
    records = list(unique.values())
    if not records or set(r["label"] for r in records) != set(config.labels):
        raise ValueError("Every target class must have examples")
    # Reused native IDs with differing bodies must not straddle partitions.
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated native IDs have different texts; resolve those groups before splitting")
    if config.mode == "full":
        splits = {"train": records, "validation": [], "test": []}
    else:
        indices = list(range(len(records)))
        remaining_count = round(len(records) * 0.1) + round(len(records) * 0.2)
        train, remaining = train_test_split(
            indices, test_size=remaining_count, random_state=config.split_seed,
            stratify=[r["label"] for r in records],
        )
        validation, test = train_test_split(
            remaining, train_size=round(len(records) * 0.1), random_state=config.split_seed,
            stratify=[records[i]["label"] for i in remaining],
        )
        splits = {name: [records[i] for i in part] for name, part in (
            ("train", train), ("validation", validation), ("test", test),
        )}
    return {"source": identity, "split_seed": config.split_seed, "mode": config.mode, "splits": splits}


class BatchCollator:
    def __init__(self, tokenizer: Any, config: TrainingConfig) -> None:
        self.tokenizer, self.config = tokenizer, config

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        cfg = self.config
        labels = torch.tensor([r["label_id"] for r in rows])
        if cfg.task == "sentence":
            encoded = self.tokenizer(
                [r["text"] for r in rows], truncation=True, max_length=cfg.max_length,
                padding=True, return_tensors="pt", return_token_type_ids=False,
            )
            return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "labels": labels}
        articles = []
        for row in rows:
            windows = self.tokenizer(
                row["text"], truncation=True, max_length=cfg.max_length, stride=cfg.overlap,
                return_overflowing_tokens=True, return_token_type_ids=False,
            )["input_ids"]
            positions = list(range(len(windows)))
            if len(windows) > cfg.max_passages:
                positions = np.linspace(0, len(windows) - 1, cfg.max_passages, dtype=int).tolist()
            articles.append((positions, [windows[i] for i in positions]))
        passage_count = max(len(windows) for _, windows in articles)
        length = max(len(tokens) for _, windows in articles for tokens in windows)
        ids = torch.full((len(rows), passage_count, length), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        positions = torch.zeros((len(rows), passage_count), dtype=torch.long)
        for i, (selected, windows) in enumerate(articles):
            for j, tokens in enumerate(windows):
                ids[i, j, :len(tokens)] = torch.tensor(tokens)
                mask[i, j, :len(tokens)] = 1
                positions[i, j] = selected[j]
        return {"input_ids": ids, "attention_mask": mask, "passage_positions": positions, "labels": labels}


class SessionTrainer:
    """One ongoing experiment, potentially spread over many short GPU runtimes."""

    def __init__(
        self, config: TrainingConfig, data_path: Path, output_dir: Path,
        *, resume_archive: Path | None = None, initial_model: str | Path | None = None,
    ) -> None:
        config.validate()
        self.cfg, self.output_dir = config, output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed(config.seed)
        data_bytes = data_path.read_bytes()
        self.data = json.loads(data_bytes)
        if self.data["mode"] != config.mode or self.data["split_seed"] != config.split_seed:
            raise ValueError("Prepared data does not match mode/split_seed")
        self.train_rows = self.data["splits"]["train"]
        self.validation_rows = self.data["splits"]["validation"]
        self.data_hash = hashlib.sha256(data_bytes).hexdigest()
        recipe = asdict(config)
        for key in ("session_seconds", "save_reserve_seconds", "checkpoint_every_seconds", "early_stopping_patience"):
            recipe.pop(key)
        self.fingerprint = hashlib.sha256(json.dumps({"recipe": recipe, "data": self.data_hash}, sort_keys=True).encode()).hexdigest()
        self.batches_per_epoch = math.ceil(len(self.train_rows) / config.batch_size)
        self.total_steps = math.ceil(self.batches_per_epoch / config.accumulation_steps) * config.epochs
        resume_root = None
        if resume_archive is not None:
            resume_root = Path(tempfile.mkdtemp(prefix="resume-", dir=output_dir))
            safe_extract(resume_archive, resume_root)
            saved_manifest = json.loads((resume_root / "manifest.json").read_text())
            if saved_manifest["fingerprint"] != self.fingerprint:
                raise ValueError("Resume recipe/data mismatch; use the original bundle or start a new experiment")
            model_path = resume_root / "model"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = (
                ArticleModel.from_pretrained(model_path) if config.task == "article"
                else AutoModelForSequenceClassification.from_pretrained(model_path)
            )
        else:
            source = str(initial_model or config.base_model)
            revision = config.model_revision if initial_model is None else None
            self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
            maps = {"id2label": dict(enumerate(config.labels)), "label2id": {v: i for i, v in enumerate(config.labels)}}
            if config.task == "article":
                self.model = ArticleModel.from_backbone(
                    source, revision=revision, dropout=config.classifier_dropout,
                    passage_microbatch=config.passage_microbatch, **maps,
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    source, revision=revision, num_labels=2, classifier_pooling="mean",
                    classifier_dropout=config.classifier_dropout, **maps,
                )
        backbone_config = self.model.config if config.task == "sentence" else self.model.encoder.config
        backbone_config.reference_compile = False
        if config.max_length > backbone_config.max_position_embeddings:
            raise ValueError("Token window exceeds the backbone's configured maximum")
        if config.gradient_checkpointing:
            backbone = self.model.encoder if config.task == "article" else self.model.base_model
            backbone.gradient_checkpointing_enable()
        self.model.to(self.device)
        self.collate = BatchCollator(self.tokenizer, config)
        groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
        for name, parameter in self.model.named_parameters():
            encoder = name.startswith("encoder.") if config.task == "article" else name.startswith(self.model.base_model_prefix + ".")
            decay = parameter.ndim > 1 and not name.endswith("bias")
            groups.setdefault((encoder, decay), []).append(parameter)
        self.optimizer = torch.optim.AdamW([
            {"params": params, "lr": config.encoder_lr if encoder else config.head_lr,
             "weight_decay": config.weight_decay if decay else 0.0}
            for (encoder, decay), params in groups.items()
        ])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(self.total_steps * config.warmup_fraction),
            num_training_steps=self.total_steps,
        )
        precision = config.precision
        if precision == "auto":
            precision = "bf16" if self.device.type == "cuda" and native_bf16_supported() else "fp16" if self.device.type == "cuda" else "fp32"
        if precision == "bf16" and self.device.type == "cuda" and not native_bf16_supported():
            raise ValueError("This GPU does not support BF16; choose auto or fp16")
        if self.device.type == "cpu" and precision == "fp16":
            raise ValueError("FP16 training requires CUDA")
        self.precision = precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        self.class_weights = torch.ones(len(config.labels), device=self.device)
        if config.task == "sentence" and config.source_weighted_loss:
            counts = np.bincount([r["label_id"] for r in self.train_rows], minlength=2)
            self.class_weights = torch.tensor(np.minimum(3, np.sqrt(len(self.train_rows) / (2 * counts))), dtype=torch.float32, device=self.device)
        self.state: dict[str, Any] = {
            "epoch": 0, "batch_cursor": 0, "global_step": 0, "phase": "train",
            "epoch_loss_sum": 0.0, "epoch_loss_weight": 0.0,
            "validation_cursor": 0, "validation_targets": [], "validation_probabilities": [],
            "history": [], "best_macro_f1": None, "training_seconds": 0.0,
        }
        if resume_root is not None:
            checkpoint = torch.load(resume_root / "trainer_state.pt", map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.state = checkpoint["state"]
            if (resume_root / "best_model").exists():
                shutil.copytree(resume_root / "best_model", output_dir / "best_model", dirs_exist_ok=True)
            if (resume_root / "evaluation_state.json").exists():
                shutil.copy2(resume_root / "evaluation_state.json", output_dir / "evaluation_state.json")
            self._restore_rng(checkpoint["rng"])
            shutil.rmtree(resume_root)
        self.last_checkpoint_at = time.monotonic()

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _rng() -> dict[str, Any]:
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    @staticmethod
    def _restore_rng(rng: dict[str, Any]) -> None:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])

    def _autocast(self) -> Any:
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16)

    def describe(self) -> dict[str, Any]:
        return {
            "task": self.cfg.task, "mode": self.cfg.mode, "device": str(self.device),
            "precision": self.precision, "parameters": sum(p.numel() for p in self.model.parameters()),
            "training_examples": len(self.train_rows), "validation_examples": len(self.validation_rows),
            "effective_batch": self.cfg.batch_size * self.cfg.accumulation_steps,
            "total_optimizer_steps": self.total_steps,
            "resumed_optimizer_step": self.state["global_step"], "phase": self.state["phase"],
        }

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def _sample_weights(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        values = [1.0] * len(rows)
        if self.cfg.task == "sentence" and self.cfg.source_weighted_loss:
            values = [{"babe": 2.0, "mbic": 1.0, "basil": 0.5}.get(r["dataset"], 1.0) for r in rows]
        target = torch.tensor([r["label_id"] for r in rows], device=self.device)
        return torch.tensor(values, device=self.device) * self.class_weights[target]

    def _training_step(self, order: list[int]) -> None:
        cfg, state = self.cfg, self.state
        start = state["batch_cursor"]
        end = min(start + cfg.accumulation_steps, self.batches_per_epoch)
        rows = [self.train_rows[i] for i in order[start * cfg.batch_size:end * cfg.batch_size]]
        denominator = self._sample_weights(rows).sum()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for offset in range(0, len(rows), cfg.batch_size):
            micro_rows = rows[offset:offset + cfg.batch_size]
            batch = self._move(self.collate(micro_rows))
            labels = batch.pop("labels")
            with self._autocast():
                logits = self.model(**batch).logits
                losses = F.cross_entropy(logits.float(), labels, reduction="none")
                numerator = (losses * self._sample_weights(micro_rows)).sum()
                loss = numerator / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; checkpoint remains at the last completed step")
            self.scaler.scale(loss).backward()
            loss_sum += numerator.detach().item()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        previous_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.get_scale() >= previous_scale:
            self.scheduler.step()
            state["global_step"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        state["batch_cursor"] = end
        state["epoch_loss_sum"] += loss_sum
        state["epoch_loss_weight"] += denominator.item()

    def _validation_step(self) -> None:
        state = self.state
        start = state["validation_cursor"]
        rows = self.validation_rows[start:start + self.cfg.batch_size]
        batch = self._move(self.collate(rows))
        labels = batch.pop("labels")
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            probabilities = self.model(**batch).logits.float().softmax(-1).cpu().tolist()
        state["validation_targets"].extend(labels.cpu().tolist())
        state["validation_probabilities"].extend(probabilities)
        state["validation_cursor"] += len(rows)

    def _finish_epoch(self) -> None:
        state = self.state
        record = {"epoch": state["epoch"] + 1,
                  "train_loss": state["epoch_loss_sum"] / max(state["epoch_loss_weight"], 1e-8)}
        if self.cfg.mode == "development":
            targets = state["validation_targets"]
            predictions = np.asarray(state["validation_probabilities"]).argmax(-1).tolist()
            score = f1_score(targets, predictions, labels=list(range(len(self.cfg.labels))), average="macro", zero_division=0)
            record.update(val_macro_f1=float(score), val_accuracy=float(accuracy_score(targets, predictions)))
            if state["best_macro_f1"] is None or score > state["best_macro_f1"]:
                state["best_macro_f1"] = float(score)
                self._save_model(self.output_dir / "best_model")
                atomic_json(self.output_dir / "best_model" / "selection.json", record)
        state["history"].append(record)
        print(json.dumps(record), flush=True)
        state.update(epoch=state["epoch"] + 1, batch_cursor=0, phase="train",
                     epoch_loss_sum=0.0, epoch_loss_weight=0.0, validation_cursor=0,
                     validation_targets=[], validation_probabilities=[])
        if state["epoch"] >= self.cfg.epochs:
            state["phase"] = "complete"
            state["stop_reason"] = "epoch_budget_completed"
        elif self.cfg.mode == "development" and self.cfg.early_stopping_patience is not None:
            best = max(state["history"], key=lambda row: row["val_macro_f1"])
            if state["epoch"] - best["epoch"] >= self.cfg.early_stopping_patience:
                state["phase"] = "complete"
                state["stop_reason"] = "validation_macro_f1_plateau"

    def _save_model(self, directory: Path) -> None:
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)

    def checkpoint(self) -> dict[str, Any]:
        """Publish one atomic archive; downloading resume.zip is sufficient to resume."""
        started = time.monotonic()
        staging = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=self.output_dir))
        try:
            self._save_model(staging / "model")
            manifest = {"format_version": 1, "fingerprint": self.fingerprint,
                        "config": asdict(self.cfg), "dataset_sha256": self.data_hash,
                        "source": self.data["source"], "labels": self.cfg.labels,
                        "progress": self.state, "precision_used": self.precision,
                        "torch_version": torch.__version__}
            atomic_json(staging / "manifest.json", manifest)
            if (self.output_dir / "evaluation_state.json").exists():
                shutil.copy2(self.output_dir / "evaluation_state.json", staging / "evaluation_state.json")
            best = self.output_dir / "best_model"
            temporary = self.output_dir / "resume.zip.tmp"
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                with archive.open("trainer_state.pt", "w", force_zip64=True) as stream:
                    torch.save({"state": self.state, "optimizer": self.optimizer.state_dict(),
                                "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                                "rng": self._rng()}, stream)
                for path in staging.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(staging))
                if best.exists() and self.cfg.mode == "development":
                    for path in best.rglob("*"):
                        if path.is_file():
                            archive.write(path, Path("best_model") / path.relative_to(best))
            os.replace(temporary, self.output_dir / "resume.zip")
            progress = {"status": self.state["phase"], "epoch": self.state["epoch"],
                        "batch_cursor": self.state["batch_cursor"],
                        "global_step": self.state["global_step"], "total_steps": self.total_steps,
                        "best_macro_f1": self.state["best_macro_f1"], "history": self.state["history"],
                        "fingerprint": self.fingerprint, "archive": "resume.zip",
                        "checkpoint_seconds": round(time.monotonic() - started, 2)}
            evaluation_path = self.output_dir / "evaluation_state.json"
            if evaluation_path.exists():
                evaluation = json.loads(evaluation_path.read_text())
                progress["evaluation_status"] = evaluation["status"]
                progress["evaluated_examples"] = len(evaluation["predictions"])
            atomic_json(self.output_dir / "progress.json", progress)
            atomic_json(self.output_dir / "training_history.json", self.state["history"])
            self.last_checkpoint_at = time.monotonic()
            return progress
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def run(self, *, session_started_at: float | None = None, max_steps: int | None = None) -> dict[str, Any]:
        started = session_started_at if session_started_at is not None else time.monotonic()
        deadline = started + self.cfg.session_seconds - self.cfg.save_reserve_seconds
        active_started = time.monotonic()
        initial_step = self.state["global_step"]
        last_report = time.monotonic()
        order_epoch, order = -1, []
        while self.state["phase"] != "complete":
            if time.monotonic() >= deadline:
                break
            if max_steps is not None and self.state["global_step"] - initial_step >= max_steps:
                break
            state = self.state
            if state["phase"] == "train":
                if state["batch_cursor"] >= self.batches_per_epoch:
                    if self.cfg.mode == "full":
                        self._finish_epoch()
                    else:
                        state["phase"] = "validation"
                    continue
                if order_epoch != state["epoch"]:
                    generator = torch.Generator().manual_seed(self.cfg.seed + state["epoch"])
                    order = torch.randperm(len(self.train_rows), generator=generator).tolist()
                    order_epoch = state["epoch"]
                self._training_step(order)
            elif state["phase"] == "validation":
                if state["validation_cursor"] >= len(self.validation_rows):
                    self._finish_epoch()
                    continue
                self._validation_step()
            else:
                raise ValueError(f"Unknown training phase: {state['phase']}")
            if time.monotonic() - last_report >= 15:
                print(f"epoch={state['epoch'] + 1}/{self.cfg.epochs} phase={state['phase']} "
                      f"step={state['global_step']}/{self.total_steps} batch={state['batch_cursor']}/{self.batches_per_epoch}", flush=True)
                last_report = time.monotonic()
            if time.monotonic() - self.last_checkpoint_at >= self.cfg.checkpoint_every_seconds:
                self.checkpoint()
        self.state["training_seconds"] += time.monotonic() - active_started
        result = self.checkpoint()
        print(f"Saved {self.output_dir / 'resume.zip'}; phase={result['status']}; step={result['global_step']}", flush=True)
        return result

    def evaluate_test(self, output_dir: Path, *, session_started_at: float | None = None) -> dict[str, Any]:
        """Run explicitly after recipe selection, in a separate evaluation session."""
        if self.cfg.mode != "development" or self.state["phase"] != "complete":
            raise ValueError("Independent test evaluation requires a completed development run")
        best = self.output_dir / "best_model"
        if not best.is_dir():
            raise FileNotFoundError("Selected best_model is missing")
        self.model.to("cpu")
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cpu()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        model = (ArticleModel.from_pretrained(best) if self.cfg.task == "article"
                 else AutoModelForSequenceClassification.from_pretrained(best)).to(self.device).eval()
        records = self.data["splits"]["test"]
        state_path = self.output_dir / "evaluation_state.json"
        evaluation = json.loads(state_path.read_text()) if state_path.exists() else {"status": "running", "predictions": []}
        saved = evaluation["predictions"]
        began = session_started_at if session_started_at is not None else time.monotonic()
        deadline = began + self.cfg.session_seconds - self.cfg.save_reserve_seconds
        last_save = time.monotonic()
        for start in range(len(saved), len(records), self.cfg.batch_size):
            if time.monotonic() >= deadline:
                break
            rows = records[start:start + self.cfg.batch_size]
            batch = self._move(self.collate(rows))
            label_ids = batch.pop("labels").cpu().tolist()
            with torch.inference_mode(), self._autocast():
                probabilities = model(**batch).logits.float().softmax(-1).cpu().tolist()
            for row, target, scores in zip(rows, label_ids, probabilities, strict=True):
                predicted = int(np.argmax(scores))
                saved.append({"id": row["id"], "text_hash": row["text_hash"],
                              "label": row["label"], "prediction": self.cfg.labels[predicted],
                              "probabilities": dict(zip(self.cfg.labels, scores, strict=True)),
                              "sources": row["sources"]})
            if time.monotonic() - last_save >= self.cfg.checkpoint_every_seconds:
                atomic_json(state_path, evaluation)
                self.checkpoint()
                last_save = time.monotonic()
        evaluation["status"] = "complete" if len(saved) == len(records) else "running"
        atomic_json(state_path, evaluation)
        self.checkpoint()
        if evaluation["status"] != "complete":
            return {"status": "paused", "evaluated": len(saved), "total": len(records)}
        targets = [self.cfg.labels.index(row["label"]) for row in saved]
        predictions = [self.cfg.labels.index(row["prediction"]) for row in saved]
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {"count": len(targets), "accuracy": float(accuracy_score(targets, predictions)),
                  "macro_f1": float(f1_score(targets, predictions, average="macro")),
                  "per_class": classification_report(targets, predictions, labels=list(range(len(self.cfg.labels))), target_names=self.cfg.labels, output_dict=True, zero_division=0),
                  "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(self.cfg.labels)))).tolist()}
        atomic_json(output_dir / "test_metrics.json", report)
        with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for prediction in saved:
                handle.write(json.dumps(prediction) + "\n")
        return report


def plot_history(history: list[dict[str, Any]], *, output_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    if not history:
        print("No complete epoch yet. Progress is saved; the next slice continues this epoch.")
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    if "val_macro_f1" in history[0]:
        axes[1].plot(epochs, [r["val_macro_f1"] for r in history], marker="o", label="Macro-F1")
        axes[1].plot(epochs, [r["val_accuracy"] for r in history], marker="o", label="Accuracy")
        axes[1].set(title="Validation", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
        axes[1].legend()
    else:
        axes[1].axis("off")
        axes[1].text(0.1, 0.5, "Full-data refit\nNo independent validation split")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
