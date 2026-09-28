"""Tests for the model contract and trainable TF-IDF reference implementation."""

from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict
from sklearn.exceptions import NotFittedError

from newsbias.models import Model, TfidfModel


def test_unsupported_operations(dataset: DatasetDict, tmp_path: Path) -> None:
    """Keep the plain base model honest when an adapter has no implementation."""
    model = Model()
    with pytest.raises(NotImplementedError):
        model.fit(dataset["train"])
    with pytest.raises(NotImplementedError):
        model.predict(dataset["test"])
    with pytest.raises(NotImplementedError):
        model.save(tmp_path)


def test_training_predictions_and_reload(dataset: DatasetDict, tmp_path: Path) -> None:
    """Prevent validation vocabulary leakage and verify bundle reload equivalence."""
    model = TfidfModel(min_df=1)
    validation = Dataset.from_dict({"text": ["validationonlyword"], "label": ["left"]})
    assert model.fit(dataset["train"], validation) is None
    vocabulary = model.pipeline.named_steps["tfidf"].vocabulary_
    assert "validationonlyword" not in vocabulary
    inputs = dataset["test"].select_columns(["text"])
    predictions = model.predict(inputs, batch_size=2)
    assert [p["label"] for p in predictions] == list(dataset["test"]["label"])
    for prediction in predictions:
        assert set(prediction["probabilities"]) == {"left", "center", "right"}
        assert sum(prediction["probabilities"].values()) == pytest.approx(1)
    assert model.save(tmp_path) is None
    assert TfidfModel.load(tmp_path).predict(inputs, batch_size=1) == predictions
    assert model.predict(inputs.select([])) == []


def test_invalid_model_inputs(dataset: DatasetDict, tmp_path: Path) -> None:
    """Raise actionable errors for unfitted prediction, bad batches, and invalid fit data."""
    model = TfidfModel()
    with pytest.raises(NotFittedError):
        model.predict(dataset["test"])
    with pytest.raises(NotFittedError):
        model.save(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        model.predict(dataset["test"], batch_size=0)
    with pytest.raises(ValueError, match="empty"):
        model.fit(dataset["train"].select([]))
    with pytest.raises(ValueError, match="two classes"):
        model.fit(dataset["train"].select([0, 1]))
