"""TF-IDF and logistic regression, fitted exclusively on training examples."""

import warnings
from pathlib import Path

import joblib
from datasets import Dataset
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from newsbias.models.base import Model, Prediction


class TfidfModel(Model):
    """Conventional text baseline pairing word TF-IDF features with logistic regression.

    Vectorizer vocabulary and IDF weights are learned only in `fit` from training
    text. The fitted sklearn pipeline can be saved as a single local joblib bundle.
    """

    def __init__(
        self,
        *,
        max_features: int = 100_000,
        ngram_max: int = 2,
        min_df: int = 2,
        c: float = 1.0,
        max_iter: int = 1000,
        seed: int = 42,
    ) -> None:
        """Set feature limits and classifier regularization before fitting.

        Defaults use up to 100,000 unigram/bigram features, sublinear term frequency,
        minimum document frequency 2, logistic-regression C=1, and seed 42.
        """
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, ngram_max),
                min_df=min_df,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                C=c, max_iter=max_iter, random_state=seed, solver="lbfgs",
            )),
        ])

    def fit(self, train: Dataset, validation: Dataset | None = None) -> None:
        """Learn the text vocabulary, IDF weights, and classifier from `train` only.

        `validation` is accepted for interface compatibility but is deliberately
        unused because this baseline has no validation-selected hyperparameters.
        Empty or single-class training data and convergence warnings are errors.
        """
        if not len(train):
            raise ValueError("Training dataset is empty")
        if len(set(train["label"])) < 2:
            raise ValueError("Training requires at least two classes")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.pipeline.fit(list(train["text"]), list(train["label"]))

    def predict(self, dataset: Dataset, *, batch_size: int = 128) -> list[Prediction]:
        """Classify texts in row order and return sklearn probability estimates.

        `batch_size` bounds calls to the vectorizer/classifier; it does not alter
        model semantics. The returned probabilities are not independently calibrated.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        check_is_fitted(self.pipeline.named_steps["classifier"])
        labels = self.pipeline.named_steps["classifier"].classes_.tolist()
        predictions: list[Prediction] = []
        for start in range(0, len(dataset), batch_size):
            texts = dataset[start:start + batch_size]["text"]
            probabilities = self.pipeline.predict_proba(texts)
            for values in probabilities:
                predictions.append({
                    "label": labels[int(values.argmax())],
                    "probabilities": dict(zip(labels, map(float, values), strict=True)),
                })
        return predictions

    def save(self, path: Path) -> None:
        """Persist the fitted vectorizer/classifier as `pipeline.joblib` in `path`.

        Saving an unfitted instance raises sklearn's `NotFittedError` rather than
        producing an unusable inference bundle.
        """
        check_is_fitted(self.pipeline.named_steps["classifier"])
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path / "pipeline.joblib")

    @classmethod
    def load(cls, path: Path) -> "TfidfModel":
        """Restore `pipeline.joblib` from a trusted local artifact directory.

        Joblib deserialization should only be used with bundles from trusted sources
        and a compatible Python/scikit-learn environment.
        """
        model = cls()
        model.pipeline = joblib.load(path / "pipeline.joblib")
        check_is_fitted(model.pipeline.named_steps["classifier"])
        return model
