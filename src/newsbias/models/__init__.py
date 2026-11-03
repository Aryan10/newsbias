"""Model inference functions for sentence bias, political leaning, and LLM classifiers."""

from newsbias.models.article_hierarchy import ArticleConfig, ArticleModel
from newsbias.models.base import Model, Prediction
from newsbias.models.bias import predict_sentence_bias, predict_sentence_bias_mock
from newsbias.models.huggingface import HuggingFaceSequenceClassifier
from newsbias.models.llm import (
    analyze_article_llm,
    classify_political_llm,
    classify_sentences_llm,
)
from newsbias.models.openai import OpenAIChatClassifier
from newsbias.models.political import (
    DEFAULT_ARTICLE_MODEL,
    DEFAULT_FALLBACK_MODEL,
    predict_political_leaning,
    predict_political_leaning_mock,
)
from newsbias.models.tfidf import TfidfModel

__all__ = [
    "ArticleConfig",
    "ArticleModel",
    "DEFAULT_ARTICLE_MODEL",
    "DEFAULT_FALLBACK_MODEL",
    "HuggingFaceSequenceClassifier",
    "Model",
    "OpenAIChatClassifier",
    "Prediction",
    "TfidfModel",
    "analyze_article_llm",
    "classify_political_llm",
    "classify_sentences_llm",
    "predict_political_leaning",
    "predict_political_leaning_mock",
    "predict_sentence_bias",
    "predict_sentence_bias_mock",
]
