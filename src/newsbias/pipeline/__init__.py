"""Pipeline modules for sentence segmentation, bias aggregation, and article analysis."""

from newsbias.pipeline.analyzer import analyze_article
from newsbias.pipeline.segmenter import split_sentences

__all__ = ["analyze_article", "split_sentences"]
