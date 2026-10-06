"""Tests for sentence segmentation and character offset tracking."""

from newsbias.pipeline.segmenter import split_sentences


def test_split_sentences_empty() -> None:
    """Empty or whitespace-only text returns empty list."""
    assert split_sentences("") == []
    assert split_sentences("   \n\t  ") == []


def test_split_sentences_single_sentence() -> None:
    """Single sentence retains correct start and end offsets."""
    text = "This is a single sentence."
    sentences = split_sentences(text)
    assert len(sentences) == 1
    assert sentences[0]["text"] == text
    assert sentences[0]["start"] == 0
    assert sentences[0]["end"] == len(text)
    assert text[sentences[0]["start"]:sentences[0]["end"]] == sentences[0]["text"]


def test_split_sentences_multiple_with_offsets() -> None:
    """Multiple sentences with irregular spacing retain exact slice boundaries."""
    text = (
        "  First sentence is here.   Second sentence follows with question? "
        "Third one ends with exclamation! Final sentence without trailing space."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 4

    for s in sentences:
        assert text[s["start"]:s["end"]] == s["text"]
        assert not s["text"].startswith(" ")
        assert not s["text"].endswith(" ")


def test_split_sentences_quoted_speech() -> None:
    """Quoted sentences and dialogue are preserved with exact offsets."""
    text = 'He said, "This policy is reckless." She replied, "We disagree."'
    sentences = split_sentences(text)
    assert len(sentences) == 2
    for s in sentences:
        assert text[s["start"]:s["end"]] == s["text"]
