"""Sentence segmentation with exact character offsets for UI text highlighting."""

import re

# Match sentence-ending punctuation plus any closing quotes/brackets followed
# by whitespace or end of string
SENTENCE_SPLIT_PATTERN = re.compile(r'([.?!]+[\'"\)\]]*(?=\s+|$))')


def split_sentences(text):
    """Split article text into sentences with exact character offsets.

    Returns a list of dicts:
    [
        {"index": 0, "text": "Sentence text", "start": 0, "end": 45},
        ...
    ]
    Guarantees: text[item["start"]:item["end"]] == item["text"]
    """
    if not text or not text.strip():
        return []

    sentences = []

    # Tokenize by punctuation boundary while keeping tracked offsets
    raw_spans = []
    last_end = 0

    for match in SENTENCE_SPLIT_PATTERN.finditer(text):
        split_point = match.end()
        raw_spans.append((last_end, split_point))
        last_end = split_point

    if last_end < len(text):
        raw_spans.append((last_end, len(text)))

    sentence_idx = 0
    for span_start, span_end in raw_spans:
        chunk = text[span_start:span_end]
        stripped = chunk.strip()
        if not stripped:
            continue

        # Find exact character offsets of stripped sentence within original text
        rel_start = chunk.find(stripped)
        start_char = span_start + rel_start
        end_char = start_char + len(stripped)

        sentences.append({
            "index": sentence_idx,
            "text": stripped,
            "start": start_char,
            "end": end_char,
        })
        sentence_idx += 1

    return sentences
