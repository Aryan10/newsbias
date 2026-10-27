"""Offline language detection for the web application's English-only specialists."""

import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache

from lingua import LanguageDetectorBuilder

SUPPORTED_TRANSFORMER_LANGUAGES = frozenset({"en"})
MIN_CONFIDENCE = 0.80


@dataclass(frozen=True)
class ArticleLanguage:
    code: str
    name: str
    confidence: float
    status: str
    languages: tuple[str, ...] = ()

    @property
    def transformers_supported(self) -> bool:
        return self.status == "detected" and self.code in SUPPORTED_TRANSFORMER_LANGUAGES

    @property
    def skip_reason(self) -> str:
        if self.status == "mixed":
            return "Mixed-language text is outside these English-only specialist checkpoints."
        if self.status != "detected":
            return "Language detection is uncertain. These checkpoints support English only."
        return f"Detected {self.name} ({self.code}). These checkpoints support English only."

    def to_dict(self) -> dict:
        return {**asdict(self), "transformers_supported": self.transformers_supported}


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


def detect_article_language(text: str) -> ArticleLanguage:
    sample = text[:20_000]
    letters = [character for character in sample if character.isalpha()]
    if len(letters) < 20:
        return ArticleLanguage("und", "Undetermined", 0.0, "uncertain")

    # If the text is in Latin script in general, keep it as English so that
    # proper nouns, acronyms, or multilingual names don't cause false non-English rejections.
    latin_count = sum("LATIN" in unicodedata.name(character, "") for character in letters)
    if latin_count / len(letters) >= 0.70:
        return ArticleLanguage("en", "English", 1.0, "detected", ("en",))

    detector = _detector()
    ranked = detector.compute_language_confidence_values(sample)
    if not ranked:
        return ArticleLanguage("und", "Undetermined", 0.0, "uncertain")
    winner = ranked[0]
    code = winner.language.iso_code_639_1.name.lower()
    name = winner.language.name.replace("_", " ").title()
    confidence = round(winner.value, 4)
    # Detect substantial code-switching between paragraphs. Short names/quotations
    # alone do not cause an otherwise English article to be marked as mixed.
    weights: dict[str, int] = {}
    for paragraph in re.split(r"\n+", sample):
        letters = sum(character.isalpha() for character in paragraph)
        if letters < 40 or len(paragraph.split()) < 5:
            continue
        values = detector.compute_language_confidence_values(paragraph)
        if values and values[0].value >= MIN_CONFIDENCE:
            paragraph_code = values[0].language.iso_code_639_1.name.lower()
            weights[paragraph_code] = weights.get(paragraph_code, 0) + letters
    if len(weights) > 1:
        total = sum(weights.values())
        significant = tuple(sorted(key for key, weight in weights.items() if weight / total >= 0.2))
        if len(significant) > 1:
            return ArticleLanguage("mul", "Mixed languages", confidence, "mixed", significant)
    letters = [character for character in sample if character.isalpha()]
    non_latin = sum("LATIN" not in unicodedata.name(character, "") for character in letters)
    if code == "en" and non_latin / len(letters) >= 0.2:
        return ArticleLanguage("mul", "Mixed languages", confidence, "mixed", ("en", "und"))
    if confidence < MIN_CONFIDENCE:
        return ArticleLanguage("und", "Undetermined", confidence, "uncertain", (code,))
    return ArticleLanguage(code, name, confidence, "detected", (code,))
