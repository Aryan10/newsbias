"""Tests for article analyzer and CLI integration."""

from pathlib import Path

import pytest

from newsbias.cli import main
from newsbias.pipeline.analyzer import analyze_article


def test_analyze_article_empty() -> None:
    """Empty text returns zeroed metrics and empty sentence list."""
    res = analyze_article("")
    assert res["bias_summary"]["total_sentences"] == 0
    assert res["sentences"] == []


def test_analyze_article_with_mock() -> None:
    """Analyze article combines sentence offsets with bias and political metrics."""
    text = (
        "Officials announced a new budget proposal. "
        "Critics attacked the reckless plan with fury."
    )
    res = analyze_article(text, use_mock=True)

    assert "political_leaning" in res
    assert res["political_leaning"]["label"] in ("left", "center", "right")

    bias = res["bias_summary"]
    assert bias["total_sentences"] == 2
    assert bias["biased_sentences_count"] == 1
    assert bias["biased_ratio"] == 0.5
    assert bias["level"] == "high"

    sentences = res["sentences"]
    assert len(sentences) == 2
    for s in sentences:
        assert text[s["start"]:s["end"]] == s["text"]


def test_cli_analyze_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify 'newsbias analyze --mock' outputs formatted breakdown."""
    monkeypatch.setattr("sys.argv", [
        "newsbias", "analyze", "The president announced reform. Critics blasted the reckless move.",
        "--mock",
    ])
    main()
    out = capsys.readouterr().out
    assert "NEWSBIAS ARTICLE ANALYSIS" in out
    assert "Political Leaning:" in out
    assert "Bias Density:" in out
    assert "Sentence Breakdown:" in out


def test_cli_analyze_json_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify 'newsbias analyze --json' outputs valid parseable JSON."""
    import json

    monkeypatch.setattr("sys.argv", [
        "newsbias", "analyze", "Neutral sentence here.", "--mock", "--json",
    ])
    main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "political_leaning" in data
    assert "bias_summary" in data
    assert "sentences" in data
