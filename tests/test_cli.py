"""Tests for unified newsbias CLI workflows, subcommands, and caching."""

import json
from pathlib import Path

import pytest
from datasets import DatasetDict

from newsbias.adapters import AdapterConfig
from newsbias.cli import (
    main,
    resolve_dataset,
    resolve_model,
)
from newsbias.experiment import ExperimentConfig, run_experiment
from newsbias.models import TfidfModel
from newsbias.remote.base import RemoteJobConfig, RemoteJobStatus, RemoteRunResult


def test_benchmark_cli_uses_configured_adapter_pair(
    dataset: DatasetDict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Route arbitrary adapter paths through the benchmark command and shared runner.

    The factory is stubbed with a fixture source and fitted test classifier, proving
    the command does not require a model-name switch in its parser.
    """
    from newsbias.benchmark_cli import main as benchmark_main

    def select_adapter(config: AdapterConfig) -> object:
        """Stand in for plugin construction based on the config's import path."""
        adapter_path = str(config["adapter"])
        if ".datasets." in adapter_path:
            class FixtureSource:
                """Provide a normalized fixture through the dataset-source contract."""

                def load(self) -> DatasetDict:
                    """Supply normalized rows without a Hub request."""
                    return dataset

            return FixtureSource()
        model = TfidfModel(min_df=1)
        model.fit(dataset["train"])
        return model

    monkeypatch.setattr("newsbias.benchmark_cli.load_adapter", select_adapter)
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/benchmark_politicalbiasbert_allsides.toml"
    )
    monkeypatch.setattr("sys.argv", [
        "newsbias-benchmark", "--config", str(config_path), "--split", "test",
        "--output-root", str(tmp_path),
    ])
    benchmark_main()
    output = capsys.readouterr().out
    assert "test: accuracy=1.0000" in output
    assert "failures=0" in output


def test_cli_runs_configured_experiment(
    dataset: DatasetDict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep the config-only training entrypoint wired to the generic experiment runner."""
    config_path = Path(__file__).resolve().parents[1] / "configs/tfidf_allsides.toml"

    def offline_run(
        config: ExperimentConfig,
        *,
        output_root: Path,
    ) -> Path:
        """Replace source acquisition while retaining the production run workflow."""
        return run_experiment(
            config,
            output_root=output_root,
            dataset=dataset,
        )

    monkeypatch.setattr("newsbias.cli.run_experiment", offline_run)
    monkeypatch.setattr("sys.argv", [
        "newsbias", "--config", str(config_path), "--split", "test",
        "--output-root", str(tmp_path),
    ])
    main()
    output = capsys.readouterr().out
    assert "Run artifacts:" in output
    assert "test: accuracy=1.0000" in output
    assert "validation:" not in output


def test_unified_cli_run_local_and_caching(
    dataset: DatasetDict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exercise alias-based local evaluation, result reuse, and forced reruns."""
    class FakeDatasetSource:
        """Provide deterministic split data so CLI behavior stays network-independent."""

        def load(self) -> DatasetDict:
            """Return the shared fixture dataset for the alias-based CLI flow."""
            return dataset

    monkeypatch.setattr(
        "newsbias.cli.load_adapter",
        lambda cfg: FakeDatasetSource() if "datasets" in cfg["adapter"] else TfidfModel(min_df=1),
    )

    # First run: cache miss, executes locally
    monkeypatch.setattr("sys.argv", [
        "newsbias", "run", "--runner", "local", "--model", "tfidf",
        "--dataset", "allsides", "--split", "test", "--output-root", str(tmp_path),
    ])
    main()
    out1 = capsys.readouterr().out
    assert "NEWSBIAS BENCHMARK REPORT" in out1
    assert "Accuracy:      1.0000" in out1
    assert "[CACHE HIT]" not in out1

    # Second run: cache hit, loads without re-running
    monkeypatch.setattr("sys.argv", [
        "newsbias", "run", "--runner", "local", "--model", "tfidf",
        "--dataset", "allsides", "--split", "test", "--output-root", str(tmp_path),
    ])
    main()
    out2 = capsys.readouterr().out
    assert "[CACHE HIT]" in out2
    assert "NEWSBIAS BENCHMARK REPORT" in out2

    # Third run: forced rerun bypasses cache
    monkeypatch.setattr("sys.argv", [
        "newsbias", "run", "--runner", "local", "--model", "tfidf",
        "--dataset", "allsides", "--split", "test", "--output-root", str(tmp_path),
        "--force-rerun",
    ])
    main()
    out3 = capsys.readouterr().out
    assert "[CACHE HIT]" not in out3
    assert "NEWSBIAS BENCHMARK REPORT" in out3


def test_unified_cli_run_kaggle_mocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Route a remote run through the runner interface and format its downloaded report."""
    class MockKaggleRunner:
        """Fake successful remote execution that materializes a summary artifact."""

        def run(
            self,
            job: RemoteJobConfig,
            *,
            wait: bool = True,
            output_dir: Path | None = None,
        ) -> RemoteRunResult:
            """Write a representative result bundle and report a completed remote job."""
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "model_name": "matous-volf/political-leaning-deberta-large",
                "dataset_name": "AllSides",
                "split": "test",
                "total_evaluated": 100,
                "accuracy": 0.82,
                "macro_f1": 0.81,
                "duration_seconds": 45.2,
                "per_class": {
                    "left": {"precision": 0.84, "recall": 0.83, "f1": 0.835, "support": 34},
                    "center": {"precision": 0.79, "recall": 0.80, "f1": 0.795, "support": 33},
                    "right": {"precision": 0.83, "recall": 0.83, "f1": 0.83, "support": 33},
                },
            }
            (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            return RemoteRunResult(
                job_id="testuser/eval-deberta",
                status=RemoteJobStatus.COMPLETE,
                duration_seconds=45.2,
                output_dir=output_dir,
            )

    monkeypatch.setattr("newsbias.remote.kaggle.KaggleRunner", MockKaggleRunner)

    monkeypatch.setattr("sys.argv", [
        "newsbias", "run", "--runner", "kaggle", "--model", "deberta-large",
        "--dataset", "allsides", "--split", "test", "--output-root", str(tmp_path),
    ])
    main()
    out = capsys.readouterr().out
    assert "NEWSBIAS BENCHMARK REPORT" in out
    assert "matous-volf/political-leaning-deberta-large" in out
    assert "Runner:      kaggle" in out
    assert "Accuracy:      0.8200" in out


def test_unified_cli_run_kaggle_no_wait(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return a submitted job handle immediately when the caller requests no wait."""
    class MockKaggleRunner:
        """Capture submission without polling or downloading a kernel."""

        def submit(self, job: RemoteJobConfig) -> str:
            """Return a deterministic test kernel ID."""
            return "testuser/eval-kernel-123"

    monkeypatch.setattr("newsbias.remote.kaggle.KaggleRunner", MockKaggleRunner)

    monkeypatch.setattr("sys.argv", [
        "newsbias", "run", "--runner", "kaggle", "--model", "politics",
        "--dataset", "allsides", "--split", "test", "--no-wait",
        "--output-root", str(tmp_path),
    ])
    main()
    out = capsys.readouterr().out
    assert "Submitted Kaggle kernel: testuser/eval-kernel-123" in out
    assert "newsbias status --id testuser/eval-kernel-123" in out


def test_unified_cli_compare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Select the newest summary for each model and show comparable split metrics."""
    b_dir = tmp_path / "benchmarks"
    b_dir.mkdir(parents=True)

    run1 = b_dir / "20261029T100000Z-11111111"
    run1.mkdir()
    (run1 / "summary.json").write_text(json.dumps({
        "model_identifier": "deberta-large",
        "split": "test",
        "sample_count": 2565,
        "metrics": {"accuracy": 0.8246, "macro_f1": 0.8194, "count": 2565},
    }), encoding="utf-8")

    run2 = b_dir / "20261029T110000Z-22222222"
    run2.mkdir()
    (run2 / "summary.json").write_text(json.dumps({
        "model_identifier": "politics",
        "split": "test",
        "sample_count": 2565,
        "metrics": {"accuracy": 0.7812, "macro_f1": 0.7740, "count": 2565},
    }), encoding="utf-8")

    monkeypatch.setattr("sys.argv", [
        "newsbias", "compare", "--split", "test", "--output-root", str(tmp_path),
    ])
    main()
    out = capsys.readouterr().out
    assert "NEWSBIAS BENCHMARK COMPARISON (Split: test)" in out
    assert "deberta-large" in out
    assert "0.8246" in out
    assert "politics" in out
    assert "0.7812" in out


def test_unified_cli_alias_resolution() -> None:
    """Keep public aliases mapped to their model/data adapter settings and metadata."""
    # SOTA 1 - Full and short aliases
    for name in (
        "political-leaning-deberta-large",
        "political-learning-deberta-large",
        "deberta-large",
    ):
        m1, cfg1, map1 = resolve_model(name)
        assert m1 == "matous-volf/political-leaning-deberta-large"
        assert map1 == "0:left,1:center,2:right"
        assert cfg1["parameters"]["id2label"] == {0: "left", 1: "center", 2: "right"}

    # SOTA 2 - Full and short aliases
    for name in ("political-leaning-politics", "political-learning-politics", "politics"):
        m2, cfg2, map2 = resolve_model(name)
        assert m2 == "matous-volf/political-leaning-politics"
        assert map2 == "0:left,1:center,2:right"

    # BERT aliases
    for name in ("political-bias-bert", "politicalbiasbert", "bert"):
        mb, cfgb, _ = resolve_model(name)
        assert mb == "bucketresearch/politicalBiasBERT"
    assert map2 == "0:left,1:center,2:right"

    # Raw HF repo
    m3, cfg3, map3 = resolve_model("my-org/my-model")
    assert m3 == "my-org/my-model"
    assert map3 is None
    assert cfg3["parameters"]["model_name"] == "my-org/my-model"

    # Invalid model
    with pytest.raises(ValueError, match="Unknown model alias"):
        resolve_model("nonexistent-model")

    # Datasets
    d1, d_cfg1, meta1 = resolve_dataset("allsides")
    assert d1 == "AllSides"
    assert meta1["text_column"] == "content"

    d2, d_cfg2, meta2 = resolve_dataset("babe")
    assert d2 == "BABE-SG2"
    assert meta2["text_column"] == "text"

    with pytest.raises(ValueError, match="Unknown dataset alias"):
        resolve_dataset("nonexistent-dataset")


def test_unified_cli_status_and_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dispatch status and download subcommands without contacting Kaggle."""
    class MockRunner:
        """Provide deterministic terminal status and a small downloaded artifact."""

        def status(self, job_id: str) -> RemoteJobStatus:
            """Return a completed status for the expected job identifier."""
            assert job_id == "testuser/test-slug"
            return RemoteJobStatus.COMPLETE

        def download(self, job_id: str, dest: Path) -> Path:
            """Create a marker file to represent downloaded remote output."""
            assert job_id == "testuser/test-slug"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "downloaded.txt").write_text("ok")
            return dest

    monkeypatch.setattr("newsbias.remote.kaggle.KaggleRunner", MockRunner)

    # Status
    monkeypatch.setattr("sys.argv", ["newsbias", "status", "--id", "testuser/test-slug"])
    main()
    out_status = capsys.readouterr().out
    assert "Kernel testuser/test-slug status: complete" in out_status

    # Download
    dl_dir = tmp_path / "dl"
    monkeypatch.setattr("sys.argv", [
        "newsbias", "download", "--id", "testuser/test-slug", "--output-dir", str(dl_dir),
    ])
    main()
    out_dl = capsys.readouterr().out
    assert "Downloaded artifacts from testuser/test-slug to" in out_dl
    assert (dl_dir / "downloaded.txt").is_file()
