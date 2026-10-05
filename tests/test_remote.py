"""Tests for remote execution abstractions, Kaggle runner, and CLI."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from newsbias.remote.base import (
    RemoteJobConfig,
    RemoteJobStatus,
)
from newsbias.remote.cli import build_parser
from newsbias.remote.kaggle import (
    KaggleRunner,
    parse_kaggle_status_output,
    slugify,
)


def test_slugify() -> None:
    """Normalize punctuation, repeated separators, and empty names for Kaggle IDs."""
    assert slugify("BERT Bias Benchmark 2026!") == "bert-bias-benchmark-2026"
    assert slugify("___A__B___") == "a-b"
    assert slugify("") == "newsbias-job"


def test_job_config_validation(tmp_path: Path) -> None:
    """Validate required entrypoint files and reject invalid resource timeouts."""
    script = tmp_path / "script.py"
    script.write_text("print('hello')", encoding="utf-8")

    job = RemoteJobConfig(name="test-job", entrypoint=script)
    assert job.name == "test-job"
    assert job.entrypoint == script
    assert job.gpu is True
    assert job.extra_files == []

    with pytest.raises(ValueError, match="must not be empty"):
        RemoteJobConfig(name="", entrypoint=script)

    with pytest.raises(FileNotFoundError, match="Entrypoint script not found"):
        RemoteJobConfig(name="test", entrypoint=tmp_path / "nonexistent.py")

    with pytest.raises(FileNotFoundError, match="Extra file or directory not found"):
        RemoteJobConfig(name="test", entrypoint=script, extra_files=[tmp_path / "missing.txt"])

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        RemoteJobConfig(name="test", entrypoint=script, timeout_seconds=0)


def test_parse_kaggle_status_output() -> None:
    """Map representative Kaggle CLI status strings, including unknown output."""
    assert parse_kaggle_status_output('status is "queued"') == RemoteJobStatus.QUEUED
    assert parse_kaggle_status_output('status is "running"') == RemoteJobStatus.RUNNING
    assert parse_kaggle_status_output('status is "complete"') == RemoteJobStatus.COMPLETE
    assert parse_kaggle_status_output('status is "error"') == RemoteJobStatus.ERROR
    assert parse_kaggle_status_output('status is "cancelRequested"') == RemoteJobStatus.CANCELLED
    assert parse_kaggle_status_output("some other text") == RemoteJobStatus.UNKNOWN


def test_kaggle_runner_resolve_username(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Resolve the account name from explicit configuration, env, and file settings."""
    runner = KaggleRunner(username="explicit_user")
    assert runner.resolve_username() == "explicit_user"

    monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
    runner_env = KaggleRunner()
    assert runner_env.resolve_username() == "env_user"

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    fake_home = tmp_path / "home"
    kaggle_dir = fake_home / ".kaggle"
    kaggle_dir.mkdir(parents=True)
    (kaggle_dir / "kaggle.json").write_text(
        json.dumps({"username": "file_user"}), encoding="utf-8"
    )
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    runner_file = KaggleRunner()
    assert runner_file.resolve_username() == "file_user"


def test_prepare_staging_bundle(tmp_path: Path) -> None:
    """Stage entrypoint, extra files, kernel metadata, and bootstrap dependencies."""
    script = tmp_path / "entry.py"
    script.write_text("print('running entry')", encoding="utf-8")

    extra_file = tmp_path / "helper.py"
    extra_file.write_text("def helper(): pass", encoding="utf-8")

    extra_dir = tmp_path / "pkg"
    extra_dir.mkdir()
    (extra_dir / "mod.py").write_text("x = 1", encoding="utf-8")

    staging = tmp_path / "staging"

    job = RemoteJobConfig(
        name="Transformer Benchmark",
        entrypoint=script,
        extra_files=[extra_file, extra_dir],
        args=["--epochs", "3"],
        pip_packages=["transformers", "torch"],
        env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        gpu=True,
    )

    runner = KaggleRunner(username="testuser")
    kernel_id = runner.prepare_staging_bundle(job, staging)

    assert kernel_id == "testuser/transformer-benchmark"
    assert (staging / "entry.py").is_file()
    assert (staging / "helper.py").is_file()
    assert (staging / "pkg" / "mod.py").is_file()
    assert (staging / "kernel-metadata.json").is_file()
    assert (staging / "__runner__.py").is_file()

    meta = json.loads((staging / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "testuser/transformer-benchmark"
    assert meta["enable_gpu"] == "true"
    assert meta["enable_internet"] == "true"

    runner_code = (staging / "__runner__.py").read_text(encoding="utf-8")
    assert "pip" in runner_code
    assert "transformers" in runner_code
    assert "HF_HUB_ENABLE_HF_TRANSFER" in runner_code
    assert "entry.py" in runner_code
    assert "['--epochs', '3']" in runner_code


def test_kaggle_runner_submit_and_status(tmp_path: Path) -> None:
    """Translate submit/status subprocess calls into a job ID and lifecycle state."""
    script = tmp_path / "entry.py"
    script.write_text("print('test')", encoding="utf-8")
    job = RemoteJobConfig(name="test-run", entrypoint=script)

    runner = KaggleRunner(username="testuser")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Kernel pushed successfully")
        job_id = runner.submit(job)
        assert job_id == "testuser/test-run"
        assert mock_run.call_args[0][0][:3] == ["kaggle", "kernels", "push"]

        mock_run.return_value = MagicMock(returncode=0, stdout='Your kernel has status "running"')
        status = runner.status(job_id)
        assert status == RemoteJobStatus.RUNNING


def test_kaggle_runner_download(tmp_path: Path) -> None:
    """Pass output retrieval to Kaggle CLI and ensure the destination is prepared."""
    runner = KaggleRunner(username="testuser")
    dest = tmp_path / "downloaded"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Output downloaded")
        res_dest = runner.download("testuser/test-run", dest)
        assert res_dest == dest
        assert dest.is_dir()
        assert mock_run.call_args[0][0][:4] == [
            "kaggle",
            "kernels",
            "output",
            "testuser/test-run",
        ]


def test_kaggle_runner_run_full(tmp_path: Path) -> None:
    """Coordinate submit, wait, and download while returning the finished artifact path."""
    script = tmp_path / "entry.py"
    script.write_text("print('test')", encoding="utf-8")
    job = RemoteJobConfig(name="e2e-job", entrypoint=script)

    runner = KaggleRunner(username="testuser")
    dest = tmp_path / "output"

    def mock_download(j: str, d: Path) -> Path:
        """Simulate Kaggle output retrieval for the orchestration test."""
        d.mkdir(parents=True, exist_ok=True)
        return d

    with (
        patch.object(runner, "submit", return_value="testuser/e2e-job"),
        patch.object(runner, "wait", return_value=RemoteJobStatus.COMPLETE),
        patch.object(runner, "download", side_effect=mock_download),
    ):
        result = runner.run(job, wait=True, output_dir=dest)

        assert result.job_id == "testuser/e2e-job"
        assert result.status == RemoteJobStatus.COMPLETE
        assert result.output_dir == dest
        assert result.duration_seconds >= 0.0


def test_cli_parsing() -> None:
    """Parse run, status, and download invocations without contacting Kaggle."""
    parser = build_parser()
    args = parser.parse_args([
        "kaggle",
        "run",
        "--script",
        "train.py",
        "--name",
        "my-run",
        "--args",
        "--epochs 5",
        "--pip",
        "torch",
        "transformers",
        "--no-wait",
    ])
    assert args.platform == "kaggle"
    assert args.action == "run"
    assert args.script == Path("train.py")
    assert args.name == "my-run"
    assert args.args == "--epochs 5"
    assert args.pip == ["torch", "transformers"]
    assert args.no_wait is True

    status_args = parser.parse_args(["kaggle", "status", "--id", "user/slug"])
    assert status_args.platform == "kaggle"
    assert status_args.action == "status"
    assert status_args.id == "user/slug"

    dl_args = parser.parse_args(
        ["kaggle", "download", "--id", "user/slug", "--output-dir", "out"]
    )
    assert dl_args.platform == "kaggle"
    assert dl_args.action == "download"
    assert dl_args.id == "user/slug"
    assert dl_args.output_dir == Path("out")
