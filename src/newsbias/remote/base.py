"""Base interfaces and protocols for executing arbitrary workloads on remote compute."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class RemoteJobStatus(StrEnum):
    """Provider-independent states used by polling and artifact orchestration."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """Tell polling code whether the job can transition any further."""
        return self in (
            RemoteJobStatus.COMPLETE,
            RemoteJobStatus.ERROR,
            RemoteJobStatus.CANCELLED,
        )


@dataclass(frozen=True)
class RemoteJobConfig:
    """Portable recipe for staging and executing one Python entrypoint remotely.

    `entrypoint` and `extra_files` are copied to the runner. `args` and
    `pip_packages` configure the subprocess environment; `gpu` and `internet`
    request provider resources. Timeout settings bound status polling.
    """

    name: str
    entrypoint: Path
    extra_files: list[Path] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    pip_packages: list[str] = field(default_factory=list)
    gpu: bool = True
    internet: bool = True
    env_vars: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 3600
    poll_interval_seconds: int = 15

    def __post_init__(self) -> None:
        """Reject incomplete job specifications before any files are staged or uploaded."""
        if not self.name.strip():
            raise ValueError("Remote job name must not be empty")
        if not self.entrypoint.is_file():
            raise FileNotFoundError(f"Entrypoint script not found: {self.entrypoint}")
        for path in self.extra_files:
            if not path.exists():
                raise FileNotFoundError(f"Extra file or directory not found: {path}")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")


@dataclass(frozen=True)
class RemoteRunResult:
    """Final job state plus any downloaded output directory and execution log."""

    job_id: str
    status: RemoteJobStatus
    duration_seconds: float
    output_dir: Path | None = None
    log_file: Path | None = None
    error_message: str | None = None


class RemoteRunner(ABC):
    """Platform boundary for staging, monitoring, and retrieving arbitrary jobs.

    Local model and dataset workflows depend on this contract instead of provider
    commands, so another remote backend can implement the same job lifecycle.
    """

    @abstractmethod
    def submit(self, job: RemoteJobConfig, staging_dir: Path | None = None) -> str:
        """Package the job inputs and submit them to a remote execution service.

        Args:
            job: Job specification containing script, dependencies, and resources.
            staging_dir: Optional custom directory to assemble bundle files.

        Returns:
            Provider-specific identifier used by status, wait, and download methods.
        """
        raise NotImplementedError

    @abstractmethod
    def status(self, job_id: str) -> RemoteJobStatus:
        """Translate the provider's current state into the shared lifecycle enum."""
        raise NotImplementedError

    @abstractmethod
    def wait(
        self,
        job_id: str,
        *,
        timeout_seconds: int = 3600,
        poll_interval_seconds: int = 15,
    ) -> RemoteJobStatus:
        """Poll at the requested interval until terminal state or timeout."""
        raise NotImplementedError

    @abstractmethod
    def download(self, job_id: str, destination: Path) -> Path:
        """Retrieve files created by a completed job into a local directory.

        Args:
            job_id: Identifier of the finished remote job.
            destination: Local directory where artifacts should be saved.

        Returns:
            Path to the directory containing downloaded artifacts.
        """
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        job: RemoteJobConfig,
        *,
        wait: bool = True,
        output_dir: Path | None = None,
    ) -> RemoteRunResult:
        """Coordinate staging, submission, optional waiting, and result retrieval.

        Implementations return a structured result even when a remote job fails so
        callers can inspect status, timing, logs, and any downloadable failure output.
        """
        raise NotImplementedError
