"""Kaggle Cloud GPU execution runner for arbitrary Python scripts and transformer jobs."""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from newsbias.remote.base import (
    RemoteJobConfig,
    RemoteJobStatus,
    RemoteRunner,
    RemoteRunResult,
)

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Normalize a display name into Kaggle's bounded, lowercase kernel-slug format."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:50] or "newsbias-job"


def parse_kaggle_status_output(output: str) -> RemoteJobStatus:
    """Map Kaggle CLI status text to the provider-independent lifecycle enum.

    Handles both legacy status format and newer KernelWorkerStatus representations.
    """
    lower = output.lower()
    if "complete" in lower or "status is complete" in lower:
        return RemoteJobStatus.COMPLETE
    if "running" in lower or "status is running" in lower:
        return RemoteJobStatus.RUNNING
    if "queued" in lower or "status is queued" in lower:
        return RemoteJobStatus.QUEUED
    if "error" in lower or "failed" in lower or "status is error" in lower:
        return RemoteJobStatus.ERROR
    if "cancel" in lower:
        return RemoteJobStatus.CANCELLED
    return RemoteJobStatus.UNKNOWN


class KaggleRunner(RemoteRunner):
    """Package arbitrary Python workloads as private Kaggle script kernels.

    This adapter delegates authentication to the installed Kaggle CLI and supports
    optional GPU, internet, pip dependencies, status polling, and artifact download.
    """

    def __init__(
        self,
        *,
        username: str | None = None,
        kaggle_cmd: str = "kaggle",
    ) -> None:
        """Set optional account/CLI overrides without reading or echoing credentials.

        When no username is supplied, `resolve_username()` checks standard Kaggle
        environment/config locations and finally asks the CLI for account metadata.
        """
        self._explicit_username = username
        self.kaggle_cmd = kaggle_cmd

    def resolve_username(self) -> str:
        """Find the account slug required for Kaggle's `username/kernel` identifiers.

        Only the public username is returned; credential contents are never logged.
        """
        if self._explicit_username:
            return self._explicit_username

        if env_user := os.getenv("KAGGLE_USERNAME"):
            return env_user.strip()

        # Check ~/.kaggle/credentials.json or ~/.kaggle/kaggle.json
        home = Path.home()
        for filename in ("credentials.json", "kaggle.json"):
            creds_file = home / ".kaggle" / filename
            if creds_file.is_file():
                try:
                    data = json.loads(creds_file.read_text(encoding="utf-8"))
                    if user := data.get("username"):
                        return str(user).strip()
                except Exception as error:
                    logger.debug("Could not read %s: %s", creds_file, error)

        # Fallback to querying kaggle kernels list
        try:
            res = subprocess.run(
                [self.kaggle_cmd, "kernels", "list", "--mine", "--page-size", "1"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in res.stdout.splitlines():
                line = line.strip()
                if line and not line.startswith("ref") and not line.startswith("-") and "/" in line:
                    slug_part = line.split()[0]
                    return slug_part.split("/")[0]
        except Exception as error:
            logger.debug("Failed querying kaggle kernels list: %s", error)

        raise RuntimeError(
            "Could not determine Kaggle username. Set KAGGLE_USERNAME in your environment "
            "or configure ~/.kaggle/kaggle.json."
        )

    def prepare_staging_bundle(self, job: RemoteJobConfig, staging_dir: Path) -> str:
        """Copy job inputs and generate Kaggle metadata plus an execution bootstrap.

        The bootstrap installs only configured packages, runs the entrypoint with its
        arguments, and streams output to Kaggle's working directory and execution log.
        The returned value is the kernel identifier that `kaggle kernels push` uses.
        """
        staging_dir.mkdir(parents=True, exist_ok=True)
        username = self.resolve_username()
        slug = slugify(job.name)
        kernel_id = f"{username}/{slug}"

        # Copy entrypoint script
        entrypoint_dest = staging_dir / job.entrypoint.name
        shutil.copy2(job.entrypoint, entrypoint_dest)

        # Copy extra files/directories
        for extra in job.extra_files:
            dest = staging_dir / extra.name
            if extra.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(extra, dest)
            else:
                shutil.copy2(extra, dest)

        # Generate kernel-metadata.json
        metadata = {
            "id": kernel_id,
            "title": slug,
            "code_file": "__runner__.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if job.gpu else "false",
            "enable_tpu": "false",
            "enable_internet": "true" if job.internet else "false",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (staging_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        # Generate __runner__.py bootstrap
        pip_installs = ""
        if job.pip_packages:
            packages_repr = repr(job.pip_packages)
            pip_installs = f"""
print(">>> [newsbias-kaggle] Installing pip packages: {job.pip_packages}...")
import subprocess, sys
cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location"] + {packages_repr}
subprocess.run(cmd, check=True)
"""

        env_exports = ""
        if job.env_vars:
            for k, v in job.env_vars.items():
                env_exports += f'os.environ["{k}"] = "{v}"\n'

        import base64

        args_repr = repr(job.args)
        entrypoint_repr = repr(job.entrypoint.name)
        entrypoint_b64 = base64.b64encode(job.entrypoint.read_bytes()).decode("ascii")

        extra_unpacks = ""
        for extra in job.extra_files:
            if extra.is_file():
                b64_content = base64.b64encode(extra.read_bytes()).decode("ascii")
                extra_unpacks += (
                    f'Path({repr(extra.name)}).write_bytes('
                    f'base64.b64decode({repr(b64_content)}))\n'
                )

        runner_script = f"""# Auto-generated remote execution bootstrap by newsbias.remote
import base64
import os
import sys
import subprocess
from pathlib import Path

working_dir = Path("/kaggle/working")
working_dir.mkdir(parents=True, exist_ok=True)
os.chdir(str(working_dir))

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

target_script = working_dir / {entrypoint_repr}
target_script.write_bytes(base64.b64decode({repr(entrypoint_b64)}))

{extra_unpacks}

{pip_installs}
{env_exports}

print(f">>> [newsbias-kaggle] Running python3 {job.entrypoint.name} with args {job.args}")
run_cmd = [sys.executable, str(target_script)] + {args_repr}

log_file = working_dir / "execution.log"
with open(log_file, "w", encoding="utf-8") as log_handle:
    proc = subprocess.Popen(
        run_cmd,
        cwd=str(working_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_handle.write(line)
        log_handle.flush()
    proc.wait()

print(f">>> [newsbias-kaggle] Finished with exit code: {{proc.returncode}}")
if proc.returncode != 0:
    sys.exit(proc.returncode)
"""
        (staging_dir / "__runner__.py").write_text(runner_script, encoding="utf-8")
        return kernel_id

    def submit(self, job: RemoteJobConfig, staging_dir: Path | None = None) -> str:
        """Build the bundle and submit it through the configured Kaggle CLI executable."""
        owns_staging = staging_dir is None
        staging_path = (
            staging_dir or Path(tempfile.mkdtemp(prefix=f"kaggle-{slugify(job.name)}-"))
        )
        try:
            kernel_id = self.prepare_staging_bundle(job, staging_path)
            logger.info("Pushing Kaggle kernel: %s from %s", kernel_id, staging_path)

            res = subprocess.run(
                [self.kaggle_cmd, "kernels", "push", "-p", str(staging_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"Kaggle kernels push failed (code {res.returncode}):\n"
                    f"stdout: {res.stdout}\nstderr: {res.stderr}"
                )
            logger.info("Kernel pushed successfully: %s", kernel_id)
            return kernel_id
        finally:
            if owns_staging and staging_path.exists():
                shutil.rmtree(staging_path, ignore_errors=True)

    def status(self, job_id: str) -> RemoteJobStatus:
        """Query one Kaggle kernel and normalize CLI output into `RemoteJobStatus`."""
        res = subprocess.run(
            [self.kaggle_cmd, "kernels", "status", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            logger.warning("Error fetching Kaggle status for %s: %s", job_id, res.stderr)
            return RemoteJobStatus.UNKNOWN
        return parse_kaggle_status_output(res.stdout)

    def wait(
        self,
        job_id: str,
        *,
        timeout_seconds: int = 3600,
        poll_interval_seconds: int = 15,
    ) -> RemoteJobStatus:
        """Poll a Kaggle kernel at the requested interval until terminal or timed out."""
        started = time.perf_counter()
        logger.info("Waiting for Kaggle job: %s (timeout: %ds)", job_id, timeout_seconds)

        while True:
            current_status = self.status(job_id)
            elapsed = time.perf_counter() - started
            logger.info(
                "Kaggle job %s status: %s (%.1fs elapsed)",
                job_id,
                current_status.value,
                elapsed,
            )

            if current_status.is_terminal:
                return current_status

            if elapsed >= timeout_seconds:
                logger.error("Timed out waiting for Kaggle job %s after %.1fs", job_id, elapsed)
                return RemoteJobStatus.UNKNOWN

            time.sleep(poll_interval_seconds)

    def download(self, job_id: str, destination: Path) -> Path:
        """Download files saved by the kernel from its `/kaggle/working` directory."""
        destination.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Kaggle kernel output for %s to %s", job_id, destination)

        res = subprocess.run(
            [self.kaggle_cmd, "kernels", "output", job_id, "-p", str(destination)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Kaggle kernels output download failed (code {res.returncode}):\n"
                f"stdout: {res.stdout}\nstderr: {res.stderr}"
            )
        return destination

    def run(
        self,
        job: RemoteJobConfig,
        *,
        wait: bool = True,
        output_dir: Path | None = None,
    ) -> RemoteRunResult:
        """Run a Kaggle job end to end and return status, logs, and local artifacts.

        With `wait=False`, submission returns immediately without downloading output.
        Otherwise the configured timeout controls polling before result retrieval.
        """
        started = time.perf_counter()
        job_id = self.submit(job)

        if not wait:
            return RemoteRunResult(
                job_id=job_id,
                status=RemoteJobStatus.QUEUED,
                duration_seconds=round(time.perf_counter() - started, 2),
            )

        final_status = self.wait(
            job_id,
            timeout_seconds=job.timeout_seconds,
            poll_interval_seconds=job.poll_interval_seconds,
        )
        duration = round(time.perf_counter() - started, 2)

        resolved_output_dir: Path | None = None
        log_file: Path | None = None
        error_msg: str | None = None

        if final_status == RemoteJobStatus.COMPLETE:
            slug = slugify(job.name)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            dest = output_dir or Path(f"runs/remote/{timestamp}-{slug}")
            self.download(job_id, dest)
            resolved_output_dir = dest
            candidate_log = dest / "execution.log"
            if candidate_log.is_file():
                log_file = candidate_log
        elif final_status == RemoteJobStatus.ERROR:
            error_msg = f"Remote Kaggle job {job_id} terminated with status ERROR."
            try:
                dest = output_dir or Path(f"runs/remote/failed-{slugify(job.name)}")
                self.download(job_id, dest)
                resolved_output_dir = dest
                candidate_log = dest / "execution.log"
                if candidate_log.is_file():
                    log_file = candidate_log
            except Exception:
                pass

        return RemoteRunResult(
            job_id=job_id,
            status=final_status,
            duration_seconds=duration,
            output_dir=resolved_output_dir,
            log_file=log_file,
            error_message=error_msg,
        )
