"""CLI utility for submitting and monitoring remote cloud GPU execution."""

import argparse
import logging
import shlex
import sys
from pathlib import Path

from newsbias.remote.base import RemoteJobConfig
from newsbias.remote.kaggle import KaggleRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("newsbias.remote")


def build_parser() -> argparse.ArgumentParser:
    """Define commands for submitting, checking, and downloading remote jobs.

    Platform-specific options are nested beneath the platform command so new remote
    backends can be added without changing the job configuration contract.
    """
    parser = argparse.ArgumentParser(
        description="Dispatch and manage arbitrary Python jobs on remote cloud GPU platforms",
    )
    subparsers = parser.add_subparsers(dest="platform", required=True)

    # Kaggle platform subcommands
    kaggle_parser = subparsers.add_parser("kaggle", help="Kaggle remote runner")
    kaggle_subs = kaggle_parser.add_subparsers(dest="action", required=True)

    # kaggle run
    run_parser = kaggle_subs.add_parser(
        "run", help="Stage, upload, and run a script on Kaggle GPU"
    )
    run_parser.add_argument(
        "--script", type=Path, required=True, help="Python entrypoint script to execute"
    )
    run_parser.add_argument(
        "--name", type=str, help="Human-readable job name (default: script stem)"
    )
    run_parser.add_argument(
        "--extra-files",
        type=Path,
        nargs="*",
        default=[],
        help="Additional scripts, modules, or config files to bundle into the kernel",
    )
    run_parser.add_argument(
        "--args",
        type=str,
        default="",
        help="Arguments passed to script, e.g. --args '--task eval --epochs 3'",
    )
    run_parser.add_argument(
        "--pip",
        nargs="*",
        default=[],
        help="Pip packages to install before running (e.g. transformers datasets accelerate)",
    )
    run_parser.add_argument("--no-gpu", action="store_true", help="Disable GPU accelerator")
    run_parser.add_argument(
        "--no-wait", action="store_true", help="Do not wait for job completion"
    )
    run_parser.add_argument(
        "--output-dir", type=Path, help="Directory to save downloaded artifacts"
    )
    run_parser.add_argument(
        "--timeout", type=int, default=3600, help="Polling timeout in seconds"
    )
    run_parser.add_argument(
        "--poll-interval", type=int, default=15, help="Status polling interval in seconds"
    )

    # kaggle status
    status_parser = kaggle_subs.add_parser(
        "status", help="Check status of a running or completed kernel"
    )
    status_parser.add_argument(
        "--id", type=str, required=True, help="Kaggle kernel id (<username>/<slug>)"
    )

    # kaggle download
    dl_parser = kaggle_subs.add_parser(
        "download", help="Download output files of a completed kernel"
    )
    dl_parser.add_argument(
        "--id", type=str, required=True, help="Kaggle kernel id (<username>/<slug>)"
    )
    dl_parser.add_argument(
        "--output-dir", type=Path, required=True, help="Local directory to write output"
    )

    return parser


def main() -> None:
    """Dispatch parsed remote commands to the selected platform runner."""
    parser = build_parser()
    args = parser.parse_args()

    if args.platform == "kaggle":
        runner = KaggleRunner()

        if args.action == "run":
            job_name = args.name or args.script.stem
            job_args = shlex.split(args.args) if args.args else []
            job = RemoteJobConfig(
                name=job_name,
                entrypoint=args.script,
                extra_files=args.extra_files,
                args=job_args,
                pip_packages=args.pip,
                gpu=not args.no_gpu,
                timeout_seconds=args.timeout,
                poll_interval_seconds=args.poll_interval,
            )
            logger.info("Submitting Kaggle job: %s (entrypoint: %s)", job.name, job.entrypoint)
            result = runner.run(job, wait=not args.no_wait, output_dir=args.output_dir)

            print("\n=== Remote Job Execution Result ===")
            print(f"Job ID:   {result.job_id}")
            print(f"Status:   {result.status.value.upper()}")
            print(f"Duration: {result.duration_seconds}s")
            if result.output_dir:
                print(f"Output:   {result.output_dir}")
            if result.log_file:
                print(f"Log:      {result.log_file}")
            if result.error_message:
                print(f"Error:    {result.error_message}")
                sys.exit(1)

        elif args.action == "status":
            st = runner.status(args.id)
            print(f"Job {args.id} status: {st.value.upper()}")

        elif args.action == "download":
            dest = runner.download(args.id, args.output_dir)
            print(f"Downloaded output for {args.id} to: {dest}")


if __name__ == "__main__":
    main()
