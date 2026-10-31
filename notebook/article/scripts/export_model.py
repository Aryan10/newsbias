"""Extract only the selected model/tokenizer for deployment, without optimizer state."""

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Allow exporting an authoritative interim checkpoint")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.resume) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["progress"]["phase"] != "complete" and not args.force:
            raise ValueError("Finish training before exporting a deployment checkpoint (or pass --force)")
        # In resume.zip, model files are located under 'model/'
        prefix = "model/"
        for member in archive.infolist():
            if member.filename.startswith(prefix) and not member.is_dir():
                destination = args.output / member.filename.removeprefix(prefix)
                if not destination.resolve().is_relative_to(args.output.resolve()):
                    raise ValueError("Unsafe archive path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    import shutil

                    shutil.copyfileobj(source, target)
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Exported {manifest['config']['task']} model to {args.output}")


if __name__ == "__main__":
    main()
