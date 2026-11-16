"""Export a complete trained finance specialist, without the shared base weights."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from model import validate_package


def export_package(package, output):
    package, output = Path(package), Path(output)
    manifest = validate_package(package)
    if manifest.get("run", {}).get("steps", 0) < 1:
        raise ValueError("This package has no recorded training steps")
    if not manifest["run"].get("learning_verified") or manifest["run"].get("optimizer_updates", 0) < 1:
        raise ValueError("No verified optimizer/LoRA updates. This is not an exportable trained finance adapter.")
    if output.exists():
        raise FileExistsError("Choose a new archive path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for relative in ["manifest.json", *sorted(manifest["sha256"])]:
            archive.write(package / relative, arcname=relative)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(export_package(args.package, args.output))
