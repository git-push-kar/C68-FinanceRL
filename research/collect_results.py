"""Collect small logs/metrics for debugging; never include model weights or venvs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def collect(root, output=None):
    root = Path(root).resolve()
    output = Path(output) if output else root / f"RETURN_TO_CHAT_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.zip"
    if output.exists():
        raise FileExistsError(output)
    sources = [root / "college_runs", root / "college_logs"]
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for source in sources:
            if not source.is_dir():
                continue
            for file in source.rglob("*"):
                if file.is_file() and file.suffix.lower() in {".json", ".csv", ".log", ".txt", ".md"} and file.stat().st_size <= 20_000_000:
                    archive.write(file, file.relative_to(root))
    return output.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(collect(Path(__file__).parent, args.output))
