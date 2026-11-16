"""Prepare/verify the exact pristine base, using SSD files before the network.

This module uses only Python's standard library so setup can run before pip.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

from config import BACKBONE_ID, BACKBONE_REVISION

MANIFEST = Path(__file__).parent / "contracts" / "base_snapshot.json"


def snapshot():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["model_id"] != BACKBONE_ID or data["revision"] != BACKBONE_REVISION:
        raise ValueError("Bundled base snapshot contract has changed")
    return data


def file_matches(path, spec):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    if "sha256" in spec:
        checksum = hashlib.sha256()
        expected = spec["sha256"]
    else:
        checksum = hashlib.sha1()
        checksum.update(f"blob {spec['size']}\0".encode())
        expected = spec["git_blob_sha1"]
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest() == expected


def verify_base(path):
    path = Path(path).resolve()
    data = snapshot()
    # Tokenizers/preprocessor are also checked when present in the snapshot.
    bad = [name for name, spec in data["files"].items() if not file_matches(path / name, spec)]
    if bad:
        raise ValueError(f"Base is incomplete or differs from pristine {BACKBONE_REVISION}: {bad}. Run base_assets.py to stage a clean copy.")
    if (path / "adapter_config.json").exists():
        raise ValueError("An adapter directory is not a pristine base")
    return {"model_id": BACKBONE_ID, "revision": BACKBONE_REVISION, "path": str(path),
            "weight_sha256": data["files"]["model.safetensors"]["sha256"]}


def candidate_sources(ssd_root):
    root = Path(ssd_root)
    return [root / "pc/LY_C68/deepfake_project/models/InternVL3-2B",
            root / "pc/C68-LR/models_v3/adapter",
            root / "deepfake/LY_C68/deepfake_project/models/InternVL3-2B"]


def stage_base(destination, sources=(), offline=False):
    destination = Path(destination).resolve()
    sources = [Path(p).resolve() for p in sources if Path(p).is_dir()]
    if any(destination == p or destination.is_relative_to(p) or p.is_relative_to(destination) for p in sources):
        raise ValueError("Destination must be separate from SSD source directories")
    destination.mkdir(parents=True, exist_ok=True)
    for name, spec in snapshot()["files"].items():
        target = destination / name
        if file_matches(target, spec):
            print(f"Verified existing {name}", flush=True)
            continue
        source = next((p / name for p in sources if file_matches(p / name, spec)), None)
        partial = target.with_name(target.name + ".partial")
        if source:
            print(f"Copying verified {name} from {source}", flush=True)
            shutil.copyfile(source, partial)
        elif offline:
            raise FileNotFoundError(f"No verified SSD copy of {name}. Retry with internet enabled.")
        else:
            url = f"https://huggingface.co/{BACKBONE_ID}/resolve/{BACKBONE_REVISION}/{name}"
            print(f"Downloading pinned {name}", flush=True)
            with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=16 * 1024 * 1024)
        if not file_matches(partial, spec):
            raise ValueError(f"Checksum mismatch for {name}; original SSD source was not changed")
        os.replace(partial, target)
    proof = verify_base(destination)
    (destination / "finance_base_identity.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="local_models/InternVL3-2B")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--ssd-root")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    sources = args.source + ([str(p) for p in candidate_sources(args.ssd_root)] if args.ssd_root else [])
    print(json.dumps(stage_base(args.destination, sources, args.offline), indent=2))
