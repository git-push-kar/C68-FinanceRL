"""Restore deduplicated historical weights; Python 3.11+, standard library only.

Download asset-manifest.json from the release first. The manifest is the trust
root: compare its SHA256 with the copy committed on main before using it.
"""
import argparse
import fnmatch
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE = 'https://github.com/git-push-kar/C68-FinanceRL/releases/download/finance-checkpoint-archive-20261116/'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Download/cache directory containing asset-manifest.json')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--only', default='*', help='Path glob, e.g. research/rl_candidate_static_native/output_seed42/checkpoint_44/*')
    parser.add_argument('--offline', action='store_true', help='Never download missing assets')
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.assets / 'asset-manifest.json').read_text())

    def asset(name):
        if Path(name).name != name or name not in manifest:
            raise ValueError('Invalid asset name')
        path = args.assets / name
        expected = manifest[name]
        if not path.exists():
            if args.offline:
                raise FileNotFoundError(path)
            temporary = path.with_suffix(path.suffix + '.partial')
            urllib.request.urlretrieve(RELEASE + name, temporary)
            if temporary.stat().st_size != expected['bytes'] or digest(temporary) != expected['sha256']:
                raise ValueError('Downloaded asset failed verification: ' + name)
            temporary.replace(path)
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Asset failed verification: ' + name)
        return path

    entries = json.loads(asset('weights-index.json').read_text())['files']
    selected = [entry for entry in entries if fnmatch.fnmatchcase(entry['path'], args.only)]
    if not selected:
        raise ValueError('No paths match --only')
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for entry in selected:
        target = (root / entry['path']).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError('Path escapes destination: ' + entry['path'])
        grouped.setdefault(entry['asset'], []).append((entry, target))
    restored = 0
    for name, group in grouped.items():
        with zipfile.ZipFile(asset(name)) as archive:
            for entry, target in group:
                if target.exists():
                    if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
                        raise FileExistsError('Refusing to overwrite differing file: ' + str(target))
                    restored += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
                    temporary = Path(output.name)
                    try:
                        with archive.open('objects/' + entry['sha256']) as source:
                            shutil.copyfileobj(source, output)
                    except BaseException:
                        output.close()
                        temporary.unlink(missing_ok=True)
                        raise
                if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
                    temporary.unlink(missing_ok=True)
                    raise ValueError('Restored object failed verification: ' + entry['path'])
                temporary.replace(target)
                restored += 1
        print(name, ': verified', len(group), 'paths', flush=True)
    print('Restored/verified', restored, 'paths in', root)


if __name__ == '__main__':
    main()
