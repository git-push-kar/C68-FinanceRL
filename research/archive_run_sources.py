"""Archive source bytes matching recorded run hashes, never guessed versions."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--source',required=True)
    args=p.parse_args();run=Path(args.run);source=Path(args.source)
    metadata=json.loads((run/'run.json').read_text())
    target=run/'source_snapshot';target.mkdir(exist_ok=True)
    report=[]
    for name,expected in metadata['source_sha256'].items():
        data=(source/name).read_bytes()
        candidates=[('current',data)]
        if name=='shared_backbone.py':
            text=data.decode('utf-8').replace('\r\n','\n')
            addition="        device = torch.device(device)\n        if device.type == 'cuda':\n            wait_for_compute_window()\n"
            previous=text.replace(addition,'        device = torch.device(device)\n',1)
            candidates.extend([('before_early_quiet_guard',previous.encode()),
                ('before_early_quiet_guard_CRLF',previous.replace('\n','\r\n').encode())])
        matching=next(((label,value) for label,value in candidates if hashlib.sha256(value).hexdigest()==expected),None)
        if matching is None:raise RuntimeError('Cannot reconstruct exact recorded source: '+name)
        label,value=matching
        destination=target/name
        if destination.exists() and destination.read_bytes()!=value:raise RuntimeError('Existing snapshot differs')
        destination.write_bytes(value)
        report.append({'name':name,'sha256':expected,'provenance':label})
    (target/'VERIFICATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'files':len(report),'snapshot':str(target)},indent=2))


if __name__=='__main__':main()
