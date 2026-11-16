"""Run a campaign command with exact source snapshot, log and exit evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import ctypes
from execution_policy import wait_for_compute_window

ROOT=Path(__file__).parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--source',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:raise ValueError('Missing Python script/arguments')
    source=Path(args.source).resolve()
    record=ROOT/'launch_records'/args.name;record.mkdir(parents=True,exist_ok=False)
    snapshot=record/'source';snapshot.mkdir()
    shutil.copy2(ROOT/'continuous_compute_policy.py',record/'continuous_compute_policy.py')
    hashes={}
    for path in source.glob('*.py'):
        data=path.read_bytes();(snapshot/path.name).write_bytes(data)
        hashes[path.name]=hashlib.sha256(data).hexdigest()
    if (source/'contracts').exists():shutil.copytree(source/'contracts',snapshot/'contracts')
    metadata={'command':[sys.executable,'-u',*command],'cwd':str(source),
        'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,
        'python':sys.version,'environment':{k:os.environ.get(k) for k in
            ['FINANCERL_EXECUTION','FINANCERL_BASE_MODEL','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        'status':'starting','note':'User lifted all clock/idle compute restrictions on 2026-11-15. Dependency and correctness guards remain.',
        'scheduling_override_sha256':hashlib.sha256((ROOT/'continuous_compute_policy.py').read_bytes()).hexdigest()}
    status=record/'status.json'
    def write():status.write_text(json.dumps(metadata,indent=2))
    write()
    wait_for_compute_window()
    with (record/'output.log').open('w',encoding='utf-8') as stream:
        if os.name=='nt':
            # Per-process keep-awake, not a persistent power-plan change.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        child=subprocess.Popen(metadata['command'],cwd=source,stdout=stream,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        metadata.update(status='running',pid=child.pid);write()
        print('RUN RECORD:',str(record),'PID:',child.pid,flush=True)
        while child.poll() is None:
            metadata['heartbeat_utc']=datetime.now(timezone.utc).isoformat();write();time.sleep(10)
        metadata.update(status='completed' if child.returncode==0 else 'failed',
            exit_code=child.returncode,finished_utc=datetime.now(timezone.utc).isoformat());write()
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print(json.dumps({'record':str(record),'status':metadata['status'],
        'exit_code':metadata['exit_code'],'finished_utc':metadata['finished_utc']},indent=2),flush=True)
    return child.returncode


if __name__=='__main__':raise SystemExit(main())
