"""Small shared I/O and one shared 10GB budget. No simulator/model imports."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

CAP = 10_000_000_000
RESERVE = 256 * 1024**2

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def replace_atomic(temporary, path):
    """Reuse the existing native trainer's bounded Windows-sharing retry."""
    deadline=time.monotonic()+2.
    while True:
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise
            time.sleep(min(.025,remaining))

def atomic_json(path, value, exclusive=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p if exclusive else p.with_name(p.name + f'.{os.getpid()}.tmp')
    with temp.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if not exclusive:
        replace_atomic(temp,p)

def snapshot(roots):
    candidates = sorted({Path(p).resolve() for p in roots}, key=lambda p: len(p.parts))
    top = []
    for p in candidates:
        if not any(p == q or p.is_relative_to(q) for q in top):
            top.append(p)
    files = {}; missing = []
    for p in top:
        if p.is_file():
            files[str(p).lower()] = p.stat().st_size
        elif p.is_dir():
            for folder, dirs, names in os.walk(p, followlinks=False):
                for name in names:
                    fp = Path(folder) / name
                    if fp.is_file():
                        files[str(fp.resolve()).lower()] = fp.stat().st_size
        else:
            missing.append(str(p))
    return {'bytes': sum(files.values()), 'files': len(files), 'missing_roots': missing,
            'scanned_roots': [str(p) for p in top]}

# A finite diagnostics root:
# one shared-resource byte snapshot at preparation, then only new-root growth.
NEW_CAP=500_000_000

def initialize_storage(root,config):
    root=Path(root).resolve();baseline=root/'storage-baseline.json'
    if baseline.exists():return check_storage(root,config,full=False)
    current=snapshot([root]);shared=snapshot([*config['shared_storage_roots'],root])
    atomic_json(baseline,dict(external_bytes=shared['bytes']-current['bytes'],at_utc=utc(),shared=shared,
        scope='One preparation byte scan only; no whole-repository/data hashing. Runtime uses only newroot growth.'))
    return check_storage(root,config,full=False)

def check_storage(root,config,reserve_bytes=0,full=False):
    root=Path(root).resolve()
    if int(config.get('shared_cap_bytes',CAP))!=CAP:raise ValueError('Sharedcapchanged')
    cap=int(config.get('new_root_cap_bytes',NEW_CAP))
    if cap!=NEW_CAP:raise ValueError('Approved500000000 newpersistentcaprequired')
    b=root/'storage-baseline.json'
    if not b.exists():raise RuntimeError('Run one scoped resource initialization before model work')
    base=read(b);current=snapshot([root]);prospective=max(0,int(reserve_bytes));reserve=max(RESERVE,int(config.get('atomic_reserve_bytes',RESERVE)))
    external=int(base['external_bytes']);guard=64*1024**2;free=shutil.disk_usage(root).free
    result=dict(shared_bytes_estimate=external+current['bytes'],new_root_bytes=current['bytes'],
        shared_cap_bytes=CAP,new_root_cap_bytes=NEW_CAP,atomic_reserve_bytes=reserve,prospective_write_bytes=prospective,
        external_growth_guard_bytes=guard,filesystem_free_bytes=free,full_scan=False,
        shared_external_reference_at_utc=base['at_utc'],at_utc=utc(),
        passed=current['bytes']+prospective<=NEW_CAP and external+current['bytes']+prospective+reserve+guard<=CAP and free>=reserve+prospective)
    if not result['passed']:raise RuntimeError('Diagnostics resource limit: '+json.dumps(result))
    return result

def save_raw(root,config,path,**arrays):
    import numpy as np
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('New predictions stay inside diagnosticsroot')
    if path.exists():raise FileExistsError('Do not overwrite/retry an existing prediction')
    size=sum(np.asarray(a).nbytes for a in arrays.values())+65536
    check_storage(root,config,reserve_bytes=size)
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    try:
        with temp.open('xb')as f:
            np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        replace_atomic(temp,path)
    finally:
        temp.unlink(missing_ok=True)
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,
        raw_precision='Original stored array dtype unchanged; no compression/quantization',compression=False)

