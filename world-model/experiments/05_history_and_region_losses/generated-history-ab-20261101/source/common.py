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

def check_storage(root, config, reserve_bytes=0, full=False):
    root = Path(root).resolve()
    if int(config.get('shared_cap_bytes', CAP)) != CAP:
        raise ValueError('This run has one shared 10,000,000,000-byte cap')
    reserve = max(RESERVE, int(config.get('atomic_reserve_bytes', RESERVE)))
    prospective = max(0, int(reserve_bytes))
    baseline = root / 'storage-baseline.json'
    current = snapshot([root])
    if full or not baseline.exists():
        shared = snapshot([*config['shared_storage_roots'], root])
        external = shared['bytes'] - current['bytes']
        atomic_json(baseline, dict(external_bytes=external, at_utc=utc(), shared=shared))
    else:
        external = read(baseline)['external_bytes']
    # Other old evidence files may acquire small receipts; reserve this explicitly.
    outside_guard = 4 * 1024**2
    concurrent_guard = 64 * 1024**2  # At most two bounded collection writers.
    free = shutil.disk_usage(root).free
    total = external + current['bytes']
    result = dict(shared_bytes_estimate=total, new_root_bytes=current['bytes'],
                  shared_cap_bytes=CAP, atomic_reserve_bytes=reserve, prospective_write_bytes=prospective,
                  concurrent_write_guard_bytes=concurrent_guard,
                  external_growth_guard_bytes=outside_guard, filesystem_free_bytes=free,
                  passed=total + reserve + prospective + outside_guard + concurrent_guard <= CAP and free >= reserve + prospective + concurrent_guard,
                  at_utc=utc(), full_scan=full)
    if not result['passed']:
        raise RuntimeError('Shared storage reserve exhausted: ' + json.dumps(result))
    return result
