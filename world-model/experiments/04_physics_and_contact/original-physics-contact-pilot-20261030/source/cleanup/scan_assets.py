"""Read-only asset inventory. Writes only its own inventory JSON; never deletes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

out = Path(__file__).resolve().parent
repo = out.parents[2]
workspace = repo.parent
started = time.perf_counter()
reparse = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
skipped = []
errors = []

def walk(base):
    stack = [base]
    while stack:
        current = stack.pop()
        try:
            for entry in os.scandir(current):
                p = Path(entry.path)
                st = entry.stat(follow_symlinks=False)
                if getattr(st, 'st_file_attributes', 0) & reparse or entry.is_symlink():
                    skipped.append(str(p))
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(p)
                elif entry.is_file(follow_symlinks=False):
                    yield p, st
        except OSError as exc:
            errors.append({'path': str(current), 'error': str(exc)})

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

binary = {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.zip', '.hdf5', '.h5', '.pkl', '.pickle', '.bin', '.zarr', '.tar', '.gz', '.7z'}
files = []
by_top = {}
by_stage = {}
for path, info in walk(repo / 'artifacts'):
    rel = path.relative_to(repo / 'artifacts')
    row = {'path': str(path), 'relative_path': str(rel), 'bytes': info.st_size,
           'suffix': path.suffix.lower(), 'file_attributes': getattr(info, 'st_file_attributes', 0),
           'sha256': sha(path) if path.suffix.lower() in binary or info.st_size >= 1024*1024 else None}
    files.append(row)
    by_top[rel.parts[0]] = by_top.get(rel.parts[0], 0) + info.st_size
    key = '/'.join(rel.parts[:2])
    by_stage[key] = by_stage.get(key, 0) + info.st_size

external = []
for sibling in workspace.iterdir():
    if sibling.is_dir() and sibling != repo and (sibling / 'artifacts').is_dir():
        entries = list(walk(sibling / 'artifacts'))
        external.append({'root': str(sibling / 'artifacts'), 'bytes': sum(s.st_size for _, s in entries),
                         'files': len(entries), 'read_only_outside_main_cleanup_scope': True,
                         'large_files': [{'path': str(p), 'bytes': s.st_size} for p, s in entries if s.st_size >= 32*1024*1024]})
baseline = json.loads((repo / 'artifacts/r3b-playground-diagnostics-20261030/baseline-storage.json').read_text(encoding='utf-8'))
roots = [Path(p) for p in baseline['roots']]
roots.append(repo / 'artifacts/original-physics-contact-pilot-20261030')
counted = {}
for root in roots:
    if root.is_file():
        counted[str(root.resolve())] = root.stat().st_size
    elif root.is_dir():
        for p, s in walk(root):
            counted[str(p.resolve())] = s.st_size
result = {'kind': 'original-physics-pilot-cleanup-readonly-scan.v1',
          'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'main_experiment_root': str(repo / 'artifacts'), 'main_experiment_bytes': sum(x['bytes'] for x in files),
          'file_count': len(files), 'files': files, 'by_top': by_top, 'by_stage': by_stage,
          'outside_main_experiment_roots': external, 'reparse_points_skipped': sorted(set(skipped)), 'errors': errors,
          'shared_roots': [str(x) for x in roots], 'shared_deduplicated_bytes': sum(counted.values()),
          'shared_cap_bytes': 3*1024**3, 'atomic_reserve_bytes': 256*1024**2,
          'disk': dict(zip(('total','used','free'), shutil.disk_usage(repo))),
          'scan_seconds': time.perf_counter()-started, 'model_calls': 0, 'physics_controls': 0, 'deleted_files': 0}
(out / 'inventory-base.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'files','shared_roots'}}, ensure_ascii=False, indent=2))
print('LARGEST_MAIN_FILES')
print(json.dumps(sorted(files, key=lambda r:r['bytes'], reverse=True)[:55], indent=2))
