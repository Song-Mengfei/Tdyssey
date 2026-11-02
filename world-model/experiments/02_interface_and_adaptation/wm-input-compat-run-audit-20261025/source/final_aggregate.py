"""Count stopped-attempt files and postrun metadata; no runtime operations."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/wm-input-compat-eval-v1-20261025'
LEDGER = ROOT/'artifacts/wm-input-compat-eval-v1-ledger-20261025'
REGISTRY = ROOT/'artifacts/wm-input-compat-eval-v1-registry'
PARTIAL = ROOT/'artifacts/wm-input-compat-partial-stats-20261025'
TARGET = AUDIT/'final-aggregate.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def count(folder):
    files = [p for p in folder.rglob('*') if p.is_file() and p != TARGET]
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files))

inventory = json.loads((AUDIT/'stopped-run-inventory.json').read_bytes())['files']
actual = {str(p.relative_to(ROOT)) for folder in (OUT, LEDGER, REGISTRY)
          for p in folder.rglob('*') if p.is_file()}
assert actual == set(inventory)
for name, row in inventory.items():
    path = ROOT/name
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']

groups = dict(results=count(OUT/'results'), guardian=count(OUT/'guardian'),
              ledger=count(LEDGER), registry=count(REGISTRY), audit=count(AUDIT),
              partial_statistics=count(PARTIAL))
runtime = sum(groups[name]['bytes'] for name in ('guardian', 'ledger', 'registry'))
base_audit = groups['audit']['bytes']
record = dict(kind='final_stopped_run_and_postrun_metadata_aggregate',
              created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              stopped_runtime_files_unchanged=True, groups=groups,
              metadata_includes_final_aggregate_itself=True,
              runtime_telemetry_bytes=runtime,
              result_limit_bytes=256*1024**2, telemetry_limit_bytes=16*1024**2,
              proof_hashes={name:sha(path) for name,path in (
                  ('before.json',AUDIT/'before.json'),('after.json',AUDIT/'after.json'),
                  ('partial-stats.json',PARTIAL/'partial-stats.json'),
                  ('stopped-run-inventory.json',AUDIT/'stopped-run-inventory.json'))},
              resume_allowed=False)
size = -1
for _ in range(10):
    groups['audit']['files'] = count(AUDIT)['files'] + 1
    groups['audit']['bytes'] = base_audit + max(size, 0)
    record['final_aggregate_bytes'] = max(size, 0)
    record['telemetry_including_postrun_bytes'] = runtime+groups['audit']['bytes']+groups['partial_statistics']['bytes']
    record['all_new_artifact_files'] = sum(row['files'] for row in groups.values())
    record['all_new_artifact_bytes'] = groups['results']['bytes']+record['telemetry_including_postrun_bytes']
    payload = (json.dumps(record, indent=2)+'\n').encode()
    if len(payload) == size:
        break
    size = len(payload)
else:
    raise AssertionError('Aggregate self-size did not converge')
assert groups['results']['bytes'] <= record['result_limit_bytes']
assert record['telemetry_including_postrun_bytes'] <= record['telemetry_limit_bytes']
assert (REGISTRY/'launcher-attempt.json').is_file() and (REGISTRY/'attempt.json').is_file()
with TARGET.open('xb') as stream:
    stream.write(payload)
print(json.dumps(record))
