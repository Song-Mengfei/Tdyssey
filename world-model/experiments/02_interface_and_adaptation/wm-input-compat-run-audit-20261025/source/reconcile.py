"""Read-only postrun reconciliation; never imports a model or image decoder."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/wm-input-compat-eval-v1-20261025'
LEDGER = ROOT / 'artifacts/wm-input-compat-eval-v1-ledger-20261025'
REGISTRY = ROOT / 'artifacts/wm-input-compat-eval-v1-registry'
GUARDIAN = OUT / 'guardian/6e9ac8b2fe0e46169c082d75e6084d94'

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

decl = read(LEDGER / 'declaration.json')
emergency = read(LEDGER / 'emergency-stop.json')
guard = read(GUARDIAN / 'stop-record.json')
seal = read(REGISTRY / 'launcher-attempt.json')
claim = read(REGISTRY / 'attempt.json')
events = []
previous = '0' * 64
for number, line in enumerate((LEDGER / 'journal.jsonl').read_bytes().splitlines(keepends=True), 1):
    row = json.loads(line)
    assert row['sequence'] == number and row['previous_sha256'] == previous
    previous = hashlib.sha256(line).hexdigest()
    events.append(row)

complete = [e['data'] for e in events if e['kind'] == 'output_complete']
reserved = [e['data'] for e in events if e['kind'] == 'output_reserved']
for row in complete:
    path = Path(row['path'])
    assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']

code = decl['binding']['code_sha256']
assets = decl['binding']['asset_sha256']
changed = [name for name, expected in {**code, **assets}.items() if sha(Path(name)) != expected]
assert not changed
assert guard['job_empty_verified_before_close']
assert emergency['state']['counts']['images'] == emergency['state']['counts']['model_loads'] == 0
assert claim['pid'] == guard['actual_worker_pid']
assert seal['nonce'] == guard['launch_seal']['nonce'] == decl['binding']['launch_seal']['nonce']

real = [read(p) for p in sorted((OUT / 'results/real').glob('*.json'))]
statuses = [[row['key'], 'missing' if row['status'] == 'missing' else 'not_attempted']
            for row in decl['plan']['calls']]
assert len(statuses) == 2304 and sum(status == 'missing' for key, status in statuses) == 36

result = dict(
    kind='postrun_read_only_reconciliation_not_a_resume',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_commit=decl['binding']['reviewed_commit'],
    authorization_id=decl['binding']['authorization_id'],
    protocol_sha256=decl['binding']['protocol_sha256'],
    t1_spec_sha256=decl['binding']['t1_spec_sha256'],
    status='stopped', first_stop_reason=emergency['reason'],
    M=dict(status='incomplete_operational_stop', passed=None, planned_rows=1072,
           available_rows=1056, original_missing_rows=16, durable_completed_rows=len(real),
           available_rows_without_durable_measurement=1056-len(real),
           completed_by_arm_shape=dict(collections.Counter(row['arm']+'-'+row['shape'] for row in real)),
           persisted_sources=sorted(set(row['source'] for row in real))),
    T=dict(status='not_attempted', passed=None, available_forecasts=576),
    L=dict(status='not_attempted', passed=None, available_forecasts=1692, original_missing_forecasts=36),
    counts=emergency['state']['counts'], native_guardian=guard,
    ledger=dict(journal_events=len(events), last_event_sha256=previous, hash_chain_valid=True,
                complete_outputs=len(complete), reserved_outputs=len(reserved),
                pending_sampler_calls=emergency['pending'], gates=emergency['state']['gates'],
                stages=emergency['state']['stage_status'],
                normal_final_exports_present={name: (OUT/'results'/name).exists() for name in
                    ('M.json', 'summary.json', 'completion.json', 'forecast-diagnostics.json',
                     'ranking.json', 'planned-status.json')}),
    bindings=dict(code_files=len(code), asset_files=len(assets), changed=changed, all_unchanged=True),
    interpretation=dict(actual_forecast_rng_pairs=0, AR_forecasts=0, TF_forecasts=0,
                        generated_shape_or_T1_effect='unknown: no forecasts',
                        full_measurement_gate='unknown: incomplete',
                        weight_adaptation_conclusion='not supported by this operational stop',
                        no_retry_or_resume=True))
(AUDIT/'run-reconciliation.json').write_text(json.dumps(result, indent=2)+'\n')
(AUDIT/'forecast-status-postrun.json').write_text(json.dumps(dict(
    kind='postrun_status_table_not_a_resume', first_stop_reason=emergency['reason'],
    rows=statuses), separators=(',', ':'))+'\n')
inventory = {}
for folder in (OUT, LEDGER, REGISTRY):
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            inventory[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=sha(path))
(AUDIT/'stopped-run-inventory.json').write_text(json.dumps(dict(
    kind='immutable_stopped_attempt_files', files=inventory), indent=2)+'\n')
print(json.dumps(dict(status=result['status'], M=result['M'], counts=result['counts'],
                      ledger=result['ledger'], bindings=result['bindings'],
                      inventory_files=len(inventory), elapsed_seconds=guard['elapsed_seconds'])))
