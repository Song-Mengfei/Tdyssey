"""Independent read-only stdlib audit of the completed evaluation, no model/native calls."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/gpc-checkpoint500-evaluation-20261026'
DELIVERY = ROOT / 'artifacts/gpc-checkpoint500-evaluation-review-20261026'
start = time.monotonic()
def read(p): return json.loads(p.read_bytes())
def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
def verify(p, declaration):
    b = p.read_bytes()
    assert len(b) == declaration['bytes'] and sha(b) == declaration['sha256'], str(p)
    return b

delivery = read(DELIVERY / 'delivery.json')
bound = sum((delivery[k] for k in ('result_tracked', 'formal_key_evidence', 'review_evidence')), [])
for d in bound: verify(ROOT / d['path'], d)
inventory = read(DELIVERY / 'output-inventory.json')
actual = {p.relative_to(OUT).as_posix() for p in OUT.rglob('*') if p.is_file()}
assert actual == set(inventory)
for name, d in inventory.items(): verify(OUT / name, d)
assert len(inventory) == 8006
total_bytes = sum(d['bytes'] for d in inventory.values())
assert total_bytes == 472896760
cfg = read(ROOT / 'configs/gpc_checkpoint500_evaluation.json')
assert sha((ROOT / 'configs/gpc_checkpoint500_evaluation.json').read_bytes()) == delivery['config_sha256']
bm = read(ROOT / cfg['benchmark_manifest'])
reg = read(ROOT / bm['regression']['path'] / 'report.json')
reg_keys = {(r['anchor'], r['candidate']) for r in reg['branches'] if r['status'] == 'complete'}
ind_keys = {(r['anchor'], r['candidate']) for r in bm['cases']}
assert len(reg_keys) == 63 and len(ind_keys) == 48 and not (reg_keys & ind_keys)
case_keys = reg_keys | ind_keys
grid = {(a,c,s,m,h) for a,c in case_keys for s in (5478,5479,5480)
        for m, hs in [('AR',(1,2,3,4)),('TF',(2,4))] for h in hs}
rows = {}
totals = defaultdict(Counter)
reasons = defaultdict(Counter)
for p in (OUT / 'measurements').glob('*.json'):
    r = read(p)
    key = (r['anchor'],r['candidate'],r['seed'],r['mode'],r['horizon'])
    role = r['role']
    assert role in ('frozen','adapted') and key in grid and (role,key) not in rows
    assert r['status'] == 'completed'
    assert r['stratum'] == ('independent' if key[:2] in ind_keys else 'regression')
    raw = (OUT / r['raw_path']).read_bytes()
    assert raw[:8] == b'\x93NUMPY\x01\x00'
    length = int.from_bytes(raw[8:10],'little')
    header = ast.literal_eval(raw[10:10+length].decode('latin1').strip())
    assert header == {'descr':'<f4','fortran_order':False,'shape':(3,96,96)}
    payload = raw[10+length:]
    assert len(payload) == 110592 and sha(payload) == r['raw_sha256']
    rows[role,key] = r
    k = '/'.join((role,r['shape'],r['stratum'],r['mode'],str(r['horizon'])))
    for group in (k, role):
        totals[group]['observed'] += 1
        for field in ('shape','state','coverage','pusher'):
            totals[group][field+'_valid'] += bool(r['measurement'][field]['valid'])
    reasons[role][r['measurement']['shape']['reason']] += 1
assert len(rows) == 3996
pairs = Counter()
for key in grid:
    f,a = rows['frozen',key],rows['adapted',key]
    for field in ('rng_entry','rng_exit','action_sha256','truth','latest','contact_by_h4','shape','stratum'):
        assert f[field] == a[field], (key,field)
    pairs['cross_weight_pairs'] += 1
    if key[3] == 'TF' or key[4] == 1:
        assert f['input_sha256'] == a['input_sha256']
        pairs['same_real_input'] += 1
    else:
        pairs['AR_own_input_different'] += f['input_sha256'] != a['input_sha256']
for role in ('frozen','adapted'):
    for a,c in case_keys:
        for s in (5478,5479,5480):
            for h in (2,4):
                ar,tf = rows[role,(a,c,s,'AR',h)],rows[role,(a,c,s,'TF',h)]
                assert all(ar[k] == tf[k] for k in ('rng_entry','rng_exit','action_sha256'))
                pairs['within_role_AR_TF_pairs'] += 1

completion = read(OUT / 'completion.json')
previous = '0'*64
pending = active = None
loads, released = [], []
events = Counter()
forecast_keys = defaultdict(list)
last_mono = 0
for n,line in enumerate((OUT/'ledger/highwater.jsonl').read_bytes().splitlines(),1):
    e = json.loads(line)
    assert e['sequence'] == n and e['previous_sha256'] == previous
    assert e['at_monotonic'] >= last_mono
    last_mono = e['at_monotonic']
    previous = sha(line+b'\n')
    kind, data = e['kind'],e['data']; events[kind] += 1
    if kind.startswith('reserve_'):
        assert pending is None
        pending = (n,kind)
        if kind == 'reserve_model_load':
            assert active is None and data['role'] == ('frozen','adapted')[len(loads)]
            active = data['role']; loads.append(active)
        else:
            assert active == data['metadata']['role']
            if kind == 'reserve_forecast':
                assert data['actual_count'] == 1 and data['euler_per_image'] == 3
                md = data['metadata']; key = (*md['case'],md['mode'],md['horizon'])
                assert key in grid
                forecast_keys[active].append(key)
            else: assert kind == 'reserve_eval' and data['batch_size'] == 1
    elif kind.startswith('complete_'):
        assert pending == (data['operation_id'],kind.replace('complete_','reserve_'))
        pending = None
    elif kind == 'model_released':
        assert pending is None and active == data['role'] and not data['cleanup']
        released.append(active); active = None
    else: assert kind in ('created','evaluation_finished')
assert n == 8064 and previous == completion['budget']['journal_tip_sha256']
assert pending is None and active is None and loads == released == ['frozen','adapted']
assert forecast_keys['frozen'] == forecast_keys['adapted']
assert len(forecast_keys['frozen']) == len(set(forecast_keys['frozen'])) == 1998
assert sha(canonical(forecast_keys['frozen'])) == completion['budget']['attempt']['forecast_order_sha256']
assert events == Counter(created=1, reserve_model_load=2,complete_model_load=2,
    reserve_eval=32,complete_eval=32,reserve_forecast=3996,complete_forecast=3996,
    model_released=2,evaluation_finished=1)
assert completion['new_optimizer_updates'] == completion['new_training_examples'] == 0
assert not completion['optimizer_state_restored'] and not completion['training_rng_restored']
loss = completion['loss_rows']
assert [(r['index'],r['noise']) for r in loss['frozen']] == [(r['index'],r['noise']) for r in loss['adapted']]
means = {k:sum(r['loss'] for r in v)/len(v) for k,v in loss.items()}

blob = (OUT/'guardian/evaluation-status.journal').read_bytes()
offset = sequence = 0
max_gap = 0
prior = None
max_payload = 0
while offset < len(blob):
    magic,length,seq = struct.unpack_from('>8sIQ',blob,offset)
    assert magic == b'GPCEVS01' and seq == sequence+1 and length <= 1536
    end = offset+20+length
    assert blob[end:end+32] == hashlib.sha256(blob[offset:end]).digest() and blob[end+32:end+36] == b'DONE'
    status = json.loads(blob[offset+20:end]); sequence = seq
    assert status['nonce'] == 'edbfc1c0d6ec40bead504443d1ea4e87' and status['actual_worker_pid'] == 21872
    if prior is not None: max_gap = max(max_gap,status['at_monotonic']-prior)
    prior = status['at_monotonic']; max_payload = max(max_payload,length); offset = end+36
assert sequence == 5800 and offset == len(blob) and max_gap < 5
assert status['phase'] == 'complete' and status['model_instances'] == 0
assert status['forecasts'] == 3996 and status['euler_steps'] == 11988 and status['optimizer_updates'] == 0
assert read(OUT/'guardian/heartbeat-error.json') == {}
stop = read(OUT/'guardian/stop-record.json')
assert stop['status'] == 'complete' and stop['worker_exit_code'] == stop['launcher_exit_code'] == 0
assert stop['job_active_processes_after_cleanup'] == 0 and not stop['single_model_lock_retained'] and not stop['cleanup_errors']
assert not stop['error'] and not stop['guard_reason'] and not any(stop['injected'].values())
assert stop['wall_seconds'] < 3600
assert status['cuda_peak_allocated_bytes'] < 512*1024**2 and status['cuda_peak_reserved_bytes'] < 1024**3
assert stop['observed_peaks']['private_bytes'] < 3*1024**3 and stop['observed_peaks']['working_set_bytes'] < 2*1024**3
after = read(DELIVERY/'after.json')
assert after['passed'] and not any(after[k] for k in ('changed_content','changed_held_stat','errors','missing','added_members','held_content_attempts'))
for d in after['old_claims']+after['base_references']+list(after['identities'].values())+[after['execution_config']]:
    verify(Path(d['path']),d)
for d in completion['budget']['attempt']['source_weights'].values(): verify(ROOT / d['path'],d)

result = dict(kind='root_independent_eval_review.v1', source_commit=delivery['result_commit'],
    delivery_sha256=sha((DELIVERY/'delivery.json').read_bytes()), bound_files=len(bound),
    full_output_inventory=dict(files=len(inventory),bytes=total_bytes,all_hashes_match=True),
    independent_grid=dict(regression=63,independent=48,forecasts=3996,paired_checks=dict(pairs)),
    ledger=dict(events=n,counts=dict(events),tip=previous,pending=None,serial_loads=loads,releases=released),
    status=dict(frames=sequence,bytes=len(blob),partial_bytes=0,max_payload=max_payload,max_gap=max_gap),
    loss_means=means,loss_ratio=means['adapted']/means['frozen'],measurement_counts=dict(totals),reasons=dict(reasons),
    guard=dict(wall_seconds=stop['wall_seconds'],cleanup_errors=[],job_active=0,lock_released=True,
        fresh_cuda_peak=status['cuda_peak_allocated_bytes'],fresh_cuda_reserved_peak=status['cuda_peak_reserved_bytes'],
        observed_host_peaks=stop['observed_peaks']),
    old_preservation=after['summary'],old_preservation_qualification='Bound executor before/after and references reviewed; main freshly hashed old claims/identities/references, not all 9245 old files. Held contents unopened.',
    execution_accepted=True,scientific_candidate_accepted=False,DPPO_allowed=False,
    conclusion='Full matched evaluation valid; candidate fails T retention, no measurable accepted L pose/coverage. H1 and TF failure exclude an AR-only explanation. Root cause not established.',
    limits='No project imports, Torch, NumPy, native calls, model loads, new measurements, predictions or physics. Raw headers/payload hashes verified, no new pixel-level diagnosis.',
    audit_development_corrections=['Initial reviewer script omitted the canonical trailing LF in ledger hash and forecast-plan digest; corrected to the audited serialization.', 'Reviewer initially treated identities mapping as list; corrected to values(). These were reviewer-script errors, no production or evidence mutations.'],
    seconds=time.monotonic()-start)
with (HERE/'review.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ('bound_files','full_output_inventory','ledger','status','loss_means','loss_ratio','execution_accepted','scientific_candidate_accepted','seconds')}))
