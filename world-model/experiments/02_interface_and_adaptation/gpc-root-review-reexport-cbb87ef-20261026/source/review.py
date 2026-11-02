"""Read-only stdlib review of sealed re-export; no raw array decoding."""
import hashlib, json, math, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/gpc-checkpoint500-offline-reexport-20261026'
OLD = ROOT / 'artifacts/gpc-checkpoint500-offline-diagnosis-20261026'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bound(base, row):
    p = base / row['path']
    assert p.stat().st_size == row['bytes'], str(p)
    assert digest(p) == row['sha256'], str(p)
def walk(x):
    if isinstance(x, float): assert math.isfinite(x)
    if isinstance(x, dict):
        for v in x.values(): walk(v)
    elif isinstance(x, list):
        for v in x: walk(v)

start = time.perf_counter()
d = read(OUT / 'delivery.json')
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip() == d['result_commit']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
for row in d['tracked_files']: bound(ROOT,row)
bound(OUT, d['inventory'])
inventory = read(OUT / d['inventory']['path'])['entries']
for row in inventory: bound(OUT,row)
assert len(inventory) == 53
for row in read(OUT/'preexecution-binding.json')['files']:
    if row['path'].startswith('docs/'):
        data=subprocess.check_output(['git','show',d['execution_commit']+':'+row['path']],cwd=ROOT)
        assert hashlib.sha256(data).hexdigest()==row['sha256']
    else: bound(ROOT,row)
for row in read(OUT/'postexecution-identities.json')['checked']: bound(ROOT,row)
assert digest(OLD/'fixed-samples.json') == d['sample_manifest_sha256']
assert not (OLD/'pixels/pixels.json').exists()
p=read(OUT/'pixels/pixels.json'); walk(p)
bound(OUT,d['full_pixel_json'])
assert p['fixed_manifest_sha256']==d['sample_manifest_sha256']
assert p['decoded_counts']==dict(physical_frames=64,prediction_frames=48,training_frames=40,training_small_state_rows=80,unknown_AR4_input_hashes=16,verified_action_hashes=48,verified_input_hashes=32)
assert len(p['branches'])==8 and len(p['train_windows'])==8
assert [r['selection']['cursor'] for r in p['train_windows']]==[24,26,0,2,31,33,1,3]
assert all(t['numeric_state_alignment'].startswith('matched') for t in p['train_windows'])
assert all(v==0 for v in p['forbidden_operations'].values())
preds=[r for b in p['branches'] for r in b['predictions']]
assert len(preds)==48
assert sum(r['input_hash_check']=='matched' for r in preds)==32
assert sum(r['input_hash_check'].startswith('unknown_AR4') for r in preds)==16
assert all(r['normalized_action_hash_check']=='matched' for r in preds)
rows=[]
for b in p['branches']:
    assert {(r['role'],r['mode'],r['horizon']) for r in b['predictions']} == {(role,mode,h) for role in ['frozen','adapted'] for mode,h in [('AR',1),('AR',4),('TF',4)]}
    for r in b['predictions']:
        m=read(ROOT/r['measurement_path'])['measurement']
        assert r['original_measurement']=={k:m[k] for k in r['original_measurement']}
        q=r['metrics']
        assert q['seed_component_count']==1 and q['component_mass_fraction_replay_difference']==0
        assert q['largest_component_mass_fraction']==m['shape']['component_mass_fraction']
        assert math.isclose(q['alpha_mass_outside_roi'],max(0.,q['object_alpha_mass']-q['alpha_mass_in_roi']),abs_tol=1e-12)
        assert math.isclose(q['alpha_fraction_outside_roi'],q['alpha_mass_outside_roi']/q['object_alpha_mass'],abs_tol=1e-12)
        if r['role']=='adapted': assert not m['state']['valid'] and m['coverage']['estimate'] is None
        if r['horizon']==1:
            rows.append(dict(anchor=b['anchor'],role=r['role'],outside_alpha=q['alpha_fraction_outside_roi'],component=q['largest_component_mass_fraction'],roi_iou=m['shape']['roi_soft_iou'],full_iou=m['shape']['full_soft_iou']))
pngs=list((OUT/'pixels').glob('*.png'))
assert len(pngs)==32
for f in pngs: assert digest(f)==digest(OLD/'pixels'/f.name)
receipts=[read(OUT/f'command-{i}-receipt.json') for i in (1,2)]
assert len(list(OUT.glob('command-*-receipt.json')))==2
assert all(r['status']=='complete' and r['exit_code']==0 and r['elapsed_seconds']<60 and r['peak_working_set_bytes']<2**30 for r in receipts)
assert sum(r['elapsed_seconds'] for r in receipts)<120
total=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
assert total<32*2**20
review=dict(kind='main_reexport_review.v1',commit=d['result_commit'],execution_commit=d['execution_commit'],tracked=5,evidence=53,identical_pngs=32,counts=p['decoded_counts'],command_seconds=sum(r['elapsed_seconds'] for r in receipts),command_peaks=[r['peak_working_set_bytes'] for r in receipts],output_bytes=total,h1_rows=rows,review_seconds=time.perf_counter()-start,scope='stdlib JSON/byte hashes/source review; no NPY/NPZ decoded, no weights/held contents, no Torch/model/native/physics',limitations=['Existing export scalar records audited against original measurement JSON; no independent raw-pixel recomputation in main review.','16 AR H4 input hashes remain unknown.','No causal training diagnosis, gate change or imagined-DPPO acceptance.'],next='One bounded training-data-only conditioning-embedding mechanism test preparation; no real execution yet.')
review['audit_script_correction']='First main audit assumed Git doc blob used LF and applied CRLF conversion; assertion failed. Corrected to compare exact Git blob bytes. No source/artifact mutation or numerical/model work.'
Path(__file__).with_name('review.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(review,ensure_ascii=True))
