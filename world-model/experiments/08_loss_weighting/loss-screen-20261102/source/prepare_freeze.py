"""One stdlib-only freeze. No model import, CUDA call, asset or environment copy."""
import ast,ctypes,json,sys,hashlib
from ctypes import wintypes
from pathlib import Path
from bounded_io import BudgetIO
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1];ART=ROOT.parent
R1=ART/'loss-rollout-ab-20261102-r1';HID=ART/'hmf-inference-diagnostics-20261102';HMF=ART/'high-motion-fit-20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
    assert not (ROOT/'frozen-config.json').exists() and not (ROOT/'execution-claim.json').exists()
    spec=ART/'retrospective-analysis-20261101/loss-literature-review-20261102/TEST_SPEC.txt'
    assert sha(spec)=='6ee0ca964127d80654800789049c14a10f7c89da4469e04e8c592a8ed0f81d71'
    assert sha(R1/'config.json')=='f652a7cc5963f4dc43bb46d8bcc38f2a5fdbaa99801f0ddf9be851311cf58293'
    parent=read(R1/'config.json');old=read(HID/'runtime-config.json')
    roots=old['shared_storage_roots']+[str(HID),str(ART/'retrospective-analysis-20261101'),str(ART/'loss-rollout-ab-20261102'),str(R1),str(ROOT)]
    io=BudgetIO(ROOT,roots)
    io.write_json('storage-plan.json',dict(shared_roots=roots,root_cap=900000000,shared_cap=10000000000,atomic=268435456,growth=67108864,metadata=16777216,
        actual_four_raw_bytes=552156384,expected_four_lightweight_bytes=139999752,conservative_weights_bytes=4*36*1024**2,metrics_bytes=4*12*1024**2,
        report_bytes=64*1024**2,source_and_owner_bytes=16*1024**2,metadata_reserved_bytes=16*1024**2,
        maximum_plan_sum=552156384+4*36*1024**2+4*12*1024**2+64*1024**2+16*1024**2+16*1024**2,
        own_byte_counter='Single-writer cached exact output lengths; rescanned at arm/eval/final boundaries.16MiB metadata reserve also isolates small owner receipt growth; no raw/weight writes outside guarded IO.'))
    io.write_bytes('SOURCE_TEST_SPEC.txt',spec.read_bytes(),exclusive=True)
    schedule=read(R1/'schedule.json');planned=dict(native_h1_batches=schedule['native_h1_batches'][:128],source=str(R1/'schedule.json'),source_sha256=sha(R1/'schedule.json'))
    assert len(planned['native_h1_batches'])==128 and all(len(x)==16 for x in planned['native_h1_batches'])
    io.write_json('schedule.json',planned,exclusive=True);io.write_bytes('evaluation-spec.json',(R1/'evaluation-spec.json').read_bytes(),exclusive=True)
    baseline=read(HMF/'evaluation/0/primary/metrics.json');parity=[];extra=[]
    for letter in ('O','C','B'):
        for qualified in (True,False):
            row=next(x for x in baseline['rows'] if x['letter']==letter and bool(x['metrics'].get('legacy_shape_assessable'))==qualified)
            ref=next(x for x in baseline['raw_files'] if x['sample_id']==row['sample_id'] and x['input_mode']==row['input_mode'])
            truth=row.get('raw_path',ref['raw_path'])
            parity.append(dict(sample_id=row['sample_id'],letter=letter,prediction_path=ref['path'],horizon=row['horizon'],anchor=int(ref['anchor']),truth_path=truth))
            extra.extend([Path(ref['path']),Path(truth)])
    io.write_json('scoring-parity-inputs.json',parity,exclusive=True)
    paths=[spec,R1/'evaluation/E2/metrics.json',R1/'evaluation/A128/metrics.json',R1/'copy-reference.json',R1/'training/A-log.json',
        HMF/'training_core.py',HMF/'data.py',HMF/'evaluation.py',HMF/'fit-standard.json',HMF/'GT-measurement-calibration.json',HMF/'selection/index.json',
        HID/'metrics.py',HID/'resource_limits.py',HID/'VERIFICATION.json',*extra,*ROOT.glob('*.py'),ROOT/'schedule.json',ROOT/'evaluation-spec.json',ROOT/'SOURCE_TEST_SPEC.txt',ROOT/'launch.ps1',ROOT/'scoring-parity-inputs.json']
    # Only necessary E2 image sources used by the saved report are additionally bound.
    e2=read(R1/'evaluation/E2/metrics.json');es=read(ROOT/'evaluation-spec.json');cases=[x for b in es['batches'] for x in b['rows']]
    chosen=[next(x for x in cases if x['selection_role']=='train' and x['letter']==l) for l in ('O','C','B')]+[x for x in cases if x['selection_role']=='original_failure']
    for row in chosen:
        paths.extend(Path(x['path']) for x in e2['raw_files'] if x['sample_id']==row['sample_id'])
    bindings={str(Path(b['path']).resolve()):b for b in parent['bindings']}
    for p in paths:bindings[str(p.resolve())]=dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
    for p,b in bindings.items():
        if sha(p)!=b['sha256']:raise RuntimeError('Existing scientific source changed: '+p)
    for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    config=dict(run_id='LOSS-LIT-SCREEN-20261102',arms=['BASE','SNR','GDL','FFL'],attempts_per_arm=128,microbatch=4,effective_batch=16,
        repo=str(REPO),python_executable=str(REPO.parent/'Tdyssey/.venv/Scripts/python.exe'),source_root=parent['source_root'],parent=parent['parent'],parent_sha256=parent['parent_sha256'],
        E2_metrics=str(R1/'evaluation/E2/metrics.json'),R1_A128_metrics=str(R1/'evaluation/A128/metrics.json'),R1_A_log=str(R1/'training/A-log.json'),copy_reference=str(R1/'copy-reference.json'),
        hid_metrics_path=str(HID/'metrics.py'),cpu_limits_path=str(HID/'resource_limits.py'),test_spec_path=str(spec),test_spec_sha256=sha(spec),
        implementation_clarification='One preflight model-only E2 load + four formal independent full-state E2 restores, one read-only CPU parent and one model construction; no four Adam copies on disk.',
        formulas=dict(BASE='original F.mse_loss full mean and order',SNR='mean_B(mean_CHW((F-r)^2)*min(.25/(s^2+.1^2),5)/(.25/(s^2+.1^2)))',GDL='base+GDL(D/.5,y/.5),alpha1,coefficient1,adjacent edges no padding',FFL='base+mean(detach(abs(complexDelta)/per_sample_channel_max_HW)*(realDelta^2+imagDelta^2));orthoFFT FP32,coefficient1'),
        caps=dict(model_image_forwards=23296,model_batch_forwards=6016,training_image_forwards=8192,evaluation_image_forwards=14976,preflight_image_forwards=128,
            eval_predictions=4992,training_attempts=512,training_updates=512,training_backward_calls=2048,preflight_gradient_calls=32,model_constructions=1,preflight_model_only_loads=1,full_arm_restores=4),
        native_recipe='E2/HMF unchanged data/history/actions/50-50cleanweak/noise/FP16forward FP32loss/AdamW1e-5/clip1',sampler=parent['sampler'],
        scoring='HID GPU basic + original qualified CPU legacy/contour/topology/axis; skip only original pre-registration unknown numeric attachments; qualification parity required; original absolute gates retained',
        preflight_math_tolerances=dict(SNR_equivalence_value_atol=1e-6,SNR_equivalence_value_rtol=1e-5,SNR_equivalence_grad_atol=1e-6,SNR_equivalence_grad_rtol=1e-4,BASE='bitwise loss and parameter gradients',scoring_primary_atol=1e-12),
        windows_task_name='Tdyssey-LLS-20261102',callback_thread_id='__LOCAL_CHAT__',callback_host_id='local',owner_thread_id='__LOCAL_CHAT__',
        stop='Any formal nonfinite/AMPskip/OOM/source/budget failure stops, no rerun or next experiment. Saved-only path/report correction permitted with original failure retained, no extra model calls.',bindings=sorted(bindings.values(),key=lambda x:x['path']))
    io.write_json('frozen-config.json',config,exclusive=True);digest=sha(ROOT/'frozen-config.json')
    io.write_json('freeze-receipt.json',dict(run_id=config['run_id'],config_sha256=digest,test_spec_sha256=sha(spec),binding_count=len(bindings),prepare_model_calls=0,
        prepare_Torch_imports=0,CPU='single BelowNormal stdlib-only',storage=io.guard(refresh=True),static_checks=read(ROOT/'STATIC_CHECKS.json')),exclusive=True)
    io.write_json('execution-release.json',dict(run_id=config['run_id'],config_sha256=digest,execution_authorized=True,source='Direct human literature/testing authorization plus finalized TEST_SPEC and implementation clarification',no_next_experiment=True),exclusive=True)
    print(json.dumps(dict(run_id=config['run_id'],config_sha256=digest,test_spec_sha256=sha(spec),bindings=len(bindings),models_constructed=0,storage=io.guard())))

if __name__=='__main__':main()

