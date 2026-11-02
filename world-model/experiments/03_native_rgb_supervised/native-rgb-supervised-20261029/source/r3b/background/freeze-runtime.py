"""Freeze the approved first-epoch configuration without Torch/model work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=ROOT/'artifacts/native-rgb-supervised-20261029'
STAGE=HERE.parent
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def binding(path):
    path=Path(path).resolve()
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),sha256=digest,bytes=path.stat().st_size)

baseline=read(STAGE/'baseline-storage.json')
prior=binding(STAGE/'prior-model-ledger.json')
paths=list((ROOT/'tdyssey').glob('native_rgb_*.py'))
paths += list((ROOT/'tdyssey/envs').glob('*.py'))
paths += [ROOT/'tdyssey'/p for p in ('gpc_world_model.py','raw_world_model_training.py','official_world_model.py','geometry.py','state.py')]
paths += list((ROOT/'scripts').glob('*r3b*.py'))
paths += [HERE/'launch.py', HERE/'launch-via-explorer.ps1', Path(__file__), STAGE/'PARENT_RELEASE.md', STAGE/'prior-model-ledger.json',
    BASE/'r3a/background-worker.py', ROOT/'configs/native_rgb_r3b_proposal.json', ROOT/'manifests/gpc_world_model.json',
    BASE/'r3a/manifest.json', BASE/'r3a/data-interface/window-catalogue.npz', BASE/'r3a/data-interface/window-catalogue.json',
    BASE/'r3a/data-interface/interface-gate.json', BASE/'r3a/r3b-proposal/eight-epoch-index-plan.npz',
    BASE/'r3a/r3b-proposal/evaluation-indices.json', BASE/'r2c3/run/last.pt',
    ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json',
    BASE/'r2c0/data/manifest.json', BASE/'r2c0/evaluation/indices.json',
    BASE/'r2c3/run/evaluation/update6000/steps3/cohort-val_motion.json']
gpc=ROOT.parent/'Tdyssey/third_party/gpc_code'
paths += list(gpc.rglob('*.py'))
paths += list((ROOT.parent/'Tdyssey/third_party/diffusion_policy/diffusion_policy/env/pusht').rglob('*.py'))
paths += list((BASE/'r2c3/run/evaluation/update6000/steps3').glob('cohort-val_motion*.npz'))
paths=sorted(set(p.resolve() for p in paths))
config=dict(kind='R3B-human-approved-first-epoch-runtime.v1',run_id='R3B-firstepoch-'+uuid.uuid4().hex,
    frozen_at_utc=datetime.now(timezone.utc).isoformat(),stage_root=str(STAGE),source_root=str(gpc),
    physics_source_root=str(ROOT.parent/'Tdyssey/third_party/diffusion_policy'),
    execution_authorized=True,training_authorized=True,is_proposal=False,first_epoch_only=True,
    max_formal_batches=1499,resume_authorized=False,require_detached_job=True,
    data_manifest=str(BASE/'r3a/manifest.json'),window_catalogue=str(BASE/'r3a/data-interface/window-catalogue.npz'),
    index_plan=str(BASE/'r3a/r3b-proposal/eight-epoch-index-plan.npz'),resume_checkpoint=str(BASE/'r2c3/run/last.pt'),
    device='cuda',max_cuda_bytes=6*1024**3,frame_cache_bytes=16*1024**2,state_cache_bytes=8*1024**2,
    engineering_max_attempts=2,max_consecutive_AMP_overflows=16,
    stage_cap_bytes=320*1024**2,shared_cap_bytes=3*1024**3,atomic_reserve_bytes=256*1024**2,
    shared_storage_roots=baseline['roots'],
    new_stage_source_files=[str(ROOT/p) for p in ('scripts/native_rgb_r3b_background.py',
        'tdyssey/native_rgb_r3b_training.py','tdyssey/native_rgb_r3b_evaluation.py','tdyssey/native_rgb_r3b_report.py',
        'tests/test_native_rgb_r3b_training.py','tests/test_native_rgb_r3b_evaluation.py','tests/test_native_rgb_r3b_background.py') if (ROOT/p).is_file()],
    ledger_path=str(BASE/'native-rgb-cost-ledger.json'),prior_ledger_copy=prior['path'],prior_ledger_sha256=prior['sha256'],
    input_bindings=[binding(p) for p in paths],
    model_time_policy='record_only_v2; no time cap; engineering/formal/evaluation charges separate',
    callback_thread_id='__LOCAL_CHAT__',completion_check_interval_minutes=30,
    next_epoch_authorized=False,CP1_CP2_training_data_included=False)
with (HERE/'runtime-config.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(config,stream,ensure_ascii=False,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(run_id=config['run_id'],input_bindings=len(paths),runtime_config=binding(HERE/'runtime-config.json'))))
