"""Save R2C0 authorization and read-only source identity; no simulator/model."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import EXPERIMENT,read,sha,atomic_json,utc,verify_sources,storage_check
OUT=EXPERIMENT/'r2c0'; PRIOR=EXPERIMENT/'r2b'
assert not (OUT/'prior-stage-state.json').exists(), 'Bootstrap is one-use'
verify_sources();s=read(EXPERIMENT/'stage-state.json')
assert s['status']=='waiting_parent' and s['current_stage']=='R2B' and s['callback']['sent']
shutil.copyfile(EXPERIMENT/'stage-state.json',OUT/'prior-stage-state.json')
shutil.copyfile(EXPERIMENT/'native-rgb-cost-ledger.json',OUT/'prior-model-ledger.json')
files=[p for p in PRIOR.rglob('*') if p.is_file()]
files.extend(EXPERIMENT/p for p in ['native-rgb-cost-ledger.json','resource-ledger.json','standing-callback-authorization.json'])
frozen=dict(kind='R2C0-readonly-R2B-and-ledger-input-identities.v1',at_utc=utc(),files=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))])
atomic_json(OUT/'input-identities.json',frozen)
decision=read(PRIOR/'parent-review/stage-decision.json');assert decision['motion_quality_accepted_for_small_learning_diagnostic_preparation'] and not decision['training_authorized_this_stage']
s['phase_r2b_callback_history']=s['callback']
s.update(current_stage='R2C0',status='R2C0_CPU_preparation',authorized_scope='Read-only R2B53 accepted selected0.1s trajectories; freeze train41/seenval12 split, motion80pct/other20pct sample draws and offline eval interfaces/previews. No physics/env/model/GPU/optimizer/training or damping comparison.',authorized_stages=['R2C0'],r2c0_preparation_authorized=True,r2c0_training_authorized=False,r2c0_physics_authorized=False,r2c0_complete=False,r2b_diagnostic_data_admission_accepted=True,formal_sampling_accepted=False,formal_full_coverage_dataset_accepted=False,parent_review_received=True,future_stages_authorized=False,training_authorized=False,next_action='Complete R2C0 CPU-only manifests/draws/eval/previews/proposal, actual authorized callback, save receipt, waiting_parent STOP')
s['budget'].update(r2c0_new_controls_cap=0,r2c0_new_model_calls_cap=0,r2c0_new_artifact_bytes_cap=32*1024**2,r2c0_proposed_next_attempts=300,r2c0_proposed_next_seconds=1200,r2c0_next_model_budget_is_authorization=False)
s['callback']=dict(required=True,sent=False,status='not_prepared',attempts=[],thread_id=s['parent_thread_id'],host_id='local',message_file='r2c0/callback-r2c0.txt',result_file='r2c0/callback-result.json',standing_authorization_file='standing-callback-authorization.json',parent_review_received=False)
atomic_json(EXPERIMENT/'stage-state.json',s)
memo='当前R2C0已由父放行仅CPU数据/接口准备：R2B53轨迹通过小规模WM学习诊断数据准入，完整触点/方向/边界覆盖未通过。固定R2B selected0.1s源，split TRAIN41/seen-val12，80%明显运动(>=2 image96px OR >=5deg) draw和评估索引/原RGB预览；不复制RGB/不混旧D数据。0新physics/env/model/CUDA/GPU/训练/扩采/阻尼验证。300attempt/1200s仅下一步拟案未授权；旧closed250/868.484模型ledger不动，物理41976/76536不变，余24/320不动。新产物32MiB、总3GiB含父审+256MiB预留。完整r2c0/PARENT_REVIEW.md。完成实际主动callback/receipt后waiting_parent STOP，不轮询父。\n\n以下为历史记录：\n\n'
p=EXPERIMENT/'SUPERVISION.md';p.write_text(memo+p.read_text(encoding='utf-8'),encoding='utf-8')
p=ROOT/'docs/STATUS.md';p.write_text('# R2C0 CPU-only preparation authorized (2026-10-29)\n\nR2B real motion and physics passed admission for a small learning diagnostic; full contact/direction/boundary coverage remains incomplete. Prepare only R2B53 source references,41TRAIN/12seen-val family split,80/20 visible-motion draws,evaluation indices and raw-RGB previews. Zero new physics/model/GPU/training.300 attempts/1200s are a proposal requiring a later parent instruction; closed250/868.484s ledger unchanged. Artifacts in supervised r2c0,<=32MiB; callbacks use standing human authorization.\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps(dict(stage='R2C0',input_files=len(files),input_identity_sha256=sha(OUT/'input-identities.json'),shared_storage=storage_check('R2C0-start',32*1024**2))))
