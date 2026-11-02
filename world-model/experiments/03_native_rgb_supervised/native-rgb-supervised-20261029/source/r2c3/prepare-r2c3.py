"""Prepare explicitly human-authorized C3 full-resume inputs, then freeze sources."""
import sys,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXP/'r2c3';R1=EXP/'r2c1r1';C2=EXP/'r2c2'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (OUT/'run-claim.json').exists(),'Never reconfigure a claimed stage'
cfgfile=ROOT/'configs/native_rgb_r2c3.json'
newfiles=[cfgfile,ROOT/'tdyssey/native_rgb_r2c3_training.py',ROOT/'scripts/native_rgb_r2c3_run.py',ROOT/'tests/test_native_rgb_r2c3_resume.py',ROOT/'tdyssey/native_rgb_r2c3_continuation.py',ROOT/'tests/test_native_rgb_r2c3_continuation.py',ROOT/'tdyssey/native_rgb_r2c3_evaluation_io.py',ROOT/'tests/test_native_rgb_r2c3_evaluation_io.py']
if '--freeze' not in sys.argv and '--freeze-v2' not in sys.argv:
    old=read(ROOT/'configs/native_rgb_r2c1r1.json');l=read(EXP/'native-rgb-cost-ledger.json')
    assert l['active_stage'] is None and l['charged_updates']==l['completed_updates']==550 and l['max_seconds'] is None
    prior=OUT/'prior-model-ledger.json';assert not prior.exists();shutil.copyfile(EXP/'native-rgb-cost-ledger.json',prior)
    refs=dict(old['input_hashes'])
    for p in [R1/'run/last.pt',ROOT/'configs/native_rgb_r2c1r1.json',R1/'source-identity.json',C2/'model-work.json',C2/'report/r2c2-report.json',C2/'R2C3_PROPOSAL.md',C2/'PARENT_TRAINING_LENGTH_NOTE.md',C2/'r2c3-proposal-review.json',OUT/'USER_CONTINUATION_AUTHORIZATION.md',EXP/'standing-callback-authorization.json',EXP/'user-model-time-policy.json']:
        refs[str(p)]=sha(p)
    config={k:old[k]for k in ('seed','effective_batch','microbatch','amp','optimizer','scheduler','grad_scaler_init_scale','gradient_clip_norm','source_root','data_manifest','evaluation_indices','calibration','max_cuda_bytes','sampler','native_call_reserve_seconds')}
    config.update(kind='native-GPC-R2C3-direct-human-full-resume-learning-curve.v1',is_proposal=False,execution_authorized=True,training_authorized=True,
        repo_root=str(ROOT),output=str(OUT),config_file=str(cfgfile),stage_state=str(EXP/'stage-state.json'),source_identity=str(OUT/'source-identity.json'),
        input_hashes=refs,parent_authorization=str(OUT/'USER_CONTINUATION_AUTHORIZATION.md'),parent_authorization_sha256=sha(OUT/'USER_CONTINUATION_AUTHORIZATION.md'),
        parent_resume_config=str(ROOT/'configs/native_rgb_r2c1r1.json'),parent_resume_config_sha256=sha(ROOT/'configs/native_rgb_r2c1r1.json'),
        resume_checkpoint=str(R1/'run/last.pt'),resume_checkpoint_sha256='3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0',resume_checkpoint_bytes=105300718,
        resume_lineage_updates=300,target_lineage_updates=6000,targets=[1000,3000,6000],target_new_complete_updates=5700,new_completed_updates=5700,
        evaluation_settings=[3,16],evaluation_batch=4,reuse_r1_root=str(R1),reuse_r2c2_root=str(C2),
        fixed_loss_checkpoints=[1000,3000,6000],fixed_loss_batch=4,no_new_300_evaluation=True,
        lr_continuation=dict(T_max=3000,eta_min=1e-5,after_T_max='hold_eta_min',restore_existing_state=True,advance_only_after_success=True),
        planned_total_lineage_draws=96000,planned_new_draws=91200,stop_on_amp_skip=True,automatic_retry=False,
        ledger_path=str(EXP/'native-rgb-cost-ledger.json'),prior_ledger_copy=str(prior),prior_ledger_sha256=sha(prior),prior_attempts=550,prior_completed_updates=550,prior_model_seconds=l['charged_seconds'],
        time_policy='record_only_v2',cap_time=None,max_model_seconds=None,global_max_seconds=None,latest_new_attempt_seconds=None,tail_reserve_seconds=None,hard_time_limits_enabled=False,user_set_model_time_budget=False,
        stage_artifact_cap_bytes=419430400,shared_cap_bytes=3221225472,atomic_reserve_bytes=268435456,
        storage_roots=list(dict.fromkeys(read(ROOT/'configs/native_rgb_r2c2.json')['storage_roots']+[str(OUT)]+[str(p)for p in newfiles])),new_stage_source_files=[str(p)for p in newfiles],
        full_checkpoint_cap_bytes=117440512,light_checkpoint_cap_bytes=37748736,
        raw_endpoint_storage='lossless_raw_references',compact_logging=True,
        physics_or_environment_construction_allowed=False,unseen_access_allowed=False,server_allowed=False,
        automatic_continuation_beyond_6000=False,model_work_includes_fixed_loss_and_CPU_endpoint_scores=True)
    save(cfgfile,config)
    print(json.dumps(dict(prepared=True,prior_sha=config['prior_ledger_sha256'],input_refs=len(refs),targets=config['targets'],new300eval=False,Torch_imported='torch'in sys.modules)))
else:
    v2='--freeze-v2' in sys.argv
    identity_file=OUT/('source-identity-v2.json' if v2 else 'source-identity.json')
    snapshot_folder='source-snapshot-v2' if v2 else 'source-snapshot'
    assert not identity_file.exists()
    prior=read(C2/'source-identity.json');files={Path(r['source']).resolve():r for r in prior['files']}
    for p in newfiles:assert p.is_file(),str(p);files[p.resolve()]={}
    rows=[]
    for p in sorted(files,key=str):
        rel=Path(snapshot_folder)/'repo'/p.relative_to(ROOT)if p.is_relative_to(ROOT)else Path(snapshot_folder)/Path(files[p]['snapshot']).relative_to('source-snapshot')
        target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        h=sha(p);assert sha(target)==h
        if files[p]:assert h==files[p]['sha256'],'Prior frozen source changed'
        rows.append(dict(source=str(p),snapshot=rel.as_posix(),sha256=h,bytes=p.stat().st_size))
    save(identity_file,dict(kind='R2C3-explicit-continuation-pre-model-freeze.v2' if v2 else 'R2C3-explicit-continuation-pre-model-freeze.v1',files=rows,at_utc=datetime.now(timezone.utc).isoformat()))
    protected={r['path']:r for r in read(C2/'input-identities.json')['files']}
    for p in sorted(C2.rglob('*')):
        if p.is_file():protected[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    for r in protected.values():assert Path(r['path']).stat().st_size==r['bytes']and sha(r['path'])==r['sha256']
    save(OUT/('input-identities-v2.json' if v2 else 'input-identities.json'),dict(kind='R2C3-protected-prior-data-and-results.v1',files=list(protected.values()),only_live_global_cost_ledger_may_change=True))
    s=read(EXP/'stage-state.json');s['r2c3_source_identity_sha256']=sha(identity_file);s['status']='R2C3_frozen_for_CPU_resource_preflight';save(EXP/'stage-state.json',s)
    print(json.dumps(dict(frozen=True,sources=len(rows),protected=len(protected),config_sha256=sha(cfgfile),source_identity_sha256=sha(identity_file),Torch_imported='torch'in sys.modules)))
