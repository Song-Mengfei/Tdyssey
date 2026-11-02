"""Build explicit CPU-only R2C2 configuration and freeze already-written sources."""
import sys,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXP/'r2c2';OLD=EXP/'r2c1r1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (OUT/'run-claim.json').exists(),'No reconfiguration of an already claimed stage'
cfgfile=ROOT/'configs/native_rgb_r2c2.json'
newfiles=[cfgfile,ROOT/'tdyssey/native_rgb_r2c2_inference.py',ROOT/'scripts/native_rgb_r2c2_run.py',ROOT/'scripts/native_rgb_r2c2_report.py',ROOT/'tests/test_native_rgb_r2c2_contracts.py']
if '--freeze' not in sys.argv:
    old=read(ROOT/'configs/native_rgb_r2c1r1.json');ledger=read(EXP/'native-rgb-cost-ledger.json')
    assert ledger['active_stage'] is None and ledger['charged_updates']==ledger['completed_updates']==550 and ledger['max_seconds'] is None
    prior=OUT/'prior-model-ledger.json'
    assert not prior.exists(),'Prepare exactly once, retain prior accounting'
    shutil.copyfile(EXP/'native-rgb-cost-ledger.json',prior)
    inputs=dict(old['input_hashes'])
    # Reference all old arrays/metrics instead of duplicating any raw data.
    for p in sorted((OLD/'run/evaluation').rglob('*')):
        if p.is_file():inputs[str(p)]=sha(p)
    for p in [OLD/'run/last.pt',OLD/'model-work.json',OLD/'source-identity.json',OLD/'report/r2c1-report.json',OLD/'parent-review/independent-audit.json',EXP/'user-model-time-policy.json',EXP/'standing-callback-authorization.json',OUT/'PARENT_REVIEW.md']:
        inputs[str(p)]=sha(p)
    config=dict(kind='native-GPC-R2C2-fixed-checkpoint-denoising-diagnostic.v1',is_proposal=False,execution_authorized=True,training_authorized=False,seed=5478,evaluation_batch=4,
        repo_root=str(ROOT),output=str(OUT),config_file=str(cfgfile),stage_state=str(EXP/'stage-state.json'),source_identity=str(OUT/'source-identity.json'),
        source_root=old['source_root'],data_manifest=old['data_manifest'],evaluation_indices=old['evaluation_indices'],calibration=old['calibration'],input_hashes=inputs,
        checkpoint=str(OLD/'run/last.pt'),checkpoint_sha256='3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0',checkpoint_bytes=105300718,checkpoint_completed_updates=300,
        reuse_root=str(OLD),new_settings=[8,16],sampler=old['sampler'],max_cuda_bytes=6442450944,
        ledger_path=str(EXP/'native-rgb-cost-ledger.json'),prior_ledger_copy=str(prior),prior_ledger_sha256=sha(prior),prior_attempts=550,prior_model_seconds=ledger['charged_seconds'],
        time_policy='record_only_v2',cap_time=None,max_model_seconds=None,hard_time_limits_enabled=False,user_set_model_time_budget=False,
        stage_artifact_cap_bytes=134217728,shared_cap_bytes=3221225472,atomic_reserve_bytes=268435456,
        storage_roots=list(dict.fromkeys(old['storage_roots']+[str(OUT)]+[str(p)for p in newfiles])),new_stage_source_files=[str(p)for p in newfiles],
        parent_authorization=str(OUT/'PARENT_REVIEW.md'),parent_authorization_sha256=sha(OUT/'PARENT_REVIEW.md'),
        parent_acceptance=str(OLD/'parent-review/independent-audit.json'),parent_acceptance_sha256=sha(OLD/'parent-review/independent-audit.json'),
        time_policy_file=str(EXP/'user-model-time-policy.json'),time_policy_sha256=sha(EXP/'user-model-time-policy.json'),
        expected_workload=dict(new_settings=2,sampler_batch_calls_per_setting=36,raw_float32_frames_per_setting=144,endpoint_score_rows_per_setting=72,denoiser_forward_calls_per_setting={'8':288,'16':576},counts_are_expectations_not_observations=True),
        finite_out_of_range_policy='Persist raw,feed raw unchanged,score existing failures; no repair or retry',real_runtime_failure_policy='Preserve trace and existing outputs;stop model work,no automatic restart',
        new_optimizer_attempts_allowed=0,duplicate_checkpoint_allowed=False,physics_or_environment_construction_allowed=False,held_out_shape_access_allowed=False,automatic_training_continuation=False,
        model_work_includes_CPU_endpoint_scores=True)
    save(cfgfile,config)
    print(json.dumps(dict(prepared=True,input_refs=len(inputs),prior_ledger_sha256=config['prior_ledger_sha256'],Torch_imported='torch'in sys.modules)))
else:
    assert not (OUT/'source-identity.json').exists(),'No overwrite of frozen identity'
    oldidentity=read(OLD/'source-identity.json');files={Path(row['source']).resolve():row for row in oldidentity['files']}
    for p in newfiles:
        assert p.is_file(),str(p)
        files[p.resolve()]={}
    rows=[]
    for p in sorted(files,key=str):
        if p.is_relative_to(ROOT):rel=Path('source-snapshot/repo')/p.relative_to(ROOT)
        else:
            priorrow=files[p];rel=Path(priorrow['snapshot'])
        target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        h=sha(p);assert sha(target)==h
        if files[p]:assert h==files[p]['sha256'],'Old source was modified'
        rows.append(dict(source=str(p),snapshot=rel.as_posix(),sha256=h,bytes=p.stat().st_size))
    save(OUT/'source-identity.json',dict(kind='R2C2-pre-model-source-freeze.v1',files=rows,at_utc=datetime.now(timezone.utc).isoformat(),old_frozen_sources_unchanged=True))
    protected={r['path']:r for r in read(OLD/'input-identities.json')['files']}
    for p in sorted(OLD.rglob('*')):
        if p.is_file():protected[str(p)]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
    for r in protected.values():assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
    save(OUT/'input-identities.json',dict(kind='R2C2-protected-old-evidence.v1',files=list(protected.values()),live_global_ledger_mutation_authorized_for_time_statistics_only=True,new_optimizer_updates=0))
    s=read(EXP/'stage-state.json');s['r2c2_source_identity_sha256']=sha(OUT/'source-identity.json');s['status']='R2C2_frozen_ready_for_CPU_preflight';save(EXP/'stage-state.json',s)
    print(json.dumps(dict(frozen=True,source_files=len(rows),protected_files=len(protected),config_sha256=sha(cfgfile),source_identity_sha256=sha(OUT/'source-identity.json'),Torch_imported='torch'in sys.modules)))
