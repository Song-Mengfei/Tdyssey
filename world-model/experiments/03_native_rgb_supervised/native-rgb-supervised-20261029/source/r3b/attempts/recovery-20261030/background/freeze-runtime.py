"""Freeze the narrowly approved recovery while keeping the old failed run intact."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tdyssey/native_rgb_r3a_data.py').is_file())
ATTEMPT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/native-rgb-supervised-20261029'
R3B=BASE/'r3b'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def binding(p):
    p=Path(p).resolve()
    with p.open('rb')as s:digest=hashlib.file_digest(s,'sha256').hexdigest()
    return dict(path=str(p),sha256=digest,bytes=p.stat().st_size)

config=read(R3B/'background/runtime-config.json');attempt=read(ATTEMPT/'attempt.json')
config.update(kind='R3B-human-approved-confirmed-loader-recovery-firstepoch.v1',run_id=attempt['run_id'],
    frozen_at_utc=datetime.now(timezone.utc).isoformat(),stage_root=str(ATTEMPT),budget_stage_root=str(R3B),
    execution_authorized=True,training_authorized=True,is_proposal=False,first_epoch_only=True,max_formal_batches=1499,
    recovery_authorized=True,resume_authorized=False,prior_ledger_copy=str(ATTEMPT/'prior-model-ledger.json'),
    prior_ledger_sha256=binding(ATTEMPT/'prior-model-ledger.json')['sha256'],
    previous_failure={**binding(R3B/'background/completion.json'),'run_id':attempt['previous_failed_run_id']},
    frozen_validation_spec=binding(R3B/'evaluation/spec.json'),
    loader_diagnostic_gate={**binding(ATTEMPT/'diagnostics/diag02-repaired/gate.json'),'run_id':attempt['run_id']},
    loader_recovery_source_sha256=binding(ROOT/'tdyssey/native_rgb_r3b_loader_recovery.py')['sha256'],
    completion_check_interval_minutes=30,next_epoch_authorized=False,previous_model_seconds=33.047)
config.pop('prelaunch_configuration_revision',None)
paths={Path(b['path'])for b in config['input_bindings']}
paths.update(ROOT/'tdyssey'/name for name in('native_rgb_r3b_loader_diagnostics.py','native_rgb_r3b_loader_recovery.py'))
paths.update([ROOT/'scripts/native_rgb_r3b_loader_diagnostic.py',ATTEMPT/'RECOVERY_RELEASE.md',ATTEMPT/'attempt.json',
    ATTEMPT/'prior-model-ledger.json',ATTEMPT/'diagnostics/root-cause.json',ATTEMPT/'diagnostics/diag02-repaired/gate.json',
    R3B/'background/completion.json',R3B/'evaluation/spec.json',Path(__file__),ATTEMPT/'background/launcher.py'])
config['input_bindings']=[binding(p)for p in sorted(paths)]
sources=set(config['new_stage_source_files'])
sources.update(str(p)for p in (ROOT/'tests').glob('test_native_rgb_r3b*.py'))
sources.update(str(ROOT/p)for p in('scripts/native_rgb_r3b_loader_diagnostic.py','tdyssey/native_rgb_r3b_loader_diagnostics.py',
    'tdyssey/native_rgb_r3b_loader_recovery.py','tests/test_native_rgb_r3b_loader_diagnostics.py','tests/test_native_rgb_r3b_loader_recovery.py'))
config['new_stage_source_files']=sorted(sources)
with(ATTEMPT/'background/runtime-config.json').open('x',encoding='utf-8',newline='\n')as s:
    json.dump(config,s,ensure_ascii=False,indent=2,sort_keys=True);s.write('\n')
state_path=BASE/'stage-state.json';state=read(state_path)
state.update(current_stage='R3B',status='r3b_recovery_verified_CPU_preparing_background',training_authorized=True,
    r3b_execution_authorized=True,r3b_training_authorized=True,r3b_recovery_authorized=True,
    r3b_current_run_id=config['run_id'],r3b_current_attempt=str(ATTEMPT),parent_can_auto_release=False,future_user_approval_required=True,
    authorized_scope='Human-approved confirmed CPU loader repair/validation and the original complete first1499batch epoch in an independent background pipeline, then actual final callback and STOP waiting_user_approval.',
    next_action='Launch the verified repaired firstepoch pipeline; observe GPU-restored two-loader validation, engineering probe and independent progress; actual brief handoff and endturn. No epoch2 or automatic retry.')
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(dict(run_id=config['run_id'],config=binding(ATTEMPT/'background/runtime-config.json'),bindings=len(paths))))
