"""Evidence-bound Windows spawn repair; original R3A dataset/renderer unchanged."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ATTEMPT=ROOT/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030'

def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def apply_spawn_repair(output=None):
    """Bypass the observed venv redirector so duplicated handles reach real PID."""
    from . import native_rgb_r3b_loader_diagnostics as diag
    import multiprocessing as mp
    from multiprocessing import spawn
    if os.name!='nt':raise RuntimeError('This confirmed recovery is Windows-specific')
    evidence_path=ATTEMPT/'diagnostics/root-cause.json'
    evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
    if evidence.get('root_cause_confirmed')is not True or evidence.get('category')!='venv_redirector_wrong_handle_target':
        raise RuntimeError('Confirmed redirector/handle evidence required')
    for item in evidence['evidence_bindings']:
        if sha(item['path'])!=item['sha256']:raise RuntimeError('Frozen cause evidence changed')
    actual=Path(diag.actual_pe()).resolve();venv=Path(sys.executable).resolve()
    if venv!=(Path(sys.prefix)/'Scripts/python.exe').resolve()or actual.parent!=Path(sys.base_prefix).resolve():
        raise RuntimeError('Existing base Python and venv identity do not match confirmed environment')
    before=dict(identity=diag.identity(),multiprocessing_executable=os.fsdecode(spawn.get_executable()))
    # Public multiprocessing API; no stdlib patch, environment replacement or
    # change to the original dataset, two-worker pool, cache or prefetch.
    mp.set_executable(str(actual));os.environ['__PYVENV_LAUNCHER__']=str(venv)
    proof=dict(kind='R3B-confirmed-Windows-spawn-repair.v1',passed=True,before=before,
        actual_base_python=str(actual),venv_python=str(venv),multiprocessing_executable=os.fsdecode(spawn.get_executable()),
        inherited_venv_launcher=os.environ['__PYVENV_LAUNCHER__'],root_cause_confirmed=True,
        evidence_path=str(evidence_path),evidence_sha256=sha(evidence_path),repair_source_sha256=sha(__file__),
        root_cause=evidence['root_cause'],repair='Start actual base PE directly with existing venv launcher identity; preserve two workers and production data behavior.')
    if output is not None:diag.json_write(Path(output)/'spawn-repair.json',proof)
    return proof


def verify_after_model_restore(session,*,status_cb=None):
    """Check actual two-process batches with model resident, without model calls."""
    import numpy as np
    from . import native_rgb_r3a_data as data
    from . import native_rgb_r3b_loader_diagnostics as diag
    output=Path(session.config['stage_root'])/'diagnostics/post-model-loader';output.mkdir(parents=True,exist_ok=True)
    if session.loader is not None:raise RuntimeError('Verify the original failure boundary before creating the loader')
    if session.order.batch_cursor or session.progress['formal_draws']:raise RuntimeError('Loader verification must precede formal consumption')
    os.environ['TDYSSEY_LOADER_DIAGNOSTIC_DIR']=str(output)
    os.environ['TDYSSEY_LOADER_DIAGNOSTIC_PARENT_PID']=str(os.getpid())
    repair=apply_spawn_repair(output)
    # Persist child diagnostics for bootstrap exceptions and later worker death.
    site_hook=diag.install_site_hook(output);diag.install_spawn_observation()
    data._initialize_worker=diag.diagnostic_initializer;data._worker_batch=diag.diagnostic_batch
    before=session.order.state_dict();started=time.monotonic();reference=None
    proof=dict(kind='R3B-loader-verified-after-GPU-restore.v1',passed=False,run_id=session.config['run_id'],
        model_resident_on_device=str(next(session.model.parameters()).device),full6000_restore_passed=session.restore_proof['passed'],
        workers=2,prefetch_batches=2,formal_draws=0,model_forward_calls=0,optimizer_attempts=0,
        repair=repair,site_hook=site_hook,batches=[],verified_worker_pids=[],parent_identity=diag.identity())
    try:
        if not proof['model_resident_on_device'].startswith('cuda'):raise RuntimeError('Actual CUDA model restore required before this boundary')
        loader=session._ensure_loader();reference=data.R3AStateDataset(**session.dataset_kwargs)
        for number,batch in enumerate(loader.iterate(session.order.batches[:4])):
            indices=batch['indices'];values=batch['model_inputs']
            expected={k:np.stack([reference[int(i)][k]for i in indices])for k in('image','action')}
            if set(values)!=set(expected)or not all(np.array_equal(values[k],expected[k])for k in expected):
                raise AssertionError('Restored-model loader differs from original indexed RGB/actions')
            identity=batch['worker_identity'];pid=batch['worker_pid']
            if identity['pid']!=pid or Path(identity['actual_process_executable']).resolve()!=Path(repair['actual_base_python']):
                raise RuntimeError('Actual Python child differs from the process owning its handles')
            if Path(identity['prefix']).resolve()!=Path(sys.prefix).resolve()or identity['torch_imported']:
                raise RuntimeError('Data worker environment changed or imported Torch')
            row=dict(number=number,worker_pid=pid,indices=list(indices),families=[session.order.rows[int(i)]['family_id']for i in indices],
                identity=identity,exact_reference_matches=True,model_input_keys=sorted(values),
                input_hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()for k,v in values.items()})
            proof['batches'].append(row);proof['verified_worker_pids']=sorted(set(proof['verified_worker_pids']+[pid]))
            if status_cb:status_cb('loader_after_model_restore_batch',dict(batch=number,worker_pid=pid,verified_batches=len(proof['batches']),formal_draws=0))
        if len(proof['verified_worker_pids'])!=2 or loader.maximum_in_flight>2:raise RuntimeError('Two bounded production workers not verified')
        if session.order.state_dict()!=before or session.progress['formal_draws']!=0:raise RuntimeError('CPU validation advanced formal cursor')
        actual={child.pid for child in diag.CHILDREN if child.is_alive()}
        if not set(proof['verified_worker_pids']).issubset(actual):raise RuntimeError('Loader child PID ownership/continued lifetime differs')
        proof.update(passed=True,verified_batches=4,CPU_only_validation_windows=64,maximum_in_flight=loader.maximum_in_flight,
            actual_child_processes=diag.child_exitcodes(),production_loader_reused_by_engineering_and_training=True)
        return proof
    except BaseException as exc:
        import traceback
        proof['failure']=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if reference is not None:reference.close()
        proof['wall_seconds']=time.monotonic()-started
        diag.json_write(output/'verification.json',proof)
