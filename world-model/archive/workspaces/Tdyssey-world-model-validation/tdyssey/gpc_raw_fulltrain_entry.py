"""Fixed authority and metadata-only preparation for one raw T/L run."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from . import gpc_adaptation as original
from .wm_compat_storage import unlinked_path

ROOT = original.ROOT
CANDIDATE = 'gpc-raw-tl-fulltrain-01'
AUTHORIZATION = '20261026-user-raw-tl-fulltrain-01'
OUTPUT = 'artifacts/gpc-raw-tl-fulltrain-20261026'
PREPARATION = 'artifacts/gpc-raw-tl-fulltrain-preparation-20261026'
CONFIG_RELATIVE = 'configs/gpc_raw_tl_fulltrain.json'
CONFIG = ROOT/CONFIG_RELATIVE
SCRIPT = 'scripts/gpc_raw_tl_fulltrain.py'
INITIAL_SHA256 = '4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892'


def configuration_template():
    return dict(kind='tdyssey.gpc_raw_tl_fulltrain.v1', candidate_id=CANDIDATE, output_dir=OUTPUT,
        data_manifest='artifacts/gpc-data-stage-a-20261026/adaptation-data.json',
        data_manifest_sha256='2a28136ca9c3b111d202f507ebab16c992b4d26a9c48d388956e390c6b2b069e',
        prefix_schedule=dict(path='artifacts/wm-sampling-b-only-20261026/training-schedule.json',
            sha256='f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a'),
        full_schedule=PREPARATION+'/training-schedule.json', development=PREPARATION+'/development.json',
        image_domain='official_raw_rgb', seed=5478,
        train=dict(max_updates=2328, max_examples=2328, batch_size=1, phase_seconds=1200,
            checkpoints=[500,2328], final_update=2328, grad_clip_norm=1., trainable_prefixes=['inner_model.'],
            objective='pinned_phase_two_one_step_residual_mse',
            optimizer=dict(name='AdamW',lr=.0001,betas=[.9,.999],eps=1e-8,weight_decay=.01,amsgrad=False)),
        evaluation=dict(roles=['initial','raw500','raw2328'], windows=16, windows_per_shape=8,
            loss_examples_per_role=16, forecasts_per_role=16, seed=5478, euler_steps=3,
            sigma_cond=None, output_clip=True, pose_and_coverage='not_measured', split='development'),
        sources=dict(initial=dict(path='artifacts/frozen-wm-assets/denoiser.pth',bytes=35043916,sha256=INITIAL_SHA256)),
        limits=dict(wall_seconds=1800, model_loads=4, active_gpu_models=1, max_updates=2328,
            max_train_examples=2328, eval_batches=48, eval_examples=48, forecasts=48, euler_steps=144,
            output_bytes=1024**3, metadata_bytes=16*1024**2, telemetry_bytes=64*1024**2,
            checkpoint_bytes=36*1024**2, cpu_threads=2, cuda_allocated_bytes=512*1024**2,
            cuda_reserved_bytes=1024**3),
        resources=dict(startup_ram_gib=4,startup_commit_gib=6,startup_gpu_gib=4,startup_disk_gib=2,
            runtime_ram_gib=1.5,runtime_commit_gib=3,runtime_gpu_gib=1.5,
            process_working_set_gib=2,process_private_gib=3,native_job_gib=3))


def validate_config(value):
    if value != configuration_template(): raise ValueError('Fixed raw T/L scientific/resource configuration differs')
    return deepcopy(value)


def identity(path,cfg):
    if unlinked_path(path) != unlinked_path(CONFIG) or cfg != configuration_template():
        raise PermissionError('Only the fixed raw T/L config/candidate/output is authorized')
    return dict(candidate_id=CANDIDATE,authorization_id=AUTHORIZATION,output_dir=OUTPUT,config_path=str(unlinked_path(CONFIG)))


def plan(config_path=CONFIG, *, claimed=False):
    from .gpc_raw_fulltrain_data import plan_data, schedule_bytes
    path = unlinked_path(config_path); cfg = validate_config(json.loads(path.read_bytes())); identity(path,cfg)
    if original.sha_file(ROOT/cfg['data_manifest']) != cfg['data_manifest_sha256']:
        raise ValueError('Frozen source adaptation manifest differs')
    inputs = plan_data(ROOT/cfg['data_manifest'],ROOT/cfg['prefix_schedule']['path'])
    plans = dict(full_schedule=inputs['full_schedule'],development=inputs['development']); identities = {}
    for key,value in plans.items():
        payload = schedule_bytes(value); frozen = unlinked_path(ROOT/cfg[key])
        if frozen.read_bytes() != payload: raise ValueError('Frozen pre-run '+key+' bytes differ')
        identities[key] = dict(path=cfg[key],bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    gaps = []
    if len(inputs['development']['entries']) != 16: gaps.append('Development lacks fixed16 windows; no held fill')
    if not claimed and unlinked_path(ROOT/OUTPUT).exists(): gaps.append('Fixed attempt already exists; no retry')
    return dict(kind='tdyssey.gpc_raw_fulltrain_plan.v1',mode='read_only',ready=not gaps,gaps=gaps,
        config=cfg,config_path=str(path),config_sha256=original.sha_file(path),inputs=inputs,
        plan_identities=identities,inputs_sha256=hashlib.sha256(original.encoded(plans)).hexdigest(),
        real_operations=0,native_resource_queries=0,execution_authorized=False)


preclaim_plan = plan


def verify_execution(prepared, *, authorization_id,reviewed_commit,authorized_config_sha256):
    if not prepared.get('ready') or authorized_config_sha256 != prepared['config_sha256']:
        raise PermissionError('Ready frozen raw plan and exact configSHA required')
    if authorization_id != AUTHORIZATION: raise PermissionError('Exact raw training authorization required')
    head = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True,timeout=5).strip()
    dirty = subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True,timeout=5).strip()
    if head != reviewed_commit or len(head) != 40 or dirty: raise PermissionError('Exact clean reviewed source required')
    value = identity(prepared['config_path'],prepared['config'])
    value.update(source_commit=head,config_sha256=prepared['config_sha256'],mode='raw_fulltrain',
        raw_inputs_sha256=prepared['inputs_sha256'],raw_plan_identities=deepcopy(prepared['plan_identities']),
        max_updates=2328,max_train_examples=2328)
    return value


def preclaim(prepared,binding,python,gpc_root,checkpoint,script,config):
    from .gpc_adaptation_guard import _assets
    python,gpc_root,checkpoint,script,config = map(unlinked_path,(python,gpc_root,checkpoint,script,config))
    if not python.is_file() or script != unlinked_path(ROOT/SCRIPT): raise PermissionError('Fixed interpreter/worker required')
    if plan(config) != prepared: raise ValueError('Raw preparation changed before claim')
    verified = verify_execution(prepared,authorization_id=binding.get('authorization_id'),
        reviewed_commit=binding.get('source_commit'),authorized_config_sha256=binding.get('config_sha256'))
    if any(binding.get(k) != v for k,v in verified.items()): raise PermissionError('Raw binding changed')
    if checkpoint != unlinked_path(ROOT/prepared['config']['sources']['initial']['path']):
        raise ValueError('Pinned initial checkpoint required')
    output = unlinked_path(ROOT/OUTPUT)
    if output.exists(): raise FileExistsError('Fixed raw identity is spent; no retry')
    assets = _assets(gpc_root,checkpoint); assets.update(raw_inputs_sha256=prepared['inputs_sha256'])
    return output,python,gpc_root,checkpoint,script,config,assets


def validate_status_record(value):
    from .gpc_adaptation_guard import monitor_reason
    if not isinstance(value,dict): raise ValueError('Raw status must be an object')
    wall,mono = value.get('at_unix'),value.get('at_monotonic')
    if type(wall) not in (int,float) or type(mono) not in (int,float): raise ValueError('Status clocks missing')
    binding = dict(deadline_unix=wall+1800,deadline_monotonic=mono+1800,
        training_deadline_unix=wall+1200,training_deadline_monotonic=mono+1200)
    reason = monitor_reason(value,binding,pid=value.get('actual_worker_pid'),nonce=value.get('nonce'),
        now_unix=wall,now_monotonic=mono,raw_fulltrain=True)
    if reason: raise ValueError('Invalid raw status frame: '+reason)


def verify_worker(path, *, native_check=True):
    from . import gpc_adaptation_guard as guard
    path = unlinked_path(path); value = guard._read(path); pid = os.getpid()
    directory = unlinked_path(ROOT/OUTPUT)/'guardian'; nonce = value.get('nonce')
    if (path != directory/'handoff.json' or value.get('kind') != 'gpc_raw_fulltrain_guard_handoff.v1'
            or value.get('prepared_ready') is not True or value.get('ownership') != 'native_job_before_ack'
            or value.get('actual_worker_pid') != pid or value.get('launcher_pid') not in (pid,os.getppid())
            or type(value.get('guard_pid')) is not int or value['guard_pid'] <= 0
            or not isinstance(nonce,str) or len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce)
            or value.get('job_memory_bytes') != 3*1024**3 or value.get('kill_on_job_close') is not True
            or value.get('output_dir') != str(directory.parent)
            or value.get('status_path') != str(directory/'evaluation-status.journal')
            or os.environ.get(guard.ENV_HANDOFF) != str(path) or os.environ.get(guard.ENV_NONCE) != nonce
            or os.environ.get(guard.ENV_PARENT) != str(value.get('guard_pid'))):
        raise PermissionError('Exact raw worker native Job ownership required')
    if (guard._read(directory/'worker.json') != dict(pid=pid,parent_pid=os.getppid(),nonce=nonce)
            or guard._read(directory/'worker-ack.json') != dict(pid=pid,nonce=nonce)
            or value.get('ack_path') != str(directory/'worker-ack.json')
            or value.get('script_path') != str(unlinked_path(ROOT/SCRIPT))
            or value.get('script_sha256') != original.sha_file(ROOT/SCRIPT)):
        raise PermissionError('Raw worker PID/ACK/source changed')
    prepared = plan(value['config_path'],claimed=True)
    verified = verify_execution(prepared,authorization_id=value['binding'].get('authorization_id'),
        reviewed_commit=value['binding'].get('source_commit'),authorized_config_sha256=value['binding'].get('config_sha256'))
    if any(value['binding'].get(k) != v for k,v in verified.items()): raise ValueError('Raw worker binding differs')
    if guard._clocks(value['binding'],time.time(),time.monotonic(),raw_fulltrain=True) != value['binding']:
        raise ValueError('Original raw deadlines changed')
    if native_check:
        from .gpc_conditioning_entry import _verify_native_job
        _verify_native_job()
    return value,prepared
