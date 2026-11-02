"""Fixed clean-history conditioning authority; no tensor-runtime import."""
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
CONFIG_RELATIVE = 'configs/gpc_conditioning_contract02.json'
CONFIG = ROOT/CONFIG_RELATIVE
CANDIDATE = 'gpc-conditioning-contract-diagnostic-02'
AUTHORIZATION = 'auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02'
OUTPUT = 'artifacts/gpc-conditioning-contract-diagnostic-20261026/'+AUTHORIZATION
SCRIPT = 'scripts/gpc_conditioning_contract.py'


def configuration_template():
    from .gpc_checkpoint_evaluation import configuration_template as evaluation_template
    from .gpc_conditioning_contract import CONTRACT_SPEC
    value = evaluation_template()
    for key in ('benchmark_manifest', 'evaluation', 'screens', 'retention'):
        value.pop(key)
    value.update(kind='tdyssey.gpc_conditioning_contract_diagnostic.v1',
        candidate_id=CANDIDATE, output_dir=OUTPUT, conditioning=deepcopy(CONTRACT_SPEC),
        forecast=dict(input_windows=8, seed=5478, mode='AR', horizon=1))
    value['limits'].update(wall_seconds=300, eval_batches=0, eval_examples=0,
        forecasts=32, euler_steps=96, output_bytes=64*1024**2)
    return value


def validate_config(value):
    if not isinstance(value, dict) or value != configuration_template():
        raise ValueError('Configuration differs from the fixed None versus .002 diagnostic')
    return deepcopy(value)


def identity(path, cfg, binding=None):
    path = unlinked_path(path); expected = unlinked_path(ROOT/CONFIG_RELATIVE)
    if path != expected or cfg.get('candidate_id') != CANDIDATE or cfg.get('output_dir') != OUTPUT:
        raise PermissionError('Only the fixed conditioning config/candidate/output is permitted')
    result = dict(candidate_id=CANDIDATE, output_dir=OUTPUT, authorization_id=AUTHORIZATION,
                  config_path=str(expected))
    if binding is not None:
        if binding.get('authorization_id') != AUTHORIZATION:
            raise PermissionError('Exact conditioning execution authorization is required')
        for key, value in result.items():
            if key in binding and binding[key] != value:
                raise PermissionError('Conditioning authority differs: '+key)
    return result


def plan(config_path=CONFIG, *, claimed=False):
    from .gpc_evaluation_assets import verify_evaluation_assets, check_prepared_assets
    from .gpc_conditioning_data import prepare_training_windows
    path = unlinked_path(config_path)
    cfg = validate_config(json.loads(path.read_bytes())); identity(path, cfg)
    inputs = prepare_training_windows(ROOT)
    assets = verify_evaluation_assets(repo_root=ROOT); check_prepared_assets(assets)
    if cfg['sources'] != assets['sources']:
        raise ValueError('Conditioning checkpoint provenance differs')
    gaps = []
    if not claimed and unlinked_path(ROOT/OUTPUT).exists():
        gaps.append('Fixed conditioning account is already spent; no resume or alternate output')
    return dict(kind='tdyssey.gpc_conditioning_contract_plan.v1', mode='read_only',
        config=cfg, config_path=str(path), config_sha256=original.sha_file(path),
        ready=not gaps, gaps=gaps, inputs=inputs,
        inputs_sha256=hashlib.sha256(original.encoded(inputs)).hexdigest(),
        assets=assets, assets_sha256=hashlib.sha256(original.encoded(assets)).hexdigest(),
        real_operations=0, native_resource_queries=0, execution_authorized=False)


def verify_execution(prepared, *, authorization_id, reviewed_commit, authorized_config_sha256):
    from .gpc_conditioning_contract import CONTRACT_SHA256
    if not prepared.get('ready') or authorized_config_sha256 != prepared.get('config_sha256'):
        raise PermissionError('Ready reviewed conditioning plan and exact config SHA are required')
    result = identity(prepared['config_path'], prepared['config'], dict(authorization_id=authorization_id))
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, timeout=5).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True, timeout=5).strip()
    if dirty or head != reviewed_commit or len(reviewed_commit) != 40:
        raise PermissionError('A clean exact reviewed source commit is required')
    result.update(source_commit=head, config_sha256=prepared['config_sha256'],
        conditioning_inputs_sha256=prepared['inputs_sha256'],
        evaluation_assets_sha256=prepared['assets_sha256'], conditioning_contract_sha256=CONTRACT_SHA256,
        mode='conditioning_only', source_checkpoint_update=500, max_updates=0, max_train_examples=0)
    return result


def preclaim(prepared, binding, python, gpc_root, checkpoint, script, config):
    from .gpc_adaptation_guard import _assets
    python, gpc_root, checkpoint, script, config = map(unlinked_path, (python, gpc_root, checkpoint, script, config))
    if not python.is_file() or script != unlinked_path(ROOT/SCRIPT) or not script.is_file():
        raise PermissionError('Explicit interpreter and fixed conditioning worker required')
    fresh = plan(config)
    if fresh != prepared:
        raise ValueError('Conditioning preparation changed before claim')
    verified = verify_execution(prepared, authorization_id=binding.get('authorization_id'),
        reviewed_commit=binding.get('source_commit'), authorized_config_sha256=binding.get('config_sha256'))
    if any(binding.get(key) != value for key, value in verified.items()):
        raise PermissionError('Conditioning binding differs before claim')
    if checkpoint != unlinked_path(ROOT/prepared['config']['sources']['frozen']['path']):
        raise ValueError('Explicit initial checkpoint differs from the pinned source')
    output = unlinked_path(ROOT/OUTPUT)
    if output.exists():
        raise FileExistsError('Conditioning account already spent; no retry')
    assets = _assets(gpc_root, checkpoint)
    assets.update(evaluation_assets_sha256=prepared['assets_sha256'],
                  conditioning_inputs_sha256=prepared['inputs_sha256'])
    return output, python, gpc_root, checkpoint, script, config, assets


def validate_status_record(value):
    """Every complete append frame is checked; only the latest frame is aged."""
    from .gpc_adaptation_guard import monitor_reason
    if not isinstance(value, dict):
        raise ValueError('Conditioning status must be an object')
    wall, mono = value.get('at_unix'), value.get('at_monotonic')
    if type(wall) not in (int, float) or type(mono) not in (int, float):
        raise ValueError('Conditioning status clocks are missing')
    reason = monitor_reason(value, dict(deadline_unix=wall+300, deadline_monotonic=mono+300),
        pid=value.get('actual_worker_pid'), nonce=value.get('nonce'), now_unix=wall,
        now_monotonic=mono, conditioning_only=True)
    if reason:
        raise ValueError('Invalid complete conditioning status: '+reason)


def verify_worker(path, *, native_check=True):
    from . import gpc_adaptation_guard as guard
    path = unlinked_path(path); value = guard._read(path); pid = os.getpid()
    directory = unlinked_path(ROOT/OUTPUT)/'guardian'
    nonce = value.get('nonce')
    if (path != directory/'handoff.json' or value.get('kind') != 'gpc_conditioning_guard_handoff.v1'
            or value.get('prepared_ready') is not True or value.get('ownership') != 'native_job_before_ack'
            or value.get('actual_worker_pid') != pid or value.get('launcher_pid') not in (pid, os.getppid())
            or type(value.get('guard_pid')) is not int or value['guard_pid'] <= 0
            or not isinstance(nonce, str) or len(nonce) != 32 or any(c not in '0123456789abcdef' for c in nonce)
            or value.get('job_memory_bytes') != 3*1024**3 or value.get('kill_on_job_close') is not True
            or value.get('output_dir') != str(directory.parent)
            or value.get('status_path') != str(directory/'evaluation-status.journal')
            or os.environ.get(guard.ENV_HANDOFF) != str(path) or os.environ.get(guard.ENV_NONCE) != nonce
            or os.environ.get(guard.ENV_PARENT) != str(value.get('guard_pid'))):
        raise PermissionError('No exact conditioning worker ownership handoff')
    if (guard._read(directory/'worker.json') != dict(pid=pid, parent_pid=os.getppid(), nonce=nonce)
            or guard._read(directory/'worker-ack.json') != dict(pid=pid, nonce=nonce)
            or value.get('ack_path') != str(directory/'worker-ack.json')
            or value.get('script_path') != str(unlinked_path(ROOT/SCRIPT))
            or value.get('script_sha256') != original.sha_file(ROOT/SCRIPT)):
        raise PermissionError('Conditioning worker PID/ACK/source changed')
    prepared = plan(value['config_path'], claimed=True)
    verified = verify_execution(prepared, authorization_id=value['binding'].get('authorization_id'),
        reviewed_commit=value['binding'].get('source_commit'), authorized_config_sha256=value['binding'].get('config_sha256'))
    if any(value['binding'].get(key) != item for key, item in verified.items()):
        raise ValueError('Conditioning source/config/input/provenance binding changed')
    if guard._clocks(value['binding'], time.time(), time.monotonic(), conditioning_only=True) != value['binding']:
        raise ValueError('Original conditioning deadlines changed')
    if native_check:
        _verify_native_job()
    return value, prepared


def _verify_native_job():
    import ctypes
    from ctypes import wintypes
    from . import policy_memory
    kernel, _ = policy_memory._libraries(); inside = wintypes.BOOL()
    kernel.GetCurrentProcess.argtypes = []; kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)) or not inside.value:
        raise RuntimeError('Conditioning worker is outside the owned Job')
    limits = policy_memory._JobExtendedLimits()
    query = kernel.QueryInformationJobObject
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    query.restype = wintypes.BOOL
    if (not query(None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None)
            or limits.JobMemoryLimit != 3*1024**3 or limits.BasicLimitInformation.LimitFlags&0x2200 != 0x2200):
        raise RuntimeError('Conditioning worker Job limits differ')
