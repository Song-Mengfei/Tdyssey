"""Standard-library-only identity and launch checks. No simulator/model imports."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

KIND = 'tdyssey-contact-background.v1'
LETTERS = 'TLOCXB'
PROTOCOL = '2ef7a9562c7451b502e23a5da08e5110349d25c8c2cfac37886bb990086b3d69'
CHECKPOINT = '23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'
PHYSICS = '768118e17ba69df4cca57c7041eaf84ecfd760466c1315451e134673585120bf'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def bound(item):
    path = Path(item['path']).resolve(strict=True)
    expected = item['sha256']
    if not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('Invalid expected SHA: ' + str(path))
    actual = sha(path)
    if actual != expected:
        raise ValueError('Identity mismatch: ' + str(path) + ' expected=' + expected + ' actual=' + actual)
    return path


def inside(path, directory):
    value = Path(path).resolve()
    if not value.is_relative_to(Path(directory).resolve()):
        raise ValueError('Write path outside pilot: ' + str(value))
    return value


def validate_partition(config, plan):
    slots = plan['slots']
    by_id = {slot['slot_id']: slot for slot in slots}
    if len(by_id) != len(slots):
        raise ValueError('Plan has duplicate slots')
    formal = [slot['slot_id'] for slot in slots if slot['phase'] == 'formal']
    if len(formal) != 360 or set(formal) != set(config['formal_slot_ids']) or len(config['formal_slot_ids']) != 360:
        raise ValueError('Formal release must bind exactly all360 original independent slots')
    shards = config['shards']
    if len(shards) != 2:
        raise ValueError('Exactly two disjoint letter shards required')
    seen_letters, labels = set(), set()
    for shard in shards:
        letters = shard['letters']
        if not letters or len(set(letters)) != len(letters) or set(letters) - set(LETTERS) or seen_letters.intersection(letters):
            raise ValueError('Shard letters must be valid and disjoint')
        seen_letters.update(letters)
        for key in ('formal_label', 'supplement_label'):
            label = shard[key]
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', label) or label in labels:
                raise ValueError('Invalid or repeated run label')
            labels.add(label)
    if seen_letters != set(LETTERS):
        raise ValueError('Two shards must cover all six letters')
    if set(config['supplement_slot_ids']) != set(LETTERS):
        raise ValueError('Freeze all six possible finite supplement sets before launch')
    for letter in LETTERS:
        ids = config['supplement_slot_ids'][letter]
        expected = {slot['slot_id'] for slot in slots if slot['phase'] == 'supplement' and slot['letter'] == letter}
        if len(ids) != 12 or len(set(ids)) != 12 or set(ids) != expected:
            raise ValueError('Supplement must bind the12 original slots per letter: ' + letter)
    return by_id


def validate_calibration_partition(config,plan,base):
    by_id={row['slot_id']:row for row in plan['slots']}
    selected=config['calibration_slot_ids']
    expected={row['slot_id'] for row in plan['slots'] if row['phase']=='calibration-v2'}
    if len(by_id)!=len(plan['slots']) or len(selected)!=18 or len(set(selected))!=18 or set(selected)!=expected:
        raise ValueError('Calibration-v2 must bind exactly its18 original plan slots once')
    if config.get('run_label')!='calibration-v2':
        raise ValueError('Calibration-v2 run label is fixed; no automatic alternate attempt')
    if any(row['phase']!='calibration-v1' for row in base['attempts']):
        raise ValueError('Calibration-v2 base ledger may contain calibration-v1 only')
    if any(sum(by_id[slot]['letter']==letter for slot in selected)!=3 for letter in LETTERS):
        raise ValueError('Calibration-v2 must contain exactly3 original slots per letter')
    return by_id


def validate_config(path, expected_sha, *, require_release, mode='formal'):
    revoked = Path(__file__).resolve().parents[1]/'OLD_MODIFIED_PHYSICS_REVOKED.json'
    if revoked.exists():
        raise RuntimeError('Modified-physics collection authorization revoked by human correction; '
                           'old calibration/formal/supplement cannot restart: '+str(revoked))
    path = Path(path).resolve(strict=True)
    if sha(path) != expected_sha:
        raise ValueError('Runtime config SHA mismatch; no work may start')
    config = read_json(path)
    if config.get('kind') != KIND or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,119}', config.get('run_id', '')):
        raise ValueError('Invalid config kind/run_id')
    if config.get('mode','formal') != mode:
        raise ValueError('Config mode mismatch; calibration cannot release formal')
    if require_release and config.get('released') is not True:
        raise ValueError('Formal release not frozen; check-only is allowed')
    if config.get('training_authorized') is not False or config.get('model_calls') != 0 or config.get('optimizer_updates') != 0:
        raise ValueError('This coordinator only performs CPU physical collection and saved reports')
    if config.get('heartbeat_seconds') != 30 or config.get('auto_retry') is not False:
        raise ValueError('Heartbeat30 seconds and no retries are mandatory')
    root = Path(config['artifact_root']).resolve(strict=True)
    background = inside(root/'background', root)
    config_directory=background if mode=='formal' else background/'calibration-v2'
    if path.parent != config_directory or Path(__file__).resolve().parent != background:
        raise ValueError('Config and coordinator must be in this pilot/background')
    collection = inside(config['collection'], root)
    if collection != root/'collection':
        raise ValueError('Unexpected collection path')
    repo = Path(config['repo']).resolve(strict=True)
    python = Path(config['python_executable']).resolve(strict=True)
    if python.name.lower() != 'python.exe' or python.parent.name != 'Scripts' or not (python.parent.parent/'pyvenv.cfg').is_file():
        raise ValueError('Use an existing Windows venv; no environment creation')
    Path(config['source_root']).resolve(strict=True)
    bindings = config['source_bindings']
    if not bindings:
        raise ValueError('Source bindings are required')
    all_bound = {str(bound(item)).casefold(): item['sha256'] for item in bindings}
    required = [Path(__file__), background/'worker.py',
                repo/'tdyssey/original_contact_pilot.py',
                repo/'tdyssey/native_rgb_contact_safe_physics.py', repo/'tdyssey/native_rgb_original_physics.py']
    if mode=='calibration-v2':
        required += [background/'calibration-worker.py',background/'calibration-launcher.py']
    else:
        required += [background/'launcher.py',repo/'tdyssey/original_contact_pilot_report.py']
    for value in required:
        if str(value.resolve()).casefold() not in all_bound:
            raise ValueError('Missing source binding: ' + str(value))
    if all_bound[str((repo/'tdyssey/native_rgb_contact_safe_physics.py').resolve()).casefold()] != PHYSICS:
        raise ValueError('Physics source changed from released v6')
    names = ['plan', 'physics_gate', 'checkpoint', 'storage_roots', 'resource_policy',
             'calibration_ledger', 'collector_script', 'notification']
    names += ['formal_policy','supplement_policy','overview_script'] if mode=='formal' else ['calibration_policy']
    paths = {key: bound(config[key]) for key in names}
    if config['checkpoint']['sha256'] != CHECKPOINT:
        raise ValueError('Frozen7499 checkpoint changed')
    gate = read_json(paths['physics_gate'])
    if gate.get('passed') is not True or gate.get('training_authorized') is not False:
        raise ValueError('Passed physics gate with training forbidden is required')
    if gate.get('protocol_sha256') != PROTOCOL or gate.get('physics_source_sha256') != PHYSICS:
        raise ValueError('Physics gate protocol/source mismatch')
    if gate.get('checkpoint_sha256') != CHECKPOINT:
        raise ValueError('Physics gate checkpoint mismatch')
    # Gate may bind physical files separately from administrative collector/report revisions.
    # Every bound gate dependency must still exist unchanged; never silently excuse a mismatch.
    for gate_path, digest in gate.get('bindings', {}).items():
        bound(dict(path=gate_path, sha256=digest))
    policy = read_json(paths['resource_policy'])
    if policy.get('shared_cap_bytes') != 3*1024**3 or policy.get('atomic_reserve_bytes') != 256*1024**2:
        raise ValueError('Human shared3GiB cap /256MiB reserve must be retained')
    if not (0 < policy.get('collection_cap_bytes', 0) <= 3*1024**3-256*1024**2):
        raise ValueError('Invalid collection operational cap')
    postprocess=policy.get('postprocess_reserve_bytes')
    if isinstance(postprocess,bool) or not isinstance(postprocess,int) or not 64*1024**2 <= postprocess <= 256*1024**2:
        raise ValueError('Freeze an explicit finite postprocess reserve before collection')
    for policy_name in (('formal_policy', 'supplement_policy') if mode=='formal' else ('calibration_policy',)):
        if not isinstance(read_json(paths[policy_name]).get('changes'), dict):
            raise ValueError('Policy must explicitly freeze strategy changes, including empty changes')
    base = read_json(paths['calibration_ledger'])
    if any(row.get('phase') not in ('calibration-v1', 'calibration-v2') for row in base['attempts']):
        raise ValueError('Frozen base ledger may contain calibration only')
    if base.get('model_calls', 0) != 0 or base.get('optimizer_updates', 0) != 0:
        raise ValueError('Unexpected model/training cost in calibration ledger')
    plan = read_json(paths['plan'])
    if mode=='formal':
        by_id = validate_partition(config, plan)
    else:
        by_id=validate_calibration_partition(config,plan,base)
    cli = paths['collector_script'].read_text(encoding='utf-8')
    for flag in ('--run-label', '--resource-policy', '--stop-file', '--letters'):
        if flag not in cli:
            raise ValueError('Collector CLI is not ready: missing ' + flag)
    return config, paths, by_id


def actual_python_pe():
    if os.name != 'nt':
        raise RuntimeError('This launcher is Windows-only')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetModuleFileNameW(None, buffer, len(buffer))
    if not length or length >= len(buffer):
        raise ctypes.WinError(ctypes.get_last_error())
    return str(Path(buffer.value).resolve())


def in_job(process_handle=None):
    if os.name != 'nt':
        raise RuntimeError('Windows Job evidence required')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    result = wintypes.BOOL()
    handle = kernel.GetCurrentProcess() if process_handle is None else process_handle
    if not kernel.IsProcessInJob(handle, None, ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bool(result.value)


def wait_launcher_exit(pid, timeout_seconds=30):
    """An actual kernel wait, not an 'exiting' text-file assertion."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: already exited PID
            return dict(pid=pid, exited=True, evidence='OpenProcess returned ERROR_INVALID_PARAMETER', at_utc=utc())
        raise ctypes.WinError(error)
    try:
        result = kernel.WaitForSingleObject(handle, int(timeout_seconds*1000))
        if result != 0:
            raise RuntimeError('Launcher has not exited before any control; wait result=' + str(result))
        return dict(pid=pid, exited=True, evidence='WaitForSingleObject WAIT_OBJECT_0', at_utc=utc())
    finally:
        kernel.CloseHandle(handle)


def child_environment(config):
    env = os.environ.copy()
    env.update(__PYVENV_LAUNCHER__=config['python_executable'], PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy',
               PYGAME_HIDE_SUPPORT_PROMPT='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return env


def storage_guard(config, *, extra_bytes=0):
    """Saved/report preflight only; collector owns atomic shared reservations.

    Deduplicate overlapping roots by actual absolute pathname, skip reparse
    entries and never remove data. Does not import the collector or physics.
    """
    resource=read_json(config['resource_policy']['path'])
    roots=read_json(config['storage_roots']['path'])
    roots=roots if isinstance(roots,list) else roots['roots']
    collection=Path(config['collection']).resolve()
    files={}
    missing=[]

    def is_reparse(path):
        try:
            value=path.stat(follow_symlinks=False)
            return path.is_symlink() or bool(getattr(value,'st_file_attributes',0)&0x400)
        except FileNotFoundError:
            return False  # A concurrently renamed atomic temporary is absent.

    for raw in [*roots,config['artifact_root']]:
        root=Path(raw).resolve()
        if not root.exists():
            missing.append(str(root));continue
        if is_reparse(root):
            raise ValueError('Resource root is a reparse point; owner audit required: '+str(root))
        if root.is_file():
            files[str(root).casefold()]=(root,root.stat().st_size)
            continue
        for directory,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=[name for name in dirs if not is_reparse(Path(directory)/name)]
            for name in names:
                path=Path(directory)/name
                if not is_reparse(path):
                    try:
                        files[str(path.resolve()).casefold()]=(path.resolve(),path.stat().st_size)
                    except FileNotFoundError:
                        continue  # No file to count after a concurrent atomic replace.
    shared=sum(size for _,size in files.values())
    own=sum(size for path,size in files.values() if path.is_relative_to(collection))
    atomic=resource.get('per_attempt_atomic_reserve_bytes',16*1024**2)
    reserve=resource['atomic_reserve_bytes']+2*atomic+extra_bytes
    free=shutil.disk_usage(collection).free
    if shared+reserve>resource['shared_cap_bytes'] or free<reserve:
        raise RuntimeError('Shared3GiB /256MiB reserve unavailable before stage; no retry or cap expansion')
    if own+2*atomic+extra_bytes>resource['collection_cap_bytes']:
        raise RuntimeError('Collection operational cap unavailable before stage; no retry or cap expansion')
    return dict(shared_bytes=shared,collection_bytes=own,disk_free_bytes=free,
                reserved_bytes=reserve,stage_extra_bytes=extra_bytes,missing_roots=missing,
                hard_shared_cap_bytes=resource['shared_cap_bytes'],
                collection_cap_bytes=resource['collection_cap_bytes'],at_utc=utc())
