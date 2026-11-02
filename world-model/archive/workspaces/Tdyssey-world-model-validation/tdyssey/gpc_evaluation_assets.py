"""Fixed evaluation sources. Import performs no I/O or tensor-runtime import.

The future loader is not execution authority: callers must first reserve a load
in a separately authorized evaluation-only account.
"""
import hashlib
import json
import math
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'artifacts/gpc-supervised-adaptation-pilot03-20261026'
INIT_SHA = '4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892'
SELECTED_SHA = '0bfd1a944c8fb01555a7ad2d01aab6f526a4805e2bb5239a96703d356c5741d3'
CONFIG_SHA = '5bc6c88efd4023dc27c3718f57655791e0271a4bc8ae8241069fef98352e4937'
SCHEDULE_SHA = 'f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a'
ORDER_SHA = '54edfd533df01652b3208996f80bf88e9f19f9806bb01ddcde93adbcaafc170d'
RULE_SHA = '909d35e2816e317ec0766291632b91cef5e0bc1545bc7da176572176637a84bc'
SPEC_SHA = 'e39bbd1f4205b369cbaaed4ac2e37dfc5d87be8e896598198e1bcb03f4931df3'
BOUNDARY_SHA = '3dc6b6e3439d7b700bcd2f8e9bc826af9dfd40e461c2d8ba6a1d7cefcc4bbe88'
WEIGHTS = {
    'frozen': ('artifacts/frozen-wm-assets/denoiser.pth', 35043916, INIT_SHA),
    'adapted': (SOURCE+'/adapted-latest.pt', 105134565, SELECTED_SHA),
}
METADATA = {
    'claim': (SOURCE+'/attempt.json', '03358e8daab32a9b28d04000a8035e334ad3953cb9eb47b02216a95fba164213'),
    'attempt': (SOURCE+'/ledger/attempt.json', '47b4c2013a97e2774109b6927b8105bcea0ed7fca20dd393d509497100f768f4'),
    'journal': (SOURCE+'/ledger/highwater.jsonl', '4be11a5f56f2679506981ebe62b5ea316e0aa2bffb08f6a8662d426a171df4a6'),
    'config': ('configs/gpc_supervised_adaptation_pilot03.json', CONFIG_SHA),
    'schedule': ('artifacts/wm-sampling-b-only-20261026/training-schedule.json', SCHEDULE_SHA),
    'schedule_binding': (SOURCE+'/training-schedule-binding.json', 'f495154662d64b221ee4cb32c3ac469dd9d86065cf7af529a404c5d05d936019'),
    'model_audit': ('artifacts/frozen-wm-assets/resolved-v2/asset-audit.json', 'd40d8ff937f54bd2352a1c7620f1f6dab930790edddd4c2db1108eb2a4db6689'),
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def source_declarations():
    result = {role: dict(path=path, bytes=size, sha256=digest)
              for role, (path, size, digest) in WEIGHTS.items()}
    result['adapted']['logical_update'] = 500
    return result


def load_model_only_state(role, assets, *, torch_module=None):
    """Future reserved load: model tensors only, no optimizer/RNG/account restore.

    No new pickle interpreter is introduced. torch.load still parses the105MB
    container metadata and creates optimizer/RNG tensor wrappers. mmap avoids
    eager storage copies, not metadata or guaranteed private-memory growth. The
    container is cleared promptly; returned model tensors keep its mapping alive.
    Runtime resource caps remain authoritative. No35MB rollback clone is made;
    callers discard a failed newly constructed model and stop.
    """
    check_prepared_assets(assets)
    if role not in WEIGHTS:
        raise ValueError('Evaluation role must be frozen or adapted')
    item = assets['weights'][role]
    if _file_identity(Path(item['path'])) != item:
        raise ValueError('Weight file changed before model-only load')
    if torch_module is None:
        import torch as torch_module
    payload = torch_module.load(item['path'], map_location='cpu', weights_only=True, mmap=True)
    state = None
    try:
        if role == 'adapted':
            if (not isinstance(payload, dict) or payload.get('kind') != 'gpc_adaptation_checkpoint.v1'
                    or type(payload.get('logical_steps')) is not int or payload['logical_steps'] != 500
                    or payload.get('attempt') != assets['source_attempt']
                    or payload.get('config_identity') != assets['source_attempt']['binding']
                    or payload.get('budget') != assets['selection_boundary']
                    or payload.get('extra_state', {}).get('training_schedule') != _schedule_identity(500)
                    or payload.get('extra_state', {}).get('next_example') != 500):
                raise ValueError('Loaded selected source provenance differs')
            state = payload.pop('model')
            payload.clear()
        else:
            state = payload
        if not isinstance(state, dict) or set(state) != set(assets['model_spec']):
            raise ValueError('Model state keys differ')
        for name, spec in assets['model_spec'].items():
            tensor = state[name]
            if (not isinstance(tensor, torch_module.Tensor) or list(tensor.shape) != spec['shape']
                    or str(tensor.dtype) != spec['dtype'] or getattr(tensor.device, 'type', None) != 'cpu'
                    or tensor.layout != torch_module.strided or tensor.requires_grad
                    or tensor.numel() != math.prod(spec['shape'])
                    or not bool(torch_module.isfinite(tensor).all())):
                raise ValueError('Model tensor shape/dtype/finite differs: '+name)
        return state
    except BaseException:
        if isinstance(state, dict):
            state.clear()
        raise
    finally:
        if role == 'adapted' and isinstance(payload, dict):
            payload.clear()
        del payload


def _file_identity(path):
    path = Path(path)
    before = path.stat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 1024:
        raise ValueError('Source is not ordinary: '+str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino):
        raise ValueError('Source changed while hashing: '+str(path))
    return dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest())


def _schedule_identity(cursor):
    return dict(sha256=SCHEDULE_SHA, order_sha256=ORDER_SHA, rule_sha256=RULE_SHA, cursor=cursor)


def _selection_from_journal(raw, entries):
    """Read a spent source as evidence; never resume or mutate its account."""
    previous = '0'*64
    reserved = completed = 0
    pending = selected = boundary = None
    last_wall = last_mono = -math.inf
    for sequence, line in enumerate(raw.splitlines(keepends=True), 1):
        row = json.loads(line)
        if (not line.endswith(b'\n') or row['sequence'] != sequence or row['previous_sha256'] != previous
                or (sequence == 1 and row['kind'] != 'created')):
            raise ValueError('Source journal chain differs')
        wall, mono = row['at_unix'], row['at_monotonic']
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (wall, mono)) or wall < last_wall or mono < last_mono:
            raise ValueError('Source journal clocks differ')
        previous = _sha(line)
        last_wall, last_mono = wall, mono
        kind, data = row['kind'], row['data']
        if kind == 'reserve_update':
            meta = data['metadata']
            if (selected is not None or pending is not None or reserved >= 500 or data['batch_size'] != 1
                    or meta.get('training_schedule') != _schedule_identity(reserved)
                    or meta.get('next_cursor') != reserved+1
                    or meta.get('dataset_indices') != [entries[reserved]['dataset_index']]):
                raise ValueError('Source update differs from frozen500 schedule')
            reserved += 1
            pending = sequence
        elif kind == 'complete_update':
            if pending is None or data['operation_id'] != pending:
                raise ValueError('Source update completion differs')
            completed += 1
            pending = None
        elif kind == 'checkpoint_committed' and data['sha256'] == SELECTED_SHA:
            if reserved != 500 or completed != 500 or pending is not None or selected is not None:
                raise ValueError('Selected source boundary is not complete500')
            boundary = data['boundary']
        elif kind == 'checkpoint_selected':
            if selected is not None or boundary is None or data['sha256'] != SELECTED_SHA:
                raise ValueError('Source selection differs')
            selected = sequence
    if selected is None or reserved != 500 or completed != 500 or pending is not None:
        raise ValueError('Source has no sole completed500 selection')
    return boundary, selected


def check_prepared_assets(assets):
    """Cheap I/O-free rejection before a new account or model-load reservation."""
    if (not isinstance(assets, dict) or assets.get('kind') != 'gpc_eval_assets.v1'
            or type(assets.get('source_logical_steps')) is not int or assets['source_logical_steps'] != 500
            or type(assets.get('new_account_updates')) is not int or assets['new_account_updates'] != 0):
        raise ValueError('Evaluation source/new-account steps differ')
    root = Path(assets['repository_root'])
    if not root.is_absolute() or assets.get('sources') != source_declarations():
        raise ValueError('Fixed relative source declarations differ')
    if set(assets['weights']) != set(WEIGHTS) or set(assets['metadata']) != set(METADATA):
        raise ValueError('Source declarations differ')
    for role, (relative, size, digest) in WEIGHTS.items():
        if assets['weights'][role] != dict(path=str(root/relative), bytes=size, sha256=digest):
            raise ValueError('Fixed weight identity differs: '+role)
    for name, (relative, digest) in METADATA.items():
        item = assets['metadata'][name]
        if item['path'] != str(root/relative) or item['sha256'] != digest or type(item['bytes']) is not int or item['bytes'] <= 0:
            raise ValueError('Fixed source metadata identity differs: '+name)
    if _sha(_canonical(assets['source_attempt'])+b'\n') != METADATA['attempt'][1]:
        raise ValueError('Embedded source account differs')
    if _sha(_canonical(assets['selection_boundary'])) != BOUNDARY_SHA:
        raise ValueError('Embedded selected500 boundary differs')
    if _sha(_canonical(assets['model_spec'])) != SPEC_SHA or len(assets['model_spec']) != 298:
        raise ValueError('Pinned key/shape/dtype specification differs')
    if type(assets['selection_sequence']) is not int or assets['selection_sequence'] != 1014:
        raise ValueError('Sole source selection event differs')


def verify_evaluation_assets(repo_root=ROOT):
    """Read fixed bytes/JSON only; no held data or tensor-runtime import."""
    root = Path(repo_root).resolve()
    weights, metadata, values = {}, {}, {}
    for role, (relative, size, digest) in WEIGHTS.items():
        item = _file_identity(root/relative)
        if item['bytes'] != size or item['sha256'] != digest:
            raise ValueError('Fixed weight bytes/SHA differ: '+role)
        weights[role] = item
    for name, (relative, digest) in METADATA.items():
        path = root/relative
        item = _file_identity(path)
        raw = path.read_bytes()
        if item['sha256'] != digest or _sha(raw) != digest:
            raise ValueError('Fixed metadata bytes/SHA differ: '+name)
        metadata[name] = item
        values[name] = raw if name == 'journal' else json.loads(raw)
    attempt, claim = values['attempt'], values['claim']
    if (claim['binding'] != attempt['binding'] or attempt['binding']['config_sha256'] != CONFIG_SHA
            or attempt['binding']['training_schedule_sha256'] != SCHEDULE_SHA
            or values['schedule_binding'] != dict(_schedule_identity(0), source_manifest_sha256=attempt['binding']['data_sha256'])):
        raise ValueError('Source claim/config/schedule binding differs')
    if len(values['schedule']['entries']) != 500:
        raise ValueError('Source schedule is not500 entries')
    boundary, selected = _selection_from_journal(values['journal'], values['schedule']['entries'])
    audit = next(row for row in values['model_audit']['checkpoint_records'] if row['sha256'] == INIT_SHA)
    spec = {name: {field: row[field] for field in ('shape', 'dtype')}
            for name, row in audit['tensor_state']['tensors'].items()}
    result = dict(kind='gpc_eval_assets.v1', repository_root=str(root), sources=source_declarations(),
        weights=weights, metadata=metadata, source_logical_steps=500, new_account_updates=0,
        source_attempt=attempt, selection_boundary=boundary, selection_sequence=selected, model_spec=spec,
        loading_scope='weights_only+mmap parses full container metadata/tensor wrappers; no private-memory guarantee')
    check_prepared_assets(result)
    return result
