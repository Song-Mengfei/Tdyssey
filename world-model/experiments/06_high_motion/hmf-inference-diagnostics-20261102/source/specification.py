"""Freeze the approved inference-only A+B specification in one serial process.

Importing this module performs no file reads, numerical work or model calls.
The root's sole low-priority process calls freeze; no scans, workers or renders.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
HMF = ROOT.parent / 'high-motion-fit-20261101'
LETTERS = ('O', 'C', 'B')
MIDPOINTS = ('250', '500', '1000', '2000')
MODES = ('real_history', 'autoregressive')
ACTIONS = ('recorded', 'hold', 'alternative')
DEFAULT_B_SEEDS = (2026110201, 2026110202, 2026110203)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _stable(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts], separators=(',', ':')).encode()).hexdigest()


def _legal_action(action):
    a = np.asarray(action)
    return a.shape == (2,) and np.isfinite(a).all() and (a >= 0).all() and (a <= 511).all()


def _write_frozen(path, value):
    from common import atomic_json
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen inference specification would change: ' + str(path))
    else:
        atomic_json(path, value)


def _single_history(parent, role, seed):
    """Choose a saved strong H1 before predictions; no physical replay."""
    with np.load(parent['raw_path'], allow_pickle=False) as z:
        states = z['states']
        passed = z['passed_actions']
        requested = z['requested_actions']
        valid = z['control_valid']
        rgb = z['rgb']
        rgb_count = len(rgb)
    candidates = []
    for phase, motion in enumerate(parent['target_motion']):
        actual_strong = motion['translation_imagepx'] >= 1 or abs(motion['signed_rotation_deg']) >= 3
        if bool(motion['strong']) != bool(actual_strong):
            raise ValueError('Saved strong H1 annotation differs from frozen thresholds')
        if not actual_strong:
            continue
        anchor = int(parent['anchor']) + phase
        if anchor < 3 or anchor + 1 >= rgb_count:
            continue
        if not valid[anchor-3:anchor+1].all():
            continue
        if not np.array_equal(passed[anchor-3:anchor+1], requested[anchor-3:anchor+1]):
            raise ValueError('Saved H1 actions were transformed')
        if not all(_legal_action(a) for a in passed[anchor-3:anchor+1]):
            continue
        hold = states[anchor, :2].copy()
        if not _legal_action(hold):
            continue  # No clipping: select a compatible saved strong history.
        candidates.append((phase, anchor, hold))
    if not candidates:
        raise ValueError('No clean strong H1 with an unclipped legal hold: ' + parent['sample_id'])
    # Earliest qualifying time keeps contact establishment visible and is fixed
    # solely by ground truth, never by a model output or counterfactual result.
    phase, anchor, hold = candidates[0]
    row = deepcopy(parent)
    row.update(sample_id='IDB-' + parent['family_id'] + '-a' + str(anchor),
        original_sample_id=parent['sample_id'], parent_anchor=int(parent['anchor']),
        parent_phase=phase, anchor=anchor, horizon=1, B_role=role,
        original_selection_role=parent['selection_role'],
        selection_role='B_action_sensitivity', true_H1_motion=deepcopy(parent['target_motion'][phase]),
        selection_seed=seed, selection_rule='First saved strong H1 with clean history and legal raw hold; recorded/hold equality does not filter histories; no output-based selection',
        actions=dict(recorded=passed[anchor].tolist(), hold=hold.tolist()),
        recorded_hold_identical=bool(np.array_equal(hold, passed[anchor])),
        past_actions_absolute=passed[anchor-3:anchor].tolist(),
        original_target_has_GT=True, alternative_actions_have_counterfactual_GT=False)
    history = rgb[anchor-3:anchor+1]
    row['history4_sha256'] = array_sha(history)
    row['past3_actions_sha256'] = array_sha(passed[anchor-3:anchor])
    return row


def build_B_rows(original_spec, selection, seed):
    """Twelve histories, one failure and three distinct trained families/letter."""
    original = {r['sample_id']: r for r in original_spec['rows']}
    chosen = []
    for letter in LETTERS:
        failures = [r for r in original_spec['rows'] if r['letter'] == letter and r['selection_role'] == 'original_failure']
        if len(failures) != 1:
            raise ValueError('Expected one original failure per letter')
        selected = [_single_history(failures[0], 'original_failure', seed)]
        for role in ('cw', 'ccw', 'translation'):
            pool = [r for r in selection['train48'] if r['letter'] == letter and r['quota'] == role]
            pool.sort(key=lambda r: _stable(seed, letter, role, r['family_id'], r['anchor']))
            if not pool:
                raise ValueError('Missing original trained quota: ' + letter + '/' + role)
            # Every selected train48 row already has >=4 strong H1. Any source
            # incompatibility is a hard error; no silent replacement/retry.
            selected.append(_single_history(original[pool[0]['sample_id']], role, seed))
        if len({r['family_id'] for r in selected}) != 4:
            raise ValueError('B history families are not distinct')
        chosen.extend(selected)
    # Alternatives come only from already selected histories' actual passed
    # actions. Prefer same-letter sources; no additional trajectory scan.
    for row in chosen:
        pool = [x for x in chosen if x['sample_id'] != row['sample_id']]
        pool.sort(key=lambda x: (x['letter'] != row['letter'], _stable(seed, 'alternative', row['sample_id'], x['sample_id'])))
        source = next((x for x in pool if not np.array_equal(x['actions']['recorded'], row['actions']['recorded'])
            and not np.array_equal(x['actions']['recorded'], row['actions']['hold'])), None)
        if source is None:
            raise ValueError('No distinct already saved legal alternative action')
        row['actions']['alternative'] = deepcopy(source['actions']['recorded'])
        row['alternative_source'] = dict(sample_id=source['sample_id'], original_sample_id=source['original_sample_id'],
            family_id=source['family_id'], raw_path=source['raw_path'], raw_sha256=source['raw_sha256'],
            metadata_path=source.get('metadata_path', str(Path(source['raw_path']).with_suffix('.json'))),
            metadata_sha256=source.get('metadata_sha256'),
            control_index=source['anchor'], requested_equals_passed=True, control_valid=True,
            provenance='An actually saved passed absolute XY target; no counterfactual ground truth')
    return chosen


def freeze(root, config):
    """Called only by the unique serial low-priority preparation process."""
    root = Path(root).resolve()
    hmf = Path(config.get('hmf_root', HMF)).resolve()
    original = read(hmf / 'evaluation-spec.json')
    selection = read(hmf / 'selection/index.json')
    summary = read(hmf / 'report/summary.json')
    if sha(hmf / 'selection/index.json') != original['selection_index_sha256']:
        raise ValueError('HMF frozen selection differs')
    completion = read(hmf / 'background/completion.json')
    if not completion['completed'] or completion['failed']:
        raise ValueError('Source HMF did not complete successfully')
    batches = [deepcopy(b) for b in original['batches'] if 12 <= b['batch_id'] <= 18]
    if [b['batch_id'] for b in batches] != list(range(12, 19)) or [len(b['rows']) for b in batches] != [4]*6 + [3]:
        raise ValueError('Original A B4/B3 layout differs')
    rows = [r for b in batches for r in b['rows']]
    if Counter(r['selection_role'] for r in rows) != Counter(dict(seen_val=24, original_failure=3)):
        raise ValueError('A requires original24seen-val plus3failures')
    checkpoint_map = {label: deepcopy(summary['checkpoint_identity'][label]) for label in ('0', *MIDPOINTS, 'final')}
    checkpoint_map['E2'] = checkpoint_map.pop('0')
    checkpoint_map['E2']['label'] = 'E2'
    for checkpoint in checkpoint_map.values():
        checkpoint['bytes'] = Path(checkpoint['path']).stat().st_size
    baseline = read(hmf / 'evaluation/0/primary/metrics.json')
    final = read(hmf / 'evaluation/final/primary/metrics.json')
    wanted = set(range(12, 19))
    expected_inputs = [p for p in baseline['input_evidence'] if p['batch_id'] in wanted]
    expected_noise = [p for p in baseline['noise_evidence'] if p['batch_id'] in wanted]
    for field, values in [('input_evidence', expected_inputs), ('noise_evidence', expected_noise)]:
        lookup = {(p['mode'], p['batch_id']): p for p in final[field]}
        if any(lookup[(p['mode'], p['batch_id'])] != p for p in values):
            raise ValueError('Existing0/final A pairing differs')
    a = dict(kind='HMF-inference-diagnostics-A.v1', rows=rows, batches=batches,
        checkpoint_labels=list(MIDPOINTS), input_modes=list(MODES), horizons=list(range(1,17)),
        expected_input_evidence=expected_inputs, expected_noise_evidence=expected_noise,
        existing_HMF_metrics={label:str((hmf/f'evaluation/{label}/primary/metrics.json').resolve()) for label in ('0', *MIDPOINTS, 'final')},
        original_train12_batch_ids=deepcopy(original['panel_batch_ids']),
        predicted_frames=3456, sampler_batch_calls=896, image_denoiser_forwards=10368,
        semantics='Only missing middle-checkpoint27cases are newly predicted. Original0/final and fixedtrain12 metrics are reused without rescoring.')
    seed = int(config.get('B_selection_seed', 2026110200))
    b_rows = build_B_rows(original, selection, seed)
    seeds = list(config.get('B_noise_seeds', DEFAULT_B_SEEDS))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('Exactly3 distinct frozen B noise seeds required')
    b_batches = [dict(batch_id=i, letter=letter, rows=[r for r in b_rows if r['letter']==letter]) for i,letter in enumerate(LETTERS)]
    noise_sets = [dict(name='seed'+str(i+1), root_seed=int(seed_value),
        batch_seeds={str(batch['batch_id']):(int(_stable(seed_value,'B',batch['batch_id'])[:8],16) % 2**32) for batch in b_batches}) for i,seed_value in enumerate(seeds)]
    b = dict(kind='HMF-inference-diagnostics-B.v1', rows=b_rows, batches=b_batches,
        checkpoint_labels=['E2','final'], actions=list(ACTIONS), noise_sets=noise_sets,
        selection_seed=seed, predicted_frames=216, sampler_batch_calls=54, image_denoiser_forwards=648,
        current_action_index=3, past_action_indices=[0,1,2],
        semantics='Only current fourth XY action changes. All3 action branches share identical saved four-frame history/past3actions/slot/actualepsilon. Recorded/hold equality is disclosed, not a history filter; that degenerate contrast cannot establish weak action influence. Keep216predictions without replacement. Alternative and hold outputs have no counterfactual GT and cannot be scored as physical correctness.')
    used = {r['raw_path']:r for r in rows+b_rows}
    bindings=[]
    def bind(path, expected=None):
        path=Path(path).resolve()
        if any(x['path']==str(path) for x in bindings):
            return
        digest=sha(path)
        if expected is not None and digest!=expected:
            raise ValueError('An actually referenced source changed: '+str(path))
        bindings.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size))
    for row in used.values():
        bind(row['raw_path'], row['raw_sha256'])
        meta=Path(row.get('metadata_path',str(Path(row['raw_path']).with_suffix('.json'))))
        bind(meta, row.get('metadata_sha256'))
    for cp in checkpoint_map.values():
        bind(cp['path'],cp['sha256'])
    for path in [hmf/'evaluation-spec.json',hmf/'selection/index.json',hmf/'report/summary.json',hmf/'GT-measurement-calibration.json',hmf/'fit-standard.json']:
        bind(path)
    for path in a['existing_HMF_metrics'].values():
        bind(path)
    a['source_HMF_spec_sha256']=sha(hmf/'evaluation-spec.json')
    a['GT_calibration_path']=str((hmf/'GT-measurement-calibration.json').resolve())
    b['GT_calibration_path']=a['GT_calibration_path']
    _write_frozen(root/'A-spec.json',a)
    _write_frozen(root/'B-spec.json',b)
    _write_frozen(root/'checkpoint-map.json',checkpoint_map)
    receipt=dict(A_spec=str(root/'A-spec.json'),B_spec=str(root/'B-spec.json'),
        checkpoint_map=str(root/'checkpoint-map.json'),A_spec_sha256=sha(root/'A-spec.json'),B_spec_sha256=sha(root/'B-spec.json'),
        planned_model_loads=6,predicted_frames=3672,image_denoiser_forwards=11016,bindings=bindings,
        model_calls_during_freeze=0,physics_calls=0,full_library_scan=False,
        CPU_policy='Run only in the unique below-normal serial process with all numerical thread limits1. No independent replay/no model warmup/no media workers.')
    _write_frozen(root/'specification-receipt.json',receipt)
    return receipt
