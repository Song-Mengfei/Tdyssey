"""Frozen R3B paired RGB evaluation; imports never load Torch or a model.

Future RGB and full object masks belong exclusively to offline scoring. Rollout
accepts only four RGB frames and four executed, normalized actions. Raw sampler
float32 values are retained once, including finite out-of-range values.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .native_rgb_r2c0_evaluation import normalize_actions, paired_seed
from .native_rgb_r3a_data import RecordedStateRenderer, load_reference, _trajectory_arrays

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/native-rgb-supervised-20261029'
MANIFEST = BASE / 'r3a/manifest.json'
CATALOGUE = BASE / 'r3a/data-interface/window-catalogue.npz'
STRESS = BASE / 'r3a/r3b-proposal/evaluation-indices.json'
OLD_FOLDER = BASE / 'r2c3/run/evaluation/update6000/steps3'
OLD_MANIFEST = BASE / 'r2c0/data/manifest.json'
OLD_INDICES = BASE / 'r2c0/evaluation/indices.json'
CALIBRATION = ROOT / 'artifacts/native-rgb-20261028/frozen-renderer-calibration.json'
CHECKPOINT = BASE / 'r2c3/run/last.pt'
OLD4 = ['R2B-new-25-O-left-bottom:12', 'R2B-new-48-B-top:5',
        'R2B-new-28-C-left:6', 'R2B-new-38-X-right:13']
SAMPLER = dict(num_steps_denoising=3, sigma_min=.002, sigma_max=5., rho=7,
               order=1, s_churn=0., s_cond=0.)
SEED = 5478


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json(path, value):
    from .native_rgb_training import atomic_json
    atomic_json(path, value)


def _uniform_anchor(family, anchors):
    """Independent local PRNG, uniform legal choice; never consults motion."""
    local = int(_digest(dict(domain='r3b-natural-family-anchor.v1', seed=SEED, family=family)), 16)
    return int(random.Random(local).choice(list(anchors)))


def _binding(path):
    return dict(path=str(Path(path).resolve()), sha256=_sha(path), bytes=Path(path).stat().st_size)


def freeze_spec(stage, *, physics_source=None):
    """Freeze all windows, memberships, context batches and seeds before t0.

    This reads numeric catalogues and provenance, without renderer/model calls.
    Re-entry must match the exact original spec; it cannot choose new windows.
    """
    stage = Path(stage).resolve()
    manifest, source_root, entries = load_reference(MANIFEST)
    if _sha(MANIFEST) != '0e91e1412f937c7d8f39473243a27a3720317a07474e730a510e628d040c0a19':
        raise ValueError('Approved R3A manifest changed')
    if _sha(CHECKPOINT) != 'c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148':
        raise ValueError('Approved original6000 checkpoint changed')
    if _sha(CATALOGUE) != 'c80581a0d7c4aee7a377a28f9833a4355307a08ef53d2a8759ca1a745d2bae6a':
        raise ValueError('Frozen full legal catalogue changed')
    with np.load(CATALOGUE, allow_pickle=False) as archive:
        catalogue = {k: archive[k].copy() for k in archive.files if k != 'selected_h1_training_indices'}
    lookup = {h: {(int(t), int(a)): i for i, (t, a) in enumerate(zip(catalogue[f'h{h}_trajectory_index'], catalogue[f'h{h}_anchor']))} for h in (1, 4)}
    rows = OrderedDict()

    def add_new(index, anchor, cohort, diagnostic_role=None):
        e = entries[index]; key = f"{e['trajectory_id']}:{anchor}"
        if e['split'] != 'seen_val' or (index, anchor) not in lookup[4]:
            raise ValueError('Only legal new seen-validation H4 windows allowed')
        if key not in rows:
            row = dict(window_id=key, record_source='R3A', trajectory_index=index,
                       trajectory_id=e['trajectory_id'], family_id=e['family_id'], letter=e['letter'],
                       anchor=anchor, source_file=e['file'], source_sha256=e['sha256'], cohorts=[], truth_motion={})
            for h in (1, 4):
                at = lookup[h][index, anchor]
                row['truth_motion'][str(h)] = dict(translation_image96_px=float(catalogue[f'h{h}_translation_image96_px'][at]),
                    signed_rotation_deg=float(catalogue[f'h{h}_signed_rotation_deg'][at]))
            rows[key] = row
        if cohort not in rows[key]['cohorts']:
            rows[key]['cohorts'].append(cohort)
        if diagnostic_role is not None:
            rows[key]['original_proposal_role'] = diagnostic_role
        return key

    natural = []
    for index, entry in enumerate(entries):
        if entry['split'] != 'seen_val':
            continue
        legal = sorted(a for t, a in lookup[4] if t == index)
        natural.append(add_new(index, _uniform_anchor(entry['family_id'], legal), 'natural'))
    if len(natural) != 144 or Counter(rows[k]['letter'] for k in natural) != Counter({l: 24 for l in 'TLOCXB'}):
        raise ValueError('Full144 natural cohort required')
    proposal = _read(STRESS); stress = []
    if len(proposal['rows']) != 16:
        raise ValueError('Original16 stress rows required')
    for row in proposal['rows']:
        key = add_new(row['trajectory_index'], row['anchor'], 'stress', row['group'])
        if key != row['window_id'] or rows[key]['source_sha256'] != row['source_sha256']:
            raise ValueError('Original stress source identity changed')
        stress.append(key)
    batches = []
    # Natural and stress calls have separate batch contexts. If a stress ID also
    # belongs to natural, both statistical memberships reference its one call.
    for cohort, cohort_ids in (('natural', natural), ('stress', [key for key in stress if key not in set(natural)])):
        for i, start in enumerate(range(0, len(cohort_ids), 4)):
            ids = cohort_ids[start:start + 4]
            batches.append(dict(context=cohort, batch_index=i, window_ids=ids,
                seeds=[paired_seed(SEED, f'R3B-{cohort}', i, ids, step) for step in range(4)]))
    old_side = _read(OLD_FOLDER / 'cohort-val_motion.json')
    if (old_side['checkpoint_sha256'] != _sha(CHECKPOINT) or old_side['num_steps_denoising'] != 3
            or not old_side['evaluation_complete'] or old_side['hard_stop']):
        raise ValueError('Old raw baseline checkpoint/sampler identity changed')
    old_rows = {r['window_id']: r for r in _read(OLD_INDICES)['cohorts']['val_motion']}
    old_entries = [e for e in _read(OLD_MANIFEST)['entries'] if e['split'] == 'seen_val']
    old_files = []
    for bi in (1, 2):
        ids = old_side['window_ids'][bi * 4:bi * 4 + 4]
        files = sorted([r for r in old_side['raw_step_files'] if r['batch_index'] == bi], key=lambda r: r['step'])
        if len(files) != 4 or [r['step'] for r in files] != list(range(4)):
            raise ValueError('Original old4 full batch context missing')
        for item in files:
            path = OLD_FOLDER / item['file']
            if _sha(path) != item['sha256'] or item['window_ids'] != ids:
                raise ValueError('Old raw baseline changed')
            with np.load(path, allow_pickle=False) as archive:
                if archive['raw'].dtype != np.float32 or archive['raw'].shape != (4, 3, 96, 96) or archive['window_ids'].tolist() != ids:
                    raise ValueError('Original old4 raw batch shape/context changed')
            old_files.append(dict(item, path=str(path.resolve())))
        batches.append(dict(context='old_val_motion', batch_index=bi, window_ids=ids,
                            seeds=[r['seed'] for r in files], original_step_records=files))
        for key in ids:
            r = old_rows[key]; entry = old_entries[r['entry_index']]
            rows[key] = dict(window_id=key, record_source='R2C0', entry_index=r['entry_index'],
                trajectory_id=r['trajectory_id'], family_id=r['family_id'], letter=r['letter'], anchor=r['anchor'],
                source_file=entry['file'], source_sha256=entry['sha256'], cohorts=['old4'] if key in OLD4 else [],
                truth_motion={'1': dict(translation_image96_px=r['translation_image96_px'], signed_rotation_deg=r['signed_rotation_deg']),
                              '4': dict(translation_image96_px=r['h4_translation_image96_px'], signed_rotation_deg=r['h4_signed_rotation_deg'])})
    physics = Path(physics_source or ROOT.parent / 'Tdyssey/third_party/diffusion_policy').resolve()
    spec = dict(kind='R3B-frozen-natural144-stress16-old4.v1', seed=SEED, sampler=SAMPLER, batch_size=4,
        rows=list(rows.values()), batches=batches, natural_ids=natural, stress_ids=stress, old4_ids=OLD4,
        old_context_ids=[k for k in rows if rows[k]['record_source'] == 'R2C0'],
        report_panel_ids={**{f'natural_{l}': [k for k in natural if rows[k]['letter'] == l][:2] for l in 'OC'}, 'old4': OLD4},
        natural_selection='One independent seeded uniform legal H4 anchor per144 seen-validation family; no motion filtering',
        same_anchor_H1_H4=True, H1_reuses_first_H4_step=True, raw_feedback_only=True, output_repairs=False,
        inputs='float32 four CHW96 RGB and four normalized executed actions only',
        offline_privileged_scoring_only=True, physics_steps=0, heldout_inputs=0, CP1_CP2_inputs=0,
        source_root=str(source_root), physics_source=str(physics), old_raw_files=old_files,
        provenance={name: _binding(path) for name, path in dict(manifest=MANIFEST, catalogue=CATALOGUE,
            stress_proposal=STRESS, old_manifest=OLD_MANIFEST, old_indices=OLD_INDICES,
            old_sidecar=OLD_FOLDER / 'cohort-val_motion.json', checkpoint6000=CHECKPOINT,
            calibration=CALIBRATION, evaluator=Path(__file__), r3a_renderer=ROOT / 'tdyssey/native_rgb_r3a_data.py',
            rgb_scorer=ROOT / 'scripts/native_rgb_stage3_score.py', physical_scorer=ROOT / 'tdyssey/native_rgb_physical_metrics.py',
            native_sample_interface=ROOT / 'tdyssey/native_rgb_r2c1r1_training.py',
            atomic_io=ROOT / 'tdyssey/native_rgb_training.py',
            physics_renderer=physics / 'diffusion_policy/env/pusht/pusht_env.py',
            physics_override=physics / 'diffusion_policy/env/pusht/pymunk_override.py').items()})
    path = stage / 'evaluation/spec.json'
    if path.exists():
        if _read(path) != spec:
            raise ValueError('Frozen evaluation spec differs; do not overwrite or reselect')
    else:
        _json(path, spec)
    return dict(spec, spec_file=str(path), spec_sha256=_sha(path))


def _spec(stage):
    path = Path(stage).resolve() / 'evaluation/spec.json'
    value = _read(path)
    for item in value['provenance'].values():
        if _sha(item['path']) != item['sha256']:
            raise ValueError(f"Frozen evaluation dependency changed: {item['path']}")
    return value


class WindowStore:
    """Bounded ephemeral rendering of saved states, never a persistent RGB bank."""
    def __init__(self, spec):
        self.spec = spec
        self.renderer = RecordedStateRenderer(spec['physics_source'], maximum_frame_cache_bytes=2 * 1024**2)
        self.entries = _read(MANIFEST)['entries']
        self.old_manifest = _read(OLD_MANIFEST)
        self.old_entries = [e for e in self.old_manifest['entries'] if e['split'] == 'seen_val']
        self.old_root = (OLD_MANIFEST.parent / self.old_manifest['source_root']).resolve()
        self.cache = OrderedDict()

    def _arrays(self, row):
        key = (row['record_source'], row['trajectory_id'])
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if row['record_source'] == 'R3A':
            entry = self.entries[row['trajectory_index']]
            path = Path(self.spec['source_root']) / row['source_file']
            values = _trajectory_arrays(path, entry)
            with np.load(path, allow_pickle=False) as z:
                values['rgb_frame_sha256'] = z['rgb_frame_sha256'].copy()
        else:
            path = self.old_root / row['source_file']
            if _sha(path) != row['source_sha256']:
                raise ValueError('Old recorded trajectory changed')
            with np.load(path, allow_pickle=False) as z:
                values = {k: z[k].copy() for k in ('rgb', 'actions', 'states')}
        self.cache[key] = values
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return values

    def frame(self, row, index):
        values = self._arrays(row)
        if row['record_source'] == 'R2C0':
            return values['rgb'][index].copy()
        rgb = self.renderer.render(row['letter'], values['states'][index], cache_key=f"{row['trajectory_id']}:{index}")
        expected = values['rgb_frame_sha256'][index].tobytes().hex()
        if _array_sha(rgb) != expected:
            raise ValueError('Saved-state RGB differs from collector original frame SHA')
        return rgb

    def full_mask(self, row, index):
        state = self._arrays(row)['states'][index]
        env = self.renderer._environment(row['letter'], state)
        env.set_render_state(state); before = env.state().copy()
        layers = env.render_layers()
        if not np.array_equal(before, env.state()):
            raise RuntimeError('Scoring layer rendering changed physics state')
        return layers['full_object_mask'].astype(np.float32, copy=True)

    def inputs(self, row):
        a = row['anchor']
        history = np.stack([self.frame(row, i) for i in range(a - 3, a + 1)]).astype(np.float32) / np.float32(255)
        actions = self._arrays(row)['actions'][a - 3:a + 4].copy()
        return history, actions

    def close(self):
        self.cache.clear(); self.renderer.close()


def _load_ref(ref):
    path = Path(ref['path'])
    if _sha(path) != ref['sha256']:
        raise ValueError('Raw prediction file SHA changed')
    with np.load(path, allow_pickle=False) as z:
        ids = z['window_ids'].tolist(); raw = z['raw']
        if raw.dtype != np.float32 or raw.shape != (len(ids), 3, 96, 96) or ids[ref['row']] != ref['window_id']:
            raise ValueError('Raw prediction reference shape/context changed')
        return raw[ref['row']].copy()


def _corners(mask):
    """Raw contour diagnostic only; never a calibrated shape-pass criterion."""
    import cv2
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return [len(cv2.approxPolyDP(c, .75, True)) for c in contours if cv2.contourArea(c) >= 2]


def _score(prediction, truth, last, full_mask, last_mask, letter, status):
    from scripts.native_rgb_stage3_score import score_image
    from .native_rgb_evaluation import palette_masks
    from .native_rgb_physical_metrics import fit_physical_pose
    from .envs.official import official_shape
    calibration = _read(CALIBRATION)
    scorer_status = 'ok' if status == 'completed' else status
    score = score_image(prediction, truth, last, calibration, letter=letter, status=scorer_status,
                        truth_full_mask=full_mask, last_full_mask=last_mask)
    unknown = score.get('legacy_shape_unknown_reason')
    shape = 'unknown'
    if status != 'completed':
        unknown = status
    elif score['hard_generation_failure']:
        shape = 'fail'; unknown = None
    elif score.get('occlusion_ambiguous') or score.get('truth_object_border_touch') or score.get('prediction_object_border_touch'):
        unknown = 'contact_occlusion_or_viewport_clipping'
    elif score.get('legacy_shape_assessable'):
        topology_failure = score.get('extra_fragments', 0) > 0 or score.get('missing_components', 0) > 0 or score.get('holes_difference', 0) > 0
        shape = 'fail' if score['legacy_shape_deformation_failure'] or topology_failure else 'pass'
        unknown = None
    fit = dict(status='unmeasurable', reason='invalid_or_missing_generation')
    predicted_corners = truth_corners = None
    if status == 'completed' and not score['hard_generation_failure']:
        try:
            fit = fit_physical_pose(prediction, full_mask, calibration, symmetry_order=official_shape(letter).symmetry_order)
        except Exception as error:
            fit = dict(status='unmeasurable', reason=f'offline_pose_exception:{type(error).__name__}:{error}')
        if letter in 'OC':
            predicted_corners = _corners(palette_masks(prediction, calibration)[0])
            truth_corners = _corners(palette_masks(truth, calibration)[0])
    measurable = fit['status'] == 'ok'
    metrics = dict(object_roi_mae=score['object_foreground_rgb_mae'], gray_iou=score['raw_mask_iou'],
        gray_contour_error_px=score['raw_contour_error_image_px'], gray_visible_centroid_error_px=score['object_position_error_image_px'],
        gray_centroid_error_px=fit.get('translation_error_px', 0) / (512 / 96) if measurable else None,
        rotation_error_deg=fit.get('angle_error_deg') if measurable else None,
        physical_measurement_status=fit['status'], physical_measurement_reason=fit.get('reason'),
        gray_drift_from_last_px=score['object_drift_from_last_image_px'],
        blue_pusher_iou=score['pusher_mask_iou'], blue_pusher_centroid_error_px=score['pusher_position_error_image_px'],
        blue_pusher_roi_mae=score['pusher_foreground_rgb_mae'], shape_status=shape, shape_unknown_reason=unknown,
        hard_generation_failure=score['hard_generation_failure'], occlusion_ambiguous=score.get('occlusion_ambiguous'),
        predicted_components=score.get('predicted_components'), truth_components=score.get('truth_components'),
        predicted_holes=score.get('predicted_holes'), truth_holes=score.get('truth_holes'),
        extra_fragments=score.get('extra_fragments'), missing_components=score.get('missing_components'),
        area_fraction_error=score.get('area_fraction_error'), tiny_fragment_pixels=score.get('tiny_fragment_pixels'),
        predicted_raw_corner_counts=predicted_corners, truth_raw_corner_counts=truth_corners,
        shape_pass_meaning='No detected failure under frozen legacy shape and exact visible topology checks; not proof of rigidity',
        rotation_meaning='Offline gray-shape fit, modulo native object symmetry; unavailable fits remain in full denominator',
        physical_fit=fit, original_score=score)
    return metrics


def _aggregate(rows):
    n = len(rows); counts = Counter(r['metrics']['shape_status'] for r in rows)
    numeric = ('object_roi_mae', 'gray_iou', 'gray_contour_error_px', 'gray_centroid_error_px', 'rotation_error_deg',
               'gray_visible_centroid_error_px', 'gray_drift_from_last_px', 'blue_pusher_iou', 'blue_pusher_centroid_error_px', 'blue_pusher_roi_mae')
    result = dict(planned_denominator=n, completed=sum(r['status'] == 'completed' for r in rows),
        hard_generation_failures=sum(r['metrics']['hard_generation_failure'] for r in rows),
        shape_pass=counts['pass'], shape_fail=counts['fail'], shape_unknown=counts['unknown'],
        shape_pass_fraction_full_denominator=counts['pass'] / n if n else None,
        shape_unknown_reasons=dict(Counter(r['metrics']['shape_unknown_reason'] for r in rows if r['metrics']['shape_status'] == 'unknown')),
        physical_measurable=sum(r['metrics']['physical_measurement_status'] == 'ok' for r in rows), metrics={})
    for name in numeric:
        values = [r['metrics'][name] for r in rows if r['metrics'][name] is not None and np.isfinite(r['metrics'][name])]
        result['metrics'][name] = dict(mean_available=float(np.mean(values)) if values else None,
            available=len(values), unavailable=n - len(values), full_planned_denominator=n)
    return result


def _summaries(rows):
    groups = []
    for cohort in ('natural', 'stress', 'old4'):
        for horizon in (1, 4):
            for method in sorted({r['method'] for r in rows}):
                selected = [r for r in rows if cohort in r['cohorts'] and r['horizon'] == horizon and r['method'] == method]
                for subgroup, subset in [('all', selected)] + [(f'letter_{l}', [r for r in selected if r['letter'] == l]) for l in 'TLOCXB'] + [
                    ('obvious_motion', [r for r in selected if r['obvious_motion']]),
                    ('small_motion', [r for r in selected if not r['obvious_motion'] and not r['nearstatic_aux']]),
                    ('nearstatic_aux', [r for r in selected if r['nearstatic_aux']])]:
                    groups.append(dict(cohort=cohort, horizon=horizon, method=method, subgroup=subgroup, **_aggregate(subset)))
    return groups


def evaluate(stage, model, source, phase, *, status_cb=None, checkpoint_sha256=None):
    """Run one paired phase using an already restored caller-owned model.

    The caller preserves all training state. Each sampler call additionally forks
    native Torch RNG. This function never loads checkpoints or advances an epoch.
    """
    if phase not in ('t0', 'epoch1'):
        raise ValueError('Only t0/epoch1 authorized')
    stage = Path(stage).resolve(); spec = _spec(stage); folder = stage / 'evaluation' / phase
    output = folder / 'metrics.json'; index_path = folder / 'index.json'
    if output.exists():
        existing = _read(output)
        if existing.get('evaluation_complete') and existing['spec_sha256'] == _sha(stage / 'evaluation/spec.json'):
            return existing
        raise RuntimeError('Existing incomplete metrics cannot be silently repeated')
    if index_path.exists():
        raise RuntimeError('Prior evaluation attempt exists; ambiguous model calls must not be silently repeated')
    if checkpoint_sha256 is not None and phase == 't0' and checkpoint_sha256 != spec['provenance']['checkpoint6000']['sha256']:
        raise ValueError('t0 must use the approved original6000 checkpoint')
    if phase == 'epoch1':
        t0_path = stage / 'evaluation/t0/metrics.json'
        if not t0_path.exists() or not _read(t0_path).get('evaluation_complete') or _read(t0_path)['spec_sha256'] != _sha(stage / 'evaluation/spec.json'):
            raise RuntimeError('Complete same-spec original6000 t0 evaluation must precede training/epoch1')
    from .native_rgb_r2c1r1_training import torch_native_sample
    sample = torch_native_sample(model, source, SAMPLER)
    store = WindowStore(spec); byid = {r['window_id']: r for r in spec['rows']}
    index = dict(kind='R3B-raw-float32-references.v1', phase=phase, spec_sha256=_sha(stage / 'evaluation/spec.json'),
        checkpoint_sha256=checkpoint_sha256, raw_files=[], endpoint_refs={k: {} for k in byid},
        statuses={k: {'1': 'not_attempted', '4': 'not_attempted'} for k in byid}, step_records=[], pending_call=None,
        stats=dict(native_sampler_batch_attempts=0, native_sampler_batch_completed=0,
            denoiser_forward_attempts=0, denoiser_forward_completed=0, raw_uncompressed_bytes=0, raw_disk_bytes=0,
            model_seconds=0., reused_old_raw_frames=0, new_physics_steps=0), evaluation_complete=False)
    handles = []
    if hasattr(model, 'inner_model'):
        handles = [model.inner_model.register_forward_pre_hook(lambda *_: index['stats'].__setitem__('denoiser_forward_attempts', index['stats']['denoiser_forward_attempts'] + 1)),
                   model.inner_model.register_forward_hook(lambda *_: index['stats'].__setitem__('denoiser_forward_completed', index['stats']['denoiser_forward_completed'] + 1))]
    was_training = model.training
    model.eval(); _json(index_path, index)
    try:
        for batch in spec['batches']:
            ids = batch['window_ids']; old = batch['context'] == 'old_val_motion'
            if old and phase == 't0':
                for item in batch['original_step_records']:
                    path = OLD_FOLDER / item['file']
                    for row_index, key in enumerate(ids):
                        if item['step'] in (0, 3):
                            horizon = str(1 if item['step'] == 0 else 4)
                            index['endpoint_refs'][key][horizon] = dict(path=str(path), sha256=item['sha256'], row=row_index, window_id=key, step=item['step'], reused=True)
                            index['statuses'][key][horizon] = 'completed'
                    index['stats']['reused_old_raw_frames'] += len(ids)
                _json(index_path, index)
                continue
            prepared = [store.inputs(byid[key]) for key in ids]
            history = np.stack([r[0] for r in prepared]); actions = normalize_actions(np.stack([r[1] for r in prepared]))
            for step in range(4):
                action4 = actions[:, step:step + 4].copy()
                record = dict(context=batch['context'], batch_index=batch['batch_index'], step=step,
                    window_ids=ids, seed=batch['seeds'][step], input_history_sha256=_array_sha(history), normalized_actions_sha256=_array_sha(action4))
                if old:
                    original = batch['original_step_records'][step]
                    # Initial history and every action must match the old6000 context.
                    # Later histories must instead be this model's own predictions.
                    if (step == 0 and record['input_history_sha256'] != original['input_history_sha256']) or record['normalized_actions_sha256'] != original['normalized_actions_sha256']:
                        raise ValueError('Original old4 batch input/action context differs')
                index['pending_call'] = record; index['stats']['native_sampler_batch_attempts'] += 1
                _json(index_path, index)
                if status_cb:
                    status_cb(dict(event='evaluation_sampler_call', phase=phase, **record, stats=index['stats'].copy()))
                started = time.monotonic()
                try:
                    raw = np.asarray(sample(history.copy(), action4, seed=record['seed']))
                finally:
                    index['stats']['model_seconds'] += time.monotonic() - started
                index['stats']['native_sampler_batch_completed'] += 1
                record.update(raw_shape=list(raw.shape), raw_dtype=str(raw.dtype), raw_sha256=_array_sha(raw))
                if raw.shape != (len(ids), 3, 96, 96) or raw.dtype != np.float32:
                    record['status'] = 'wrong_shape_or_dtype'; index['step_records'].append(record)
                    for key in ids:
                        index['statuses'][key]['4'] = record['status']
                        if step == 0:
                            index['statuses'][key]['1'] = record['status']
                    raise RuntimeError('Sampler output must remain original float32 NCHW96')
                filename = f"{batch['context']}-batch{batch['batch_index']:02d}-step{step}.npz"
                path = folder / filename; temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, raw=raw, window_ids=np.asarray(ids))
                    stream.flush(); os.fsync(stream.fileno())
                deadline = time.monotonic() + 2
                while True:
                    try:
                        temporary.replace(path); break
                    except PermissionError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(.025)
                file = dict(path=str(path), sha256=_sha(path), bytes=path.stat().st_size,
                    window_ids=ids, step=step, batch_index=batch['batch_index'], context=batch['context'])
                index['raw_files'].append(file)
                index['stats']['raw_uncompressed_bytes'] += raw.nbytes; index['stats']['raw_disk_bytes'] += file['bytes']
                statuses = ['completed' if np.isfinite(r).all() else 'nonfinite' for r in raw]
                record.update(status='completed' if all(s == 'completed' for s in statuses) else 'nonfinite',
                    finite_out_of_range_rows=[key for key, r in zip(ids, raw) if np.isfinite(r).all() and (r.min() < 0 or r.max() > 1)])
                if step in (0, 3):
                    horizon = str(1 if step == 0 else 4)
                    for row_index, key in enumerate(ids):
                        index['endpoint_refs'][key][horizon] = dict(path=str(path), sha256=file['sha256'], row=row_index, window_id=key, step=step)
                        index['statuses'][key][horizon] = statuses[row_index]
                index['step_records'].append(record); index['pending_call'] = None; _json(index_path, index)
                if not all(s == 'completed' for s in statuses):
                    raise RuntimeError('Nonfinite raw output: stop without GT repair or retry')
                history = np.concatenate((history[:, 1:], raw[:, None]), axis=1)
        if handles and (index['stats']['denoiser_forward_attempts'] != index['stats']['native_sampler_batch_attempts'] * 3
                or index['stats']['denoiser_forward_completed'] != index['stats']['native_sampler_batch_completed'] * 3):
            raise RuntimeError('Observed native sampler forwards differ from frozen3-step protocol')
        rows = []
        copy_cache = {}
        if phase == 'epoch1':
            copy_cache = {(r['window_id'], r['horizon']): r for r in _read(stage / 'evaluation/t0/metrics.json')['rows'] if r['method'] == 'copy_last'}
        for number, row in enumerate(spec['rows']):
            a = row['anchor']; last = store.frame(row, a); last_mask = store.full_mask(row, a)
            for horizon in (1, 4):
                motion = row['truth_motion'][str(horizon)]; translation = motion['translation_image96_px']; rotation = abs(motion['signed_rotation_deg'])
                truth = store.frame(row, a + horizon); full_mask = store.full_mask(row, a + horizon)
                for method in ('copy_last', phase):
                    key = (row['window_id'], horizon)
                    if method == 'copy_last' and key in copy_cache:
                        rows.append(copy_cache[key]); continue
                    ref = index['endpoint_refs'][row['window_id']].get(str(horizon))
                    status = 'completed' if method == 'copy_last' else index['statuses'][row['window_id']][str(horizon)]
                    prediction = last.astype(np.float32) / np.float32(255) if method == 'copy_last' else _load_ref(ref) if ref else None
                    metric = _score(prediction, truth, last, full_mask, last_mask, row['letter'], status)
                    rows.append(dict(window_id=row['window_id'], letter=row['letter'], cohorts=row['cohorts'], horizon=horizon,
                        method=method, status=status, metrics=metric, raw_reference=None if method == 'copy_last' else ref,
                        translation_image96_px=translation, abs_rotation_deg=rotation, signed_rotation_deg=motion['signed_rotation_deg'],
                        obvious_motion=translation >= 2 or rotation >= 5, nearstatic_aux=translation < .1 and rotation < .1,
                        denominator_retained=True, shape_status=metric['shape_status']))
            if number % 4 == 0 and status_cb:
                status_cb(dict(event='evaluation_scoring', phase=phase, completed_windows=number + 1, planned_windows=len(spec['rows']), stats=index['stats'].copy()))
        index['evaluation_complete'] = True; index['renderer_stats'] = store.renderer.stats.copy(); _json(index_path, index)
        result = dict(kind='R3B-paired-evaluation-metrics.v1', phase=phase, evaluation_complete=True,
            spec_sha256=index['spec_sha256'], rows=rows, summaries=_summaries(rows), stats=index['stats'],
            index_file=str(index_path), raw_storage='Deduplicated float32 raw steps only; H1/H4 references; old6000 external references',
            shape_denominator='Pass/fail/unknown all retain full planned denominator',
            physical_measurement='Offline fit only, explicit None for unavailable centroid/rotation, modulo native symmetry',
            natural_planned_windows=144, stress_planned_windows=16, old4_planned_windows=4)
        _json(output, result)
        return result
    except Exception as error:
        index['error'] = f'{type(error).__name__}: {error}'; index['renderer_stats'] = store.renderer.stats.copy(); _json(index_path, index)
        # A stopped run still has every planned row. Keep already scored rows;
        # all other rows explicitly unavailable, never silently smaller cohorts.
        completed_rows = {(r['window_id'], r['horizon'], r['method']): r for r in locals().get('rows', [])}
        failure_rows = []
        white = np.full((3, 96, 96), 255, dtype=np.uint8); empty = np.zeros((96, 96), dtype=np.float32)
        unavailable = _score(None, white, white, empty, empty, 'T', 'evaluation_stopped')
        for row in spec['rows']:
            for horizon in (1, 4):
                motion = row['truth_motion'][str(horizon)]; translation = motion['translation_image96_px']; rotation = abs(motion['signed_rotation_deg'])
                for method in ('copy_last', phase):
                    key = row['window_id'], horizon, method
                    if key in completed_rows:
                        failure_rows.append(completed_rows[key]); continue
                    failure_rows.append(dict(window_id=row['window_id'], letter=row['letter'], cohorts=row['cohorts'],
                        horizon=horizon, method=method, status='evaluation_stopped', metrics=unavailable,
                        raw_reference=None if method == 'copy_last' else index['endpoint_refs'][row['window_id']].get(str(horizon)),
                        translation_image96_px=translation, abs_rotation_deg=rotation, signed_rotation_deg=motion['signed_rotation_deg'],
                        obvious_motion=translation >= 2 or rotation >= 5, nearstatic_aux=translation < .1 and rotation < .1,
                        denominator_retained=True, shape_status='unknown', failure_reason=index['error']))
        _json(output, dict(kind='R3B-paired-evaluation-metrics.v1', phase=phase, evaluation_complete=False,
            spec_sha256=index['spec_sha256'], rows=failure_rows, summaries=_summaries(failure_rows), stats=index['stats'],
            index_file=str(index_path), error=index['error'], natural_planned_windows=144, stress_planned_windows=16, old4_planned_windows=4))
        raise
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training); store.close()


def load_window_RGB(stage, window_id):
    """On-demand original RGB for reports; never launches inference or physics."""
    spec = _spec(stage); row = next(r for r in spec['rows'] if r['window_id'] == window_id)
    store = WindowStore(spec)
    try:
        a = row['anchor']
        return dict(window=row, history=np.stack([store.frame(row, i) for i in range(a - 3, a + 1)]),
            last=store.frame(row, a), truth_h1=store.frame(row, a + 1), truth_h4=store.frame(row, a + 4))
    finally:
        store.close()


def load_case_images(stage, phase, window_id):
    """Raw H1/H4 plus source RGB; no clipping, resampling, registration or repair."""
    value = load_window_RGB(stage, window_id)
    index = _read(Path(stage) / 'evaluation' / phase / 'index.json')
    refs = index['endpoint_refs'][window_id]
    value.update(prediction_h1=_load_ref(refs['1']) if '1' in refs else None,
                 prediction_h4=_load_ref(refs['4']) if '4' in refs else None,
                 status=index['statuses'][window_id], raw_references=refs)
    return value
