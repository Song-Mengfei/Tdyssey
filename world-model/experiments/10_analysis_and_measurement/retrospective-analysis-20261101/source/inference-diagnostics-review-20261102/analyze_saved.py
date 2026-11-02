"""Read-only independent audit of existing A/B results; stdlib only, no models."""
import collections
import hashlib
import itertools
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT.parent.parent
RUN = ART / 'hmf-inference-diagnostics-20261102'
HMF = ART / 'high-motion-fit-20261101'
LABELS = ['0', '250', '500', '1000', '2000', 'final']
METRICS = ['raw_mask_iou', 'object_position_error_image_px', 'raw_contour_error_image_px']

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def mean(values):
    values = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(values) if values else None

def close(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a - b) < 1e-10

result = read(RUN / 'inference/result.json')
specA = read(RUN / 'A-spec.json')
specB = read(RUN / 'B-spec.json')
config = read(RUN / 'runtime-config.json')
config_sha = hashlib.sha256((RUN / 'runtime-config.json').read_bytes()).hexdigest()
assert config_sha == result['config_sha256']
assert result['budget']['counts']['completed_predictions'] == 3672
assert result['budget']['counts']['image_forward_attempts'] == 11016
assert result['budget']['active_call'] is None
assert all(result[k] == 0 for k in ['optimizer_updates', 'backward_calls', 'physical_steps', 'new_data'])

# Reaggregate source score files, independently of the owner's curve summaries.
old = {label: read(HMF / f'evaluation/{label}/primary/metrics.json') for label in LABELS}
train12 = {r['sample_id'] for r in old['250']['rows'] if r['selection_role'] == 'train'}
assert len(train12) == 12
rows = []
for label, data in old.items():
    for row in data['rows']:
        if row['selection_role'] != 'train' or row['sample_id'] in train12:
            rows.append(dict(row, checkpoint=label))
for batch in result['A_batches']:
    rows.extend(read(pathlib.Path(batch['metrics']))['rows'])
assert len(rows) == 7488
assert len({(r['checkpoint'], r['sample_id'], r['input_mode'], r['horizon']) for r in rows}) == len(rows)
grouped = collections.defaultdict(list)
for row in rows:
    role = 'fixed_train12' if row['selection_role'] == 'train' else row['selection_role']
    grouped[(row['checkpoint'], role, row['letter'], row['input_mode'], row['horizon'])].append(row)
curves = []
for (label, role, letter, mode, horizon), values in grouped.items():
    curves.append(dict(checkpoint=label, role=role, letter=letter, mode=mode, horizon=horizon,
                       families=len({x['family_id'] for x in values}), count=len(values),
                       metrics={m: mean([x['metrics'].get(m) for x in values]) for m in METRICS},
                       shape=dict(collections.Counter(x['visible_shape_status'] for x in values)),
                       topology=dict(collections.Counter(x['visible_topology_status'] for x in values)),
                       rotation=dict(collections.Counter(x['rotation']['status'] for x in values))))
write('INDEPENDENT_CURVES.json', curves)
motion = {(r['sample_id'], i + 1): s for r in specA['rows']
          for i, s in enumerate(r.get('strong_flags', [t['strong'] for t in r['target_motion']]))}
strong_curves = []
for (label, role, letter, mode, horizon), values in grouped.items():
    if mode != 'real_history' or horizon != 1:
        continue
    selected = [r for r in rows if r['checkpoint'] == label and r['letter'] == letter
                and r['input_mode'] == mode
                and ('fixed_train12' if r['selection_role'] == 'train' else r['selection_role']) == role
                and (r.get('strong_H1', False) or motion.get((r['sample_id'], r['horizon']), False))]
    strong_curves.append(dict(checkpoint=label, role=role, letter=letter, targets=len(selected),
                             metrics={m: mean([r['metrics'].get(m) for r in selected]) for m in METRICS}))
write('STRONG_RH_CURVES.json', strong_curves)
own = {(r['checkpoint'], r['role'], r['letter'], r['mode'], r['horizon']): r for r in curves}
owner_curves = read(RUN / 'report/A-all16-curves.json')
compared = 0
for row in owner_curves:
    subset = row['subset']
    if row['checkpoint'] not in LABELS or not subset.startswith('H') or not subset[1:].isdigit():
        continue
    key = (row['checkpoint'], row['role'], row['letter'], row['mode'], int(subset[1:]))
    if key not in own:
        continue
    for metric in METRICS:
        assert close(own[key]['metrics'][metric], row['metrics'][metric]['mean']), (key, metric)
        compared += 1

endpoint = []
for role, letter, mode in itertools.product(['fixed_train12', 'seen_val', 'original_failure'], 'OCB', ['real_history', 'autoregressive']):
    seq = [own[(label, role, letter, mode, 16)] for label in LABELS]
    endpoint.append(dict(role=role, letter=letter, mode=mode,
                         iou=[x['metrics']['raw_mask_iou'] for x in seq],
                         position=[x['metrics']['object_position_error_image_px'] for x in seq],
                         best_iou_checkpoint=LABELS[max(range(6), key=lambda i: seq[i]['metrics']['raw_mask_iou'])],
                         best_position_checkpoint=LABELS[min(range(6), key=lambda i: seq[i]['metrics']['object_position_error_image_px'])]))
write('ENDPOINT_TRAJECTORIES.json', endpoint)

paired = []
for role, letter, mode, horizon in itertools.product(['fixed_train12', 'seen_val', 'original_failure'], 'OCB', ['real_history', 'autoregressive'], [1, 4, 8, 16]):
    v0 = grouped[('0', role, letter, mode, horizon)]
    vf = {x['sample_id']: x for x in grouped[('final', role, letter, mode, horizon)]}
    counts = {}
    for metric in METRICS:
        deltas = [vf[x['sample_id']]['metrics'][metric] - x['metrics'][metric] for x in v0]
        sign = 1 if metric == 'raw_mask_iou' else -1
        counts[metric] = dict(better=sum(d * sign > 1e-12 for d in deltas),
                              worse=sum(d * sign < -1e-12 for d in deltas),
                              tied=sum(abs(d) <= 1e-12 for d in deltas), deltas=deltas)
    paired.append(dict(role=role, letter=letter, mode=mode, horizon=horizon, counts=counts))
write('PAIRED_CHANGES.json', paired)

# Check recorded epsilon and input evidence against frozen expected A and B pairings.
noise_checks = input_checks = 0
for batch in result['A_batches']:
    r = batch['record']
    assert r['batch_slots'] == list(range(len(r['rows'])))
    for k, v in r['expected_input_evidence'].items():
        actual = r[k] if k in ('mode', 'batch_id') else r['input_evidence'][k]
        assert actual == v, (r['record_id'], k)
        input_checks += 1
    expected_noise = r['expected_noise_evidence']
    expected_steps = expected_noise['steps'] if isinstance(expected_noise, dict) else expected_noise
    assert len(r['noise_evidence']) == len(expected_steps) == 16
    for actual, expected in zip(r['noise_evidence'], expected_steps):
        for k in ['step', 'seed', 'batch_size', 'actual_initial_epsilon_sha256', 'per_case_initial_epsilon_sha256']:
            assert actual[k] == expected[k], (r['record_id'], k)
            noise_checks += 1
        assert actual['extra_random_draws'] == 0
        assert actual['original_value_returned']
bgroups = collections.defaultdict(list)
for batch in result['B_batches']:
    r = batch['record']
    bgroups[(r['batch_id'], r['noise_id'])].append(r)
for key, records in bgroups.items():
    assert len(records) == 6
    assert len({r['input_evidence']['initial_history_sha256'] for r in records}) == 1
    assert len({tuple(r['batch_slots']) for r in records}) == 1
    assert len({tuple(r['input_evidence']['sample_ids']) for r in records}) == 1
    assert len({r['noise_evidence'][0]['actual_initial_epsilon_sha256'] for r in records}) == 1
    noise_checks += 5
for row in specB['rows']:
    assert row['true_H1_motion']['strong']
    assert all(0 <= v <= 511 for action in row['actions'].values() for v in action)
assert len({r['family_id'] for r in specB['rows']}) == 12

# Rebuild action/noise contrasts from individual saved pair measurements.
B = read(RUN / 'report/B-action-noise-comparisons.json')
bmetrics = ['object_centroid_change_imagepx', 'object_mask_change', 'object_contour_change_imagepx', 'object_visible_region_rgb_mae_excluding_blue']
b_summary = []
for row in B['rows']:
    case, label, pair = row['case_id'], row['checkpoint'], row['action_pair']
    source = [p for p in B['raw_pairs'] if p['case_id'] == case and p['checkpoint'] == label]
    effects = [p for p in source if p['kind'] == 'same_noise_different_action' and [p['left']['action'], p['right']['action']] == pair]
    noises = [p for p in source if p['kind'] == 'same_action_different_noise' and p['left']['action'] in pair]
    assert len(effects) == 3 and len(noises) == 6
    out = dict(case_id=case, checkpoint=label, letter=row['letter'], action_pair=pair, metrics={})
    for metric in bmetrics:
        action = mean([p['metrics'][metric] for p in effects])
        noise = mean([p['metrics'][metric] for p in noises])
        supplied = row['metrics'][metric]
        assert close(action, supplied['action_effect']['mean'])
        assert close(noise, supplied['noise_reference_mean'])
        ratio = action / noise if noise else None
        assert close(ratio, supplied['action_to_noise_ratio'])
        out['metrics'][metric] = dict(action_mean=action, noise_mean=noise, ratio=ratio,
                                      action_mean_gt_noise_mean=action > noise,
                                      min_action_gt_max_noise=min(p['metrics'][metric] for p in effects) > max(p['metrics'][metric] for p in noises))
    b_summary.append(out)
write('INDEPENDENT_ACTION_NOISE.json', b_summary)

accuracy = []
for label, split, letter in itertools.product(['E2', 'final'], ['train', 'seen_val'], 'OCB'):
    values = [r for r in B['recorded_GT_accuracy'] if (r['checkpoint'], r['split'], r['letter']) == (label, split, letter)]
    accuracy.append(dict(checkpoint=label, split=split, letter=letter, families=len({r['family_id'] for r in values}),
                         noise_images=len(values), metrics={m: mean([x['metrics'].get(m) for x in values]) for m in METRICS}))
write('B_RECORDED_GT_ACCURACY.json', accuracy)

# Only six new raw archives are hashed; no repeat full-data/checkpoint audit.
raw_checks = []
for section in ['A', 'B']:
    for letter in 'OCB':
        candidates = [x for x in result[f'{section}_batches'] if x['record']['rows'][0]['letter'] == letter]
        selected = candidates[-1]
        info = selected['prediction']
        f = pathlib.Path(info['path'])
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        assert digest == info['sha256']
        raw_checks.append(dict(section=section, letter=letter, path=str(f), bytes=f.stat().st_size, sha256=digest, passed=True))
completion = read(RUN / 'background/completion.json')
storage = read(RUN / 'background/STORAGE_LIMIT_FAILURE.json')
audit = dict(run_id=result['run_id'], runtime_sha256=config_sha, source_score_rows=len(rows),
             curve_metric_means_compared=compared, input_fields_compared=input_checks,
             recorded_noise_fields_compared=noise_checks, B_pair_groups=len(bgroups),
             B_action_noise_rows_reaggregated=len(b_summary), raw_samples=raw_checks,
             budget=result['budget'], CPU=result['CPU'], timings=result['timings'],
             worker_wall_seconds=completion['wall_seconds'], own_CPU_seconds=completion['own_CPU_seconds'],
             overall_failed=completion['failed'], failure=completion['failure'], storage=storage,
             phase_at_read=read(RUN / 'PHASE_STATE.json'),
             audit_models=0, audit_new_predictions=0, audit_physics=0,
             audit_method='stdlib serial, BelowNormal inherited; saved score arithmetic and six new raw hashes only')
write('AUDIT.json', audit)
print(json.dumps(dict(curves=len(curves), compared=compared, inputs=input_checks, noise_fields=noise_checks, raw_samples=len(raw_checks),
                      AR16=[x for x in endpoint if x['mode']=='autoregressive'],
                      B_recorded_hold=[dict(checkpoint=l, cases=len(v), centroid_all_mean_gt_noise=all(x['metrics'][bmetrics[0]]['action_mean_gt_noise_mean'] for x in v),
                                            conservative_count=sum(x['metrics'][bmetrics[0]]['min_action_gt_max_noise'] for x in v),
                                            action_centroid_range=[min(x['metrics'][bmetrics[0]]['action_mean'] for x in v),max(x['metrics'][bmetrics[0]]['action_mean'] for x in v)])
                                       for l in ['E2','final'] for v in [[x for x in b_summary if x['checkpoint']==l and x['action_pair']==['recorded','hold']]]]), ensure_ascii=False, indent=2))
