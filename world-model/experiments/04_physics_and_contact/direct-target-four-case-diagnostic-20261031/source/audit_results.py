"""Independent saved-only audit of the approved four-case continuation.

No environment, Body, Shape or Space is constructed. This module imports no
simulation or model implementation and never invokes physics or inference.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

CASES = ('DT-O-01-0-b0', 'DT-T-00-0-b0', 'DT-L-05-0-b0', 'DT-C-06-1-b0')
CAPS = dict(zip(CASES, (8, 8, 8, 11)))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_raw(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    encoded = zlib.decompress(result.pop('telemetry_zlib').tobytes())
    expected = result.pop('telemetry_sha256', None)
    if expected is not None and expected.tobytes() != hashlib.sha256(encoded).digest():
        raise ValueError('Telemetry checksum differs: '+str(path))
    result.update(json.loads(encoded))
    return result


def executed_rows(data):
    return [row for control in data['telemetry']
            for row in control.get('substeps', []) if row.get('executed')]


def numeric_difference(actual, expected):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return float('inf')
    return float(np.max(np.abs(actual-expected))) if actual.size else 0.


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def audit(root, write_output=True):
    root = Path(root).resolve()
    old_root = root.parent/'direct-target-acceptance-20261031'
    diagnosis = root.parent/'direct-target-diagnosis-review-20261031'
    approval = read(root/'APPROVAL.json')
    config = read(root/'runtime-config.json')
    plan = read(old_root/'plan.json')
    planned = {item['attempt_id']: item for item in plan['attempts']}
    errors, warnings, rows = [], [], []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check([item['id'] for item in approval['cases']] == list(CASES), 'Approval case order differs')
    check(approval['max_attempts'] == 4 and approval['total_control_cap'] == 35
          and approval['total_regular_substep_cap'] == 350, 'Approval budget differs')
    check(approval['training_eligible'] is False and approval['model_calls'] == 0
          and approval['optimizer_updates'] == 0, 'Approval admits training or model work')
    for relative, expected in approval['source_specification_sha256'].items():
        check(sha(diagnosis/relative) == expected, 'Approved specification changed: '+relative)

    binding_count = 0
    for binding in config.get('bindings', []):
        path = Path(binding['path'])
        check(path.is_file() and sha(path) == binding['sha256'], 'Runtime source binding changed: '+str(path))
        binding_count += 1

    old_verified = 0
    with (old_root/'report/raw-manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        for entry in csv.DictReader(stream):
            path = Path(entry['path'])
            check(path.is_file() and path.stat().st_size == int(entry['bytes'])
                  and sha(path) == entry['sha256'], 'Old raw changed: '+entry['attempt_id'])
            old_verified += 1
    check(old_verified == 120, 'Original raw manifest does not contain120 records')
    old_meta_verified = 0
    for binding in read(diagnosis/'SOURCE_MANIFEST.json')['files']:
        path = Path(binding['path'])
        if path.parent == old_root/'collection/attempts':
            check(path.is_file() and sha(path) == binding['sha256'], 'Old metadata/labels changed: '+path.name)
            old_meta_verified += 1
    check(old_meta_verified == 120, 'Original metadata baseline does not contain120 records')

    original_sources = read(old_root/'runtime-config.json')['bindings']
    for binding in original_sources:
        path = Path(binding['path'])
        check(path.is_file() and sha(path) == binding['sha256'], 'Old collection source/config changed: '+str(path))

    actual_npz = sorted((root/'results').glob('*.npz')) if (root/'results').exists() else []
    check(len(actual_npz) <= 4, 'More than four new physical raw records')
    check(all(path.stem in CASES for path in actual_npz), 'Unapproved fifth physical case')
    actual_controls = actual_substeps = 0
    initial_calls = regular_calls = 0
    for path in actual_npz:
        ident = path.stem
        data = load_raw(path)
        old = load_raw(old_root/'collection/raw'/path.name)
        meta_path = root/'results'/(ident+'.json')
        check(meta_path.is_file(), 'Missing new metadata: '+ident)
        meta = read(meta_path) if meta_path.exists() else {}
        source_meta = read(old_root/'collection/attempts'/(ident+'.json'))
        new_initial = meta.get('initial_identity')
        old_initial = source_meta['initial_identity']
        local_polygons = [Polygon(item['geometry']['vertices'])
                          for item in old_initial['physics']['shapes']
                          if item['role'] == 'object' and item['geometry']['kind'] == 'poly']
        local_geometry = unary_union(local_polygons)
        vertices = np.concatenate([np.asarray(p.exterior.coords[:-1], float) for p in local_polygons])
        if new_initial:
            check(new_initial.get('physics_sha256') == old_initial.get('physics_sha256'), 'Physics identity mismatch: '+ident)
            check(new_initial.get('protocol_sha256') == old_initial.get('protocol_sha256'), 'Physics protocol mismatch: '+ident)
        else:
            check(False, 'Missing actual physics identity: '+ident)

        actions = data['requested_actions']
        n = len(actions)
        check(n <= CAPS[ident], 'Case exceeds its control budget: '+ident)
        check(actions.shape == (n, 2), 'Malformed action shape: '+ident)
        check(np.isfinite(actions).all() and (actions >= 0).all() and (actions <= 512).all(), 'Action outside original domain: '+ident)
        source_actions_equal = bool(np.array_equal(actions, np.asarray(planned[ident]['actions'], float)[:n]))
        passed_actions_equal = bool(np.array_equal(actions, data.get('passed_actions')))
        check(source_actions_equal, 'Actions differ from original frozen plan: '+ident)
        check(passed_actions_equal, 'Requested differs from actual upstream input: '+ident)
        check(data['states'].shape == (n+1, 15), 'State/action alignment differs: '+ident)
        check(data['rgb'].shape == (n+1, 3, 96, 96), 'RGB/action alignment differs: '+ident)
        check(len(data['telemetry']) == n, 'Control telemetry length differs: '+ident)
        native = executed_rows(data)
        expected = executed_rows(old)
        check(len(native) <= CAPS[ident]*10, 'Case exceeds its substep budget: '+ident)
        old_count = len(expected)
        prefix_count = min(old_count, len(native))
        prefix_errors = []
        for index, (actual, prior) in enumerate(zip(native[:prefix_count], expected[:prefix_count])):
            for name in ('pre_state', 'integrator_pre_state', 'post_state'):
                prior_value = prior.get(name, prior.get('pre_state') if name == 'integrator_pre_state' else None)
                actual_value = actual.get(name, actual.get('pre_state') if name == 'integrator_pre_state' else None)
                error = numeric_difference(actual_value, prior_value)
                prefix_errors.append(error)
                check(error <= 1e-8, f'Prefix {name} differs at {ident} micro{index}: {error}')
            check(actual.get('dt') == prior.get('dt') == .01, 'Prefix timestep changed: '+ident)
        complete_old_controls = len(old['requested_actions'])-1
        # The old final frame is a partial control; compare it separately.
        boundary_frames = min(complete_old_controls+1, len(data['rgb']))
        check(np.array_equal(data['rgb'][:boundary_frames], old['rgb'][:boundary_frames]), 'Old complete-boundary RGB differs: '+ident)
        check(numeric_difference(data['states'][:boundary_frames], old['states'][:boundary_frames]) <= 1e-8,
              'Old complete-boundary state differs: '+ident)
        terminal_rgb = data.get('legacy_terminal_rgb')
        if terminal_rgb is None:
            terminal_rgb = data.get('prefix_terminal_rgb_actual')
        terminal_rgb_matches = terminal_rgb is not None and np.array_equal(terminal_rgb, old['rgb'][-1])
        check(terminal_rgb_matches, 'Missing or different old partial-terminal RGB snapshot: '+ident)

        # Check the pinned external PD update, including the unobserved future.
        before = data['states'][0].copy()
        pd_max_error = 0.
        geometry_max_error = 0.
        hard_rows = []
        flat_index = 0
        for index, info in enumerate(data['telemetry']):
            target = actions[index]
            for sub in info.get('substeps', []):
                if not sub.get('executed'):
                    continue
                dt = float(sub['dt'])
                check(dt == .01, 'Actual integrator dt differs: '+ident)
                expected_pre = before.copy()
                expected_pre[5:7] += (100.*(target-before[:2])-20.*before[5:7])*dt
                pd_error = numeric_difference(sub['pre_state'], expected_pre)
                pd_max_error = max(pd_max_error, pd_error)
                check(pd_error <= 1e-8, 'Original external PD update differs: '+ident)
                before = np.asarray(sub['post_state'], float)
                if np.isfinite(before).all():
                    th = before[4]
                    rotation = np.asarray([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                    actual_vertices = vertices@rotation.T+before[2:4]
                    bounds = np.asarray([actual_vertices[:, 0].min(), actual_vertices[:, 1].min(),
                                         actual_vertices[:, 0].max(), actual_vertices[:, 1].max()])
                    local_center = rotation.T@(before[:2]-before[2:4])
                    point = Point(*local_center)
                    signed = -local_geometry.boundary.distance(point) if local_geometry.contains(point) else local_geometry.distance(point)
                    x, y = before[:2]
                    calculated_depths = [max(0., 7.-bounds[0], 7.-bounds[1], bounds[2]-504., bounds[3]-504.),
                                         max(0., 15.-signed), max(0., 22.-x, 22.-y, x-489., y-489.)]
                    geometry_error = numeric_difference(sub['penetrations'], calculated_depths)
                    geometry_max_error = max(geometry_max_error, geometry_error)
                    check(geometry_error <= 1e-8, 'Independent actual-shape depth mismatch: '+ident)
                    if local_geometry.contains(point):
                        region = 'solid'
                    else:
                        holes = [Polygon(ring) for ring in local_geometry.interiors]
                        region = next(('hole-'+str(i) for i, hole in enumerate(holes) if hole.contains(point)), 'outside')
                    check(sub.get('pusher_region') == region, 'Independent pusher region mismatch: '+ident)
                    expected_legacy = bool(max(calculated_depths) > 1.+1e-8 or region == 'solid' or region.startswith('hole-'))
                    check(sub.get('would_stop_legacy') is expected_legacy, 'Legacy observation marker differs: '+ident)
                    hard = []
                    if calculated_depths[2] > 1.+1e-8:
                        hard.append('AW_overlap_above_1_world_px')
                    if region == 'solid' or region.startswith('hole-'):
                        hard.append('illegal_pusher_center_region')
                    if min(bounds[:2]) < 3. or max(bounds[2:]) > 508.:
                        hard.append('object_outer_wall_envelope_crossed')
                    check(set(sub.get('diagnostic_hard_reasons', [])) == set(hard), 'Diagnostic hard-stop logic differs: '+ident)
                    check(numeric_difference(sub.get('object_collision_vertex_bounds'), bounds) <= 1e-8, 'Actual vertex envelope differs: '+ident)
                    if hard:
                        hard_rows.append(flat_index)
                flat_index += 1
        check(not hard_rows or hard_rows == [len(native)-1], 'Execution continued after a diagnostic hard stop: '+ident)

        times = data.get('control_time_seconds')
        seconds = sum(float(sub['dt']) for sub in native)
        durations = [sum(float(sub['dt']) for sub in row.get('substeps', []) if sub.get('executed'))
                     for row in data['telemetry']]
        if times is not None:
            check(len(times) == n+1 and np.allclose(times, [0., *np.cumsum(durations)], rtol=0, atol=1e-9), 'Saved timeline differs from actual integration: '+ident)
        else:
            check(False, 'Missing actual control timeline: '+ident)
        cost = meta.get('cost', {})
        if cost:
            check(cost.get('actual_controls') == n, 'Metadata control cost mismatch: '+ident)
            check(cost.get('actual_substeps') == len(native), 'Metadata substep cost mismatch: '+ident)
            initial_calls += int(cost.get('initialization_integrator_calls', 0))
            regular_calls += int(cost.get('regular_integrator_calls', len(native)))
        check(meta.get('training_eligible') is False, 'Diagnostic raw is not explicitly training-ineligible: '+ident)
        raw_receipt = meta.get('raw') or {}
        if raw_receipt.get('sha256'):
            check(sha(path) == raw_receipt['sha256'], 'New raw receipt hash differs: '+ident)
        actual_controls += n
        actual_substeps += len(native)
        rows.append(dict(attempt_id=ident, actual_controls=n, actual_substeps=len(native), simulated_seconds=seconds,
            source_prefix_substeps=old_count, checked_prefix_substeps=prefix_count,
            all_old_prefix_observed=prefix_count == old_count,
            prefix_state_max_abs_error=max(prefix_errors, default=None) if all(np.isfinite(prefix_errors)) else 'nonfinite_or_shape_mismatch',
            old_partial_terminal_RGB_exact=terminal_rgb_matches, external_PD_max_abs_error=pd_max_error,
            independent_geometry_max_abs_error=geometry_max_error, hard_stop_microsteps=hard_rows,
            original_physics_identity_matches=bool(new_initial and new_initial.get('physics_sha256') == old_initial.get('physics_sha256')),
            requested_equals_passed_equals_original_plan=source_actions_equal and passed_actions_equal, raw_sha256=sha(path)))

    check(actual_controls <= 35 and actual_substeps <= 350, 'Unified budget exceeded')
    check(regular_calls == actual_substeps, 'Integrator call count differs from preserved executed substeps')
    # Missing cases can be correct after an engineering prefix mismatch. They
    # remain visible and are never replaced by a fifth attempt.
    unstarted = [ident for ident in CASES if ident not in {row['attempt_id'] for row in rows}]
    if unstarted:
        warnings.append('Partial cohort; unstarted cases are not silently replaced: '+','.join(unstarted))
    result = dict(kind='independent-four-case-saved-audit.v1', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        passed=not errors, errors=errors, warnings=warnings, case_results=rows,
        all_selected_cases_saved=not unstarted, unstarted_cases=unstarted,
        actual_controls=actual_controls, actual_substeps=actual_substeps,
        regular_integrator_calls=regular_calls, initialization_integrator_calls=initial_calls,
        original120_raw_hashes_verified=old_verified, original42_source_bindings_verified=len(original_sources),
        original120_metadata_and_legacy_labels_verified=old_meta_verified,
        diagnostic_runtime_bindings_verified=binding_count,
        audit_physics_calls=0, audit_model_calls=0, audit_optimizer_updates=0,
        training_eligible=False)
    if write_output:
        (root/'independent-audit.json').write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
