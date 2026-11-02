"""Saved-only direct-target acceptance audit; never constructs a simulator.

Motion is privileged ground truth, not a learned prediction. All planned attempts
remain in the index, including initial-state failures and unstarted attempts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import zlib

import numpy as np

from .envs.official import official_shape
from .native_rgb_env import shape_geometry
from .original_contact_pilot_resources import ResourceReservation

LETTERS = tuple('TLOCXB')
HORIZONS = (1, 4, 8, 16)
OBVIOUS = {1: (1., 3.), 4: (2., 5.), 8: (4., 10.), 16: (8., 20.)}
SCALE = 96./512.
TRANSLATION_BINS = [0., .1, .5, 1., 2., 4., 8., 16., 32., None]
ROTATION_BINS = [0., .5, 1., 3., 5., 10., 20., 45., 90., None]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def _csv(rows):
    stream = io.StringIO(newline='')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue().encode('utf-8-sig')


def distribution(values, bins=None):
    values = np.asarray(list(values), float)
    good = values[np.isfinite(values)]
    result = dict(N=len(values), measurable=len(good), unknown=len(values)-len(good))
    result.update(dict(zip(('P10', 'P50', 'P90', 'max'), map(float, np.quantile(good, [.1, .5, .9, 1.]))))
                  if len(good) else dict.fromkeys(('P10', 'P50', 'P90', 'max')))
    if bins:
        result.update(bin_edges=bins, bin_counts=np.histogram(good,
                      [np.inf if item is None else item for item in bins])[0].tolist())
    return result


def centers(letter, states):
    states = np.asarray(states, float)
    local = np.asarray(official_shape(letter).centroid, float)
    c, s = np.cos(states[:, 4]), np.sin(states[:, 4])
    return states[:, 2:4]+np.column_stack((c*local[0]-s*local[1], s*local[0]+c*local[1]))


@lru_cache(maxsize=6)
def _geometry(letter):
    if letter not in LETTERS:
        raise ValueError('This acceptance report supports only T/L/O/C/X/B')
    return shape_geometry(letter)


def _actual_arc(letter, local):
    """Map actual object contact to exterior arclength; holes stay separate."""
    from shapely.geometry import Point, LineString
    geometry = _geometry(letter)
    rings = [LineString(geometry.exterior.coords), *[LineString(r.coords) for r in geometry.interiors]]
    point = Point(*local)
    distances = [ring.distance(point) for ring in rings]
    index = int(np.argmin(distances))
    if distances[index] > 2.:
        return dict(ring=-1, arc16=-1, distance_world_px=float(distances[index]))
    ring = rings[index]
    return dict(ring=index, arc16=min(15, int(16*ring.project(point)/ring.length)),
                distance_world_px=float(distances[index]))


def load_attempt(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    payload = zlib.decompress(data.pop('telemetry_zlib').tobytes())
    if hashlib.sha256(payload).digest() != data.pop('telemetry_sha256').tobytes():
        raise ValueError(f'Saved telemetry hash mismatch: {path}')
    data.update(json.loads(payload))
    return data


def validate_timeline(data):
    n = len(data['requested_actions'])
    if data['states'].shape != (n+1, 15) or data['rgb'].shape != (n+1, 3, 96, 96):
        raise ValueError('Saved states/RGB do not align with actions')
    if len(data['control_valid']) != n or len(data['telemetry']) != n:
        raise ValueError('Saved validity/telemetry do not align with actions')
    passed = data.get('passed_actions')
    if passed is None or not np.array_equal(passed, data['requested_actions']):
        raise ValueError('Requested targets differ from values passed to original environment')
    if not np.isfinite(data['requested_actions']).all() or (data['requested_actions'] < 0).any() or (data['requested_actions'] > 512).any():
        raise ValueError('Saved requested targets exceed original action_space [0,512]')
    for name in ('frame_indices', 'control_time_seconds', 'segment_ids'):
        if len(data[name]) != n+1:
            raise ValueError(f'Saved {name} does not align with frames')
    durations = [sum(float(s.get('dt', 0.)) for s in row.get('substeps', []) if s.get('executed'))
                 for row in data['telemetry']]
    if not np.allclose(np.diff(data['control_time_seconds']), durations, rtol=0, atol=1e-9):
        raise ValueError('Frame timestamps differ from actual executed native substeps')
    if any(bool(data['control_valid'][i]) and (len(row.get('substeps', [])) != 10 or abs(durations[i]-.1) > 1e-9)
           for i, row in enumerate(data['telemetry'])):
        raise ValueError('A purported legal control is not ten actual native substeps')
    return n


def legal_window(data, anchor, horizon):
    n = len(data['control_valid'])
    lo, end = anchor-3, anchor+horizon
    if lo < 0 or end > n:
        return False
    if not np.all(data['control_valid'][lo:end]):
        return False
    if not np.all(np.diff(data['frame_indices'][lo:end+1]) == 1):
        return False
    if len(np.unique(data['segment_ids'][lo:end+1])) != 1:
        return False
    return bool(np.allclose(np.diff(data['control_time_seconds'][lo:end+1]), .1, rtol=0, atol=1e-9)
                and np.isfinite(data['states'][lo:end+1]).all())


def _positive(event):
    return bool(event.get('phase') == 'post_solve'
                and np.linalg.norm(event.get('total_impulse', [0., 0.])) > 0.)


def states_last(data):
    return data['states'][-1].tolist() if len(data['states']) else None


def _unknown_motion_tail(attempt, tail):
    substeps = [s for s in tail.get('telemetry', {}).get('substeps', []) if s.get('executed')]
    seconds = sum(float(s.get('dt', 0.)) for s in substeps)
    def timed(kind, positive=False):
        return sum(float(s['dt'])*any(e.get('kind') == kind and e.get('phase') == 'post_solve'
            and (not positive or _positive(e)) for e in s.get('contacts', [])) for s in substeps)
    return dict(attempt_id=attempt['attempt_id'], family_id=attempt['family_id'], letter=attempt['letter'],
        scenario=attempt['scenario'], control=tail['control'], valid=False, planned_phase='engineering_unframed_terminal',
        category='invalid', actual_seconds=seconds, actual_substeps=len(substeps),
        AO_positive_seconds=timed('agent_object', True), AO_pair_seconds=timed('agent_object'),
        OW_pair_seconds=timed('object_wall'), OW_positive_seconds=timed('object_wall', True), AW_pair_seconds=timed('agent_wall'),
        invalid_substep_seconds=seconds, contact_and_push_seconds=0., contact_blocked_or_small_seconds=0.,
        translation_world_px=None, translation_image_px=None, signed_rotation_deg=None, rotation_deg=None,
        pusher_motion_world_px=None, pusher_motion_image_px=None, target_distance_from_pusher_world_px=None,
        target_change_world_px=None, target_x=tail['action'][0], target_y=tail['action'][1],
        max_penetration_OW_AO_AW_world_px=None, actual_wall_contact_seconds={'unknown': timed('object_wall')},
        unknown_motion_reason='Saved unframed terminal state is nonfinite/unavailable; actual time/contact retained, no fabricated position or RGB')


def control_metrics(attempt, data):
    letter = attempt['letter']
    states, actions = data['states'], data['requested_actions']
    positions = centers(letter, states)
    rows, contact_points, arcs = [], [], []
    for index, info in enumerate(data['telemetry']):
        substeps = [row for row in info.get('substeps', []) if row.get('executed')]
        seconds = positive = ao_pair = ow_pair = ow_positive = aw_pair = invalid_seconds = 0.
        native_translation_path = native_rotation_path = 0.
        wall_seconds = Counter()
        penetration = np.zeros(3)
        for micro, sub in enumerate(substeps):
            dt = float(sub['dt'])
            if abs(dt-.01) > 1e-12:
                raise ValueError('Saved original substep dt is not .01 seconds')
            seconds += dt
            events = [event for event in sub.get('contacts', []) if event.get('phase') == 'post_solve']
            ao = [event for event in events if event.get('kind') == 'agent_object' and _positive(event)]
            ow = [event for event in events if event.get('kind') == 'object_wall']
            positive += dt*bool(ao)
            ao_pair += dt*any(event.get('kind') == 'agent_object' for event in events)
            ow_pair += dt*bool(ow)
            ow_positive += dt*any(_positive(event) for event in ow)
            aw_pair += dt*any(event.get('kind') == 'agent_wall' for event in events)
            invalid_seconds += dt*bool(sub.get('valid') is False)
            penetration = np.maximum(penetration, np.asarray(sub.get('penetrations', [0., 0., 0.]), float))
            after = np.asarray(sub['post_state'], float)
            before = np.asarray(sub.get('pre_state', sub.get('integrator_pre_state')), float)
            if before.shape == (15,) and np.isfinite(before).all() and np.isfinite(after).all():
                pair_centers = centers(letter, np.asarray([before, after]))
                native_translation_path += float(np.linalg.norm(pair_centers[1]-pair_centers[0]))
                native_rotation_path += abs(float(np.rad2deg(after[4]-before[4])))
            c, s = math.cos(after[4]), math.sin(after[4])
            hit_arcs = set()
            for event in ao:
                object_first = event.get('shapes_order', [None])[0] == 'object'
                for point in event.get('points', []):
                    world = point.get('object_world') or point.get('point_a' if object_first else 'point_b')
                    if world is None:
                        continue
                    delta = np.asarray(world)-after[2:4]
                    local = np.asarray((c*delta[0]+s*delta[1], -s*delta[0]+c*delta[1]))
                    arc = _actual_arc(letter, local)
                    hit_arcs.add((arc['ring'], arc['arc16']))
                    selected_local = attempt.get('selected_contact', {}).get('point_local')
                    selected_arc = _actual_arc(letter, selected_local) if selected_local is not None else None
                    contact_points.append(dict(attempt_id=attempt['attempt_id'], family_id=attempt['family_id'],
                        letter=letter, control=index, microstep=micro, local_x=float(local[0]),
                        local_y=float(local[1]), world_x=float(world[0]), world_y=float(world[1]),
                        selected_point_distance_world_px=float(np.linalg.norm(local-np.asarray(selected_local))) if selected_local is not None else None,
                        selected_exterior_arc16=selected_arc['arc16'] if selected_arc and selected_arc['ring'] == 0 else None,
                        same_selected_exterior_arc16=bool(selected_arc and selected_arc['ring'] == 0 and arc['ring'] == 0 and selected_arc['arc16'] == arc['arc16']), **arc))
            if ao and not hit_arcs:
                hit_arcs.add((-1, -1))
            for ring, arc16 in sorted(hit_arcs):
                arcs.append(dict(attempt_id=attempt['attempt_id'], family_id=attempt['family_id'],
                                 letter=letter, control=index, ring=ring, arc16=arc16, seconds=dt))
            sides = set()
            for event in ow:
                object_first = event.get('shapes_order', [None])[0] == 'object'
                for point in event.get('points', []):
                    world = point.get('point_a' if object_first else 'point_b')
                    if world is not None:
                        gaps = dict(left=abs(world[0]-7), right=abs(world[0]-504),
                                    top=abs(world[1]-7), bottom=abs(world[1]-504))
                        side = min(gaps, key=gaps.get)
                        if gaps[side] <= 2.:
                            sides.add(side)
            if ow and not sides:
                sides.add('unknown')
            for side in sides:
                wall_seconds[side] += dt
            for vertical in ('top', 'bottom'):
                for horizontal in ('left', 'right'):
                    if vertical in sides and horizontal in sides:
                        wall_seconds[horizontal+'-'+vertical] += dt
        translation = float(np.linalg.norm(positions[index+1]-positions[index]))
        rotation = float(np.rad2deg(states[index+1, 4]-states[index, 4]))
        pusher_move = float(np.linalg.norm(states[index+1, :2]-states[index, :2]))
        moving = translation*SCALE >= .1 or abs(rotation) >= .5
        valid = bool(data['control_valid'][index])
        contact = positive > 0
        decisions = data.get('decisions', [])
        phases = attempt.get('planned_phase', attempt.get('planned_phases', []))
        phase = decisions[index].get('phase') if len(decisions) > index else (phases[index] if len(phases) > index else 'unknown')
        category = ('invalid' if not valid else 'contact_and_push' if contact and moving
                    else 'contact_blocked_or_small' if contact else 'object_wall_no_positive_AO' if ow_pair
                    else 'searching_without_contact' if any(word in str(phase) for word in ('entry', 'approach', 'route', 'acquir'))
                    else 'empty_pusher_motion' if pusher_move*SCALE >= .1
                    else 'no_contact_object_motion' if moving else 'static')
        rows.append(dict(attempt_id=attempt['attempt_id'], family_id=attempt['family_id'], letter=letter,
            scenario=attempt['scenario'], control=index, valid=valid, planned_phase=phase, category=category,
            actual_seconds=seconds, actual_substeps=len(substeps), AO_positive_seconds=positive,
            AO_pair_seconds=ao_pair, OW_pair_seconds=ow_pair, OW_positive_seconds=ow_positive,
            AW_pair_seconds=aw_pair, invalid_substep_seconds=invalid_seconds,
            contact_and_push_seconds=seconds if contact and moving and valid else 0.,
            contact_blocked_or_small_seconds=seconds if contact and not moving and valid else 0.,
            translation_world_px=translation, translation_image_px=translation*SCALE,
            native_cumulative_translation_world_px=native_translation_path,
            native_cumulative_rotation_deg=native_rotation_path,
            signed_rotation_deg=rotation, rotation_deg=abs(rotation),
            pusher_motion_world_px=pusher_move, pusher_motion_image_px=pusher_move*SCALE,
            target_distance_from_pusher_world_px=float(np.linalg.norm(actions[index]-states[index, :2])),
            target_change_world_px=float(np.linalg.norm(actions[index]-actions[index-1])) if index else None,
            target_x=float(actions[index, 0]), target_y=float(actions[index, 1]),
            max_penetration_OW_AO_AW_world_px=penetration.tolist(), actual_wall_contact_seconds=dict(wall_seconds)))
    return rows, contact_points, arcs


def window_metrics(attempt, data, controls):
    positions = centers(attempt['letter'], data['states'])
    # Pymunk's body.angle is continuous already. Applying np.unwrap to large
    # legitimate per-control rotations would erase complete turns.
    angles = np.asarray(data['states'][:, 4], float)
    n = len(data['requested_actions'])
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    angle_steps = np.abs(np.rad2deg(np.diff(angles)))
    order = official_shape(attempt['letter']).symmetry_order
    period = 360./order
    rows = []
    blocks = attempt.get('blocks', [])
    all_precommitted = attempt.get('all_actions_precommitted', False)
    for horizon in HORIZONS:
        for anchor in range(3, n-horizon+1):
            if not legal_window(data, anchor, horizon):
                continue
            translation = float(np.linalg.norm(positions[anchor+horizon]-positions[anchor]))
            signed_angle = float(np.rad2deg(angles[anchor+horizon]-angles[anchor]))
            # All actions fixed before scene execution are known at every anchor.
            committed = all_precommitted or any(int(block['start']) <= anchor and anchor+horizon <= int(block['end'])
                                                for block in blocks)
            rows.append(dict(attempt_id=attempt['attempt_id'], family_id=attempt['family_id'],
                letter=attempt['letter'], scenario=attempt['scenario'], anchor=anchor, horizon=horizon,
                common_H16_anchor=legal_window(data, anchor, 16), action_causality='anchor_committed' if committed else 'feedback_recorded',
                translation_world_px=translation, translation_image_px=translation*SCALE,
                cumulative_translation_world_px=float(steps[anchor:anchor+horizon].sum()),
                cumulative_translation_image_px=float(steps[anchor:anchor+horizon].sum()*SCALE),
                native_cumulative_translation_world_px=float(sum(r['native_cumulative_translation_world_px'] for r in controls[anchor:anchor+horizon])),
                native_cumulative_translation_image_px=float(sum(r['native_cumulative_translation_world_px'] for r in controls[anchor:anchor+horizon])*SCALE),
                maximum_translation_world_px=float(np.linalg.norm(positions[anchor:anchor+horizon+1]-positions[anchor], axis=1).max()),
                signed_rotation_deg=signed_angle, rotation_deg=abs(signed_angle),
                cumulative_rotation_deg=float(angle_steps[anchor:anchor+horizon].sum()),
                native_cumulative_rotation_deg=float(sum(r['native_cumulative_rotation_deg'] for r in controls[anchor:anchor+horizon])),
                maximum_rotation_deg=float(np.abs(np.rad2deg(angles[anchor:anchor+horizon+1]-angles[anchor])).max()),
                symmetry_equivalent_rotation_deg=abs((signed_angle+period/2)%period-period/2), symmetry_order=order,
                AO_positive_seconds=float(sum(row['AO_positive_seconds'] for row in controls[anchor:anchor+horizon])),
                OW_pair_seconds=float(sum(row['OW_pair_seconds'] for row in controls[anchor:anchor+horizon])),
                obvious_translation=translation*SCALE >= OBVIOUS[horizon][0],
                obvious_rotation=abs(signed_angle) >= OBVIOUS[horizon][1],
                diagnostic_only=True, training_eligible=False))
    return rows


def _recontact(controls, lo=0, hi=None):
    """A positive AO period, >=2 controls without AO pair, then positive AO."""
    stage, gap = 0, 0
    for row in controls[lo:hi]:
        if stage == 0 and row['AO_positive_seconds'] > 0:
            stage = 1
        elif stage == 1:
            if row['AO_pair_seconds'] == 0:
                gap += 1
                if gap >= 2:
                    stage = 2
            else:
                gap = 0
        elif stage == 2 and row['AO_positive_seconds'] > 0:
            return True
    return False


def _summary_group(attempts, controls, windows):
    observed_seconds = sum(row['actual_seconds'] for row in controls)
    has_cost = any(a.get('_cost') for a in attempts)
    seconds = sum(float(a.get('_cost', {}).get('simulated_seconds', 0.)) for a in attempts) if has_cost else observed_seconds
    unknown_seconds = max(0., seconds-observed_seconds)
    positive = sum(row['AO_positive_seconds'] for row in controls)
    result = dict(planned_attempts=len(attempts), independent_families=len({a['family_id'] for a in attempts}),
        attempted=sum(a.get('_attempted', False) for a in attempts),
        raw_saved=sum(a.get('_raw_saved', False) for a in attempts),
        completed=sum(a.get('_complete', False) for a in attempts),
        invalid_terminated=sum(a.get('_invalid', False) for a in attempts),
        executed_controls=sum(int(a.get('_cost', {}).get('actual_controls', 0)) for a in attempts) if has_cost else len(controls),
        legal_controls=sum(int(a.get('_cost', {}).get('actual_valid_controls', 0)) for a in attempts) if has_cost else sum(row['valid'] for row in controls),
        actual_substeps=sum(int(a.get('_cost', {}).get('actual_substeps', 0)) for a in attempts) if has_cost else sum(row['actual_substeps'] for row in controls),
        executed_seconds=seconds, observed_control_records=len(controls), observed_seconds=observed_seconds,
        missing_contact_telemetry_seconds=unknown_seconds,
        AO_positive_seconds=positive, AO_positive_seconds_complete=unknown_seconds < 1e-8,
        AO_positive_fraction_all_executed=positive/seconds if seconds and unknown_seconds < 1e-8 else None,
        zero_contact_or_unstarted_attempts=sum(not a.get('_AO_positive_seconds', 0.) for a in attempts),
        OW_pair_seconds=sum(row['OW_pair_seconds'] for row in controls),
        OW_positive_seconds=sum(row['OW_positive_seconds'] for row in controls),
        contact_and_push_control_seconds=sum(row['contact_and_push_seconds'] for row in controls),
        contact_and_push_definition='positive AO during control AND (centroid translation >=0.1 imagepx OR abs raw rotation >=0.5deg); entire control duration; distinct from positive microstep time',
        control_categories=dict(Counter(row['category'] for row in controls)),
        control_category_seconds={key: sum(row['actual_seconds'] for row in controls if row['category'] == key)
                                  for key in sorted({row['category'] for row in controls})},
        target_distance_from_pusher_world_px=distribution(row['target_distance_from_pusher_world_px'] for row in controls),
        target_change_world_px=distribution(row['target_change_world_px'] for row in controls if row['target_change_world_px'] is not None),
        actual_pusher_motion_world_px=distribution(row['pusher_motion_world_px'] for row in controls),
        horizons={}, paired_common_H16={})
    # Pushing phase is an explicit plan label, never inferred from success.
    push = [row for row in controls if any(word in str(row['planned_phase']) for word in ('push', 'drive', 'main', 'wall', 'slide', 'recontact'))]
    push_seconds = sum(row['actual_seconds'] for row in push)
    result['planned_pushing_seconds'] = push_seconds
    result['planned_pushing_AO_positive_seconds'] = sum(row['AO_positive_seconds'] for row in push)
    result['planned_pushing_AO_positive_fraction'] = result['planned_pushing_AO_positive_seconds']/push_seconds if push_seconds else None
    for horizon in HORIZONS:
        for paired, destination in ((False, 'horizons'), (True, 'paired_common_H16')):
            rows = [row for row in windows if row['horizon'] == horizon and (not paired or row['common_H16_anchor'])]
            stats = dict(legal_history4_windows=len(rows), independent_families=len({row['family_id'] for row in rows}),
                unique_attempts=len({row['attempt_id'] for row in rows}),
                action_causality=dict(Counter(row['action_causality'] for row in rows)),
                obvious_translation_count=sum(row['obvious_translation'] for row in rows),
                obvious_rotation_count=sum(row['obvious_rotation'] for row in rows),
                obvious_translation_threshold_image_px=OBVIOUS[horizon][0],
                obvious_rotation_threshold_deg=OBVIOUS[horizon][1],
                clockwise_raw_rotation_count=sum(row['signed_rotation_deg'] >= OBVIOUS[horizon][1] for row in rows),
                counterclockwise_raw_rotation_count=sum(row['signed_rotation_deg'] <= -OBVIOUS[horizon][1] for row in rows))
            for field in ('translation_world_px', 'translation_image_px', 'cumulative_translation_world_px',
                          'cumulative_translation_image_px', 'maximum_translation_world_px', 'rotation_deg',
                          'native_cumulative_translation_world_px', 'native_cumulative_translation_image_px',
                          'native_cumulative_rotation_deg',
                          'signed_rotation_deg', 'cumulative_rotation_deg', 'maximum_rotation_deg',
                          'symmetry_equivalent_rotation_deg'):
                stats[field] = distribution((row[field] for row in rows),
                    TRANSLATION_BINS if field.endswith('image_px') else ROTATION_BINS if field.endswith('_deg') and not field.startswith('signed') else None)
            result[destination][str(horizon)] = stats
    return result


def select_examples(attempts, controls, windows):
    """Fixed actual-outcome qualification; absent positives remain unavailable."""
    examples = []
    by_id = {attempt['attempt_id']: attempt for attempt in attempts}
    control_groups = {key: [row for row in controls if row['attempt_id'] == key] for key in by_id}
    for letter in LETTERS:
        available = [row for row in windows if row['letter'] == letter]
        for kind in ('translation', 'clockwise', 'counterclockwise', 'boundary', 'recontact'):
            if kind == 'translation':
                candidates = [row for row in available if row['obvious_translation']]
                score = lambda r: (-r['horizon'], -r['translation_image_px']/OBVIOUS[r['horizon']][0], r['attempt_id'], r['anchor'])
            elif kind in ('clockwise', 'counterclockwise'):
                sign = 1 if kind == 'clockwise' else -1  # screen y down
                candidates = [row for row in available if sign*row['signed_rotation_deg'] >= OBVIOUS[row['horizon']][1]]
                score = lambda r: (-r['horizon'], -abs(r['signed_rotation_deg'])/OBVIOUS[r['horizon']][1], r['attempt_id'], r['anchor'])
            elif kind == 'boundary':
                candidates = [row for row in available if row['OW_pair_seconds'] > 0]
                score = lambda r: (-r['horizon'], -r['OW_pair_seconds'], r['attempt_id'], r['anchor'])
            else:
                candidates = [row for row in available if row['horizon'] == 16 and
                              _recontact(control_groups[row['attempt_id']], row['anchor'], row['anchor']+16)]
                score = lambda r: (r['attempt_id'], r['anchor'])
            selected = min(candidates, key=score) if candidates else None
            examples.append(dict(letter=letter, kind=kind, available=bool(selected),
                reason=None if selected else 'No complete legal qualifying saved window; not replaced or relabelled',
                selected=selected))
    rng = random.Random(5478)
    failed = sorted(a['attempt_id'] for a in attempts if a.get('_raw_saved') and not a.get('_complete'))
    empty = sorted(a['attempt_id'] for a in attempts if a.get('_raw_saved') and not a.get('_AO_positive_seconds', 0.))
    for kind, pool in (('failure', failed), ('zero_contact', empty)):
        for attempt_id in rng.sample(pool, min(2, len(pool))):
            examples.append(dict(letter=by_id[attempt_id]['letter'], kind=kind, available=True,
                                 selected=dict(attempt_id=attempt_id, anchor=None, horizon=None)))
    return examples


def _png(canvas):
    import cv2
    ok, packed = cv2.imencode('.png', canvas)
    if not ok:
        raise RuntimeError('PNG encoder failed')
    return packed.tobytes()


def _picture(rgb):
    import cv2
    return cv2.cvtColor(rgb.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)


def truth_board(attempt, data, anchor):
    import cv2
    canvas = np.full((272, 1040, 3), 248, np.uint8)
    cv2.putText(canvas, 'SAVED TRUE RGB | no model prediction | '+attempt['attempt_id'], (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1, cv2.LINE_AA)
    for column, offset in enumerate((0, 1, 4, 8, 16)):
        image = cv2.resize(_picture(data['rgb'][anchor+offset]), (192, 192), interpolation=cv2.INTER_NEAREST)
        x = 8+column*208
        canvas[60:252, x:x+192] = image
        cv2.putText(canvas, f't+{offset} | frame {anchor+offset}', (x, 49), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (30, 30, 30), 1, cv2.LINE_AA)
    return _png(canvas)


def _video(path, attempt, data, rows, lease):
    import cv2
    import imageio_ffmpeg
    limit = 3*1024**2
    lease.ensure_capacity(limit+1024**2)
    path.parent.mkdir(parents=True, exist_ok=True)
    old_size = path.stat().st_size if path.exists() else 0
    process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '576x688', '-r', '10', '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-fs', str(limit), str(path)], stdin=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    error = None
    position = centers(attempt['letter'], data['states'])
    try:
        for index, rgb in enumerate(data['rgb']):
            canvas = np.full((688, 576, 3), 248, np.uint8)
            picture = cv2.resize(_picture(rgb), (480, 480), interpolation=cv2.INTER_NEAREST)
            target_index = min(index, len(data['requested_actions'])-1)
            if target_index >= 0:
                targets = data['requested_actions'][:target_index+1]*480/512
                if len(targets) > 1:
                    cv2.polylines(picture, [np.rint(targets).astype(np.int32)], False, (100, 180, 210), 1, cv2.LINE_AA)
                target = tuple(np.rint(targets[-1]).astype(int))
                cv2.drawMarker(picture, target, (30, 30, 220), cv2.MARKER_CROSS, 13, 2)
                actual = tuple(np.rint(data['states'][index, :2]*480/512).astype(int))
                cv2.circle(picture, actual, 16, (220, 90, 20), 1)
            canvas[198:678, 48:528] = picture
            previous = rows[index-1] if index and index-1 < len(rows) else None
            translation = float(np.linalg.norm(position[index]-position[0])*SCALE)
            angle = float(np.rad2deg(data['states'][index, 4]-data['states'][0, 4]))
            lines = ['ORIGINAL100Hz solver10 | SAVED TRUE RGB | no model',
                f"{attempt['attempt_id']} | {attempt['scenario']}",
                f"frame {index}/{len(data['rgb'])-1} | saved t={data['control_time_seconds'][index]:.3f}s | 10fps",
                f'centroid net={translation:.3f} imagepx | raw angle={angle:+.2f}deg (screen +CW)',
                f"previous control AO+={previous['AO_positive_seconds']:.2f}s OW={previous['OW_pair_seconds']:.2f}s" if previous else 'initial frame; history consists of real controls',
                'red cross/path=planned target | blue ring=actual pusher',
                'INVALID terminal partial: actual timestamp shown; held one display frame' if previous and not previous['valid'] else 'No skipped/interpolated frames; overlays are display only']
            for line, content in enumerate(lines):
                cv2.putText(canvas, content, (10, 23+25*line), cv2.FONT_HERSHEY_SIMPLEX, .43,
                            (30, 40, 50), 1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
    except BaseException as exc:
        error = exc
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        code = process.wait()
        if path.exists():
            lease.commit_external_write(path, previous_size=old_size)
    if error or code or not path.exists() or path.stat().st_size >= limit:
        raise RuntimeError('Bounded real-time media encoding failed; partial evidence preserved') from error
    capture = cv2.VideoCapture(str(path))
    fps, count = capture.get(cv2.CAP_PROP_FPS), 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count != len(data['rgb']) or abs(fps-10) > 1e-6:
        raise RuntimeError('Actual decoded media frames/fps do not match complete saved record')
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size, fps=fps, frames=count,
                video_seconds=count/fps, executed_simulation_seconds=float(data['control_time_seconds'][-1]),
                timing='Each saved frame once at10fps. The initial/end frame adds0.1s; invalid partial shown at actual timestamp, no altered metric timing.')


def contact_heatmap(letter, points, arcs, attempts, controls):
    import cv2
    canvas = np.full((720, 720, 3), 250, np.uint8)
    geometry = _geometry(letter)
    x0, y0, x1, y1 = geometry.bounds
    factor = min(400/(x1-x0), 400/(y1-y0))
    def xy(values):
        return np.rint((np.asarray(values)-[x0, y0])*factor+[140, 70]).astype(np.int32)
    cv2.fillPoly(canvas, [xy(geometry.exterior.coords)], (175, 180, 185))
    for ring in geometry.interiors:
        cv2.fillPoly(canvas, [xy(ring.coords)], (250, 250, 250))
    for point in points:
        cv2.circle(canvas, tuple(xy([point['local_x'], point['local_y']])), 2, (20, 50, 220), -1)
    from shapely.geometry import LineString
    exterior = LineString(geometry.exterior.coords)
    counts = Counter()
    families = defaultdict(set)
    for arc in arcs:
        if arc['ring'] == 0:
            counts[arc['arc16']] += arc['seconds']
            families[arc['arc16']].add(arc['family_id'])
    for arc in range(16):
        p = exterior.interpolate((arc+.5)/16*exterior.length)
        cv2.putText(canvas, str(arc), tuple(xy([p.x, p.y])), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (20, 100, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, letter+' | ACTUAL positive-impulse contact (local coordinates)', (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    for index in range(16):
        column, row = index//8, index%8
        cv2.putText(canvas, f'arc{index:02d}: {counts[index]:.2f}s | {len(families[index])} families',
                    (10+column*350, 505+row*23), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Closed O/B holes excluded from required exterior coverage; unknown reachability retained.',
                (8, 707), cv2.FONT_HERSHEY_SIMPLEX, .36, (40, 40, 40), 1, cv2.LINE_AA)
    return _png(canvas)


def generate_report(root):
    root = Path(root).resolve()
    config = read(root/'runtime-config.json')
    plan = read(root/'plan.json')
    roots_path = Path(config.get('storage_roots_json', config.get('storage_roots', root/'storage-roots.json')))
    roots_value = read(roots_path)
    shared_roots = roots_value['roots'] if isinstance(roots_value, dict) else roots_value
    policy = Path(config.get('resource_policy', config.get('resource_policy_path', root/'policies/resource-policy.json')))
    attempts = [dict(attempt) for attempt in plan['attempts']]
    controls, windows, points, arcs, hashes, sources = [], [], [], [], [], {}
    by_family = defaultdict(set)
    collection = root/'collection'
    ledger = read(collection/'ledger.json') if (collection/'ledger.json').exists() else {}
    ledger_attempts = {a['attempt_id']: a for a in ledger.get('attempts', [])}
    terminal = read(collection/'completion.json') if (collection/'completion.json').exists() else {}
    unframed_tails = []
    for attempt in attempts:
        attempt_id = attempt['attempt_id']
        metadata_path = collection/'attempts'/f'{attempt_id}.json'
        raw_path = collection/'raw'/f'{attempt_id}.npz'
        metadata = read(metadata_path) if metadata_path.exists() else {}
        cost = metadata.get('cost', ledger_attempts.get(attempt_id, {}))
        status = metadata.get('status', cost.get('status', 'unstarted'))
        attempt.update(_attempted=bool(metadata) or attempt_id in ledger_attempts, _raw_saved=raw_path.exists(),
            _complete=bool(metadata.get('complete', metadata.get('completed', False))),
            _invalid=bool(metadata.get('first_invalid') or metadata.get('first_invalid_substep') or 'invalid' in status),
            _status=status, _metadata=metadata, _cost=cost)
        tail = cost.get('unframed_terminal_control')
        if tail:
            unframed_tails.append(dict(attempt_id=attempt_id, family_id=attempt['family_id'], letter=attempt['letter'], **tail))
        split = attempt.get('split', 'development_acceptance')
        by_family[attempt['family_id']].add(split)
        if not raw_path.exists():
            continue
        data = load_attempt(raw_path)
        n = validate_timeline(data)
        planned = np.asarray(attempt.get('actions', []), float)
        if planned.shape[1:] != (2,) or not np.array_equal(data['requested_actions'], planned[:n]):
            raise ValueError(f'Saved actions differ from frozen plan: {attempt_id}')
        actual_hash = digest(raw_path)
        metadata_raw = metadata.get('raw', metadata.get('raw_record', {}))
        if isinstance(metadata_raw, dict) and metadata_raw.get('sha256') and metadata_raw['sha256'] != actual_hash:
            raise ValueError(f'Saved raw hash differs from actual receipt: {attempt_id}')
        rows, actual_points, actual_arcs = control_metrics(attempt, data)
        actual_windows = window_metrics(attempt, data, rows)
        attempt['_AO_positive_seconds'] = sum(row['AO_positive_seconds'] for row in rows)
        attempt['_recontact_actual'] = _recontact(rows)
        attempt['_invalid'] = attempt['_invalid'] or bool(np.any(~data['control_valid']))
        attempt['_controls'] = n
        controls.extend(rows)
        points.extend(actual_points)
        arcs.extend(actual_arcs)
        windows.extend(actual_windows)
        sources[attempt_id] = raw_path
        hashes.append(dict(attempt_id=attempt_id, family_id=attempt['family_id'], path=str(raw_path),
                           sha256=actual_hash, bytes=raw_path.stat().st_size, saved_controls=n,
                           saved_frames=len(data['rgb']), requested_equals_passed=True, equals_frozen_plan=True))
        if tail:
            # A saved engineering tail is audit evidence, never an RGB frame or
            # window. Its real integration and contact time remain in totals.
            info = tail.get('telemetry', {})
            substeps = [s for s in info.get('substeps', []) if s.get('executed')]
            before = substeps[0].get('pre_state', substeps[0].get('integrator_pre_state')) if substeps else states_last(data)
            after = tail.get('post_state')
            if before is not None and after is not None and np.isfinite(np.asarray([before, after], float)).all():
                tail_data = dict(states=np.asarray([before, after], float), requested_actions=np.asarray([tail['action']], float),
                    control_valid=np.asarray([False]), telemetry=[info], decisions=[dict(phase='engineering_unframed_terminal')])
                tail_rows, tail_points, tail_arcs = control_metrics(attempt, tail_data)
                for r in tail_rows+tail_points+tail_arcs:
                    r['control'] = tail['control']
                controls.extend(tail_rows); points.extend(tail_points); arcs.extend(tail_arcs)
            else:
                controls.append(_unknown_motion_tail(attempt, tail))
            attempt['_AO_positive_seconds'] = sum(row['AO_positive_seconds'] for row in controls if row['attempt_id'] == attempt_id)
    if any(len(value) != 1 for value in by_family.values()):
        raise ValueError('A scene family crosses dataset splits')
    summary = dict(kind='direct-target-saved-truth-acceptance-report.v1', run_id=config.get('run_id'),
        runtime_config_sha256=digest(root/'runtime-config.json'), plan_sha256=digest(root/'plan.json'),
        model_calls=0, optimizer_updates=0, report_physics_calls=0, training_eligible=False,
        interpretation='Privileged saved ground truth only; overlapping windows are not independent families; no learned prediction or training claim.',
        original_action_domain=[0, 512], old_model_action_domain=[0, 511],
        action_compatibility_todo='Raw world targets retained without512->511 clipping. Existing model a/511 normalization is unchanged and needs separately approved compatibility work.',
        contact_impulse_definition='Strictly positive Euclidean total_impulse in post_solve AO event, actual executed .01s substeps; zero and missing distinguished by raw telemetry.',
        diagnostic_contact_push_definition='contact AND (translation>=0.1 imagepx OR abs raw rotation>=0.5deg) per actual control. Not an admission/pass threshold.',
        planned_attempt_denominator=len(attempts), all=_summary_group(attempts, controls, windows),
        by_letter={}, by_scenario={}, by_letter_scenario={}, coverage={}, branch_pairs=[], unavailable=[],
        cost={'simulation_seconds_not_cpu_seconds': sum(row['actual_seconds'] for row in controls),
              'regular_executed_substeps': sum(row['actual_substeps'] for row in controls),
              'initialization_substeps': sum(a['_metadata'].get('reset_or_initialization_substeps', a['_cost'].get('initialization_integrator_calls', 0)) for a in attempts),
              'attempt_wall_seconds': sum(float(a['_cost'].get('wall_seconds', 0.)) for a in attempts),
              'collector_wall_seconds': terminal.get('wall_seconds'),
              'ledger_actual_controls': ledger.get('actual_controls'), 'ledger_actual_valid_controls': ledger.get('actual_valid_controls'),
              'ledger_actual_substeps': ledger.get('actual_substeps'), 'ledger_simulated_seconds': ledger.get('simulated_seconds'),
              'ledger_initialization_integrator_calls': ledger.get('initialization_integrator_calls'),
              'ledger_initialization_integrator_seconds': ledger.get('initialization_integrator_seconds'),
              'ledger_regular_integrator_calls': ledger.get('regular_integrator_calls'),
              'ledger_regular_integrator_seconds': ledger.get('regular_integrator_seconds')},
        collector_terminal=terminal, unframed_terminal_controls=unframed_tails,
        unframed_note='Actual calls and telemetry retained in metadata/cost. Missing RGB is never copied or manufactured. All windows stop at the last real saved frame.')
    summary['cost']['audit_control_count_matches_ledger'] = ledger.get('actual_controls') in (None, len(controls))
    summary['cost']['audit_substep_count_matches_ledger'] = ledger.get('actual_substeps') in (None, sum(r['actual_substeps'] for r in controls))
    summary['cost']['audit_seconds_match_ledger'] = ledger.get('simulated_seconds') is None or abs(ledger['simulated_seconds']-sum(r['actual_seconds'] for r in controls)) < 1e-8
    for dimension, destination in (('letter', 'by_letter'), ('scenario', 'by_scenario')):
        for key in sorted({attempt[dimension] for attempt in attempts}):
            aa = [a for a in attempts if a[dimension] == key]
            cc = [r for r in controls if r[dimension] == key]
            ww = [r for r in windows if r[dimension] == key]
            summary[destination][key] = _summary_group(aa, cc, ww)
    for letter in LETTERS:
        summary['by_letter_scenario'][letter] = {}
        for scenario in sorted({a['scenario'] for a in attempts if a['letter'] == letter}):
            summary['by_letter_scenario'][letter][scenario] = _summary_group(
                [a for a in attempts if a['letter'] == letter and a['scenario'] == scenario],
                [r for r in controls if r['letter'] == letter and r['scenario'] == scenario],
                [r for r in windows if r['letter'] == letter and r['scenario'] == scenario])
        aa = [a for a in attempts if a['letter'] == letter]
        rr = [r for r in arcs if r['letter'] == letter]
        cc = [r for r in controls if r['letter'] == letter]
        arc_stats = [dict(arc16=index, positive_contact_seconds=sum(r['seconds'] for r in rr if r['ring'] == 0 and r['arc16'] == index),
                         independent_families=len({r['family_id'] for r in rr if r['ring'] == 0 and r['arc16'] == index}),
                         reachability='unknown_not_removed_from_denominator') for index in range(16)]
        wall = {side: sum(r['actual_wall_contact_seconds'].get(side, 0.) for r in cc)
                for side in ('left', 'right', 'top', 'bottom', 'left-top', 'right-top', 'left-bottom', 'right-bottom', 'unknown')}
        poses = Counter()
        family_seen = set()
        for a in aa:
            if a['family_id'] in family_seen:
                continue
            family_seen.add(a['family_id'])
            angle = a.get('initialization', {}).get('angle', (a.get('initial_state15') or [0]*5)[4])
            poses[int((float(angle)%(2*math.pi))/(2*math.pi)*8)%8] += 1
        summary['coverage'][letter] = dict(exterior_arc16=arc_stats,
            unobserved_exterior_arcs=[r['arc16'] for r in arc_stats if not r['positive_contact_seconds']],
            closed_holes='Not externally reachable required targets for O/B. Actual inner-ring hits remain in contact-points.csv.',
            initial_pose_bins8=[poses[i] for i in range(8)], actual_wall_contact_seconds=wall,
            missing_wall_sides=[side for side in ('left', 'right', 'top', 'bottom') if not wall[side]],
            missing_wall_corners=[side for side in ('left-top', 'right-top', 'left-bottom', 'right-bottom') if not wall[side]],
            wall_note='Post-solve object-wall arbiter pair and actual contact coordinates; positive impulse time is separate. Near-wall intent is not contact. Corner requires both wall sides during same microstep.',
            actual_recontact_attempts=[a['attempt_id'] for a in aa if a.get('_recontact_actual')],
            selected_point_distance_world_px=distribution(p['selected_point_distance_world_px'] for p in points if p['letter'] == letter and p['selected_point_distance_world_px'] is not None),
            actual_contact_points=len([p for p in points if p['letter'] == letter]),
            actual_points_in_selected_exterior_arc16=sum(p['same_selected_exterior_arc16'] for p in points if p['letter'] == letter),
            selected_point_note='Continuous actual-local to planned-local point distance and same16-exterior-arc counts, not an invented selected-point success threshold. Any-contact time is separately retained.',
            clockwise_note='Positive raw angle is screen clockwise (y down); symmetry-equivalent angle is separately reported.')
    # Full native reconstruction/common prefix has already been charged to each
    # attempt; saved equality is independently audited here, never replayed.
    for family_id in sorted(by_family):
        family = [a for a in attempts if a['family_id'] == family_id and a['attempt_id'] in sources]
        if len(family) < 2:
            continue
        base = load_attempt(sources[family[0]['attempt_id']])
        for a in family[1:]:
            data = load_attempt(sources[a['attempt_id']])
            prefix = min(3, len(base['requested_actions']), len(data['requested_actions']))
            same = prefix == 3 and np.array_equal(base['states'][:4], data['states'][:4]) and np.array_equal(base['rgb'][:4], data['rgb'][:4]) and np.array_equal(base['requested_actions'][:3], data['requested_actions'][:3])
            complete = len(base['states']) >= 20 and len(data['states']) >= 20 and legal_window(base, 3, 16) and legal_window(data, 3, 16)
            summary['branch_pairs'].append(dict(family_id=family_id, baseline=family[0]['attempt_id'], branch=a['attempt_id'],
                full_common_history_equal=same, H16_legal_both=complete,
                outcome_centroid_separation_H16_image_px=float(np.linalg.norm(centers(a['letter'], base['states'])[19]-centers(a['letter'], data['states'])[19])*SCALE) if complete else None,
                outcome_raw_angle_separation_H16_deg=abs(float(np.rad2deg(base['states'][19, 4]-data['states'][19, 4]))) if complete else None))
    examples = select_examples(attempts, controls, windows)
    summary['unavailable'] = [dict(letter=e['letter'], kind=e['kind'], reason=e['reason']) for e in examples if not e['available']]
    summary['selection_rule'] = 'Qualify actual legal threshold first, prefer H16 then8/4/1, greatest normalized actual magnitude, attempt ID then anchor. Boundary requires actual OW; recontact requires contact, >=2 AO-pair-free controls, contact within one legal H16. Failure/zero-contact use fixed seed5478.'
    report = root/'report'
    media = {}
    with ResourceReservation(root, shared_roots, policy, 'direct-target-saved-only-report', role='postprocess') as lease:
        for filename, values in (('controls.csv', controls), ('windows.csv', windows),
                                 ('paired-common-H16.csv', [r for r in windows if r['common_H16_anchor']]),
                                 ('contact-points.csv', points), ('contact-arcs.csv', arcs),
                                 ('raw-manifest.csv', hashes)):
            lease.write_bytes(report/filename, _csv(values))
        index = []
        for a in attempts:
            index.append(dict(attempt_id=a['attempt_id'], family_id=a['family_id'], letter=a['letter'], scenario=a['scenario'],
                branch=a.get('branch', a.get('branch_id')), status=a['_status'], attempted=a['_attempted'], complete=a['_complete'],
                raw_saved=a['_raw_saved'], controls=a.get('_controls', 0), invalid_terminated=a['_invalid'],
                AO_positive_seconds=a.get('_AO_positive_seconds', 0.), recontact_actual=a.get('_recontact_actual', False),
                metadata_path=str(collection/'attempts'/f"{a['attempt_id']}.json"),
                raw_path=str(sources.get(a['attempt_id'], '')), first_invalid=a['_metadata'].get('first_invalid', a['_cost'].get('first_invalid')),
                error=a['_cost'].get('error'), actual_controls_including_unframed=a['_cost'].get('actual_controls'),
                actual_substeps_including_unframed=a['_cost'].get('actual_substeps'), unframed_terminal_control=bool(a['_cost'].get('unframed_terminal_control'))))
        lease.write_bytes(report/'attempt-index.csv', _csv(index))
        for letter in LETTERS:
            lease.write_bytes(report/f'contact-heatmap-{letter}.png', contact_heatmap(letter,
                [p for p in points if p['letter'] == letter], [p for p in arcs if p['letter'] == letter],
                [a for a in attempts if a['letter'] == letter], [r for r in controls if r['letter'] == letter]))
        for example in examples:
            if not example['available']:
                continue
            selected = example['selected']
            attempt_id = selected['attempt_id']
            attempt = next(a for a in attempts if a['attempt_id'] == attempt_id)
            data = load_attempt(sources[attempt_id])
            if attempt_id not in media:
                rows = [row for row in controls if row['attempt_id'] == attempt_id]
                media[attempt_id] = _video(report/'videos'/f'{attempt_id}.mp4', attempt, data, rows, lease)
            example['media'] = media[attempt_id]
            anchor = selected.get('anchor')
            if anchor is not None and legal_window(data, anchor, 16):
                board = report/'truth-boards'/f'{attempt_id}-t{anchor}.png'
                lease.write_bytes(board, truth_board(attempt, data, anchor))
                example['truth_board'] = str(board)
        summary['examples'] = examples
        summary['media_unique_full_attempt_videos'] = len(media)
        summary['guard_snapshot'] = lease.snapshot()
        lease.write_bytes(report/'summary.json', _json(summary))
        body = _markdown(summary)
        lease.write_bytes(root/'REPORT.md', body.encode('utf-8'))
        lease.write_bytes(report/'index.html', _html(summary, body).encode('utf-8'))
        manifest = [dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
                    for path in sorted(report.rglob('*')) if path.is_file() and path.name != 'final-manifest.json']
        manifest.append(dict(path=str(root/'REPORT.md'), bytes=(root/'REPORT.md').stat().st_size, sha256=digest(root/'REPORT.md')))
        lease.write_bytes(report/'final-manifest.json', _json(dict(run_id=config.get('run_id'), files=manifest,
            raw_manifest=hashes, report_physics_calls=0, model_calls=0, optimizer_updates=0)))
        lease.write_bytes(report/'resource-receipt.json', _json(lease.snapshot()))
    return summary


def _num(value, digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def _markdown(summary):
    all_rows = summary['all']
    lines = ['# Direct-target acceptance: saved ground truth', '',
        f"Run: `{summary['run_id']}`. Planned {all_rows['planned_attempts']} attempts / {all_rows['independent_families']} independent scene families; "
        f"attempted {all_rows['attempted']}, raw saved {all_rows['raw_saved']}, complete {all_rows['completed']}, invalid terminated {all_rows['invalid_terminated']}.", '',
        f"Actual {all_rows['executed_controls']} controls / {all_rows['legal_controls']} legal controls / {all_rows['actual_substeps']} native substeps; "
        f"{_num(all_rows['executed_seconds'])} simulated seconds (not CPU seconds). Model calls=0, optimizer updates=0; this report executes no physics.", '',
        'These are true observations, not model predictions. No collection expansion or training is released by this report.', '',
        '## Motion at identical H16-capable anchors', '',
        'Each cell is P10 / P50 / P90. Centroid displacement uses actual geometry area centroid, not body origin. Rotation is raw unwrapped absolute net angle; symmetry-equivalent and cumulative angles are in summary.json and CSV. Overlapping windows are not independent.', '',
        '|Letter|H|Independent families|Legal paired windows|Net translation imagepx|Net rotation deg|Obvious translation / rotation|',
        '|---|---:|---:|---:|---|---|---|']
    for letter in LETTERS:
        for horizon in HORIZONS:
            item = summary['by_letter'][letter]['paired_common_H16'][str(horizon)]
            t = item['translation_image_px']
            r = item['rotation_deg']
            lines.append(f"|{letter}|{horizon}|{item['independent_families']}|{item['legal_history4_windows']}|"
                f"{' / '.join(_num(t[key]) for key in ('P10','P50','P90'))}|"
                f"{' / '.join(_num(r[key]) for key in ('P10','P50','P90'))}|"
                f"{item['obvious_translation_count']} / {item['obvious_rotation_count']}|")
    lines += ['', 'Own-horizon denominators (including valid short prefixes) and per-letter/per-scenario distributions are in summary.json; paired-common-H16.csv uses the same anchor set across all four horizons.', '',
        '## Contact and complete denominators', '',
        '|Letter|Attempted / planned|Complete|AO positive time / all executed|AO positive fraction|OW pair seconds|Contact-and-push control seconds|',
        '|---|---|---:|---|---:|---:|---:|']
    for letter in LETTERS:
        row = summary['by_letter'][letter]
        lines.append(f"|{letter}|{row['attempted']} / {row['planned_attempts']}|{row['completed']}|"
            f"{_num(row['AO_positive_seconds'])} / {_num(row['executed_seconds'])}|{_num(row['AO_positive_fraction_all_executed'])}|"
            f"{_num(row['OW_pair_seconds'])}|{_num(row['contact_and_push_control_seconds'])}|")
    lines += ['', 'AO contact ratio counts actual post-solve positive impulse microstep time / all executed simulation time. Attempts with no executed time stay in attempt counts, not an invented time denominator. Missing contact telemetry makes the full ratio unavailable, with exact missing seconds disclosed; observed contact is never passed off as complete. Contact-and-push is a separate control diagnostic: contact AND (>=0.1 imagepx translation OR >=0.5deg rotation). It does not establish sufficient motion. Static, blocked/small contact, empty motion, searching, invalid and wall-only categories retain all actual controls.', '',
        'Display thresholds, translation / rotation separately: H1 1px / 3deg; H4 2px / 5deg; H8 4px / 10deg; H16 8px / 20deg. These select media, not data admission. Missing examples remain unavailable.', '',
        '## Coverage and missing evidence', '']
    for letter in LETTERS:
        item = summary['coverage'][letter]
        lines.append(f"- {letter}: unobserved exterior arcs {item['unobserved_exterior_arcs']}; "
            f"missing actual wall sides {item['missing_wall_sides']}; missing actual corners {item['missing_wall_corners']}; "
            f"actual continuous recontact attempts {len(item['actual_recontact_attempts'])}.")
    lines += ['', 'All16 exterior bins stay visible, with unknown reachability retained. O/B closed holes are not required externally accessible targets. A boundary-intended scene is not labelled actual wall contact without telemetry. Corner coverage requires both actual wall sides during one microstep.', '', 'Unavailable qualifying media:']
    lines += [f"- {row['letter']} {row['kind']}: {row['reason']}." for row in summary['unavailable']] or ['- None.']
    lines += ['', '## Media, identity and limits', '',
        'Open report/index.html for full-attempt real10Hz videos, actual contact heatmaps and same-anchor t/t+1/t+4/t+8/t+16 truth boards. Every saved frame appears once; an invalid final partial is explicitly timestamped and held for one display frame. Display overlays never enter raw RGB.', '',
        'Full attempts: report/attempt-index.csv. Raw identity and request=passed=frozen-plan checks: report/raw-manifest.csv. Complete controls/contacts/windows: report/*.csv. All outcomes, branch equality and actual outcome separation, per-scene statistics, action distances/changes and actual pusher motion are separate in report/summary.json.', '',
        f"Original targets use [0,512]. Existing model uses [0,511] and a/511 normalization; no model interface was changed, no512->511 target clipping applied. Later compatibility work requires approval.", '',
        'Sampling uses privileged geometry/state only for planning and these true-data audits. No learned-model result can be inferred. Subsequent formal collection and training require user review.', '']
    return '\n'.join(lines)


def _html(summary, body):
    parts = ['<!doctype html><meta charset="utf-8"><title>Direct-target true-data acceptance</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:0 20px}pre{white-space:pre-wrap}video{width:min(100%,480px)}img{max-width:100%}section{margin:28px 0;padding:16px;border:1px solid #ccc}a{overflow-wrap:anywhere}</style>',
        '<h1>Saved true observations — no model prediction</h1>',
        '<p><a href="../REPORT.md">Report</a> · <a href="summary.json">Complete statistics</a> · <a href="attempt-index.csv">All attempts</a> · <a href="paired-common-H16.csv">Paired H1/H4/H8/H16</a></p>',
        '<pre>'+html.escape(body)+'</pre>']
    for letter in LETTERS:
        parts.append(f'<h2>{letter}: actual contact coverage</h2><img width="720" src="contact-heatmap-{letter}.png">')
        for example in (row for row in summary['examples'] if row['letter'] == letter):
            parts.append('<section><h3>'+html.escape(example['kind'])+'</h3>')
            if not example['available']:
                parts.append('<p>Unavailable: '+html.escape(example['reason'])+'</p></section>')
                continue
            selected = example['selected']
            parts.append('<p>'+html.escape(selected['attempt_id'])+'; full actual attempt, normal10fps.</p>')
            parts.append('<video controls preload="metadata" src="videos/'+html.escape(Path(example['media']['path']).name)+'"></video>')
            if example.get('truth_board'):
                parts.append('<p>Same-anchor saved truth: t / t+1 / t+4 / t+8 / t+16</p><img src="truth-boards/'+html.escape(Path(example['truth_board']).name)+'">')
            if selected.get('horizon'):
                parts.append('<pre>'+html.escape(json.dumps(selected, indent=2))+'</pre>')
            parts.append('</section>')
    return '\n'.join(parts)
