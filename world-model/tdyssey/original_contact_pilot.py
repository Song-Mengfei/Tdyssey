"""Bounded six-letter candidate collection; no learning or model imports.

Geometry/state are privileged collection priors and audit labels, never predictor
inputs. All original images, actions and actual microstep telemetry are retained.
Physics execution requires a separately completed gate and frozen plan.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import zlib

import numpy as np
from shapely import affinity
from shapely.geometry import Point

from .native_rgb_env import shape_geometry
from .native_rgb_r3a_collection import area_center, rot, near_initial, segments
from .native_rgb_r2b_sampling import (
    geometry_at, wall_gaps, contact_frame, _stage, _free, free_route,
    _line, _departure_route, _candidate, _room,
)

LETTERS = tuple('TLOCXB')
LOCATIONS = ('center', 'left', 'right', 'top', 'bottom', 'left-top',
             'right-top', 'left-bottom', 'right-bottom')
SCENARIOS = ('translation_nearCOM', 'rotation_positive', 'rotation_negative',
             'mixed_positive', 'mixed_negative', 'grazing', 'static')
COUNTS = {'train_candidate': (20, 8, 8, 4, 4, 2, 2),
          'seen_validation': (5, 2, 2, 1, 1, 1, 0)}
RULES = dict(
    kind='original-contact-pilot64.v1', letters=list(LETTERS),
    controls=64, control_seconds=.1, image_size=96, world_size=512,
    train_per_letter=48, validation_per_letter=12,
    calibration_per_letter=3, calibration_revision_maximum=1,
    supplement_per_letter_maximum=12,
    scenario_counts={k: list(v) for k, v in COUNTS.items()},
    boundary_scene_fraction=.25,
    phases=[[0, 2, 'approach'], [2, 28, 'push'], [28, 30, 'stop'],
            [30, 34, 'withdraw'], [34, 36, 'recontact'], [36, 64, 'push']],
    exterior_arcs_only=True, closed_holes='not reachable from exterior; not targeted',
    collection_priors='actual current state, union geometry and past contact; privileged; not model input',
    push_advance_world_px=24., entry_advance_world_px=12.,
    departure_distances_world_px=[48., 33., 18.],
    target_distance_world_px=48., target_change_world_px=96.,
    current_contact_tracking='fixed body-local contact, current pose; refresh after 3 lost contacts or 3 wall-stall controls',
    static_prefreeze_candidates_per_slot=128,
    cross_split_near=dict(centroid_world_px=8., symmetry_angle_deg=5., local_pusher_world_px=16.),
    valid_contact_motion_control_target=.60,
    useful_noncontact_or_static_target=[.10, .20], boundary_target=[.20, .30],
    noise_floor_control_image_px=.1, noise_floor_control_angle_deg=.1,
    obvious_translation_image_px=2., obvious_rotation_deg=5.,
    translation_bins_image_px=[0., .5, 2., 5., None],
    rotation_bins_deg=[0., 1., 5., 15., None],
    supplement_trigger='per-letter H1 translation<.5 image px exceeds50%, OR no H1 rotation>=5deg trajectory; report all attempts',
    collection_cap_bytes=512*1024**2, video_cap_bytes=64*1024**2,
    shared_cap_bytes=3*1024**3, atomic_reserve_bytes=256*1024**2,
    per_attempt_atomic_reserve_bytes=16*1024**2,
    models=0, optimizer_updates=0, hidden_physics_lookahead=0,
    candidate_status='not authorized for training',
)
POLICY_EXECUTION_VERSION = 'original-contact-pilot64-confirmed-acquisition.v2'
COLLECTION_IO_VERSION = 'original-contact-pilot-partitioned-reserved-io.v2'


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def phase_for(index, scenario):
    if not 0 <= index < 64:
        raise ValueError('Exactly64 requested controls; never repeat to fill')
    if scenario == 'static':
        return 'static'
    return next(phase for lo, hi, phase in RULES['phases'] if lo <= index < hi)


def select_contact(letter, state, scenario, arc_start, location, *, exclude_arc=None):
    """Finite present-geometry selection; no simulator or future rollout."""
    geometry = geometry_at(letter, state)
    free = _free(geometry)
    role = ('translation_1' if scenario == 'translation_nearCOM' else
            scenario if scenario.startswith('rotation_') else 'generic')
    arcs = segments(letter)
    wall = min(wall_gaps(geometry), key=wall_gaps(geometry).get)
    inward = {'left': np.array((1., 0.)), 'right': np.array((-1., 0.)),
              'top': np.array((0., 1.)), 'bottom': np.array((0., -1.))}[wall]
    candidates = []
    for offset in range(len(arcs)):
        segment = arcs[(int(arc_start) + offset) % len(arcs)]
        if int(segment['id']) == exclude_arc:
            continue
        contact, _ = _candidate(letter, state, geometry, segment, role, 'center', free)
        if contact is None:
            continue
        beta = (float(contact['drive_tangent_ratio']) if scenario == 'translation_nearCOM'
                else .35 if scenario == 'mixed_positive' else -.35 if scenario == 'mixed_negative'
                else .6 * (1 if arc_start % 2 else -1) if scenario == 'grazing' else 0.)
        normal = np.asarray(contact['outward_normal_world'])
        direction = -normal + beta*np.array((-normal[1], normal[0]))
        direction /= np.linalg.norm(direction)
        room = _room(geometry, direction)
        if room < (24 if scenario == 'translation_nearCOM' else 12):
            continue
        wall_dot = float(direction @ inward)
        # Boundary scenes include along-wall and departure. The original local
        # role stays intact; no response or motion quality is consulted.
        if location != 'center' and wall_dot < -.1:
            continue
        contact = deepcopy(contact)
        contact['drive_tangent_ratio'] = beta
        contact['drive_direction_world'] = direction.tolist()
        contact['translation_room_world_px'] = room
        contact['boundary_direction_inward_dot'] = wall_dot
        contact['selection_scope'] = 'present geometry; no observed outcome'
        along = location != 'center' and scenario in ('grazing', 'mixed_positive', 'mixed_negative')
        candidates.append(((int(along and abs(wall_dot) > .5), offset), contact))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _scene(letter, split, index, seed, location, scenario):
    rng = np.random.default_rng(seed)
    angle = float((index % 12 + rng.uniform(.05, .95))*math.pi/6)
    rotated = affinity.rotate(shape_geometry(letter), angle, origin=(0, 0), use_radians=True)
    x0, y0, x1, y1 = rotated.bounds
    center = rng.uniform(174., 337., size=2)
    origin = center - np.array(((x0+x1)/2, (y0+y1)/2))
    gaps = rng.uniform(4., 18., size=2)
    if 'left' in location:
        origin[0] = 7 + gaps[0] - x0
    if 'right' in location:
        origin[0] = 504 - gaps[0] - x1
    if 'top' in location:
        origin[1] = 7 + gaps[1] - y0
    if 'bottom' in location:
        origin[1] = 504 - gaps[1] - y1
    state = np.zeros(15, np.float64)
    state[2:4] = origin
    state[4] = angle
    arc_start = int(rng.integers(len(segments(letter))))
    contact = select_contact(letter, state, scenario, arc_start, location)
    if contact is None:
        return None, 'no_feasible_exterior_role_contact'
    state[:2] = contact['staging_world']
    if not ((state[:2] >= 22).all() and (state[:2] <= 489).all()):
        return None, 'pusher_staging_outside_inner_wall'
    if shape_geometry(letter).contains(Point(*contact['point_local'])):
        return None, 'target_not_union_boundary'
    cog = np.array((0., 45.)) if letter == 'T' else np.zeros(2)
    return dict(seed=int(seed), initial_state15=state.tolist(),
                initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                initialization=dict(center=(origin + rot(angle)@cog).tolist(),
                                    angle=angle, pusher=state[:2].tolist()),
                selected_contact=contact, arc_start=arc_start,
                initial_space=location, scenario=scenario), None


def build_plan(seed=2026103017):
    """Freeze all scenes, phases, finite retries and splits before physics."""
    slots, rejections = [], []
    accepted = {letter: [] for letter in LETTERS}
    signatures = set()
    phases = [('calibration-v1', 'calibration', 3),
              ('calibration-v2', 'calibration', 3),
              ('formal', 'train_candidate', 48),
              ('formal', 'seen_validation', 12),
              ('supplement', 'train_candidate', 12)]
    for phase_index, (phase, split, count) in enumerate(phases):
        for li, letter in enumerate(LETTERS):
            scenarios = ([name for name, n in zip(SCENARIOS, COUNTS[split]) for _ in range(n)]
                         if phase == 'formal' else
                         ['translation_nearCOM', 'rotation_positive', 'rotation_negative'] if count == 3 else
                         ['translation_nearCOM']*4 + ['rotation_positive']*3 + ['rotation_negative']*3 + ['mixed_positive', 'mixed_negative'])
            rng = np.random.default_rng(seed + phase_index*1009 + li*37)
            scenarios = [scenarios[int(i)] for i in rng.permutation(count)]
            boundary_count = count//4 if count >= 12 else 1
            locations = ['center']*count
            boundary_indices = rng.permutation(count)[:boundary_count]
            for bi, index in enumerate(boundary_indices):
                locations[int(index)] = LOCATIONS[1+(bi+li+phase_index*3) % 8]
            for index in range(count):
                slot_id = f'OPC-{phase}-{split}-{letter}-{index:03d}'
                for static_try in range(RULES['static_prefreeze_candidates_per_slot']):
                    scene_seed = seed + phase_index*1000000 + li*100000 + index*128 + static_try
                    scene, reason = _scene(letter, split, index, scene_seed, locations[index], scenarios[index])
                    if scene is not None:
                        state = np.asarray(scene['initial_state15'], np.float64)
                        if scene['initial_state_sha256'] in signatures:
                            reason = 'duplicate_initial_state'
                        elif split == 'seen_validation' and any(old_split == 'train_candidate' and near_initial(letter, old, state)
                                                               for old_split, old in accepted[letter]):
                            reason = 'cross_split_near_initial'
                        elif split == 'train_candidate' and any(old_split == 'seen_validation' and near_initial(letter, old, state)
                                                                 for old_split, old in accepted[letter]):
                            reason = 'cross_split_near_initial'
                        else:
                            signatures.add(scene['initial_state_sha256'])
                            accepted[letter].append((split, state))
                            scene['static_try'] = static_try
                            slots.append(dict(slot_id=slot_id, family_id='F-'+slot_id,
                                              phase=phase, split=split, letter=letter,
                                              index=index, **scene))
                            break
                    rejections.append(dict(slot_id=slot_id, static_try=static_try, seed=scene_seed, reason=reason))
                else:
                    raise RuntimeError('Prefreeze geometry catalogue exhausted: '+slot_id)
    return dict(kind='original-contact-pilot-independent-scenes.v1', created_at_utc=utc(),
                rules=RULES, rules_sha256=identity(RULES), seed=seed, slots=slots,
                static_rejections=rejections, primary_trajectories=360,
                primary_requested_controls=23040, calibration_max_trajectories=36,
                supplement_max_trajectories=72, maximum_total_requested_controls=29952,
                no_trajectory_reuse=True, split_before_physics=True,
                transformed_derived_trajectories=False, training_authorized=False)


class PilotPolicy:
    """Present/past-state feedback, with no hidden simulator lookahead."""
    def __init__(self, env, slot, rules=None):
        self.env, self.slot = env, slot
        self.rules = RULES if rules is None else rules
        self.contact = deepcopy(slot['selected_contact'])
        self.departure_contact = deepcopy(self.contact)
        self.awaiting_acquisition = True
        self.acquisition_no_route = 0
        self.last_route_kind = None
        self.last_selected_contact_seconds = 0.
        self.last_action = None
        self.release_target = None
        self.lost = 0
        self.stall = 0
        self.refresh_count = 0
        self.last_phase = None

    def route(self, geometry, start, target, departure_contact=None):
        self.last_route_kind = None
        path = free_route(geometry, start, target)
        if path is not None:
            self.last_route_kind = 'free_route'
        if path is None and _departure_route(geometry, start, target):
            path = [np.asarray(start).tolist(), np.asarray(target).tolist()]
            self.last_route_kind = 'direct_departure'
        if path is None:
            state = np.asarray(self.env.state(), np.float64)
            source = departure_contact or self.departure_contact
            point, normal, _ = contact_frame(state, source)
            for distance in self.rules['departure_distances_world_px']:
                departure = point+normal*distance
                if not _departure_route(geometry, start, departure):
                    continue
                rest = free_route(geometry, departure, target)
                if rest is None:
                    continue
                candidate = [np.asarray(start).tolist(), *rest]
                if all(_free(geometry).covers(_line(a, b)) or _departure_route(geometry, np.asarray(a), np.asarray(b))
                       for a, b in zip(candidate, candidate[1:])):
                    path = candidate
                    self.last_route_kind = 'retained_contact_normal_departure_bridge'
                    break
        return path

    def next_action(self, index):
        self.last_route_kind = None
        state = np.asarray(self.env.state(), np.float64)
        geometry = geometry_at(self.slot['letter'], state)
        phase = phase_for(index, self.slot['scenario'])
        start = state[:2]
        if phase != self.last_phase:
            self.release_target = None
            if phase in ('approach', 'withdraw', 'recontact'):
                self.awaiting_acquisition = True
                self.acquisition_no_route = 0
        self.last_phase = phase
        changed = None
        refresh_reason = ('wall_contact_stall3' if self.stall >= 3 else
                          'acquisition_no_route3' if self.awaiting_acquisition and self.acquisition_no_route >= 3 else
                          'lost_contact3' if not self.awaiting_acquisition and self.lost >= 3 else None)
        if phase in ('push', 'recontact') and refresh_reason is not None:
            fresh = select_contact(self.slot['letter'], state, self.slot['scenario'],
                                   self.slot['arc_start']+index*3, self.slot['initial_space'],
                                   exclude_arc=int(self.contact['arc_id']))
            if fresh is not None:
                self.contact = fresh
                self.refresh_count += 1
                changed = refresh_reason
                self.awaiting_acquisition = True
                self.acquisition_no_route = 0
                self.lost = self.stall = 0
        point, normal, tangent = contact_frame(state, self.contact)
        staging = _stage(geometry, point, normal)
        raw, mode = start.copy(), 'hold'
        if phase in ('static', 'stop'):
            pass
        elif phase == 'withdraw':
            if self.release_target is None:
                for distance in self.rules['departure_distances_world_px']:
                    candidate = point + normal*distance
                    if self.route(geometry, start, candidate) is not None:
                        self.release_target = candidate
                        break
            if self.release_target is not None:
                path = self.route(geometry, start, self.release_target)
                if path is not None:
                    raw, mode = np.asarray(path[1]), 'free_route'
        elif staging is not None:
            stage, center, _ = staging
            if self.awaiting_acquisition and np.linalg.norm(start-center) > 8 and np.linalg.norm(start-stage) > 4:
                path = self.route(geometry, start, stage)
                if path is not None:
                    raw, mode = np.asarray(path[1]), 'free_route'
                    self.acquisition_no_route = 0
                else:
                    self.acquisition_no_route += 1
            else:
                entering = self.awaiting_acquisition
                advance = self.rules['entry_advance_world_px'] if entering or phase != 'push' else self.rules['push_advance_world_px']
                beta = 0. if entering else self.contact['drive_tangent_ratio']
                raw = center - normal*advance + tangent*advance*beta
                mode = 'contact_entry_intent' if entering else 'contact_push_intent'
                self.acquisition_no_route = 0
        elif phase in ('approach', 'push', 'recontact'):
            self.acquisition_no_route += 1
        action = np.asarray(raw, float)
        displacement = action-start
        length = float(np.linalg.norm(displacement))
        if length > self.rules['target_distance_world_px']:
            action = start + displacement*self.rules['target_distance_world_px']/length
        if self.last_action is not None:
            change = action-self.last_action
            length = float(np.linalg.norm(change))
            if length > self.rules['target_change_world_px']:
                action = self.last_action + change*self.rules['target_change_world_px']/length
        action = np.clip(action, 0., 511.)
        # A route preview is geometry-only. Contact safety is enforced by the
        # common environment contract, not by deleting troublesome records here.
        self.last_action = action.copy()
        return action, dict(phase=phase, mode=mode, selected_arc_id=int(self.contact['arc_id']),
                            selected_point_local=self.contact['point_local'],
                            selected_normal_local=self.contact['outward_normal_local'],
                            drive_tangent_ratio=float(self.contact['drive_tangent_ratio']),
                            raw_sampler_target=np.asarray(raw).tolist(),
                            requested_action=action.tolist(), target_refresh_reason=changed,
                            target_refresh_count=self.refresh_count,
                            acquisition_required=bool(self.awaiting_acquisition),
                            acquisition_no_route_controls=self.acquisition_no_route,
                            prior_selected_contact_seconds=self.last_selected_contact_seconds,
                            route_kind=self.last_route_kind,
                            departure_source_point_local=self.departure_contact['point_local'],
                            policy_execution_version=POLICY_EXECUTION_VERSION,
                            privileged_current_state=True)

    def observe(self, before, after, info):
        rows = info.get('substeps', [])
        ao = any(e.get('kind') == 'agent_object' and e.get('phase') == 'post_solve'
                 for row in rows for e in row.get('contacts', []))
        ow = any(e.get('kind') == 'object_wall' and e.get('phase') == 'post_solve'
                 for row in rows for e in row.get('contacts', []))
        movement = np.linalg.norm(area_center(self.slot['letter'], after)-area_center(self.slot['letter'], before))*96/512
        turn = abs(float(after[4]-before[4]))*180/math.pi
        self.lost = 0 if ao else self.lost+1
        self.stall = self.stall+1 if ao and ow and movement < .1 and turn < .1 else 0
        selected_seconds = 0.
        terminal_selected = False
        executed = [row for row in rows if row.get('executed')]
        for row_index, row in enumerate(executed):
            matched = False
            state = np.asarray(row.get('post_state', after), np.float64)
            selected_world_normal = rot(state[4])@np.asarray(self.contact['outward_normal_local'])
            for event in row.get('contacts', []):
                if event.get('phase') != 'post_solve' or event.get('kind') != 'agent_object':
                    continue
                impulse = np.asarray(event.get('total_impulse', [0., 0.]), float)
                if not np.isfinite(impulse).all() or np.linalg.norm(impulse) <= 1e-8:
                    continue
                object_first = event['shapes_order'][0] == 'object'
                outward = np.asarray(event['normal'], float)*(1 if object_first else -1)
                if float(outward@selected_world_normal) < math.cos(math.pi/4):
                    continue
                for point in event.get('points', []):
                    local = point.get('object_local')
                    if local is None:
                        world = point.get('object_world')
                        if world is None:
                            world = point.get('point_a' if object_first else 'point_b')
                        if world is None:
                            continue
                        local = rot(state[4]).T@(np.asarray(world)-state[2:4])
                    if np.linalg.norm(np.asarray(local)-self.contact['point_local']) <= 24.:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                selected_seconds += float(row.get('dt', 0.))
            if row_index == len(executed)-1:
                terminal_selected = matched
        self.last_selected_contact_seconds = selected_seconds
        if selected_seconds >= .05-1e-9 and terminal_selected:
            self.awaiting_acquisition = False
            self.acquisition_no_route = 0
            self.departure_contact = deepcopy(self.contact)


def storage_roots_for_scan(roots):
    """Remove redundant nested walks without changing the resolved coverage."""
    present, missing = {}, []
    for raw in roots:
        path = Path(raw).resolve()
        if not path.exists():
            missing.append(str(path))
        else:
            present[str(path).casefold()] = path
    primary = []
    for path in sorted(present.values(), key=lambda item: (len(item.parts), str(item).casefold())):
        if any(parent.is_dir() and path.is_relative_to(parent) for parent in primary):
            continue
        primary.append(path)
    return primary, missing


def storage_snapshot(roots):
    """Deduplicate files and nested roots; never follow interior reparse links."""
    files = {}
    primary, missing = storage_roots_for_scan(roots)
    for root in primary:
        if root.is_file():
            files[str(root).casefold()] = root.stat().st_size
            continue
        for parent, directories, names in os.walk(root, followlinks=False):
            directories[:] = [name for name in directories if not (Path(parent)/name).is_symlink()
                               and not (getattr((Path(parent)/name).stat(), 'st_file_attributes', 0) & 1024)]
            for name in names:
                file = Path(parent)/name
                if not file.is_symlink():
                    files[str(file.resolve()).casefold()] = file.stat().st_size
    return dict(bytes=sum(files.values()), files=len(files), missing_roots=missing,
                scanned_roots=[str(root) for root in primary])


def _storage_guard(output, shared_roots, extra=0):
    own = storage_snapshot([output])
    shared = storage_snapshot([*shared_roots, output])
    free = __import__('shutil').disk_usage(output).free
    reserve = RULES['atomic_reserve_bytes'] + RULES['per_attempt_atomic_reserve_bytes'] + extra
    if own['bytes'] + RULES['per_attempt_atomic_reserve_bytes'] + extra > RULES['collection_cap_bytes']:
        raise RuntimeError('Collection512MiB operational storage cap reached; preserve prior records')
    if shared['bytes'] + reserve > RULES['shared_cap_bytes'] or free < reserve:
        raise RuntimeError('Shared3GiB/atomic storage reserve not available; no control executed')
    return dict(collection=own, shared=shared, disk_free_bytes=free, reserved_bytes=reserve)


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_attempt(path, states, actions, valid, rgb, telemetry, decisions, *, reservation=None):
    path = Path(path)
    payload = canonical(_safe_json(dict(telemetry=telemetry, decisions=decisions)))
    arrays = dict(states=np.asarray(states, np.float64).reshape(-1, 15),
                  requested_actions=np.asarray(actions, np.float64).reshape(-1, 2),
                  control_valid=np.asarray(valid, bool), rgb=np.asarray(rgb, np.uint8),
                  telemetry_zlib=np.frombuffer(zlib.compress(payload, 6), np.uint8),
                  telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(), np.uint8))
    path.parent.mkdir(parents=True, exist_ok=True)
    if reservation is None:
        temporary = path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    else:
        packed = io.BytesIO()
        np.savez_compressed(packed, **arrays)
        reservation.write_bytes(path, packed.getvalue())
    return dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha(path),
                state_sha256=hashlib.sha256(arrays['states'].tobytes()).hexdigest(),
                rgb_sha256=hashlib.sha256(arrays['rgb'].tobytes()).hexdigest())


def load_attempt(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    payload = zlib.decompress(arrays.pop('telemetry_zlib').tobytes())
    if hashlib.sha256(payload).digest() != arrays.pop('telemetry_sha256').tobytes():
        raise ValueError('Saved telemetry hash mismatch')
    arrays.update(json.loads(payload))
    return arrays


def normalize_letters(letters):
    values = tuple(LETTERS if letters is None else letters)
    if not values or len(set(values)) != len(values) or any(letter not in LETTERS for letter in values):
        raise ValueError('Use a nonempty compact unique subset of TLOCXB, without separators')
    return tuple(letter for letter in LETTERS if letter in values)


def execution_io_paths(output, phase, run_label=None):
    output = Path(output).resolve()
    if run_label is not None:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', run_label) or run_label.casefold() in {
                'con', 'prn', 'aux', 'nul', *[f'com{i}' for i in range(1, 10)], *[f'lpt{i}' for i in range(1, 10)]}:
            raise ValueError('Unsafe or reserved run label')
        directory = (output/'runs'/run_label).resolve()
        if not directory.is_relative_to(output):
            raise ValueError('Run directory escapes collection root')
        return dict(directory=directory, identity=directory/'identity.json',
                    ledger=directory/'ledger.json', events=directory/'events.jsonl',
                    progress=directory/'progress.json', completion=directory/'completion.json')
    return dict(directory=output, identity=output/(phase+'-run-identity.json'),
                ledger=output/'ledger.json', events=output/'events.jsonl',
                progress=output/'progress.json', completion=output/(phase+'-completion.json'))


def selected_slots(plan, phase, letters):
    if phase not in ('calibration-v1', 'calibration-v2', 'formal', 'supplement'):
        raise ValueError('Unknown frozen collection phase')
    allowed = normalize_letters(letters)
    result = [row for row in plan['slots'] if row['phase'] == phase and row['letter'] in allowed]
    if not result:
        raise ValueError('Selected frozen phase/letters have no scenes')
    return result


def _append_execution_event(path, event):
    with Path(path).open('a', encoding='utf-8') as journal:
        journal.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
        journal.flush()
        os.fsync(journal.fileno())


def collect(plan_path, output, phase, physics_gate, source_root, shared_roots, *,
            rules_path=None, letters=None, run_label=None, resource_policy=None, stop_file=None):
    """Frozen disjoint partitions, shared reservations, no failed-scene retry."""
    revoked = (Path(__file__).resolve().parents[1]/'artifacts'/
               'original-physics-contact-pilot-20261030'/'OLD_MODIFIED_PHYSICS_REVOKED.json')
    if revoked.exists():
        raise RuntimeError('Modified-physics collector authorization revoked; use the separately '
                           'verified strict-original collector, never this v6 collector: '+str(revoked))
    from .original_contact_pilot_resources import ResourceReservation
    import tdyssey.original_contact_pilot_resources as resources_module
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = execution_io_paths(output, phase, run_label)
    paths['directory'].mkdir(parents=True, exist_ok=True)
    if run_label is not None and (paths['identity'].exists() or paths['ledger'].exists()):
        raise FileExistsError('Named run identity/ledger already exists; no retry or reset')
    if run_label is not None and resource_policy is None:
        raise ValueError('Partitioned workers require an explicit frozen resource policy')
    letters = normalize_letters(letters)
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if identity(plan['rules']) != plan['rules_sha256'] or plan['rules'] != RULES:
        raise ValueError('Frozen scene plan/rules changed')
    gate_path = Path(physics_gate).resolve()
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    if gate.get('passed') is not True or gate.get('training_authorized') is not False:
        raise RuntimeError('A separate passed execution/physics gate is required')
    mismatches = [path for path, digest in gate.get('bindings', {}).items()
                  if not Path(path).exists() or file_sha(path) != digest]
    if mismatches:
        raise RuntimeError('Gate bindings changed before execution: '+repr(mismatches))
    module = importlib.import_module('tdyssey.native_rgb_contact_safe_physics')
    protocol_sha = module.PROTOCOL_SHA256
    if gate.get('protocol_sha256') != protocol_sha:
        raise RuntimeError('Gate protocol differs from actual safe factory')
    policy_rules = deepcopy(RULES)
    if rules_path is not None:
        revision = json.loads(Path(rules_path).read_text(encoding='utf-8'))
        if revision.get('policy_execution_version') not in (None, POLICY_EXECUTION_VERSION):
            raise ValueError('Frozen policy revision names a different execution version')
        allowed = {'push_advance_world_px', 'entry_advance_world_px', 'departure_distances_world_px',
                   'target_distance_world_px', 'target_change_world_px'}
        if set(revision['changes']) - allowed:
            raise ValueError('Only recorded contact-action strategy fields may change')
        policy_rules.update(revision['changes'])
    if phase == 'calibration-v2' and rules_path is None:
        raise RuntimeError('The sole revised calibration requires a frozen policy revision')
    if phase == 'supplement':
        release = output/'supplement-release.json'
        if not release.exists() or not json.loads(release.read_text(encoding='utf-8'))['released']:
            raise RuntimeError('Bounded supplement release is absent')
    selected = selected_slots(plan, phase, letters)
    stop_path = None if stop_file is None else Path(stop_file).resolve()
    run_identity = dict(phase=phase, run_label=run_label, letters=list(letters),
        selected_slot_ids=[row['slot_id'] for row in selected],
        selected_slot_ids_sha256=identity([row['slot_id'] for row in selected]),
        plan_sha256=file_sha(plan_path), physics_gate_sha256=file_sha(gate_path),
        physics_protocol_sha256=protocol_sha, physics_source_sha256=file_sha(module.__file__),
        collection_source_sha256=file_sha(__file__), resource_helper_source_sha256=file_sha(resources_module.__file__),
        policy_revision_sha256=None if rules_path is None else file_sha(rules_path),
        resource_policy_sha256=None if resource_policy is None else file_sha(resource_policy),
        resource_policy_path=None if resource_policy is None else str(Path(resource_policy).resolve()),
        stop_file=None if stop_path is None else str(stop_path),
        policy_rules_sha256=identity(policy_rules), policy_execution_version=POLICY_EXECUTION_VERSION,
        io_version=COLLECTION_IO_VERSION,
        action_contract='Requested absolute target remains the model action; endpoint/velocity intervention is separate telemetry.',
        model_calls=0, models=0, optimizer_updates=0)
    if paths['identity'].exists():
        if json.loads(paths['identity'].read_text(encoding='utf-8')) != run_identity:
            raise ValueError('Existing phase identity differs; no changed-source continuation')
    owner_prefix = f'{run_label or phase}:pid{os.getpid()}'
    def lease(owner, reserve=None):
        return ResourceReservation(output, shared_roots, resource_policy,
            owner_prefix+':'+owner, role='collection', reserve_bytes=reserve)
    def write_reserved_json(reservation, path, value):
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
        reservation.write_bytes(path, encoded)
    with lease('initialize', 1024**2) as reservation:
        write_reserved_json(reservation, paths['identity'], run_identity)
        ledger = json.loads(paths['ledger'].read_text(encoding='utf-8')) if paths['ledger'].exists() else dict(
            kind='original-contact-pilot-actual-cost.v2', run_label=run_label,
            attempts=[], actual_controls=0, actual_substeps=0, actual_physics_seconds=0.,
            model_calls=0, optimizer_updates=0)
        write_reserved_json(reservation, paths['ledger'], ledger)
    start = time.perf_counter()
    attempted_this_run = []
    initial_controls = ledger['actual_controls']
    initial_substeps = ledger['actual_substeps']
    graceful_stop = False
    failure = None
    exception_to_raise = None
    try:
        for slot in selected:
            if stop_path is not None and stop_path.exists():
                graceful_stop = True
                break
            attempt_dir = output/'trajectories'/slot['slot_id']
            claim = attempt_dir/'claim.json'
            if claim.exists():
                # Keep legacy unlabelled completed-slot references, while new
                # named workers must never claim another partition's old slot.
                if run_label is None and (attempt_dir/'metadata.json').exists():
                    continue
                raise RuntimeError('Existing scene claim cannot be retried: '+slot['slot_id'])
            with lease(slot['slot_id']) as reservation:
                attempt_dir.mkdir(parents=True, exist_ok=True)
                cost = dict(slot_id=slot['slot_id'], phase=phase, split=slot['split'],
                            letter=slot['letter'], run_label=run_label,
                            started_at_utc=utc(), reserved_controls=64, actual_controls=0,
                            status='reserved', plan_sha256=run_identity['plan_sha256'],
                            physics_protocol_sha256=protocol_sha,
                            policy_rules_sha256=identity(policy_rules),
                            policy_execution_version=POLICY_EXECUTION_VERSION)
                # Exclusive creation, even if a caller misconfigures overlapping
                # partitions. Do not overwrite any earlier attempt claim.
                with claim.open('x', encoding='utf-8') as handle:
                    json.dump(cost, handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                ledger['attempts'].append(cost)
                attempted_this_run.append(slot['slot_id'])
                write_reserved_json(reservation, paths['ledger'], ledger)
                states, actions, valids, images, telemetry, decisions = [], [], [], [], [], []
                env = None
                initial_identity = None
                attempt_failure = None
                failed_image_reused = False
                wall_start = time.perf_counter()
                try:
                    env = module.make_contact_safe_env(source_root, slot['letter'])
                    image = env.reset_scene(**slot['initialization'])
                    states.append(np.asarray(env.state(), np.float64).copy())
                    images.append(np.asarray(image, np.uint8).copy())
                    if not np.allclose(states[0], slot['initial_state15'], rtol=0, atol=1e-8):
                        raise RuntimeError('Initial construction differs from frozen scene')
                    initial_identity = env.initial_identity()
                    policy = PilotPolicy(env, slot, policy_rules)
                    for index in range(64):
                        before = np.asarray(env.state(), np.float64).copy()
                        action, decision = policy.next_action(index)
                        _append_execution_event(paths['events'], dict(event='control_requested', slot_id=slot['slot_id'],
                            control=index, requested_action=action.tolist(), run_label=run_label, at_utc=utc()))
                        call_start = time.perf_counter()
                        try:
                            image, info = env.step(action)
                        except Exception as exc:
                            info = getattr(exc, 'info', None) or getattr(exc, 'evidence', None) or getattr(exc, 'details', {})
                            info = dict(info or {}, valid=False, exception=repr(exc))
                            info.setdefault('substeps', getattr(env, 'last_substeps', []))
                            attempt_failure = repr(exc)
                            try:
                                if not np.isfinite(env.state()).all():
                                    raise RuntimeError('nonfinite failure state cannot render')
                                image = env.render_clean()
                            except Exception:
                                image = images[-1]
                                failed_image_reused = True
                                info['failure_RGB'] = 'Unavailable: last visible frame retained as explicitly invalid evidence, not a new observation or training frame.'
                        after = np.asarray(env.state(), np.float64).copy()
                        info = dict(info, collector_wall_seconds=time.perf_counter()-call_start)
                        duration = sum(float(row.get('dt', 0.)) for row in info.get('substeps', []) if row.get('executed'))
                        valid = bool(info.get('valid', False) and info.get('substeps')
                                     and abs(duration-.1) < 1e-7 and np.isfinite(after).all())
                        actions.append(np.asarray(action, np.float64).copy())
                        states.append(after)
                        valids.append(valid)
                        images.append(np.asarray(image, np.uint8).copy())
                        telemetry.append(_safe_json(info))
                        decisions.append(_safe_json(decision))
                        ledger['actual_controls'] += 1
                        ledger['actual_substeps'] += sum(bool(row.get('executed')) for row in info.get('substeps', []))
                        ledger['actual_physics_seconds'] += duration
                        cost['actual_controls'] = len(actions)
                        _append_execution_event(paths['events'], dict(event='control_returned', slot_id=slot['slot_id'],
                            control=index, valid=valid, actual_substeps=len(info.get('substeps', [])),
                            actual_seconds=duration, run_label=run_label, at_utc=utc()))
                        if len(actions) % 8 == 0 or not valid:
                            save_attempt(attempt_dir/'raw.npz', states, actions, valids, images, telemetry, decisions,
                                         reservation=reservation)
                        if not valid:
                            attempt_failure = attempt_failure or 'invalid_or_partial_physical_control'
                            break
                        policy.observe(before, after, info)
                except Exception as exc:
                    attempt_failure = repr(exc)
                saved = (save_attempt(attempt_dir/'raw.npz', states, actions, valids, images, telemetry, decisions,
                                      reservation=reservation) if states else None)
                complete = len(actions) == 64 and all(valids) and attempt_failure is None
                cost.update(status='completed' if complete else 'failed', finished_at_utc=utc(), actual_controls=len(actions))
                metadata = dict(slot=slot, reservation=cost, raw=saved, complete=complete,
                    failure=attempt_failure, actual_valid_controls=sum(valids), initial_identity=initial_identity,
                    wall_seconds=time.perf_counter()-wall_start, requested_controls=64,
                    policy_rules=policy_rules, policy_execution_version=POLICY_EXECUTION_VERSION,
                    protocol_sha256=protocol_sha, run_label=run_label,
                    resource_policy_sha256=run_identity['resource_policy_sha256'],
                    policy_revision_sha256=run_identity['policy_revision_sha256'],
                    model_calls=0, models=0, optimizer_updates=0,
                    contiguous_frames=not failed_image_reused,
                    invalid_failure_RGB_is_last_visible_frame=failed_image_reused,
                    hidden_replay_or_filled_frames=False, future_training_authorized=False)
                write_reserved_json(reservation, attempt_dir/'metadata.json', metadata)
                write_reserved_json(reservation, paths['ledger'], ledger)
                write_reserved_json(reservation, paths['progress'], dict(
                    phase=phase, run_label=run_label, letters=list(letters), last_slot=slot['slot_id'],
                    attempts_this_run=len(attempted_this_run), actual_controls=ledger['actual_controls'],
                    actual_substeps=ledger['actual_substeps'], updated_at_utc=utc(),
                    model_calls=0, optimizer_updates=0))
                print(json.dumps(dict(slot=slot['slot_id'], complete=complete, controls=len(actions),
                                      failure=attempt_failure, bytes=saved['bytes'] if saved else 0)), flush=True)
                if not complete:
                    raise RuntimeError('Substantive collection failure; preserve partial and stop: '+slot['slot_id']+': '+str(attempt_failure))
    except Exception as exc:
        failure = repr(exc)
        exception_to_raise = exc
    all_selected_finished = all((output/'trajectories'/row['slot_id']/'metadata.json').exists()
        and json.loads((output/'trajectories'/row['slot_id']/'metadata.json').read_text(encoding='utf-8'))['complete']
        for row in selected)
    completed = failure is None and not graceful_stop and all_selected_finished
    completion = dict(**{key: run_identity[key] for key in (
        'phase', 'run_label', 'letters', 'plan_sha256', 'physics_protocol_sha256',
        'physics_gate_sha256', 'resource_policy_sha256', 'policy_revision_sha256')},
        completed=completed, failed=failure is not None, graceful_stop=graceful_stop,
        failure=failure, finished_at_utc=utc(), wall_seconds=time.perf_counter()-start,
        selected_slots=len(selected), actual_attempts=len(attempted_this_run),
        actual_controls=ledger['actual_controls']-initial_controls,
        actual_substeps=ledger['actual_substeps']-initial_substeps,
        ledger_total_controls=ledger['actual_controls'], ledger_total_substeps=ledger['actual_substeps'],
        unstarted_slots=[row['slot_id'] for row in selected if not (output/'trajectories'/row['slot_id']/'claim.json').exists()],
        model_calls=0, optimizer_updates=0)
    # Small terminal evidence must survive a data-allocation failure; the data
    # guard has retained the administrative/atomic margins throughout execution.
    write_json(paths['completion'], completion)
    if exception_to_raise is not None:
        raise exception_to_raise
    return completion
