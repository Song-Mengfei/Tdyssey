"""Bounded requested-action collection in the unmodified pinned PushT loop.

Only sampling and observation are added. The original environment executes every
PD update and 0.01s integrator call. An illegal observed state ends its trajectory;
it is retained as diagnostic evidence and is never repaired or retried.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zlib

import numpy as np
from shapely.geometry import Point, Polygon

from . import original_contact_pilot as saved
from . import native_rgb_original_physics as original
from .native_rgb_r2a_env import R2AStepAborted
from .native_rgb_r2b_sampling import geometry_at, wall_gaps, contact_frame, _room
from .native_rgb_r3a_collection import near_initial
from .original_contact_pilot_resources import ResourceReservation

LETTERS = saved.LETTERS
POLICY_VERSION = 'strict-original-requested-target-policy.v1'
RULES = dict(deepcopy(saved.RULES),
    kind='strict-original-contact-pilot64.v1',
    physics_protocol_sha256=original.PROTOCOL_SHA256,
    push_advance_world_px=12., entry_advance_world_px=2.,
    target_distance_world_px=24., target_change_world_px=24.,
    departure_distances_world_px=[33., 48., 18.],
    boundary_initial_gap_world_px=[18., 32.],
    wall_room_refresh_world_px=24.,
    requested_target_inner_bounds=[23., 488.],
    numerical_overlap_valid_max_world_px=1.,
    invalid_center='solid or any closed hole: exterior starts only',
    intervention='none: requested targets pass unchanged to original PD',
    invalid_trajectory='stop on first observed illegal substep; retain complete attempt and legal prefix; no replacement',
    collection_cap_bytes=1536*1024**2,
    candidate_status='pending user training approval; protocol eligibility is separate')


class OriginalStateInvalid(R2AStepAborted):
    """Observed original state is illegal; no physical value was changed."""


def state_invalidity(state, penetrations, region):
    reasons = []
    if not np.isfinite(np.asarray(state, float)).all():
        reasons.append('nonfinite_state')
    depths = np.asarray(penetrations, float)
    if not np.isfinite(depths).all():
        reasons.append('nonfinite_penetration')
    elif len(depths) and float(np.max(depths)) > RULES['numerical_overlap_valid_max_world_px']+1e-8:
        reasons.append('overlap_above_1_world_px')
    if region == 'solid' or region.startswith('hole-'):
        reasons.append('illegal_pusher_center_region')
    return reasons


class StrictObservedOriginalEnv(original.OriginalPhysicsEnv):
    """Original control/physics, plus a terminal observation-only validity guard."""
    def _point_region(self):
        local = Point(*self.block.world_to_local(self.agent.position))
        if self.local_geometry.contains(local):
            return 'solid'
        for index, hole in enumerate(self._observed_holes):
            if hole.contains(local):
                return f'hole-{index}'
        return 'outside'

    def reset_scene(self, *args, **kwargs):
        rgb = super().reset_scene(*args, **kwargs)
        pieces = list(self.local_geometry.geoms) if hasattr(self.local_geometry, 'geoms') else [self.local_geometry]
        self._observed_holes = [Polygon(ring) for piece in pieces for ring in piece.interiors]
        reasons = state_invalidity(self.state(), self.penetrations(), self._point_region())
        if reasons:
            self._terminal = True
            raise OriginalStateInvalid('Invalid frozen initial scene', dict(valid=False, initial_invalid=True,
                reasons=reasons, initial_state=self.state().tolist(), substeps=[]))
        return rgb

    def _observe_step(self, dt):
        # The parent calls the actual bound Space.step(dt), unchanged. No velocity,
        # position, force, solver, timestep, controller or callback return is edited.
        super()._observe_step(dt)
        row = self.last_substeps[-1]
        row['pusher_region'] = self._point_region()
        row['invalid_reasons'] = state_invalidity(row['post_state'], row['penetrations'], row['pusher_region'])
        row['valid'] = not row['invalid_reasons']
        if row['invalid_reasons']:
            raise OriginalStateInvalid('Original environment entered an illegal observed state', dict(
                valid=False, aborted=True, first_invalid_substep=int(row['index']),
                reasons=row['invalid_reasons'], substeps=self.last_substeps,
                physics_protocol_sha256=original.PROTOCOL_SHA256))


def make_strict_original_env(source_root, letter):
    return StrictObservedOriginalEnv(source_root, letter)


def build_plan(seed=2026103017):
    """Freeze all independent poses before any physics, with larger wall gaps."""
    plan = saved.build_plan(seed)
    signatures = set()
    accepted = {letter: [] for letter in LETTERS}
    for slot in plan['slots']:
        state = np.asarray(slot['initial_state15'], np.float64)
        geometry = geometry_at(slot['letter'], state)
        gaps = wall_gaps(geometry)
        delta = np.zeros(2)
        # This changes an initial sampling scene, never a running environment.
        rng = np.random.default_rng(slot['seed']+1729)
        requested_gaps = rng.uniform(*RULES['boundary_initial_gap_world_px'], size=2)
        for name, axis, sign in (('left', 0, 1), ('right', 0, -1), ('top', 1, 1), ('bottom', 1, -1)):
            if name in slot['initial_space']:
                delta[axis] += sign*(float(requested_gaps[axis])-gaps[name])
        state[:2] += delta
        state[2:4] += delta
        slot['initial_state15'] = state.tolist()
        slot['initial_state_sha256'] = __import__('hashlib').sha256(state.tobytes()).hexdigest()
        slot['initialization']['center'] = (np.asarray(slot['initialization']['center'])+delta).tolist()
        slot['initialization']['pusher'] = state[:2].tolist()
        slot['initial_boundary_gap_world_px'] = wall_gaps(geometry_at(slot['letter'], state))
        for key in ('point_world', 'contact_center_world', 'staging_world'):
            slot['selected_contact'][key] = (np.asarray(slot['selected_contact'][key])+delta).tolist()
        slot['selected_contact']['free_approach_path'] = [
            (np.asarray(point)+delta).tolist() for point in slot['selected_contact']['free_approach_path']]
        slot['slot_id'] = slot['slot_id'].replace('OPC-', 'SOP-', 1)
        slot['family_id'] = 'F-'+slot['slot_id']
        if slot['initial_state_sha256'] in signatures:
            raise RuntimeError('Frozen scene duplicate after boundary placement')
        if not ((state[:2] >= 23.).all() and (state[:2] <= 488.).all()):
            raise RuntimeError('Frozen staging falls outside requested-action inner bounds')
        for old_split, old_state in accepted[slot['letter']]:
            if {slot['split'], old_split} == {'train_candidate', 'seen_validation'} and near_initial(slot['letter'], old_state, state):
                raise RuntimeError('Cross-split near duplicate after strict boundary placement')
        signatures.add(slot['initial_state_sha256'])
        accepted[slot['letter']].append((slot['split'], state.copy()))
    plan.update(kind='strict-original-contact-independent-scenes.v1', rules=RULES,
        rules_sha256=saved.identity(RULES), policy_execution_version=POLICY_VERSION,
        physics_protocol_sha256=original.PROTOCOL_SHA256,
        prior_modified_physics_data_reused=False, training_eligible=False,
        stage_release='Only explicit new original-protocol gates may release a phase')
    return plan


class RequestedTargetPolicy(saved.PilotPolicy):
    """Slower entry, bounded requests and geometric wall departure only."""
    def __init__(self, env, slot, rules=None):
        super().__init__(env, slot, rules or RULES)

    def next_action(self, index):
        state = np.asarray(self.env.state(), float)
        geometry = geometry_at(self.slot['letter'], state)
        _, normal, tangent = contact_frame(state, self.contact)
        direction = -normal+self.contact['drive_tangent_ratio']*tangent
        direction /= np.linalg.norm(direction)
        room = _room(geometry, direction)
        refresh = None
        if saved.phase_for(index, self.slot['scenario']) == 'push' and room < self.rules['wall_room_refresh_world_px']:
            fresh = saved.select_contact(self.slot['letter'], state, self.slot['scenario'],
                self.slot['arc_start']+index*3, self.slot['initial_space'], exclude_arc=int(self.contact['arc_id']))
            if fresh is not None:
                self.contact = fresh
                self.awaiting_acquisition = True
                self.acquisition_no_route = 0
                self.lost = self.stall = 0
                self.refresh_count += 1
                refresh = 'geometric_drive_room_under24'
        action, decision = super().next_action(index)
        # An absolute requested target is the sole input to the original PD.
        # This is intentionally saved as the action, never called an executed
        # velocity intervention or concealed behind a different model action.
        action = np.clip(action, *self.rules['requested_target_inner_bounds'])
        self.last_action = action.copy()
        decision.update(requested_action=action.tolist(), policy_execution_version=POLICY_VERSION,
            original_control_and_physics_unchanged=True, geometric_drive_room_world_px=float(room),
            target_refresh_reason=refresh or decision.get('target_refresh_reason'),
            execution_intervention=False)
        return action, decision


def valid_prefix_ranges(mask):
    """Return all contiguous half-open legal intervals; no window bridges gaps."""
    values = np.asarray(mask, bool).tolist()
    ranges, begin = [], None
    for index, valid in enumerate(values+[False]):
        if valid and begin is None:
            begin = index
        elif not valid and begin is not None:
            ranges.append([begin, index])
            begin = None
    return ranges


def save_attempt(path, states, actions, valid, rgb, telemetry, decisions, *, reservation):
    """Lossless original arrays plus explicit actual time/segment labels."""
    path = Path(path)
    payload = saved.canonical(saved._safe_json(dict(telemetry=telemetry, decisions=decisions)))
    durations = [sum(float(row.get('dt', 0.)) for row in info.get('substeps', [])
        if row.get('executed')) for info in telemetry]
    arrays = dict(states=np.asarray(states, np.float64).reshape(-1, 15),
        requested_actions=np.asarray(actions, np.float64).reshape(-1, 2),
        control_valid=np.asarray(valid, bool), rgb=np.asarray(rgb, np.uint8),
        frame_indices=np.arange(len(states), dtype=np.int64),
        control_time_seconds=np.asarray([0., *np.cumsum(durations)], np.float64),
        segment_ids=np.zeros(len(states), np.int64),
        telemetry_zlib=np.frombuffer(zlib.compress(payload, 6), np.uint8),
        telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(), np.uint8))
    packed = io.BytesIO()
    np.savez_compressed(packed, **arrays)
    reservation.write_bytes(path, packed.getvalue())
    return dict(file=path.name, bytes=path.stat().st_size, sha256=saved.file_sha(path),
        state_sha256=hashlib.sha256(arrays['states'].tobytes()).hexdigest(),
        rgb_sha256=hashlib.sha256(arrays['rgb'].tobytes()).hexdigest(),
        frame_time_contract='Cumulative executed original0.01s substeps; invalid final partial is not a full10Hz frame')


def _reserved_json(lease, path, value):
    lease.write_bytes(path, (json.dumps(saved._safe_json(value), ensure_ascii=False,
        indent=2, allow_nan=False)+'\n').encode('utf-8'))


def process_job_evidence():
    evidence = dict(pid=os.getpid(), platform=os.name, checked_at_utc=saved.utc())
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        kernel.IsProcessInJob.restype = wintypes.BOOL
        in_job = wintypes.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(in_job)):
            raise ctypes.WinError(ctypes.get_last_error())
        evidence['is_process_in_job'] = bool(in_job.value)
    else:
        evidence.update(is_process_in_job=None, windows_job_check_applicable=False)
    return evidence


def verify_gate(plan_path, gate_path, policy_path, phase):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    gate = json.loads(Path(gate_path).read_text(encoding='utf-8'))
    policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    if plan.get('kind') != 'strict-original-contact-independent-scenes.v1' or plan.get('rules') != RULES:
        raise ValueError('Not the independently frozen strict-original plan')
    if saved.identity(plan['rules']) != plan.get('rules_sha256'):
        raise ValueError('Strict original frozen rules hash differs')
    if gate.get('passed') is not True or gate.get('strict_original_restored_verified') is not True or gate.get('training_authorized') is not False:
        raise ValueError('Actual strict-original restoration/identity gate is required')
    if gate.get('protocol_sha256') != original.PROTOCOL_SHA256:
        raise ValueError('Gate must bind actual original100Hz/solver10 protocol')
    if gate.get('phase') != phase or gate.get('plan_sha256') != saved.file_sha(plan_path):
        raise ValueError('Gate does not release this exact plan/phase')
    if gate.get('policy_sha256') != saved.file_sha(policy_path):
        raise ValueError('Gate does not bind the frozen requested-action policy')
    bindings = gate.get('bindings', {})
    required = {str(Path(__file__).resolve()), str(Path(original.__file__).resolve()), str(Path(saved.__file__).resolve())}
    if not required <= set(bindings):
        raise ValueError('Gate lacks actual collector/original/sampling source bindings')
    for path, digest in bindings.items():
        if not Path(path).is_file() or saved.file_sha(path) != digest:
            raise ValueError('Gate source/evidence binding changed: '+path)
    if policy.get('policy_execution_version') != POLICY_VERSION:
        raise ValueError('Frozen policy execution identity differs')
    if policy.get('base_rules_sha256') != saved.identity(RULES):
        raise ValueError('Frozen requested-action base rules differ')
    changes = policy.get('changes', {})
    allowed = {'push_advance_world_px', 'entry_advance_world_px', 'target_distance_world_px',
        'target_change_world_px', 'departure_distances_world_px', 'wall_room_refresh_world_px'}
    if set(changes)-allowed:
        raise ValueError('Only requested-action sampling choices may change')
    for key in set(changes)-{'departure_distances_world_px'}:
        if not isinstance(changes[key], (float, int)) or not 0 < changes[key] <= 48:
            raise ValueError('Requested-action policy outside frozen finite bounds')
    return plan, gate, dict(deepcopy(RULES), **changes)


def collect(plan_path, output, phase, physics_gate, source_root, shared_roots, *,
            policy_path, resource_policy, letters=None, run_label=None, stop_file=None,
            run_id=None, runtime_config_sha256=None, require_detached=False):
    """Attempt each released frozen scene once, retaining illegal partials."""
    plan_path, gate_path, policy_path = map(lambda p: Path(p).resolve(), (plan_path, physics_gate, policy_path))
    plan, gate, rules = verify_gate(plan_path, gate_path, policy_path, phase)
    process_evidence = process_job_evidence()
    if require_detached and (os.name != 'nt' or process_evidence['is_process_in_job'] is not False):
        raise RuntimeError('Independent background collector must actually be outside Windows Job before any scene')
    if require_detached and (not run_id or not runtime_config_sha256):
        raise ValueError('Detached collection requires immutable run_id and runtime-config SHA')
    if gate.get('run_id') is not None and run_id is not None and gate['run_id'] != run_id:
        raise ValueError('Actual run_id differs from frozen release gate')
    selected = saved.selected_slots(plan, phase, letters)
    if not run_label:
        raise ValueError('An independent named run is mandatory; no legacy continuation')
    output = Path(output).resolve()
    paths = saved.execution_io_paths(output, phase, run_label)
    paths['directory'].mkdir(parents=True, exist_ok=True)
    if any(paths[key].exists() for key in ('identity', 'ledger', 'completion')):
        raise FileExistsError('Strict-original run already exists; no retry/reset')
    stop_path = None if stop_file is None else Path(stop_file).resolve()
    identity = dict(kind='strict-original-contact-run.v1', phase=phase, run_label=run_label,
        letters=list(saved.normalize_letters(letters)), selected_slot_ids=[row['slot_id'] for row in selected],
        run_id=run_id or gate.get('run_id', run_label), plan_sha256=saved.file_sha(plan_path),
        physics_gate_sha256=saved.file_sha(gate_path), physics_protocol_sha256=original.PROTOCOL_SHA256,
        policy_sha256=saved.file_sha(policy_path), resource_policy_sha256=saved.file_sha(resource_policy),
        physics_source_sha256=saved.file_sha(original.__file__), collection_source_sha256=saved.file_sha(__file__),
        policy_rules_sha256=saved.identity(rules), policy_execution_version=POLICY_VERSION,
        runtime_config_sha256=runtime_config_sha256, process_evidence=process_evidence,
        require_detached=require_detached,
        upstream_source_root=str(Path(source_root).resolve()),
        stop_file=str(stop_path) if stop_path else None, training_eligible=False,
        old_modified_protocol_allowed=False, model_calls=0, optimizer_updates=0)
    ledger = dict(kind='strict-original-contact-actual-cost.v1', run_id=identity['run_id'],
        attempts=[], actual_controls=0, actual_valid_controls=0, actual_substeps=0,
        simulated_seconds=0., model_calls=0, optimizer_updates=0)
    def reservation(name, reserve=None):
        return ResourceReservation(output, shared_roots, resource_policy,
            f'{run_label}:pid{os.getpid()}:{name}', role='collection', reserve_bytes=reserve)
    with reservation('initialize', 1024**2) as lease:
        _reserved_json(lease, paths['identity'], identity)
        _reserved_json(lease, paths['ledger'], ledger)
    started = time.perf_counter()
    failure, stopped = None, False
    try:
        for slot in selected:
            if stop_path and stop_path.exists():
                stopped = True
                break
            directory = output/'trajectories'/slot['slot_id']
            if (directory/'claim.json').exists():
                raise FileExistsError('Frozen scene already claimed; no replacement or retry')
            with reservation(slot['slot_id']) as lease:
                directory.mkdir(parents=True, exist_ok=True)
                cost = dict(slot_id=slot['slot_id'], phase=phase, split=slot['split'], letter=slot['letter'],
                    run_label=run_label, status='reserved', started_at_utc=saved.utc(), actual_controls=0,
                    actual_valid_controls=0, actual_substeps=0, simulated_seconds=0.)
                with (directory/'claim.json').open('x', encoding='utf-8') as handle:
                    json.dump(cost, handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                ledger['attempts'].append(cost)
                _reserved_json(lease, paths['ledger'], ledger)
                states, actions, valids, images, telemetry, decisions = [], [], [], [], [], []
                env, initial, attempt_error, expected_invalid = None, None, None, False
                failure_rgb_reused = False
                attempt_started = time.perf_counter()
                try:
                    env = make_strict_original_env(source_root, slot['letter'])
                    image = env.reset_scene(**slot['initialization'])
                    states.append(np.asarray(env.state(), np.float64).copy())
                    images.append(np.asarray(image, np.uint8).copy())
                    initial = env.initial_identity()
                    if not np.allclose(states[0], slot['initial_state15'], rtol=0, atol=1e-8):
                        raise RuntimeError('Frozen scene construction differs')
                    policy = RequestedTargetPolicy(env, slot, rules)
                    for index in range(64):
                        if stop_path and stop_path.exists():
                            stopped = True
                            break
                        before = env.state().copy()
                        action, decision = policy.next_action(index)
                        saved._append_execution_event(paths['events'], dict(event='control_requested',
                            slot_id=slot['slot_id'], control=index, requested_action=action.tolist(), at_utc=saved.utc()))
                        control_started = time.perf_counter()
                        try:
                            image, info = env.step(action)
                        except Exception as exc:
                            expected_invalid = isinstance(exc, OriginalStateInvalid)
                            info = dict(getattr(exc, 'info', {}) or {}, valid=False, exception=repr(exc))
                            info.setdefault('substeps', getattr(env, 'last_substeps', []))
                            attempt_error = repr(exc)
                            try:
                                image = env.render_clean()
                            except Exception:
                                image, failure_rgb_reused = images[-1], True
                                info['failure_RGB'] = 'Unavailable; last visible frame reused as invalid diagnostic only'
                        rows = info.get('substeps', [])
                        after = np.asarray(env.state(), np.float64).copy()
                        substeps = sum(bool(row.get('executed')) for row in rows)
                        duration = sum(float(row.get('dt', 0.)) for row in rows if row.get('executed'))
                        valid = bool(info.get('valid') and substeps == 10 and len(rows) == 10
                            and all(float(row['dt']) == .01 and row.get('valid', True) for row in rows)
                            and abs(duration-.1) < 1e-9 and np.isfinite(after).all())
                        info.update(collector_wall_seconds=time.perf_counter()-control_started,
                            protocol_sha256=original.PROTOCOL_SHA256, requested_action=action.tolist(),
                            execution_intervention=False, valid=valid)
                        actions.append(action.copy()); states.append(after); valids.append(valid)
                        images.append(np.asarray(image, np.uint8).copy())
                        telemetry.append(saved._safe_json(info)); decisions.append(saved._safe_json(decision))
                        for container in (cost, ledger):
                            container['actual_controls'] += 1
                            container['actual_valid_controls'] += int(valid)
                            container['actual_substeps'] += substeps
                            container['simulated_seconds'] += duration
                        saved._append_execution_event(paths['events'], dict(event='control_returned',
                            slot_id=slot['slot_id'], control=index, valid=valid, actual_substeps=substeps,
                            actual_seconds=duration, at_utc=saved.utc()))
                        if len(actions) % 8 == 0 or not valid:
                            save_attempt(directory/'raw.npz', states, actions, valids, images,
                                telemetry, decisions, reservation=lease)
                        if not valid:
                            attempt_error = attempt_error or 'Invalid or incomplete original control'
                            break
                        policy.observe(before, after, info)
                except Exception as exc:
                    attempt_error = repr(exc)
                    expected_invalid = isinstance(exc, OriginalStateInvalid)
                finally:
                    if env is not None:
                        env.close()
                raw = save_attempt(directory/'raw.npz', states, actions, valids, images,
                    telemetry, decisions, reservation=lease) if states else None
                complete = len(actions) == 64 and all(valids) and not attempt_error
                cost.update(status='completed' if complete else 'stopped_by_user' if stopped else
                    'invalid_original_state' if expected_invalid else 'failed', finished_at_utc=saved.utc(),
                    wall_seconds=time.perf_counter()-attempt_started)
                metadata = dict(kind='strict-original-contact-attempt.v1', slot=slot, reservation=deepcopy(cost),
                    raw=raw, complete=complete, failure=attempt_error, expected_original_invalidity=expected_invalid,
                    actual_valid_controls=sum(valids), initial_identity=initial, requested_controls=64,
                    wall_seconds=cost['wall_seconds'], policy_rules=rules, policy_execution_version=POLICY_VERSION,
                    policy_sha256=identity['policy_sha256'], protocol_sha256=original.PROTOCOL_SHA256,
                    physics_source_sha256=identity['physics_source_sha256'],
                    collector_source_sha256=identity['collection_source_sha256'],
                    resource_policy_sha256=identity['resource_policy_sha256'], run_label=run_label,
                    contiguous_frames=not failure_rgb_reused, valid_contiguous_control_ranges=valid_prefix_ranges(valids),
                    invalid_failure_RGB_is_last_visible_frame=failure_rgb_reused,
                    hidden_replay_or_filled_frames=False, old_physics_data=False,
                    physics_transition_modified=False, training_eligible=False, pending_user_approval=True,
                    model_calls=0, optimizer_updates=0)
                _reserved_json(lease, directory/'metadata.json', metadata)
                _reserved_json(lease, paths['ledger'], ledger)
                _reserved_json(lease, paths['progress'], dict(phase=phase, run_label=run_label,
                    actual_attempts=len(ledger['attempts']), actual_controls=ledger['actual_controls'],
                    actual_valid_controls=ledger['actual_valid_controls'], actual_substeps=ledger['actual_substeps'],
                    simulated_seconds=ledger['simulated_seconds'], last_slot=slot['slot_id'], updated_at_utc=saved.utc()))
                print(json.dumps(dict(slot=slot['slot_id'], complete=complete, controls=len(actions),
                    valid_controls=sum(valids), substeps=cost['actual_substeps'], failure=attempt_error)), flush=True)
                if stopped:
                    break
                if not complete and not expected_invalid:
                    raise RuntimeError('Unexpected collection failure; preserve partial and stop: '+str(attempt_error))
    except Exception as exc:
        failure = repr(exc)
    # The reserved administrative margin preserves the real cost ledger even
    # when a larger data write is denied. A pending/reserved attempt is explicit,
    # and never masquerades as a fully preserved/eligible trajectory.
    saved.write_json(paths['ledger'], ledger)
    completion = dict(**identity, finished_at_utc=saved.utc(), wall_seconds=time.perf_counter()-started,
        completed=failure is None and not stopped and len(ledger['attempts']) == len(selected),
        all_slots_attempted=len(ledger['attempts']) == len(selected),
        all_trajectories_complete=len(ledger['attempts']) == len(selected) and all(
            cost['status'] == 'completed' for cost in ledger['attempts']),
        failed=failure is not None, graceful_stop=stopped, failure=failure,
        selected_slots=len(selected), actual_attempts=len(ledger['attempts']),
        complete_trajectories=sum(cost['status'] == 'completed' for cost in ledger['attempts']),
        invalid_trajectories=sum(cost['status'] == 'invalid_original_state' for cost in ledger['attempts']),
        actual_controls=ledger['actual_controls'], actual_valid_controls=ledger['actual_valid_controls'],
        actual_substeps=ledger['actual_substeps'], simulated_seconds=ledger['simulated_seconds'],
        raw_preservation_complete=all(cost['status'] != 'reserved' for cost in ledger['attempts']),
        pending_raw_attempts=[cost['slot_id'] for cost in ledger['attempts'] if cost['status'] == 'reserved'],
        ledger=str(paths['ledger']), unstarted_slots=[row['slot_id'] for row in selected
            if not (output/'trajectories'/row['slot_id']/'claim.json').exists()])
    saved.write_json(paths['completion'], completion)
    if failure:
        raise RuntimeError(failure)
    return completion
