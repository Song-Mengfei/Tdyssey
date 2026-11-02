"""One approved, finite CPU acceptance collection in the pinned original loop.

The only inherited behavior is original physics plus observation-only validity.
No old requested-target policy is instantiated. Targets are never transformed.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zlib

import numpy as np

from .native_rgb_original_physics import PROTOCOL_SHA256
from .native_rgb_r2a_env import R2AStepAborted
from .strict_original_contact_pilot import StrictObservedOriginalEnv, OriginalStateInvalid, process_job_evidence
from .original_contact_pilot import _safe_json, storage_snapshot
from .original_contact_pilot_resources import ResourceReservation

LIMITS = dict(attempts=120, controls_per_attempt=64, controls=7680,
              regular_substeps=76800, new_bytes=256*1024**2,
              shared_bytes=3*1024**3, atomic_reserve_bytes=256*1024**2)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write(path, value, lease=None, exclusive=False):
    path = Path(path)
    data = (json.dumps(_safe_json(value), indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
    if lease is not None:
        lease.write_bytes(path, data)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path if exclusive else path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('xb' if exclusive else 'wb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    if not exclusive:
        tmp.replace(path)


class DirectTargetOriginalEnv(StrictObservedOriginalEnv):
    """Accept original action-space endpoints; execute pinned step verbatim."""
    def step(self, action):
        if not self._reset or self._terminal or self._in_step:
            raise RuntimeError('A live original-physics session is required')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all() or (action < 0).any() or (action > 512).any():
            raise ValueError('Original action_space requires finite absolute xy in [0,512]')
        # Independent copies prove both the requested and actual upstream input.
        self.last_passed_action = action.copy()
        self._action = action.copy(); self.last_substeps = []; self._in_step = True
        self._base.contact_points = []; self._base.contact_angles = []; self._base.substep_penetrations = []
        try:
            self._upstream.step(self._base, action)
            self._base.control_step += 1
            if len(self.last_substeps) != 10 or not np.isfinite(self.state()).all():
                raise RuntimeError('Incomplete/nonfinite original control')
            return self.render_clean(), dict(valid=True, physics_substeps=10,
                control_step=self.control_step, substeps=self.last_substeps,
                requested_action=self._action.tolist(), passed_action=self.last_passed_action.tolist(),
                protocol_sha256=PROTOCOL_SHA256, execution_intervention=False)
        except BaseException as exc:
            self._terminal = True
            if isinstance(exc, R2AStepAborted):
                raise
            raise R2AStepAborted(str(exc), dict(valid=False, aborted=True,
                substeps=self.last_substeps, control_step=self.control_step)) from exc
        finally:
            self._in_step = False


@contextmanager
def counted_original_steps(cost):
    """Observe calls including construction/reset, without introducing calls."""
    import pymunk
    original = pymunk.Space.step
    phase = {'value': 'initialization'}
    def counted(space, dt):
        prefix = phase['value']
        # Count an invoked integrator even if its own execution raises.
        cost[prefix+'_integrator_calls'] += 1
        cost[prefix+'_integrator_seconds'] += float(dt)
        return original(space, dt)
    pymunk.Space.step = counted
    try:
        yield phase
    finally:
        pymunk.Space.step = original


def valid_windows(mask, times, segments, horizon):
    """Four true history frames, contiguous legal history and future actions."""
    mask = np.asarray(mask, bool); times = np.asarray(times); segments = np.asarray(segments)
    return [t for t in range(3, len(mask)-horizon+1)
            if mask[t-3:t+horizon].all()
            and np.all(segments[t-3:t+horizon+1] == segments[t])
            and np.allclose(np.diff(times[t-3:t+horizon+1]), .1, rtol=0, atol=1e-8)]


def save_raw(path, states, actions, passed, valid, images, telemetry, decisions, lease):
    payload = json.dumps(_safe_json(dict(telemetry=telemetry, decisions=decisions)),
                         separators=(',', ':'), sort_keys=True, allow_nan=False).encode()
    durations = [sum(float(s.get('dt', 0)) for s in row.get('substeps', []) if s.get('executed')) for row in telemetry]
    arrays = dict(states=np.asarray(states, np.float64).reshape(-1, 15),
        requested_actions=np.asarray(actions, np.float64).reshape(-1, 2),
        passed_actions=np.asarray(passed, np.float64).reshape(-1, 2),
        control_valid=np.asarray(valid, bool), rgb=np.asarray(images, np.uint8).reshape(-1, 3, 96, 96),
        frame_indices=np.arange(len(states), dtype=np.int64), segment_ids=np.zeros(len(states), np.int64),
        control_time_seconds=np.asarray([0., *np.cumsum(durations)], np.float64),
        telemetry_zlib=np.frombuffer(zlib.compress(payload, 6), np.uint8),
        telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(), np.uint8))
    if not (len(arrays['states']) == len(arrays['rgb']) == len(actions)+1):
        raise RuntimeError('Unaligned raw sequence; no fabricated frames')
    if not np.array_equal(arrays['requested_actions'], arrays['passed_actions']):
        raise RuntimeError('Requested target differs from actual upstream input')
    packed = io.BytesIO(); np.savez_compressed(packed, **arrays)
    lease.write_bytes(path, packed.getvalue())
    return dict(file=str(Path(path).resolve()), bytes=Path(path).stat().st_size, sha256=sha(path),
        requested_equals_passed=True, windows={str(h):len(valid_windows(valid, arrays['control_time_seconds'], arrays['segment_ids'], h)) for h in (1,4,8,16)})


def validate_config(path, digest):
    path = Path(path).resolve(strict=True)
    if sha(path) != digest:
        raise ValueError('Frozen runtime-config SHA mismatch')
    c = read(path)
    if c['kind'] != 'direct-target-acceptance.v1' or c['limits'] != LIMITS:
        raise ValueError('Wrong acceptance identity or finite budget')
    if c['training_authorized'] or c['model_calls'] != 0 or c['optimizer_updates'] != 0 or c['auto_retry'] or c['next_phase_released']:
        raise ValueError('Scope exceeds approved CPU collection')
    if c['physics_protocol_sha256'] != PROTOCOL_SHA256:
        raise ValueError('Original physics protocol identity mismatch')
    for binding in c['bindings']:
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Frozen source or evidence changed: '+binding['path'])
    repo=Path(c['repo']).resolve()
    required={Path(c[k]).resolve() for k in ('plan','resource_policy','storage_roots','approval_evidence')}
    required.update(repo/p for p in ('tdyssey/direct_target_collection.py','tdyssey/direct_target_policy.py',
        'tdyssey/direct_target_report.py','scripts/direct_target_collect.py','scripts/direct_target_background.py',
        'scripts/direct_target_report.py','tdyssey/native_rgb_original_physics.py','tdyssey/native_rgb_env.py',
        'tdyssey/strict_original_contact_pilot.py','tdyssey/envs/official.py','tdyssey/letters.py',
        'tdyssey/original_contact_pilot_resources.py','manifests/official_dp.json'))
    required.update(Path(c['source_root'])/p for p in ('diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py'))
    if not required <= {Path(b['path']).resolve() for b in c['bindings']}:
        raise ValueError('Required runtime source/approval/plan/physics binding missing')
    plan = read(c['plan']); attempts = plan['attempts']
    if len(attempts) != 120 or len(set(a['attempt_id'] for a in attempts)) != 120:
        raise ValueError('Exactly frozen120 unique attempt slots required')
    if len(set(a['family_id'] for a in attempts)) != 96 or set(a['letter'] for a in attempts) != set('TLOCXB'):
        raise ValueError('Wrong family/letter scope')
    for a in attempts:
        if not a.get('plan_valid'):
            raise ValueError('Frozen plan contains an invalid pre-execution target design: '+a['attempt_id'])
        values = np.asarray(a['actions'], np.float64)
        if values.shape != (64, 2) or not np.isfinite(values).all() or (values<0).any() or (values>512).any():
            raise ValueError('Frozen action plan outside original domain')
    return c, plan


def collect(config_path, digest, require_detached=True):
    c, plan = validate_config(config_path, digest)
    root = Path(c['artifact_root']); output = root/'collection'; output.mkdir(parents=True, exist_ok=True)
    process = process_job_evidence()
    if require_detached and process.get('is_process_in_job') is not False:
        raise RuntimeError('Collector must be outside Windows Job before construction')
    if (output/'ledger.json').exists():
        raise FileExistsError('Existing attempt ledger; no retry or replacement')
    roots = read(c['storage_roots'])['roots']
    ledger = dict(run_id=c['run_id'], config_sha256=digest, attempts=[], actual_controls=0,
        actual_valid_controls=0, actual_substeps=0, simulated_seconds=0., initialization_integrator_calls=0,
        initialization_integrator_seconds=0., regular_integrator_calls=0, regular_integrator_seconds=0., model_calls=0, optimizer_updates=0)
    write(output/'identity.json', dict(run_id=c['run_id'],config_sha256=digest,process=process, started_at_utc=utc(),
        action_domain=[0,512], old_model_domain=[0,511], model_normalization_changed=False,
        physics_protocol_sha256=PROTOCOL_SHA256), exclusive=True)
    started = time.perf_counter(); failure = None; stopped = False
    for slot in plan['attempts']:
        if Path(c['stop_file']).exists():
            stopped = True; break
        try:
            with ResourceReservation(root, roots, c['resource_policy'], c['run_id']+':'+slot['attempt_id']) as lease:
                cost = dict(attempt_id=slot['attempt_id'],family_id=slot['family_id'],letter=slot['letter'],
                    scenario=slot['scenario'],status='claimed', actual_controls=0,actual_valid_controls=0,
                    actual_substeps=0,simulated_seconds=0., initialization_integrator_calls=0,
                    initialization_integrator_seconds=0.,regular_integrator_calls=0,regular_integrator_seconds=0.,started_at_utc=utc())
                claim = output/'claims'/(slot['attempt_id']+'.json')
                write(claim,cost,exclusive=True)
                ledger['attempts'].append(cost); write(output/'ledger.json',ledger,lease)
                states=[]; actions=[]; passed=[]; valid=[]; images=[]; telemetry=[]; decisions=[]
                error=None; expected_invalid=False; initial=None; env=None; first_invalid=None; matched=None
                attempt_start=time.perf_counter()
                try:
                    with counted_original_steps(cost) as counter:
                        env = DirectTargetOriginalEnv(c['source_root'],slot['letter'])
                        image = env.reset_scene(**slot['initialization'])
                        initial = env.initial_identity(); states.append(env.state().copy()); images.append(image.copy())
                        if not np.allclose(states[0],slot['initial_state15'],rtol=0,atol=1e-8):
                            raise RuntimeError('Frozen initial state differs from actual construction')
                        # Original action-space is inspected, never changed.
                        if not (np.array_equal(env.action_space.low,[0,0]) and np.array_equal(env.action_space.high,[512,512])):
                            raise RuntimeError('Pinned original action_space differs')
                        counter['value']='regular'
                        for index, target in enumerate(slot['actions']):
                            if Path(c['stop_file']).exists():
                                stopped=True; break
                            if ledger['actual_controls'] >= LIMITS['controls']:
                                raise RuntimeError('Unified physical control cap reached')
                            action=np.asarray(target,np.float64)
                            info={}; control_start=time.perf_counter()
                            try:
                                image,info=env.step(action)
                            except Exception as exc:
                                error=repr(exc); expected_invalid=isinstance(exc,OriginalStateInvalid)
                                info=dict(getattr(exc,'info',{}) or {},valid=False,exception=error)
                                info.setdefault('substeps',env.last_substeps)
                                image=None
                            rows=info.get('substeps',[])
                            count=sum(bool(s.get('executed')) for s in rows)
                            duration=sum(float(s.get('dt',0)) for s in rows if s.get('executed'))
                            legal=bool(info.get('valid') and count==10 and len(rows)==10 and all(s.get('valid',True) for s in rows)
                                       and abs(duration-.1)<1e-9 and np.isfinite(env.state()).all())
                            info.update(requested_action=action.tolist(),passed_action=env.last_passed_action.tolist(),
                                valid=legal,control_wall_seconds=time.perf_counter()-control_start,execution_intervention=False)
                            for dest in (cost,ledger):
                                dest['actual_controls']+=1;dest['actual_valid_controls']+=int(legal)
                                dest['actual_substeps']+=count;dest['simulated_seconds']+=duration
                            if not legal:
                                first_invalid=dict(control=index,substep=info.get('first_invalid_substep'),
                                    reasons=info.get('reasons'),substeps_executed=count)
                            if image is None:
                                try:image=env.render_clean()
                                except Exception as render_error:
                                    # Preserve every real call and failed terminal state even without an RGB frame.
                                    # Raw NPZ stops at the previous available frame; this tail is explicitly excluded from windows.
                                    cost['unframed_terminal_control']=dict(control=index,action=action.tolist(),
                                        passed_action=env.last_passed_action.tolist(),post_state=_safe_json(env.state()),
                                        telemetry=_safe_json(info),RGB_available=False,renderer_error=repr(render_error))
                                    raise RuntimeError('Terminal RGB unavailable; actual call retained in attempt cost') from render_error
                            actions.append(action.copy());passed.append(env.last_passed_action.copy());valid.append(legal)
                            states.append(env.state().copy());images.append(image.copy());telemetry.append(_safe_json(info))
                            decisions.append(dict(control=index,phase=slot['planned_phase'][index],action_plan_frozen_before_initialization=True,
                                source='frozen absolute world XY',future_state_used=False))
                            # Branches are full independent construction+prefix replay, never public snapshots.
                            if index==2:
                                prefix_state=np.asarray(states);prefix_rgb=np.asarray(images);prefix_actions=np.asarray(actions)
                                baseline=slot.get('branch_reference_attempt_id',slot['attempt_id'])
                                if baseline != slot['attempt_id']:
                                    with np.load(output/'raw'/(baseline+'.npz'),allow_pickle=False) as previous:
                                        matched=bool(len(previous['control_valid'])>=3 and previous['control_valid'][:3].all() and
                                            np.array_equal(prefix_state,previous['states'][:4]) and
                                            np.array_equal(prefix_rgb,previous['rgb'][:4]) and np.array_equal(prefix_actions,previous['requested_actions'][:3]))
                                    if not matched:
                                        raise RuntimeError('Branch common state/RGB/action prefix differs')
                                else:
                                    matched=True
                            if len(actions)%8==0 or not legal:
                                save_raw(output/'raw'/(slot['attempt_id']+'.npz'),states,actions,passed,valid,images,telemetry,decisions,lease)
                            if not legal:
                                error=error or 'Invalid original state';break
                except Exception as exc:
                    error=repr(exc);expected_invalid=isinstance(exc,OriginalStateInvalid)
                    if expected_invalid and first_invalid is None:
                        first_invalid=dict(initial=True,**_safe_json(getattr(exc,'info',{})))
                    if env is not None and not states:
                        try:
                            initial=env.initial_identity()
                            initial_rgb=env.render_clean()
                            states.append(env.state().copy());images.append(initial_rgb.copy())
                        except Exception as initial_evidence_error:
                            cost['initial_evidence_error']=repr(initial_evidence_error)
                            try:cost['actual_initial_state']=_safe_json(env.state())
                            except Exception:pass
                finally:
                    if env is not None:
                        try:env.close()
                        except Exception as close_error:
                            cost['close_error']=repr(close_error)
                            error=error or 'Environment close failed: '+repr(close_error)
                            expected_invalid=False
                ledger['initialization_integrator_calls']+=cost['initialization_integrator_calls']
                ledger['initialization_integrator_seconds']+=cost['initialization_integrator_seconds']
                ledger['regular_integrator_calls']+=cost['regular_integrator_calls']
                ledger['regular_integrator_seconds']+=cost['regular_integrator_seconds']
                if cost['regular_integrator_calls'] != cost['actual_substeps']:
                    cost['integrator_telemetry_mismatch']=True
                    error=error or 'Actual integrator calls differ from preserved telemetry'
                    expected_invalid=False
                raw=save_raw(output/'raw'/(slot['attempt_id']+'.npz'),states,actions,passed,valid,images,telemetry,decisions,lease) if states else None
                complete=len(actions)==64 and all(valid) and error is None
                cost.update(status='completed' if complete else 'stopped_by_user' if stopped else 'invalid_original_state' if expected_invalid else 'engineering_failed',
                    wall_seconds=time.perf_counter()-attempt_start,finished_at_utc=utc(),error=error,first_invalid=first_invalid)
                metadata=dict(attempt=slot,run_id=c['run_id'],config_sha256=digest,complete=complete,raw=raw,cost=deepcopy(cost),
                    initial_identity=initial,common_prefix_matched=matched,first_invalid=first_invalid,
                    training_eligible=False,pending_user_review=True,model_calls=0,optimizer_updates=0,
                    request_equals_passed=raw['requested_equals_passed'] if raw else None,physics_transition_modified=False,
                    reset_or_initialization_substeps=cost['initialization_integrator_calls'],original_action_domain=[0,512],
                    prior_modified_physics_data_reused=False,hidden_replay=False,filled_frames=False)
                write(output/'attempts'/(slot['attempt_id']+'.json'),metadata,lease)
                write(output/'ledger.json',ledger,lease)
                write(output/'progress.json',dict(run_id=c['run_id'],config_sha256=digest,attempts=len(ledger['attempts']),
                    actual_controls=ledger['actual_controls'],last_attempt=cost,updated_at_utc=utc()),lease)
                print(json.dumps(dict(attempt=slot['attempt_id'],status=cost['status'],controls=len(actions),substeps=cost['actual_substeps'])),flush=True)
                if stopped:break
                if not complete and not expected_invalid:
                    raise RuntimeError('Engineering failure; no extra attempt or replacement: '+str(error))
        except Exception as exc:
            failure=repr(exc);break
    completion=dict(run_id=c['run_id'],config_sha256=digest,finished_at_utc=utc(),wall_seconds=time.perf_counter()-started,
        completed=failure is None and not stopped and len(ledger['attempts'])==120,failed=failure is not None,failure=failure,graceful_stop=stopped,
        actual_attempts=len(ledger['attempts']),complete_trajectories=sum(a['status']=='completed' for a in ledger['attempts']),
        invalid_trajectories=sum(a['status']=='invalid_original_state' for a in ledger['attempts']),
        actual_controls=ledger['actual_controls'],actual_valid_controls=ledger['actual_valid_controls'],actual_substeps=ledger['actual_substeps'],
        simulated_seconds=ledger['simulated_seconds'],initialization_integrator_calls=ledger['initialization_integrator_calls'],
        initialization_integrator_seconds=ledger['initialization_integrator_seconds'],unstarted_attempts=[s['attempt_id'] for s in plan['attempts'][len(ledger['attempts']):]],
        regular_integrator_calls=ledger['regular_integrator_calls'],regular_integrator_seconds=ledger['regular_integrator_seconds'],
        raw_preservation_complete=all((output/'attempts'/(a['attempt_id']+'.json')).is_file() for a in ledger['attempts']),
        pending_raw_attempts=[a['attempt_id'] for a in ledger['attempts'] if not (output/'attempts'/(a['attempt_id']+'.json')).is_file()],
        shared_storage=storage_snapshot([*roots,root]),new_storage=storage_snapshot([root]),model_calls=0,optimizer_updates=0,
        next_phase_released=False,training_authorized=False)
    write(output/'ledger.json',ledger);write(output/'completion.json',completion)
    return completion
