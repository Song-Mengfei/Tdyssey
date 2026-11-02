"""Bounded new-data collection; pinned original step, passive event observer.

Geometric state is used only for privileged task planning/quality evidence.
Every 16-action block is committed before its first physical call.  Nothing in
this file changes a body, collision response, PD update, or requested target.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import zlib

import numpy as np
from shapely.geometry import LineString, Point

from tdyssey.direct_target_collection import DirectTargetOriginalEnv, counted_original_steps
from tdyssey.direct_target_policy import (_placed, _choose_contact, _diameter,
    _distance_endpoint, _endpoint, _line, _rotation, geometry_at, validate_targets)
from tdyssey.native_rgb_env import shape_geometry, STATE_COLUMNS
from tdyssey.native_rgb_original_physics import OriginalPhysicsEnv, PROTOCOL_SHA256
from tdyssey.native_rgb_r2a_env import R2AStepAborted
from tdyssey.strict_original_contact_pilot import state_invalidity, OriginalStateInvalid

LETTERS = tuple('TLOCXB')
TASKS = ('translation', 'rotation_cw', 'rotation_ccw', 'surface_change',
         'stop_response', 'depart_recontact', 'boundary', 'negative')
LENGTHS = (35, 19, 19, 35, 35, 51, 51, 19)
EPS = 1e-8


def utc():
    return datetime.now(timezone.utc).isoformat()


def _safe(value):
    if isinstance(value, dict): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, np.ndarray): return _safe(value.tolist())
    if isinstance(value, np.generic): return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def atomic_json(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_safe(value), ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode()
    temp = path if exclusive else path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('xb' if exclusive else 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if not exclusive: temp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _storage(root, config, reserve=0):
    # The stage-owned helper implements the newly authorized 10G *shared* cap.
    from common import check_storage
    return check_storage(Path(root), config, reserve_bytes=int(reserve))


def build_plan(seed=547810315):
    """Pure geometry, no environment creation: 1536 independent frozen families."""
    families = []
    boundaries = ('left','right','top','bottom','left-top','right-top','left-bottom','right-bottom')
    for replicate in range(32):
        for li, letter in enumerate(LETTERS):
            D = _diameter(shape_geometry(letter))
            for ti, task in enumerate(TASKS):
                scene_seed = int(seed)+replicate*100000+li*1000+ti*11
                rng = np.random.default_rng(scene_seed)
                pose_bin = (replicate+ti+li) % 8
                angle = (pose_bin+float(rng.uniform(.08,.92)))*math.pi/4
                boundary = boundaries[(replicate+li)%8] if task == 'boundary' else None
                state, geom = _placed(letter,angle,rng.uniform(190.,322.,2),boundary,float(rng.uniform(3.,9.)))
                policy_task = 'hold_turn_withdraw' if task == 'stop_response' else task
                contact, pusher, direction = _choose_contact(letter,state,geom,policy_task,
                    (replicate*5+li+ti)%16,rng,boundary)
                state[:2] = pusher
                center = state[2:4]+_rotation(angle) @ np.asarray((0.,45.) if letter=='T' else (0.,0.))
                family_id = f'AC-{letter}-{ti:02d}-{replicate:02d}'
                families.append(dict(attempt_id=family_id,family_id=family_id,letter=letter,task=task,
                    scenario=task,seed=scene_seed,replicate=replicate,pose_bin=pose_bin,
                    split='seen_val' if replicate%8==0 else 'train',max_controls=LENGTHS[ti],
                    initialization=dict(center=center.tolist(),pusher=pusher.tolist(),angle=angle),
                    initial_state15=state.tolist(),selected_contact=contact,
                    initial_boundary=boundary,geometry_diameter_world_px=D,
                    distance_bin=(replicate+ti+li)%3,
                    initial_drive_direction=direction.tolist(),
                    initial_drive_direction_local=(_rotation(-angle) @ direction).tolist(),
                    mode='moving',branch_id=0))
    if len(families)!=1536 or len({x['family_id'] for x in families})!=1536:
        raise AssertionError('Frozen independent family table changed')
    return dict(kind='accelerated-original-contact-tasks.v1',seed=int(seed),attempts=families,
        initial_families=1536,reserve_attempts=464,letters=list(LETTERS),task_lengths=dict(zip(TASKS,LENGTHS)),
        split_rule='within each letter/task, replicates0/8/16/24 seen_val; all other replicates train',
        branches=False,physics_protocol_sha256=PROTOCOL_SHA256,
        action_domain=[0,512],target_clipping=False,
        block_commitment='3 real history holds; subsequent16-target blocks committed before execution',
        reserve_rule='only observed clean motion/contact support deficits; same planner; no replacement/retry')


def _native_contact(row, kind, side=None):
    events = [e for e in row.get('contacts',[]) if e.get('kind')==kind and e.get('phase')=='post_solve']
    if side is not None:
        # New telemetry binds the actual unmodified Segment to its wall side.
        # Historical saved rows lack that field. Their wall contact point must
        # uniquely identify one radius2 capsule; an ambiguous corner stays unknown.
        def event_side(event):
            if event.get('wall_side') is not None:return event['wall_side']
            order=event.get('shapes_order',[])
            if 'wall' not in order:return None
            point_key='point_a' if order.index('wall')==0 else 'point_b'
            candidates=[]
            for p in event.get('points',[]):
                q=p.get(point_key)
                if q is None:continue
                x,y=map(float,q)
                candidates.extend(s for s,d in (('left',abs(x-5.)),('right',abs(x-506.)),
                    ('top',abs(y-5.)),('bottom',abs(y-506.))) if d<=2.+EPS)
            unique=set(candidates)
            return next(iter(unique)) if len(unique)==1 else None
        events = [e for e in events if event_side(e)==side]
    return bool(events)


class ObservationStop(R2AStepAborted):
    """Expected geometric hard stop or unresolved contact observation."""


class WatchTracker:
    """Pure event classification; fixture-testable without a physical world."""
    def __init__(self):
        self.events=[]; self.active={}; self.first_warning_control=None

    def update(self, row, control, slop, block_end, separated):
        depth=np.asarray(row['penetrations'],float)
        pairs=[('AO','agent_object',1,None)]
        if depth[0]>1.+EPS:
            side=min(row['signed_wall_gaps'],key=row['signed_wall_gaps'].get)
            pairs.append(('OW:'+side,'object_wall',0,side))
        # Preserve an ongoing wall-side observation after its depth drops.
        for key in list(self.active):
            if key.startswith('OW:') and not any(p[0]==key for p in pairs):
                pairs.append((key,'object_wall',0,key.split(':',1)[1]))
        stop=None
        for key,kind,di,side in pairs:
            d=float(depth[di])
            if key not in self.active and d>1.+EPS:
                event=dict(event_id=len(self.events),pair=key,first_control=int(control),
                    first_warning_control=int(control),first_substep=int(row['index']),
                    first_global_microstep=int(row['global_microstep']),first_depth=d,peak_depth=d,
                    block_end_control=int(block_end),status='WATCH',observed_microsteps=0,
                    continuous_native_contact=True)
                self.events.append(event);self.active[key]=event
                if self.first_warning_control is None:self.first_warning_control=int(control)
            event=self.active.get(key)
            if event is None: continue
            native=_native_contact(row,kind,side)
            event['observed_microsteps']+=1;event['peak_depth']=max(event['peak_depth'],d)
            event['continuous_native_contact'] &= native
            gap=bool(separated(key,side))
            event['last_depth']=d;event['last_control']=int(control);event['last_substep']=int(row['index'])
            if native and event['continuous_native_contact'] and not gap and d<=slop+EPS:
                event.update(status='RECOVERED_CONTACT_CANDIDATE',terminal_depth=d,
                    terminal_global_microstep=int(row['global_microstep']))
                del self.active[key]
            elif gap or not native:
                event.update(status='SEPARATED_UNRESOLVED',geometric_separation=gap,
                    terminal_global_microstep=int(row['global_microstep']))
                del self.active[key];stop='SEPARATED_UNRESOLVED'
        return stop

    def end_block(self, control):
        if not self.active:return False
        for event in self.active.values():
            event.update(status='UNRESOLVED_AT_BLOCK_END',terminal_control=int(control))
        self.active.clear();return True


class ContactTaskEnv(DirectTargetOriginalEnv):
    def __init__(self, source_root, letter, *, max_substeps=510):
        self.tracker=WatchTracker();self.block_end=3;self.microsteps=0
        self.max_substeps=int(max_substeps);self.hard_stop=None
        super().__init__(source_root,letter)

    def _record(self, arbiter, phase):
        previous=len(self._events)
        super()._record(arbiter,phase)
        if len(self._events)==previous:return
        event=self._events[-1]
        if event.get('kind') not in ('object_wall','agent_wall'):return
        try:
            wall=next(s for s in arbiter.shapes if self._role(s.body)=='wall')
            a=wall.body.local_to_world(wall.a);b=wall.body.local_to_world(wall.b)
            if abs(a.x-5.)<=EPS and abs(b.x-5.)<=EPS:side='left'
            elif abs(a.x-506.)<=EPS and abs(b.x-506.)<=EPS:side='right'
            elif abs(a.y-5.)<=EPS and abs(b.y-5.)<=EPS:side='top'
            elif abs(a.y-506.)<=EPS and abs(b.y-506.)<=EPS:side='bottom'
            else:raise RuntimeError('Unrecognized original wall Segment')
            event['wall_side']=side
        except Exception as exc:self._callback_error=repr(exc)

    def _bounds(self):
        vertices=[]
        for shape in self.block.shapes:
            if not hasattr(shape,'get_vertices') or float(shape.radius)!=0.:
                raise RuntimeError('Original zero-radius polygon geometry changed')
            vertices.extend(tuple(self.block.local_to_world(p)) for p in shape.get_vertices())
        a=np.asarray(vertices,float)
        return [float(a[:,0].min()),float(a[:,1].min()),float(a[:,0].max()),float(a[:,1].max())]

    def _observe_step(self, dt):
        if self.microsteps>=self.max_substeps:
            raise RuntimeError('Finite per-attempt substep allowance exceeded')
        previous=len(self.last_substeps)
        try: OriginalPhysicsEnv._observe_step(self,dt)
        finally:
            if len(self.last_substeps)>previous and self.last_substeps[-1].get('executed'):
                self.microsteps+=1
        row=self.last_substeps[-1];row['global_microstep']=self.microsteps-1
        row['control_index']=int(self.control_step)
        state=np.asarray(row.get('post_state',[]),float);depth=np.asarray(row.get('penetrations',[]),float)
        finite=state.shape==(15,) and np.isfinite(state).all() and depth.shape==(3,) and np.isfinite(depth).all()
        region=self._point_region() if finite else 'unknown_nonfinite'
        bounds=self._bounds() if finite else None
        hard=[]
        if not finite:hard.append('nonfinite')
        if region=='solid' or region.startswith('hole-'):hard.append('illegal_pusher_center_region')
        if finite and depth[2]>1.+EPS:hard.append('AW_overlap_above_1')
        if bounds is not None and (min(bounds[:2])<3. or max(bounds[2:])>508.):
            hard.append('object_outer_wall_envelope_crossed')
        row.update(pusher_region=region,object_collision_vertex_bounds=bounds,
            legacy_invalid_reasons=state_invalidity(state,depth,region),
            would_stop_legacy=bool(state_invalidity(state,depth,region)),
            hard_reasons=hard,valid=not hard)
        if hard:
            self.hard_stop=dict(status='HARD_GEOMETRY_STOP',control=int(self.control_step),
                substep=int(row['index']),global_microstep=self.microsteps-1,reasons=hard,
                penetrations=depth.tolist(),object_collision_vertex_bounds=bounds)
            raise ObservationStop('Explicit original-geometry violation',dict(valid=False,
                hard_stop=self.hard_stop,substeps=self.last_substeps))
        def separated(key,side):
            if key=='AO':return self.world_geometry().distance(Point(*state[:2]))-15.>EPS
            return float(row['signed_wall_gaps'][side])>EPS
        status=self.tracker.update(row,int(self.control_step),float(self.space.collision_slop),self.block_end,separated)
        row['warning_active']=bool(self.tracker.active)
        if status:
            raise ObservationStop(status,dict(valid=False,status=status,substeps=self.last_substeps))


def _domain_goal(start,direction,length):
    target=_endpoint(start,direction,length)
    if target is None:raise ValueError('Committed task endpoint outside original0..512 domain')
    return target


def _free_route(start, geom, D, rng):
    """Select a genuine empty straight path; no impossible wall-corner targets."""
    candidates=[]
    for angle in rng.uniform(-math.pi,math.pi,48):
        direction=np.asarray((math.cos(angle),math.sin(angle)))
        end=_endpoint(start,direction,float(rng.uniform(.18,.95))*D)
        if end is None or (end<22.000001).any() or (end>488.999999).any():continue
        if geom.distance(LineString((start,end)))>=15.000001:candidates.append(end)
    if not candidates:raise ValueError('No pre-committed empty route available for this fixed scene')
    return max(candidates,key=lambda p:float(np.linalg.norm(p-start)))


def block_targets(slot, block_index, state):
    """Pure planner reading only the state at this block's commitment boundary."""
    rng=np.random.default_rng(int(slot['seed'])+100000007*(block_index+1))
    start=np.asarray(state[:2],float);geom=geometry_at(slot['letter'],state)
    D=float(slot['geometry_diameter_world_px']);task=slot['task']
    direction=_rotation(float(state[4]))@np.asarray(slot['initial_drive_direction_local'],float)
    mode='linear_absolute_targets';purpose=task
    if task=='negative':
        if slot['replicate']%4==0:goal=start.copy();purpose='stationary_negative'
        else:goal=_free_route(start,geom,D,rng);purpose='no_contact_pusher_motion'
    elif task=='stop_response' and block_index==1:
        goal=start.copy();purpose='stop_response'
    elif task=='depart_recontact' and block_index==1:
        goal=_domain_goal(start,-direction,.28*D);purpose='withdraw'
    elif task=='depart_recontact' and block_index==2:
        nearest=geom.exterior.interpolate(geom.exterior.project(Point(*start)))
        direction=np.asarray((nearest.x,nearest.y))-start
        norm=float(np.linalg.norm(direction))
        if norm<=1e-12:raise ValueError('Recontact direction unavailable at fixed block boundary')
        direction/=norm
        goal,detail=_distance_endpoint(start,direction,D,slot['distance_bin'],rng)
        if goal is None:raise ValueError(detail['reason'])
        purpose='recontact'
    elif task=='boundary':
        boundary=slot['initial_boundary']
        toward=np.asarray((-1 if 'left' in boundary else 1 if 'right' in boundary else 0,
                           -1 if 'top' in boundary else 1 if 'bottom' in boundary else 0),float)
        toward/=np.linalg.norm(toward)
        if block_index==0:
            gaps=dict(left=geom.bounds[0]-7.,right=504.-geom.bounds[2],top=geom.bounds[1]-7.,bottom=504.-geom.bounds[3])
            gap=min(gaps[s] for s in ('left','right','top','bottom') if s in boundary)
            goal=_domain_goal(start,toward,max(0.,gap)+float(rng.uniform(.18,.35))*D)
            purpose='object_approach_wall'
        elif block_index==1:
            tangent=np.asarray((-toward[1],toward[0]))
            if slot['replicate']%2:tangent=-tangent
            goal=_domain_goal(start,tangent,float(rng.uniform(.18,.48))*D);purpose='along_wall'
        else:goal=_domain_goal(start,-toward,.28*D);purpose='leave_wall'
    elif task=='surface_change' and block_index==1:
        # A route around the actual exterior offset changes the contact side.
        # Committed waypoints, not collision correction or within-block feedback.
        ring=geom.buffer(18.,resolution=16).exterior
        origin=float(ring.project(Point(*start)));length=float(ring.length)
        sign=1 if slot['replicate']%2==0 else -1
        coordinates=[]
        for fraction in np.arange(1,17)/16:
            point=ring.interpolate((origin+sign*.30*length*fraction)%length)
            coordinates.append((point.x,point.y))
        actions=validate_targets(coordinates)
        return actions,dict(purpose='change_contact_surface',planner=mode,
            requested_length_fraction=.30,uses_current_boundary_state=True)
    else:
        goal,detail=_distance_endpoint(start,direction,D,slot['distance_bin'],rng)
        if goal is None:raise ValueError(detail['reason'])
        purpose='continue_push' if block_index else task
    actions=validate_targets(_line(start,goal,16))
    return actions,dict(purpose=purpose,planner=mode,start=start.tolist(),endpoint=goal.tolist(),
        target_distance_world_px=float(np.linalg.norm(goal-start)),uses_current_boundary_state=True)


def _save_raw(path, states, images, actions, passed, valid, telemetry, segments):
    payload=json.dumps(_safe(dict(telemetry=telemetry)),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    durations=[sum(float(s.get('dt',0)) for s in x.get('substeps',[]) if s.get('executed')) for x in telemetry]
    contact=[any(e.get('kind')=='agent_object' and e.get('phase')=='post_solve' and
        np.linalg.norm(e.get('total_impulse',[0,0]))>0 for s in x.get('substeps',[]) for e in s.get('contacts',[])) for x in telemetry]
    arrays=dict(states=np.asarray(states,np.float64).reshape(-1,15),rgb=np.asarray(images,np.uint8).reshape(-1,3,96,96),
        requested_actions=np.asarray(actions,np.float64).reshape(-1,2),passed_actions=np.asarray(passed,np.float64).reshape(-1,2),
        control_valid=np.asarray(valid,bool),contact_positive=np.asarray(contact,bool),frame_indices=np.arange(len(states)),
        control_time_seconds=np.asarray([0.,*np.cumsum(durations)],np.float64),times=np.asarray([0.,*np.cumsum(durations)],np.float64),
        segment_ids=np.zeros(len(states),np.int64),telemetry_zlib=np.frombuffer(zlib.compress(payload,6),np.uint8),
        telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(),np.uint8))
    if len(states)!=len(images) or len(states)!=len(actions)+1:raise RuntimeError('Unaligned actual sequence')
    if not np.array_equal(arrays['requested_actions'],arrays['passed_actions']):raise RuntimeError('Target execution changed')
    packed=io.BytesIO();np.savez_compressed(packed,**arrays)
    temp=Path(path).with_name(Path(path).name+f'.{os.getpid()}.tmp')
    with temp.open('wb') as stream:stream.write(packed.getvalue());stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    return dict(path=str(Path(path).resolve()),file=str(Path(path).resolve()),bytes=Path(path).stat().st_size,sha256=digest(path))


def _run_attempt(root_text,config,slot):
    """Called only by the bounded owner; no retries or hidden reference runs."""
    root=Path(root_text);out=root/'collection'/'attempts';out.mkdir(parents=True,exist_ok=True)
    aid=slot['attempt_id'];began=time.perf_counter();env=None
    cost=dict(actual_controls=0,actual_substeps=0,simulated_seconds=0.,initialization_integrator_calls=0,
        initialization_integrator_seconds=0.,regular_integrator_calls=0,regular_integrator_seconds=0.)
    states=[];images=[];actions=[];passed=[];valid=[];telemetry=[];segments=[]
    initial=None;error=None;status='completed';terminal=None;engineering=False
    try:
        with counted_original_steps(cost) as counter:
            env=ContactTaskEnv(config.get('physics_source_root',config.get('source_root')),slot['letter'],max_substeps=slot['max_controls']*10)
            image=env.reset_scene(**slot['initialization']);initial=env.initial_identity()
            states.append(env.state().copy());images.append(image.copy())
            if not np.allclose(states[0],slot['initial_state15'],rtol=0,atol=1e-8):
                raise RuntimeError('Frozen initial state differs from original construction')
            counter['value']='regular'
            history=np.repeat(states[0][:2][None,:],3,axis=0)
            queue=history;queue_start=0
            segments.append(dict(start_control=0,end_control=3,committed_before_control=0,
                purpose='real_history_hold',actions=history.tolist(),committed_actions=history.tolist()))
            for control in range(slot['max_controls']):
                if (root/'background'/'STOP.json').exists():status='stopped_by_user';break
                if control==3 or (control>3 and (control-3)%16==0):
                    if env.tracker.end_block(control):status='UNRESOLVED_AT_BLOCK_END';break
                    bi=(control-3)//16
                    try:queue,detail=block_targets(slot,bi,env.state().copy())
                    except ValueError as exc:
                        status='TASK_PLAN_UNAVAILABLE';error=str(exc);break
                    queue_start=control;env.block_end=control+16
                    segments.append(dict(start_control=control,end_control=control+16,
                        committed_before_control=control,actions=queue.tolist(),committed_actions=queue.tolist(),**detail))
                    # Durable pre-execution commitment, no future observations.
                    atomic_json(out/(aid+'.commitments.json'),dict(attempt_id=aid,segments=segments))
                target=queue[control-queue_start].copy();image=None;info={};failure=None
                try:image,info=env.step(target)
                except R2AStepAborted as exc:
                    failure=exc;info=dict(getattr(exc,'info',{}) or {});info.setdefault('substeps',env.last_substeps)
                rows=info.get('substeps',[]);count=sum(bool(s.get('executed')) for s in rows)
                duration=sum(float(s.get('dt',0)) for s in rows if s.get('executed'))
                cost['actual_controls']+=1;cost['actual_substeps']+=count;cost['simulated_seconds']+=duration
                if image is None:image=env.render_clean()
                hard=env.hard_stop
                clean=bool(count==10 and duration>.099999999 and failure is None and
                    env.tracker.first_warning_control is None and hard is None)
                states.append(env.state().copy());images.append(image.copy());actions.append(target)
                passed.append(env.last_passed_action.copy());valid.append(clean);telemetry.append(info)
                if not np.array_equal(target,env.last_passed_action):raise RuntimeError('Original input transformed')
                if failure is not None:
                    if isinstance(failure,ObservationStop):
                        status=hard['status'] if hard else str(info.get('status','SEPARATED_UNRESOLVED'))
                        terminal=_safe(info);break
                    raise failure
            if status=='completed' and env.tracker.end_block(len(actions)):
                status='UNRESOLVED_AT_BLOCK_END'
    except OriginalStateInvalid as exc:
        status='INVALID_INITIAL_SCENE';error=str(exc);terminal=_safe(getattr(exc,'info',{}))
    except BaseException as exc:
        status='engineering_failed';error=repr(exc);engineering=True
        terminal=_safe(getattr(exc,'info',{}))
        if env is not None and cost['actual_controls']>len(actions):
            # A failed renderer must not erase the actual terminal physical call.
            cost['unframed_terminal_control']=dict(control=len(actions),
                requested_action=_safe(locals().get('target')),
                passed_action=_safe(getattr(env,'last_passed_action',None)),
                actual_post_state=_safe(env.state()),telemetry=_safe(locals().get('info',{})),
                RGB_available=False)
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:error=error or repr(exc);engineering=True;status='engineering_failed'
    events=deepcopy(env.tracker.events) if env is not None else []
    first_warning=env.tracker.first_warning_control if env is not None else None
    first_hard=(env.hard_stop or {}).get('control') if env is not None else None
    affected=min([x for x in (first_warning,first_hard) if x is not None],default=len(valid))
    for k in range(affected,len(valid)):valid[k]=False
    clean_controls=next((i for i,v in enumerate(valid) if not v),len(valid))
    if cost['actual_substeps']!=cost['regular_integrator_calls']:
        engineering=True;status='engineering_failed';error=error or 'Native calls differ from saved telemetry'
    raw=None
    cost.update(wall_seconds=time.perf_counter()-began,finished_at_utc=utc(),status=status)
    # Small durable physical-cost receipt precedes large/raw writes. If those
    # writes fail, the owner reports this accurate partial, never retries it.
    atomic_json(out/(aid+'.cost.json'),dict(attempt_id=aid,run_id=config['run_id'],cost=cost,
        pending_raw=True,engineering_failed=engineering,error=error,terminal=terminal,
        first_warning_control=first_warning,events=events))
    _storage(root,config,16*1024**2)
    if states:raw=_save_raw(out/(aid+'.npz'),states,images,actions,passed,valid,telemetry,segments)
    cost.update(wall_seconds=time.perf_counter()-began,finished_at_utc=utc(),status=status)
    meta=dict(run_id=config['run_id'],attempt_id=aid,family_id=slot['family_id'],letter=slot['letter'],
        task=slot['task'],scenario=slot['task'],split=slot['split'],seed=slot['seed'],attempt=slot,
        status=status,engineering_failed=engineering,error=error,raw=raw,cost=cost,initial_identity=initial,
        clean_control_count=clean_controls,first_warning_control=first_warning,first_affected_control=affected,
        events=events,hard_stop=env.hard_stop if env is not None else None,terminal=terminal,
        segments=segments,blocks=segments,state_columns=list(STATE_COLUMNS),
        request_equals_passed=bool(np.array_equal(np.asarray(actions),np.asarray(passed))),
        physics_protocol_sha256=PROTOCOL_SHA256,physics_transition_modified=False,
        candidate_training_prefix_only=True,old_data_reused=False,filled_frames=False,
        model_calls=0,optimizer_updates=0)
    atomic_json(out/(aid+'.json'),meta)
    atomic_json(out/(aid+'.cost.json'),dict(attempt_id=aid,run_id=config['run_id'],cost=cost,
        pending_raw=False,metadata=str((out/(aid+'.json')).resolve()),raw=raw))
    return meta


def collect(root,config):
    root=Path(root);output=root/'collection';output.mkdir(parents=True,exist_ok=True)
    if (output/'execution-claim.json').exists():raise FileExistsError('Already claimed; no automatic retry')
    plan_path=Path(config.get('collection_plan',root/'collection-plan.json'))
    plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
    slots=plan['attempts'];limits=config.get('limits',{})
    max_attempts=int(limits.get('attempts',2000));max_controls=int(limits.get('controls',102000))
    max_substeps=int(limits.get('regular_substeps',1020000))
    if len(slots)>max_attempts or any(x['max_controls']>51 for x in slots):raise ValueError('Physical scope exceeded')
    if sum(x['max_controls'] for x in slots)>max_controls or sum(x['max_controls']*10 for x in slots)>max_substeps:
        raise ValueError('Unified physical allowance exceeded')
    _storage(root,config,32*1024**2)
    atomic_json(output/'execution-claim.json',dict(run_id=config['run_id'],started_at_utc=utc(),
        maximum_attempts=len(slots),maximum_controls=sum(x['max_controls'] for x in slots),
        model_calls=0,optimizer_updates=0),exclusive=True)
    ledger=dict(run_id=config['run_id'],attempts=[],actual_controls=0,actual_substeps=0,
        simulated_seconds=0.,initialization_integrator_calls=0,initialization_integrator_seconds=0.,
        regular_integrator_calls=0,regular_integrator_seconds=0.,actual_valid_controls=0,
        model_calls=0,optimizer_updates=0)
    began=time.perf_counter();failure=None;claimed=0;completed={};pending={}
    workers=max(1,min(2,int(config.get('collection_workers',2))))
    def record(index,meta):
        completed[index]=meta
        ledger['attempts'].append(dict(attempt_id=meta['attempt_id'],family_id=meta['family_id'],
            letter=meta['letter'],task=meta['task'],split=meta['split'],
            clean_control_count=meta['clean_control_count'],**meta['cost']))
        for key in ('actual_controls','actual_substeps','simulated_seconds','initialization_integrator_calls',
            'initialization_integrator_seconds','regular_integrator_calls','regular_integrator_seconds'):
            ledger[key]+=meta['cost'][key]
        ledger['actual_valid_controls']+=meta['clean_control_count']
        ledger['attempts'].sort(key=lambda x:x['attempt_id'])
        atomic_json(output/'ledger.json',ledger)
        atomic_json(output/'progress.json',dict(run_id=config['run_id'],finished_attempts=len(completed),
            claimed_attempts=claimed,last_attempt=meta['attempt_id'],last_status=meta['status'],
            actual_controls=ledger['actual_controls'],actual_substeps=ledger['actual_substeps'],updated_at_utc=utc()))
        print(json.dumps(dict(stage='collection',attempt=meta['attempt_id'],status=meta['status'],
            clean=meta['clean_control_count'],controls=meta['cost']['actual_controls'])),flush=True)
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            while claimed<len(slots) or pending:
                while claimed<len(slots) and len(pending)<workers and failure is None and not (root/'background'/'STOP.json').exists():
                    slot=slots[claimed];_storage(root,config,32*1024**2)
                    atomic_json(output/'claims'/(slot['attempt_id']+'.json'),dict(run_id=config['run_id'],
                        slot_index=claimed,slot=slot,claimed_at_utc=utc()),exclusive=True)
                    future=pool.submit(_run_attempt,str(root),config,slot);pending[future]=claimed;claimed+=1
                if not pending:break
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    index=pending.pop(future)
                    try:meta=future.result();record(index,meta)
                    except BaseException as exc:failure=failure or repr(exc);continue
                    if meta['engineering_failed']:failure=failure or meta['error'] or 'Engineering failure'
                if failure is not None and not pending:break
    except BaseException as exc:failure=failure or repr(exc)
    completion=dict(ledger,actual_attempts=claimed,finished_attempts=len(completed),
        requested_attempts=len(slots),completed=failure is None and len(completed)==len(slots),
        failed=failure is not None,failure=failure,graceful_stop=(root/'background'/'STOP.json').exists(),
        wall_seconds=time.perf_counter()-began,finished_at_utc=utc(),raw_preservation_complete=len(completed)==claimed,
        pending_attempts=[slots[i]['attempt_id'] for i in range(claimed) if i not in completed],
        no_retry=True,training_automatically_next=True)
    atomic_json(output/'completion.json',completion)
    return completion
