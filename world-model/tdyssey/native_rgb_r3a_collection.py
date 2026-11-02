"""Independent selected-physics scenes and compact actual-state collection.

Geometry/state is privileged collection/audit metadata. No model, future-state
search, altered physics loop, stored RGB library, or trajectory replay exists.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from shapely import affinity
from .envs.official import official_shape
from .native_rgb_data import contour_segments
from .native_rgb_env import shape_geometry,STATE_COLUMNS
from .native_rgb_r2b_sampling import (geometry_at,wall_gaps,contact_frame,free_route,
    _stage,_free,_bounded,_line,_contact_command_ok,_departure_route,_candidate)
from .native_rgb_selected_physics import SELECTED_PROTOCOL_SHA256

LETTERS=tuple("TLOCXB")
LOCATIONS=("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
SCENARIOS=("translation_nearCOM","rotation_positive","rotation_negative","release_recontact","grazing","wall_probe","static")
SCENARIO_COUNTS={"train":(40,24,24,24,8,6,2),"seen_val":(6,4,4,6,2,1,1)}
PHASES=("approach","push","withdraw","recontact","static")
MODES=("hold","free_route","contact_push_intent")
CONTACT_KINDS=("agent_object","object_wall","agent_wall")
CONTACT_COLUMNS=("object_world_x","object_world_y","object_local_x","object_local_y","outward_normal_x","outward_normal_y",
    "distance","impulse_x","impulse_y","total_ke","signed_normal_lever")
RULES=dict(kind="R3A-current-state48-independent-scenes.v1",protocol_sha256=SELECTED_PROTOCOL_SHA256,
    controls=48,control_dt=.1,substeps_per_control=10,physics_dt=.01,
    main_phases=[[0,3,"approach"],[3,22,"push"],[22,26,"withdraw"],[26,30,"recontact"],[30,48,"push"]],
    release_phases=[[0,3,"approach"],[3,18,"push"],[18,26,"withdraw"],[26,31,"recontact"],[31,48,"push"]],
    target_distance_cap=64.,target_change_cap=96.,penetration_limit=2.,
    stage_distances=[33.,18.],advance_push=48.,advance_approach=24.,grazing_tangent_ratio=.8,
    first_push_and_second_push_both_active=True,lost_contact_before_recovery=2,
    recovery_controls_before_geometry_arc_refresh=4,initial_yaw_bins=12,
    initial_near_duplicate=dict(area_centroid_world_px=8.,symmetry_angle_deg=5.,local_pusher_world_px=16.),
    prefreeze_static_candidate_limit=64,physical_replacements_per_slot=2,
    static_rejections="Geometry feasibility/cross-split-near only, before any physics and before freeze; never motion quality",
    motion_threshold="H1 area-centroid translation>=2 image96px OR abs unwrapped rotation>=5deg; unchanged",
    RGB_policy="Original reset/step internally render; returned RGB hashed then discarded, never persisted")


def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def identity(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rot(angle):
    c,s=math.cos(float(angle)),math.sin(float(angle));return np.asarray(((c,-s),(s,c)),np.float64)
@lru_cache(maxsize=6)
def shape_spec(letter):return official_shape(letter)
@lru_cache(maxsize=6)
def segments(letter):return tuple(r for r in contour_segments(letter,8) if r['ring']==0)
def area_center(letter,state):return np.asarray(state[2:4])+rot(state[4])@np.asarray(shape_spec(letter).centroid)


def near_initial(letter,first,second):
    a,b=np.asarray(first),np.asarray(second);period=2*math.pi/shape_spec(letter).symmetry_order
    angle=abs((a[4]-b[4]+period/2)%period-period/2)*180/math.pi
    return bool(np.linalg.norm(area_center(letter,a)-area_center(letter,b))<=8 and angle<=5
        and np.linalg.norm(rot(a[4]).T@(a[:2]-a[2:4])-rot(b[4]).T@(b[:2]-b[2:4]))<=16)


def phase_for(control,scenario):
    if type(control)is not int or not 0<=control<48:raise ValueError('Exactly48 controls per scene')
    if scenario=='static':return 'static'
    table=RULES['release_phases'] if scenario=='release_recontact' else RULES['main_phases']
    return next(name for lo,hi,name in table if lo<=control<hi)


def choose_contact(letter,state,scenario,arc_start,*,along_wall=False):
    """Finite current-pose geometry only; first feasible cyclic arc, no rollout."""
    geometry=geometry_at(letter,state);free=_free(geometry);arcs=segments(letter)
    role='translation_1' if scenario=='translation_nearCOM' else scenario if scenario.startswith('rotation_') else 'generic'
    gaps=wall_gaps(geometry);wall=min(gaps,key=gaps.get)
    inward={'left':np.array((1.,0.)),'right':np.array((-1.,0.)),'top':np.array((0.,1.)),'bottom':np.array((0.,-1.))}[wall]
    for offset in range(len(arcs)):
        segment=arcs[(int(arc_start)+offset)%len(arcs)]
        candidate,_=_candidate(letter,state,geometry,segment,role,'center',free)
        if candidate is None:continue
        direction=np.asarray(candidate['drive_direction_world']);room=candidate['translation_room_world_px']
        if scenario=='wall_probe':
            if along_wall and (abs(float(direction@inward))>.35 or room<16):continue
            if not along_wall and float(direction@inward)>-.5:continue
        elif room<16:continue
        if scenario=='grazing':candidate['drive_tangent_ratio']=RULES['grazing_tangent_ratio']*(1 if arc_start%2==0 else -1)
        if scenario=='release_recontact':candidate['drive_tangent_ratio']=.35*(1 if arc_start%2==0 else -1)
        keys=('arc_id','point_local','outward_normal_local','staging_world','staging_distance_world_px',
            'signed_lever_world_px','resultant_signed_lever_world_px','physical_local_CoG','drive_tangent_ratio')
        return {key:deepcopy(candidate[key]) for key in keys}
    return None


def scene_candidate(letter,split,index,attempt,static_try,seed_base,scenario):
    serial=LETTERS.index(letter)*1000+(0 if split=='train' else 500)+index*3+attempt
    seed=int(seed_base)%1000000000+1000000000+serial*64+static_try
    rng=np.random.default_rng(seed);yaw_bin=index%12
    region_index=(index//12+yaw_bin*4+(0 if split=='train' else 3))%9
    location=LOCATIONS[region_index];angle=(yaw_bin+float(rng.uniform(.08,.92)))*math.pi/6
    local=shape_geometry(letter);geometry=affinity.rotate(local,angle,origin=(0,0),use_radians=True)
    x0,y0,x1,y1=geometry.bounds
    center=rng.uniform(194.,317.,size=2);origin=center-np.array(((x0+x1)/2,(y0+y1)/2))
    gap=rng.uniform(2.,12.,size=2) if scenario=='wall_probe' else rng.uniform(36.,64.,size=2)
    if 'left' in location:origin[0]=7+gap[0]-x0
    if 'right' in location:origin[0]=504-gap[0]-x1
    if 'top' in location:origin[1]=7+gap[1]-y0
    if 'bottom' in location:origin[1]=504-gap[1]-y1
    state=np.zeros(15,np.float64);state[2:4]=origin;state[4]=angle
    arc_start=int(rng.integers(len(segments(letter))))
    contact=choose_contact(letter,state,scenario,arc_start)
    if contact is None:return None,'no_role_feasible_exterior_contact'
    state[:2]=contact['staging_world'];cog=np.array((0.,45.)) if letter=='T' else np.zeros(2)
    return dict(seed=seed,attempt_index=attempt,static_try=static_try,initial_yaw_bin=yaw_bin,initial_space=location,
        scenario=scenario,arc_start=arc_start,selected_contact=contact,initial_state15=state.tolist(),
        initialization=dict(center=(origin+rot(angle)@cog).tolist(),angle=float(angle),pusher=state[:2].tolist())),None


def build_plan(config,*,progress_callback=None):
    """Split and all primary/replacement scenes freeze before any initialization."""
    if config.get('train_per_letter',128)!=128 or config.get('val_per_letter',24)!=24:raise ValueError('Exactly128TRAIN+24newval per letter')
    slots=[];rejections=[];accepted={l:[] for l in LETTERS};seeds=set();exact=set()
    for split in ('train','seen_val'):
        count=128 if split=='train' else 24
        for letter in LETTERS:
            scenario_list=[name for name,n in zip(SCENARIOS,SCENARIO_COUNTS[split]) for _ in range(n)]
            # Fixed interleaving avoids assigning each scenario one contiguous
            # yaw/position block. No actual trajectory quality is consulted.
            rng=np.random.default_rng(int(config.get('seed',20261030))+LETTERS.index(letter)*37+(0 if split=='train' else 1001))
            order=rng.permutation(count);scenario_order=[scenario_list[int(i)] for i in order]
            trans=scenario_order.index('translation_nearCOM')
            scenario_order[0],scenario_order[trans]=scenario_order[trans],scenario_order[0]
            for index in range(count):
                sid=f'R3A-{split}-{letter}-{index:03d}';family='F-'+sid;attempts=[]
                for attempt in range(3):
                    for static_try in range(RULES['prefreeze_static_candidate_limit']):
                        candidate,reason=scene_candidate(letter,split,index,attempt,static_try,config.get('seed',20261030),scenario_order[index])
                        if candidate is not None:
                            state=np.asarray(candidate['initial_state15'],np.float64);signature=hashlib.sha256(state.tobytes()).hexdigest()
                            if candidate['seed'] in seeds:raise RuntimeError('Independent seed collision')
                            if signature in exact:reason='exact_initial_state_duplicate'
                            elif split=='seen_val' and any(old_split=='train' and near_initial(letter,state,s) for old_split,s in accepted[letter]):reason='cross_split_initial_near_duplicate'
                            else:
                                seeds.add(candidate['seed']);exact.add(signature);accepted[letter].append((split,state.copy()));attempts.append(candidate);break
                        rejections.append(dict(slot_id=sid,attempt=attempt,static_try=static_try,reason=reason))
                    else:raise RuntimeError('Static independent-scene candidate catalogue exhausted: '+sid)
                slots.append(dict(slot_id=sid,family_id=family,letter=letter,split=split,index=index,scenario=scenario_order[index],attempts=attempts))
            if progress_callback is not None:progress_callback(dict(split=split,letter=letter,frozen_slots=len(slots),static_rejections=len(rejections)))
    gate=[next(row for row in slots if row['letter']==letter and row['split']=='train' and row['index']==0) for letter in LETTERS]
    slots=gate+[row for row in slots if row not in gate]
    return dict(kind='R3A-prephysics-independent-scene-plan.v1',protocol_sha256=SELECTED_PROTOCOL_SHA256,
        seed=config.get('seed',20261030),rules=RULES,rules_sha256=identity(RULES),slots=slots,
        static_rejections=rejections,successful_scene_target=912,planned_controls_no_failure=43776,
        max_physical_attempts=2736,max_controls_including_all_replacements=131328,
        split_frozen_before_any_physics=True,source_trajectory_reuse=False,derived_rotated_mirrored_family=False,
        cross_split_declared_near_candidates=0,initial_pose_candidate_count=2736)


class ContactPolicy:
    """48-step current-state-only feedback; second half contains real pushing."""
    def __init__(self,env,slot,candidate):
        self.env,self.slot,self.candidate=env,slot,candidate;self.contact=deepcopy(candidate['selected_contact'])
        self.last_action=None;self.release_target=None;self.next_index=0;self.pending=None
        self.lost_contact=0;self.recovery=0;self.phase=None

    def route(self,geometry,start,target):
        path=free_route(geometry,start,target)
        if path is None and _departure_route(geometry,start,target):path=[np.asarray(start).tolist(),np.asarray(target).tolist()]
        return path

    def next_action(self,index):
        if index!=self.next_index or self.pending is not None:raise ValueError('Exactly one action/observe pair per actual control')
        state=np.asarray(self.env.state(),np.float64)
        if state.shape!=(15,) or not np.isfinite(state).all():raise ValueError('Invalid actual current state')
        scenario=self.slot['scenario'];phase=phase_for(index,scenario);geometry=geometry_at(self.slot['letter'],state);start=state[:2]
        if phase!=self.phase:
            self.recovery=0
            if phase!='withdraw':self.release_target=None
            self.phase=phase
        point,normal,tangent=contact_frame(state,dict(point_local=self.contact['point_local'],outward_normal_local=self.contact['outward_normal_local']))
        staging=_stage(geometry,point,normal)
        if phase in ('push','recontact') and (staging is None or self.recovery>=4 or scenario=='wall_probe' and index==30):
            fresh=choose_contact(self.slot['letter'],state,scenario,self.candidate['arc_start']+index//4,along_wall=scenario=='wall_probe' and index>=26)
            if fresh is not None:
                self.contact=fresh;self.recovery=0
                point,normal,tangent=contact_frame(state,dict(point_local=fresh['point_local'],outward_normal_local=fresh['outward_normal_local']))
                staging=_stage(geometry,point,normal)
        raw=start.copy();mode='hold';verified=True
        if phase=='static':pass
        elif phase=='withdraw':
            if self.release_target is None:
                for distance in (64.,48.,33.,18.):
                    desired=point+normal*distance
                    if self.route(geometry,start,desired) is not None:self.release_target=desired;break
            if self.release_target is not None:
                path=self.route(geometry,start,self.release_target)
                if path is not None:raw=np.asarray(path[1]);mode='free_route'
        elif staging is not None:
            stage,center,_=staging;recover=self.lost_contact>=2
            if (phase in ('approach','recontact') or recover) and np.linalg.norm(start-center)>8 and np.linalg.norm(start-stage)>4:
                path=self.route(geometry,start,stage)
                if path is not None:raw=np.asarray(path[1]);mode='free_route'
                if recover:self.recovery+=1
            else:
                advance=48. if phase=='push' else 24.
                raw=center-normal*advance+tangent*advance*self.contact['drive_tangent_ratio'];mode='contact_push_intent'
        action=_bounded(raw,start,self.last_action)
        if mode=='free_route':verified=_free(geometry).covers(_line(start,action)) or _departure_route(geometry,start,action)
        elif mode=='contact_push_intent':verified=staging is not None and _contact_command_ok(geometry,start,action,staging[1])
        if not verified:action=_bounded(start,start,self.last_action);mode='hold'
        decision=dict(phase=PHASES.index(phase),mode=MODES.index(mode),arc_id=self.contact['arc_id'],
            point_local=self.contact['point_local'],normal_local=self.contact['outward_normal_local'],normal_world=normal.tolist(),
            beta=float(self.contact['drive_tangent_ratio']),signed_lever=float(self.contact['signed_lever_world_px']),
            raw_target=np.asarray(raw).tolist(),geometry_verified=bool(verified),state_before_sha256=hashlib.sha256(state.tobytes()).hexdigest())
        self.pending=(state.copy(),np.asarray(action).copy(),decision);self.last_action=np.asarray(action).copy()
        return np.asarray(action,np.float64),decision

    def observe(self,before,action,after,info):
        if self.pending is None or not np.array_equal(before,self.pending[0]) or not np.array_equal(action,self.pending[1]):raise ValueError('Observed transition differs from issued action')
        count=info.get('agent_object_contact_substeps',info.get('contact',0))
        self.lost_contact=0 if count>0 else self.lost_contact+1
        self.pending=None;self.next_index+=1


class CompactAttempt:
    """Bounded numeric evidence; no repeated telemetry JSON and no RGB payload."""
    def __init__(self,letter,initial_state,rgb):
        self.letter=letter;self.states=[np.asarray(initial_state,np.float64).copy()];self.actions=[];self.attempted=[];self.valid=[]
        self.substates=[self.states[0].copy()];self.available=[True];self.subcontrol=[];self.sublocal=[];self.substatus=[]
        self.forces=[];self.velocities=[];self.penetrations=[];self.flags=[];self.offsets=[0]
        self.contacts=[];self.tags=[];self.arc_offsets=[0];self.arc_ids=[];self.decisions=[];self.walltimes=[]
        self.rgb_hashes=[];self.rgb_digest=hashlib.sha256();self.internal_rgb_returns=0;self.add_rgb(rgb)
    def add_rgb(self,rgb):
        rgb=np.asarray(rgb)
        if rgb.shape!=(3,96,96) or rgb.dtype!=np.uint8:raise ValueError('Original renderer must return uint8 CHW96')
        raw=rgb.tobytes();self.rgb_hashes.append(np.frombuffer(hashlib.sha256(raw).digest(),np.uint8).copy());self.rgb_digest.update(raw);self.internal_rgb_returns+=1
    def append(self,action,decision,substeps,*,completed,after=None,rgb=None,wall_seconds=0.):
        control=len(self.attempted);self.attempted.append(np.asarray(action,np.float64));self.decisions.append(deepcopy(decision));self.walltimes.append(float(wall_seconds))
        cog=np.array((0.,45.)) if self.letter=='T' else np.zeros(2)
        for sub in substeps:
            global_index=len(self.subcontrol);post=sub.get('post_state');known=post is not None
            self.substates.append(np.asarray(post,np.float64) if known else np.full(15,np.nan));self.available.append(known)
            self.subcontrol.append(control);self.sublocal.append(int(sub.get('index',0)))
            self.substatus.append(2 if sub.get('engine_exception') else 1 if sub.get('executed') else 0)
            force=sub.get('exact_formula_reconstruction',{})
            self.forces.append([*force.get('unclipped_force',[np.nan]*2),*force.get('clipped_force',[np.nan]*2),*sub.get('clipped_body_force',[np.nan]*2)])
            self.velocities.append(sub.get('velocity_after_cap',[np.nan]*2));self.penetrations.append(sub.get('penetrations') or [np.nan]*3)
            flags=[0,0,0]
            for event in sub.get('contacts',[]):
                kind=CONTACT_KINDS.index(event['kind']);phase=0 if event['phase']=='pre_solve' else 1;flags[kind]|=1<<phase
                normal=np.asarray(event['normal'],np.float64);object_first=event['shapes_order'][0]=='object'
                if 'object' in event['shapes_order'] and not object_first:normal=-normal
                for point in event.get('points',[]):
                    values=np.zeros(11,np.float32);mask=0
                    if point.get('object_world') is not None:values[:2]=point['object_world'];mask|=1
                    if point.get('object_local') is not None:values[2:4]=point['object_local'];mask|=2
                    values[4:6]=normal;values[6]=point['distance']
                    if event.get('total_impulse') is not None:values[7:9]=event['total_impulse'];mask|=4
                    if event.get('total_ke') is not None:values[9]=event['total_ke'];mask|=8
                    if kind==0 and phase==1 and known and point.get('object_world') is not None:
                        st=np.asarray(post);com=st[2:4]+rot(st[4])@cog;r=np.asarray(point['object_world'])-com
                        values[10]=r[0]*(-normal[1])-r[1]*(-normal[0]);mask|=16
                    self.contacts.append(values);self.tags.append((global_index,kind,phase,int(object_first),mask))
                    self.arc_ids.extend(map(int,point.get('arc_ids',[])));self.arc_offsets.append(len(self.arc_ids))
            self.flags.append(flags)
        self.offsets.append(len(self.subcontrol))
        if completed:
            self.states.append(np.asarray(after,np.float64).copy());self.actions.append(np.asarray(action,np.float64).copy());self.valid.append(True);self.add_rgb(rgb)
    def arrays(self):
        n=len(self.actions);c=len(self.attempted);m=len(self.subcontrol);k=len(self.contacts)
        decision=self.decisions
        return dict(states=np.asarray(self.states,np.float64).reshape(n+1,15),actions=np.asarray(self.actions,np.float64).reshape(n,2),
            attempted_actions=np.asarray(self.attempted,np.float64).reshape(c,2),control_valid=np.asarray(self.valid,bool),control_time_seconds=np.arange(n+1,dtype=np.float64)*.1,
            rgb_frame_sha256=np.asarray(self.rgb_hashes,np.uint8).reshape(n+1,32),
            substep_states=np.asarray(self.substates,np.float64).reshape(m+1,15),substep_state_available=np.asarray(self.available,bool),
            substep_control_index=np.asarray(self.subcontrol,np.uint8),substep_local_index=np.asarray(self.sublocal,np.uint8),substep_status=np.asarray(self.substatus,np.int8),
            substep_forces=np.asarray(self.forces,np.float32).reshape(m,6),substep_velocity_after_cap=np.asarray(self.velocities,np.float32).reshape(m,2),
            substep_penetrations=np.asarray(self.penetrations,np.float32).reshape(m,3),contact_flags=np.asarray(self.flags,np.uint8).reshape(m,3),
            control_substep_offsets=np.asarray(self.offsets,np.uint16),contact_values=np.asarray(self.contacts,np.float32).reshape(k,11),
            contact_tags=np.asarray(self.tags,np.int16).reshape(k,5),contact_arc_offsets=np.asarray(self.arc_offsets,np.int32),contact_arc_ids=np.asarray(self.arc_ids,np.int16),
            control_phase=np.asarray([d['phase'] for d in decision],np.uint8),control_mode=np.asarray([d['mode'] for d in decision],np.uint8),
            control_arc_id=np.asarray([d['arc_id'] for d in decision],np.int16),control_point_local=np.asarray([d['point_local'] for d in decision],np.float32).reshape(c,2),
            control_normal_local=np.asarray([d['normal_local'] for d in decision],np.float32).reshape(c,2),control_normal_world=np.asarray([d['normal_world'] for d in decision],np.float32).reshape(c,2),
            control_beta=np.asarray([d['beta'] for d in decision],np.float32),control_signed_lever=np.asarray([d['signed_lever'] for d in decision],np.float32),
            control_raw_target=np.asarray([d['raw_target'] for d in decision],np.float32).reshape(c,2),control_geometry_verified=np.asarray([d['geometry_verified'] for d in decision],bool),
            control_wall_seconds=np.asarray(self.walltimes,np.float64))


def save_npz(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**arrays);stream.flush();__import__('os').fsync(stream.fileno())
    temp.replace(path)


def verify_saved_attempt(path,*,complete=False):
    with np.load(path,allow_pickle=False) as z:
        n=len(z['actions']);c=len(z['attempted_actions']);m=len(z['substep_control_index'])
        if z['states'].dtype!=np.float64 or z['states'].shape!=(n+1,15) or z['actions'].shape!=(n,2):raise ValueError('Saved complete state/action timeline differs')
        if any(name in z.files for name in ('rgb','images','frames')):raise ValueError('Large RGB payload forbidden')
        if z['rgb_frame_sha256'].shape!=(n+1,32) or len(z['control_valid'])!=n or len(z['control_substep_offsets'])!=c+1:raise ValueError('Saved compact evidence lengths differ')
        if complete and (n!=48 or c!=48 or m!=480 or not z['control_valid'].all() or not np.isfinite(z['states']).all() or not np.isfinite(z['substep_states']).all()):raise ValueError('Complete48-control scene evidence incomplete')
        if complete and not np.array_equal(z['states'],z['substep_states'][::10]):raise ValueError('Control state15 differs from actual100Hz chain')
        return dict(completed_controls=n,attempted_controls=c,substeps=m,state_sha256=hashlib.sha256(z['states'].tobytes()).hexdigest(),
            action_sha256=hashlib.sha256(z['actions'].tobytes()).hexdigest(),rgb_frame_hash_sequence_sha256=hashlib.sha256(z['rgb_frame_sha256'].tobytes()).hexdigest())
