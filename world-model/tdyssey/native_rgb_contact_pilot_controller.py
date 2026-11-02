"""CONTACT-PILOT1: recorded-contact feedback, with unchanged native physics.

Only current geometry/state and completed past control observations are read.
This module never creates or steps an environment, loads a model or renders.
Historical R3A policy/source and its planned task schedule remain unchanged.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
import numpy as np

from .native_rgb_r3a_collection import (PHASES,MODES,phase_for,segments,rot,area_center,identity)
from .native_rgb_r2b_sampling import (geometry_at,wall_gaps,contact_frame,_stage,_free,
    free_route,_bounded,_line,_contact_command_ok,_departure_route,_candidate,_room)
from .native_rgb_selected_physics import SELECTED_PROTOCOL_SHA256

RULES=dict(kind='CONTACT-PILOT1-current-and-past-contact-feedback.v1',
    selected_protocol_sha256=SELECTED_PROTOCOL_SHA256,controls=48,control_dt=.1,physics_dt=.01,
    task_schedule='exact historical phase_for(control,scenario), including unchanged fixed withdraw windows and recontact start26',
    impulse_epsilon=1e-8,establish_contact_substeps=5,establish_requires_terminal_AO=True,
    lost_contact_controls=2,loss_definition='two controls without stable5-of10 terminal contact and nonzero valid impulse',
    establish_selected_point_tolerance_world_px=24.,establish_selected_normal_tolerance_degrees=45.,
    minimum_selected_target_dwell_controls=4,
    no_route_progress_world_px=.5,no_progress_controls_before_switch=3,
    unconfirmed_route_controls_before_switch=2,
    route_eta_assumed_world_px_per_s=100.,route_eta_extra_controls=2,
    route_eta_scope='distance/speed estimate only; productive route never switched solely for elapsed time',
    wall_nearstatic_controls_before_escape=3,
    nearstatic_translation_image96_px=.1,nearstatic_abs_rotation_deg=.1,
    observed_point_max_change_world_px_per_control=12.,observed_normal_max_change_degrees=45.,
    observed_normal_large_change_confirmations=2,
    feedforward_current_contact_velocity_seconds=.1,feedforward_cap_world_px=16.,
    normal_advance_push_world_px=48.,normal_advance_entry_world_px=24.,
    withdraw_distances_world_px=[64.,48.,33.,18.],
    candidate_minimum_drive_room_world_px=16.,wall_escape_minimum_inward_dot=-.05,
    wall_escape_preferred_abs_inward_dot=.65,
    route_scope='current-pose free visibility route plus confirmed monotone-clearance departure bridge; no dynamics lookahead',
    candidate_scope='finite exterior arcs only; task role first, generic wall escape fallback disclosed',
    target_selection='shortest confirmed current-pose route, then tangential escape preference, arc id; no measured future quality',
    no_reset_no_early_stop_no_replacement=True,models=0,lookahead_physics_steps=0)
RULES_SHA256=identity(RULES)


def trace_observation(info,previous_contact=None):
    """Per-substep event presence; never sums duplicated arbiter impulses."""
    rows=info.get('substeps')
    if not isinstance(rows,list) or len(rows)!=10 or not all(r.get('executed') and not r.get('engine_exception') for r in rows):
        raise ValueError('Controller requires ten known actual substeps, not aggregate intent/contact counts')
    counts=dict(AO=0,OW=0,AW=0,valid_impulse=0,nonzero_impulse=0)
    representative=None;terminal=False
    for si,row in enumerate(rows):
        events=[e for e in row.get('contacts',[])if e.get('phase')=='post_solve']
        ao=[e for e in events if e.get('kind')=='agent_object']
        counts['AO']+=bool(ao);counts['OW']+=any(e.get('kind')=='object_wall'for e in events)
        counts['AW']+=any(e.get('kind')=='agent_wall'for e in events)
        if si==9:terminal=bool(ao)
        valid=[]
        for event in ao:
            vector=np.asarray(event.get('total_impulse',[]),float)
            if vector.shape==(2,)and np.isfinite(vector).all():valid.append((float(np.linalg.norm(vector)),event))
        counts['valid_impulse']+=bool(valid)
        positive=[v for v in valid if v[0]>RULES['impulse_epsilon']]
        counts['nonzero_impulse']+=bool(positive)
        if positive:
            _,event=max(positive,key=lambda pair:pair[0])
            points=[p for p in event.get('points',[])if p.get('object_local')is not None]
            if points:
                old=np.asarray(previous_contact['point_local'])if previous_contact else np.zeros(2)
                point=min(points,key=lambda p:(float(np.linalg.norm(np.asarray(p['object_local'])-old)),tuple(p['object_local'])))
                normal=np.asarray(event['normal'],float)
                if event.get('shapes_order',[None])[0]!='object':normal=-normal
                representative=dict(point_local=list(map(float,point['object_local'])),normal_world=normal.tolist(),
                    arc_ids=list(map(int,point.get('arc_ids',event.get('arc_ids',[])))))
    return dict(**counts,terminal_AO=terminal,representative=representative,
        established=counts['AO']>=RULES['establish_contact_substeps']and terminal and counts['nonzero_impulse']>0)


class ContactPilotPolicy:
    """One command/observation per real control; no authority to step/reset."""
    def __init__(self,env,slot,candidate,rules=None):
        if rules is not None and rules!=RULES:raise ValueError('Frozen pilot controller rules differ')
        if slot['letter']not in 'TLOCXB':raise ValueError('Pilot is six TRAIN letters only')
        self.env,self.slot,self.candidate=env,deepcopy(slot),deepcopy(candidate)
        self.contact=deepcopy(candidate['selected_contact']);self.last_action=None;self.pending=None;self.next_index=0
        self.history=[];self.established=False;self.awaiting_selected_contact=True;self.lost_contact=0;self.wall_stall=0
        self.no_progress=0;self.unconfirmed_routes=0;self.target_dwell=0;self.recovery_elapsed=0
        self.release_target=None;self.task_phase=None;self.escape_active=False
        self.last_observation=None;self.normal_proposal=None;self.normal_proposal_count=0
        self.target_revision=0;self.last_replan_reason=None;self.eta_controls=None

    def route(self,geometry,start,target):
        """A verified departure bridge avoids declaring contact-shell starts stuck."""
        start,target=np.asarray(start),np.asarray(target)
        path=free_route(geometry,start,target)
        if path is not None:return path
        if _departure_route(geometry,start,target):return [start.tolist(),target.tolist()]
        state=np.asarray(self.env.state(),float);point,normal,_=contact_frame(state,self.contact)
        staging=_stage(geometry,point,normal)
        if staging is not None:
            bridge=staging[0]
            if _departure_route(geometry,start,bridge):
                rest=free_route(geometry,bridge,target)
                if rest is not None:return [start.tolist(),*rest]
        return None

    @staticmethod
    def route_length(path):
        return float(np.linalg.norm(np.diff(np.asarray(path,float),axis=0),axis=1).sum())

    def _choose_replacement(self,state,geometry,*,wall_escape):
        scenario=self.slot['scenario'];role=('translation_1'if scenario=='translation_nearCOM'else
            scenario if scenario.startswith('rotation_')else 'generic')
        wall=min(wall_gaps(geometry),key=wall_gaps(geometry).get)
        inward={'left':np.array((1.,0.)),'right':np.array((-1.,0.)),
                'top':np.array((0.,1.)),'bottom':np.array((0.,-1.))}[wall]
        free=_free(geometry)
        for selected_role in ([role,'generic']if wall_escape and role!='generic'else[role]):
            feasible=[]
            for segment in segments(self.slot['letter']):
                if int(segment['id'])==int(self.contact['arc_id']):continue
                fresh,_=_candidate(self.slot['letter'],state,geometry,segment,selected_role,'center',free)
                if fresh is None:continue
                if scenario in('release_recontact','grazing'):
                    fresh['drive_tangent_ratio']=self.candidate['selected_contact'].get('drive_tangent_ratio',0.)
                direction=-np.asarray(fresh['outward_normal_world'])+fresh['drive_tangent_ratio']*np.array((-fresh['outward_normal_world'][1],fresh['outward_normal_world'][0]))
                direction/=np.linalg.norm(direction);dot=float(direction@inward)
                if _room(geometry,direction)<RULES['candidate_minimum_drive_room_world_px']:continue
                if wall_escape and dot<RULES['wall_escape_minimum_inward_dot']:continue
                path=self.route(geometry,state[:2],fresh['staging_world'])
                if path is None:continue
                priority=(int(wall_escape and abs(dot)>RULES['wall_escape_preferred_abs_inward_dot']),self.route_length(path),int(fresh['arc_id']))
                feasible.append((priority,fresh,selected_role!=role))
            if feasible:
                _,fresh,fallback=min(feasible,key=lambda row:row[0])
                keys=('arc_id','point_local','outward_normal_local','staging_world','staging_distance_world_px',
                      'signed_lever_world_px','resultant_signed_lever_world_px','physical_local_CoG','drive_tangent_ratio')
                result={k:deepcopy(fresh[k])for k in keys};result['wall_escape_generic_role_fallback']=fallback
                return result
        return None

    def _replace_if_needed(self,state,geometry,staging):
        if self.target_dwell<RULES['minimum_selected_target_dwell_controls']:return None
        reason=None
        if self.wall_stall>=RULES['wall_nearstatic_controls_before_escape']:reason='AO_OW_nearstatic3_wall_escape'
        elif self.no_progress>=RULES['no_progress_controls_before_switch']:reason='route_no_progress3'
        elif self.unconfirmed_routes>=RULES['unconfirmed_route_controls_before_switch']:reason='route_unconfirmed2'
        if reason is None:return None
        fresh=self._choose_replacement(state,geometry,wall_escape=reason.startswith('AO_OW'))
        self.last_replan_reason=reason+('_selected'if fresh is not None else '_no_confirmed_alternative')
        # A failed geometric search is also debounced, rather than repeated at100Hz.
        self.target_dwell=0;self.no_progress=0;self.unconfirmed_routes=0
        if fresh is not None:
            self.contact=fresh;self.target_revision+=1;self.established=False;self.awaiting_selected_contact=True;self.lost_contact=2
            self.escape_active=reason.startswith('AO_OW');self.wall_stall=0;self.eta_controls=None
        return self.last_replan_reason

    def _drive_target(self,state,center,normal,tangent,advance):
        local=np.asarray(self.contact['point_local']);cog=np.array((0.,45.))if self.slot['letter']=='T'else np.zeros(2)
        offset=rot(state[4])@(local-cog)
        contact_velocity=state[7:9]+state[9]*np.array((-offset[1],offset[0]))
        lead=contact_velocity*RULES['feedforward_current_contact_velocity_seconds'];length=np.linalg.norm(lead)
        if length>RULES['feedforward_cap_world_px']:lead*=RULES['feedforward_cap_world_px']/length
        beta=float(self.contact['drive_tangent_ratio'])
        if self.slot['scenario']=='translation_nearCOM':
            r=local-cog;n=np.asarray(self.contact['outward_normal_local']);t=np.array((-n[1],n[0]))
            normal_lever=r[0]*(-n[1])-r[1]*(-n[0]);denom=r[0]*t[1]-r[1]*t[0]
            if abs(denom)>1e-9:beta=float(np.clip(-normal_lever/denom,-.5,.5))
        return center-normal*advance+tangent*advance*beta+lead,beta

    def _current_levers(self,beta=None):
        cog=np.array((0.,45.))if self.slot['letter']=='T'else np.zeros(2)
        r=np.asarray(self.contact['point_local'])-cog;n=np.asarray(self.contact['outward_normal_local']);t=np.array((-n[1],n[0]))
        normal=float(r[0]*(-n[1])-r[1]*(-n[0]))
        beta=float(self.contact['drive_tangent_ratio'])if beta is None else float(beta)
        return normal,float(normal+beta*(r[0]*t[1]-r[1]*t[0]))

    def next_action(self,index):
        if index!=self.next_index or self.pending is not None:raise ValueError('Exactly one action/observe pair per actual control')
        state=np.asarray(self.env.state(),np.float64)
        if state.shape!=(15,)or not np.isfinite(state).all():raise ValueError('Invalid current state15')
        task=phase_for(index,self.slot['scenario']);geometry=geometry_at(self.slot['letter'],state);start=state[:2]
        if task!=self.task_phase:
            if task=='withdraw':self.release_target=None
            if self.task_phase=='withdraw':self.established=False;self.awaiting_selected_contact=True;self.lost_contact=2;self.escape_active=False;self.eta_controls=None
            self.task_phase=task
        point,normal,tangent=contact_frame(state,self.contact);staging=_stage(geometry,point,normal)
        reason=None
        if task not in('withdraw','static'):
            reason=self._replace_if_needed(state,geometry,staging)
            point,normal,tangent=contact_frame(state,self.contact);staging=_stage(geometry,point,normal)
        raw=start.copy();mode='hold';actual='unconfirmed_hold';path=None;distance=None;beta=float(self.contact['drive_tangent_ratio'])
        if task=='static':actual='static'
        elif task=='withdraw':
            actual='withdraw'
            if self.release_target is None:
                for gap in RULES['withdraw_distances_world_px']:
                    target=point+normal*gap
                    if self.route(geometry,start,target)is not None:self.release_target=target.copy();break
            if self.release_target is not None:
                path=self.route(geometry,start,self.release_target)
                if path is not None:raw=np.asarray(path[1]);mode='free_route'
        elif staging is not None:
            stage,center,_=staging
            advance=RULES['normal_advance_push_world_px']if self.established else RULES['normal_advance_entry_world_px']
            drive,beta=self._drive_target(state,center,normal,tangent,advance)
            bounded_drive=_bounded(drive,start,self.last_action)
            if _contact_command_ok(geometry,start,bounded_drive,center):
                raw=drive;mode='contact_push_intent';actual='contact_follow'if self.established else 'recovery_entry'if index>=3 else'initial_acquisition'
                distance=float(np.linalg.norm(start-center))
            else:
                path=self.route(geometry,start,stage)
                if path is not None:
                    raw=np.asarray(path[1]);mode='free_route';distance=self.route_length(path)
                    actual='wall_escape_route'if self.escape_active else'recovery_route'if index>=3 else'initial_acquisition'
        action=_bounded(raw,start,self.last_action)
        verified=True
        if mode=='free_route':verified=_free(geometry).covers(_line(start,action))or _departure_route(geometry,start,action)
        elif mode=='contact_push_intent':verified=staging is not None and _contact_command_ok(geometry,start,action,staging[1])
        if not verified:action=_bounded(start,start,self.last_action);mode='hold';actual='unconfirmed_hold'
        if task not in('withdraw','static'):
            self.unconfirmed_routes=self.unconfirmed_routes+1 if mode=='hold'else 0
            if distance is not None and self.eta_controls is None:
                self.eta_controls=int(math.ceil(distance/(RULES['route_eta_assumed_world_px_per_s']*.1)))+RULES['route_eta_extra_controls']
        signed_lever,resultant_lever=self._current_levers(beta)
        decision=dict(phase=PHASES.index(task),planned_task_phase=task,controller_state=actual,
            mode=MODES.index(mode),arc_id=int(self.contact['arc_id']),point_local=deepcopy(self.contact['point_local']),
            normal_local=deepcopy(self.contact['outward_normal_local']),normal_world=normal.tolist(),beta=beta,
            signed_lever=signed_lever,resultant_signed_lever=resultant_lever,raw_target=np.asarray(raw).tolist(),geometry_verified=bool(verified),
            state_before_sha256=hashlib.sha256(state.tobytes()).hexdigest(),controller_rules_sha256=RULES_SHA256,
            actual_contact_established_before_action=bool(self.established),previous_observation=None if self.last_observation is None else{key:value for key,value in self.last_observation.items()if key!='representative'},
            replan_reason=reason,target_revision=self.target_revision,target_dwell_controls=self.target_dwell,
            route_remaining_world_px=distance,route_eta_controls=self.eta_controls,route_elapsed_controls=self.recovery_elapsed,
            no_progress_controls=self.no_progress,unconfirmed_route_controls=self.unconfirmed_routes,
            wall_stall_controls=self.wall_stall,route=None if path is None else path,
            wall_escape_generic_role_fallback=bool(self.contact.get('wall_escape_generic_role_fallback',False)),
            future_state_used=False,lookahead_physics_steps=0,task_budget_extended=False)
        self.pending=(state.copy(),action.copy(),decision);self.last_action=action.copy()
        return action,decision

    def _track_actual_contact(self,after,observation):
        item=observation['representative']
        if item is None:return
        point=np.asarray(item['point_local'],float);old=np.asarray(self.contact['point_local'],float)
        delta=point-old;length=np.linalg.norm(delta)
        if length>RULES['observed_point_max_change_world_px_per_control']:point=old+delta*(RULES['observed_point_max_change_world_px_per_control']/length)
        normal=rot(after[4]).T@np.asarray(item['normal_world']);norm=np.linalg.norm(normal)
        if norm<1e-9:return
        normal/=norm;old_normal=np.asarray(self.contact['outward_normal_local']);angle=math.degrees(math.acos(float(np.clip(normal@old_normal,-1,1))))
        if angle>RULES['observed_normal_max_change_degrees']:
            if self.normal_proposal is not None and float(normal@self.normal_proposal)>.999:self.normal_proposal_count+=1
            else:self.normal_proposal=normal.copy();self.normal_proposal_count=1
            if self.normal_proposal_count<RULES['observed_normal_large_change_confirmations']:return
        self.normal_proposal=None;self.normal_proposal_count=0
        self.contact['point_local']=point.tolist();self.contact['outward_normal_local']=normal.tolist()
        if item['arc_ids']:self.contact['arc_id']=min(item['arc_ids'])
        signed,resultant=self._current_levers()
        self.contact['signed_lever_world_px']=signed;self.contact['resultant_signed_lever_world_px']=resultant

    def observe(self,before,action,after,info):
        if self.pending is None:raise ValueError('observe requires pending actual action')
        before,after=np.asarray(before,np.float64),np.asarray(after,np.float64)
        if not np.array_equal(before,self.pending[0])or not np.array_equal(action,self.pending[1])or after.shape!=(15,)or not np.isfinite(after).all():
            raise ValueError('Observed actual transition differs from command/state')
        observed=trace_observation(info,self.contact);decision=self.pending[2]
        accepted=observed['established'];item=observed['representative']
        if accepted and self.awaiting_selected_contact:
            accepted=False
            if item is not None:
                point_distance=float(np.linalg.norm(np.asarray(item['point_local'])-self.contact['point_local']))
                normal=rot(after[4]).T@np.asarray(item['normal_world']);normal/=max(float(np.linalg.norm(normal)),1e-12)
                normal_dot=float(normal@np.asarray(self.contact['outward_normal_local']))
                accepted=point_distance<=RULES['establish_selected_point_tolerance_world_px']and normal_dot>=math.cos(math.radians(RULES['establish_selected_normal_tolerance_degrees']))
        if accepted:
            self.established=True;self.lost_contact=0;self.recovery_elapsed=0;self.eta_controls=None;self.escape_active=False
            self.awaiting_selected_contact=False
            if decision['planned_task_phase']!='withdraw':self._track_actual_contact(after,observed)
        else:
            self.lost_contact+=1
            if self.lost_contact>=RULES['lost_contact_controls']:self.established=False
        task=decision['planned_task_phase']
        translation=float(np.linalg.norm(area_center(self.slot['letter'],after)-area_center(self.slot['letter'],before))*96/512)
        rotation=float((after[4]-before[4])*180/math.pi)
        near=translation<RULES['nearstatic_translation_image96_px']and abs(rotation)<RULES['nearstatic_abs_rotation_deg']
        self.wall_stall=self.wall_stall+1 if task not in('withdraw','static')and observed['AO']>0 and observed['OW']>0 and near else 0
        seeking=task not in('withdraw','static')and not accepted
        direction=np.asarray(decision['raw_target'])-before[:2];length=float(np.linalg.norm(direction))
        progress=0. if length<1e-9 else float((after[:2]-before[:2])@(direction/length))
        if seeking:
            self.recovery_elapsed+=1
            self.no_progress=self.no_progress+1 if progress<RULES['no_route_progress_world_px']else 0
        else:self.no_progress=0
        observed.update(translation_image96_px=translation,signed_rotation_deg=rotation,nearstatic=bool(near),selected_contact_established=bool(accepted),
            progress_along_command_world_px=progress,actual_controller_contact_established_after_observation=self.established)
        self.history.append(dict(control=self.next_index,decision=decision,observed=observed))
        self.last_observation=observed;self.target_dwell+=1;self.next_index+=1;self.pending=None


# A short alias keeps root's collector agnostic to naming while preserving the
# historical ContactPolicy class in its original untouched module.
ContactPolicy=ContactPilotPolicy
