"""CP2: transactional current-geometry routes and validated contact targets.

The recorder retains raw contact rows. This controller never rewrites those
observations, constructs a simulator, steps physics, or calls a model.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
import numpy as np
from shapely.geometry import Point

from .native_rgb_contact_pilot_controller import ContactPilotPolicy as CP1Policy,trace_observation
from .native_rgb_r3a_collection import PHASES,MODES,phase_for,segments,rot,area_center,identity
from .native_rgb_r2b_sampling import geometry_at,wall_gaps,contact_frame,_stage,_free,free_route,_bounded,_line,_contact_command_ok,_departure_route,_candidate,_room
from .native_rgb_env import shape_geometry
from .native_rgb_selected_physics import SELECTED_PROTOCOL_SHA256

RULES=dict(kind='CONTACT-PILOT2-transactional-contact-target.v1',selected_protocol_sha256=SELECTED_PROTOCOL_SHA256,
    controls=48,control_dt=.1,physics_dt=.01,impulse_epsilon=1e-8,
    stable_contact_substeps=5,stable_contact_requires_terminal=True,lost_contact_controls=2,
    task_schedule='historical phase_for unchanged; release withdrawal18..25,other22..25,recontact26',
    target_boundary_tolerance_world_px=.75,target_arc_projection_tolerance_world_px=.75,
    target_normal_in_out_probe_world_px=1.,target_normal_minimum_arc_dot=.85,
    observed_pair_max_jump_world_px=24.,requested_entry_lateral_tolerance_world_px=2.,
    requested_entry='normal only,zero tangent and lead until actual positive impulse requested-arc stable5-of10+terminal',
    requested_unconfirmed_controls_before_explicit_fallback=6,requested_incidental_controls_before_explicit_fallback=3,
    requested_scope='initial_requested_arc immutable; fallback never relabels requested-hit success',
    target_minimum_dwell_controls=4,no_progress_world_px=.5,no_progress_controls=3,unconfirmed_route_controls=2,
    wall_stall_controls=3,route_arrival_world_px=4.,route_eta_speed_world_px_s=100.,route_eta_extra_controls=2,
    repeated_wall_risk_mitigation_before_reacquire=2,
    route_transaction='contact target+departure source+waypoints+revision committed together; replan current geometry every command',
    observation_target='raw pair unchanged; executable point/normal/arc validated together against complete exterior union and staging',
    current_follow_advance_world_px=48.,entry_advance_world_px=24.,
    withdraw_distances_world_px=[64.,48.,33.,18.],candidate_minimum_drive_room_world_px=16.,
    nearstatic_translation_image96_px=.1,nearstatic_abs_rotation_deg=.1,
    safety_scope='all actual bounded commands gated by separate frozen CP2 current-state wall-risk heuristic; mode never exempts inward action',
    risk_contact_context='current verified committed departure source for an actual departure segment; otherwise current executable target',
    progress_reference='issued final action,not cancelled raw candidate',
    no_physics_lookahead=True,no_task_deletion=True,no_model_interface=True)
RULES_SHA256=identity(RULES)


def target_identity(contact):
    return identity({k:contact[k]for k in('arc_id','point_local','outward_normal_local')})


def validate_target_pair(letter,state,contact):
    """Validate an unchanged observed/requested pair, never clip one component."""
    try:
        point=np.asarray(contact['point_local'],float);normal=np.asarray(contact['outward_normal_local'],float)
        arc=int(contact['arc_id']);all_arcs={int(a['id']):a for a in segments(letter)}
        if point.shape!=(2,)or normal.shape!=(2,)or not np.isfinite(point).all()or not np.isfinite(normal).all():
            return dict(valid=False,reason='nonfinite_or_shape')
        length=float(np.linalg.norm(normal))
        if not .999<=length<=1.001:return dict(valid=False,reason='normal_not_unit')
        normal=normal/length;local=shape_geometry(letter);outer=local.exterior;p=Point(*point)
        if outer.distance(p)>RULES['target_boundary_tolerance_world_px']:return dict(valid=False,reason='point_not_full_union_exterior')
        if arc not in all_arcs:return dict(valid=False,reason='arc_not_outer')
        segment=all_arcs[arc];position=float(outer.project(p));perimeter=float(outer.length)
        lo=float(segment['arc_start'])-RULES['target_arc_projection_tolerance_world_px']
        hi=float(segment['arc_end'])+RULES['target_arc_projection_tolerance_world_px']
        if not any(lo<=position+offset<=hi for offset in(-perimeter,0.,perimeter)):
            return dict(valid=False,reason='point_arc_mismatch')
        if float(normal@np.asarray(segment['normal']))<RULES['target_normal_minimum_arc_dot']:
            return dict(valid=False,reason='normal_arc_mismatch')
        probe=RULES['target_normal_in_out_probe_world_px']
        if local.buffer(-1e-6).covers(Point(*(point+normal*probe)))or not local.buffer(.05).covers(Point(*(point-normal*probe))):
            return dict(valid=False,reason='normal_not_outward_union_pair')
        geometry=geometry_at(letter,state);world,world_normal,_=contact_frame(state,dict(point_local=point,outward_normal_local=normal))
        staging=_stage(geometry,world,world_normal)
        if staging is None:return dict(valid=False,reason='pair_has_no_legal_union_staging')
        return dict(valid=True,reason='full_union_pair_and_arc_valid',staging=staging,
            boundary_distance_world_px=float(outer.distance(p)),arc_projection=position)
    except (KeyError,TypeError,ValueError):return dict(valid=False,reason='malformed_pair')


def _raw_points(info):
    """Keep source substep and point indices so raw NPZ rows remain auditable."""
    points=[]
    for si,row in enumerate(info['substeps']):
        for ei,event in enumerate(row.get('contacts',[])):
            if event.get('phase')!='post_solve'or event.get('kind')!='agent_object':continue
            vector=np.asarray(event.get('total_impulse',[]),float)
            valid=vector.shape==(2,)and np.isfinite(vector).all()
            positive=bool(valid and np.linalg.norm(vector)>RULES['impulse_epsilon'])
            normal=np.asarray(event.get('normal',[]),float)
            if normal.shape!=(2,)or not np.isfinite(normal).all():continue
            if event.get('shapes_order',[None])[0]!='object':normal=-normal
            for pi,point in enumerate(event.get('points',[])):
                if point.get('object_local')is None:continue
                points.append(dict(local_substep=si,event_index=ei,point_index=pi,
                    point_local=list(map(float,point['object_local'])),normal_world=normal.tolist(),
                    arc_ids=list(map(int,point.get('arc_ids',[]))),valid_impulse=bool(valid),positive_impulse=positive))
    return points


class ContactPilot2Policy(CP1Policy):
    def __init__(self,env,slot,candidate,rules=None):
        if rules is not None and rules!=RULES:raise ValueError('Frozen CP2 rules changed')
        super().__init__(env,slot,candidate)
        self.initial_requested_contact=deepcopy(self.contact);self.initial_requested_arc=int(self.contact['arc_id'])
        self.requested_contact_substeps=0;self.requested_positive_substeps=0;self.requested_initial_established=False
        self.requested_unconfirmed=0;self.requested_incidental_controls=0;self.requested_fallback_reason=None
        self.route_transaction=None;self.departure_contact=deepcopy(self.contact)
        self.raw_observed_contact=None;self.target_validation=None;self.last_pair_rejection=None
        self.withdrawal_release_verified=False;self.recovery_observed_after_release=False
        self.wall_risk_mitigation_controls=0

    def _route_packet(self,state,geometry,start,target,source_contact=None,*,revision=None):
        start,target=np.asarray(start,float),np.asarray(target,float);source=deepcopy(source_contact or self.departure_contact or self.contact)
        path=free_route(geometry,start,target);kind='free_route'
        if path is None and _departure_route(geometry,start,target):path=[start.tolist(),target.tolist()];kind='direct_departure'
        if path is None:
            point,normal,_=contact_frame(state,source);stage=_stage(geometry,point,normal)
            if stage is not None and _departure_route(geometry,start,stage[0]):
                rest=free_route(geometry,stage[0],target)
                if rest is not None:path=[start.tolist(),*rest];kind='retained_departure_bridge'
        if path is None:return None
        if not all(_free(geometry).covers(_line(a,b))or _departure_route(geometry,np.asarray(a),np.asarray(b))for a,b in zip(path,path[1:])):
            return None
        return dict(kind=kind,target_revision=self.target_revision if revision is None else revision,
            target_identity=target_identity(self.contact),departure_source=source,waypoints=deepcopy(path),
            goal_world=target.tolist(),current_state_sha256=hashlib.sha256(np.asarray(state,np.float64).tobytes()).hexdigest(),
            route_identity=identity(dict(source=target_identity(source),path=path,revision=self.target_revision if revision is None else revision)))

    def route(self,geometry,start,target):
        packet=self._route_packet(np.asarray(self.env.state(),float),geometry,start,target)
        return None if packet is None else packet['waypoints']

    def _choose_replacement(self,state,geometry,*,wall_escape):
        scenario=self.slot['scenario'];role='translation_1'if scenario=='translation_nearCOM'else scenario if scenario.startswith('rotation_')else'generic'
        wall=min(wall_gaps(geometry),key=wall_gaps(geometry).get)
        inward={'left':np.array((1.,0.)),'right':np.array((-1.,0.)),'top':np.array((0.,1.)),'bottom':np.array((0.,-1.))}[wall]
        free=_free(geometry)
        for chosen_role in([role,'generic']if wall_escape and role!='generic'else[role]):
            choices=[]
            for segment in segments(self.slot['letter']):
                if int(segment['id'])==int(self.contact['arc_id']):continue
                fresh,_=_candidate(self.slot['letter'],state,geometry,segment,chosen_role,'center',free)
                if fresh is None:continue
                if scenario in('release_recontact','grazing'):fresh['drive_tangent_ratio']=self.initial_requested_contact.get('drive_tangent_ratio',0.)
                n=np.asarray(fresh['outward_normal_world']);direction=-n+fresh['drive_tangent_ratio']*np.array((-n[1],n[0]));direction/=np.linalg.norm(direction)
                dot=float(direction@inward)
                if _room(geometry,direction)<RULES['candidate_minimum_drive_room_world_px']or wall_escape and dot<-.05:continue
                valid=validate_target_pair(self.slot['letter'],state,fresh)
                if not valid['valid']:continue
                packet=self._route_packet(state,geometry,state[:2],fresh['staging_world'],self.departure_contact,revision=self.target_revision+1)
                if packet is None:continue
                packet['target_identity']=target_identity(fresh)
                priority=(int(wall_escape and abs(dot)>.65),self.route_length(packet['waypoints']),int(fresh['arc_id']))
                choices.append((priority,fresh,packet,chosen_role!=role))
            if choices:
                _,fresh,packet,fallback=min(choices,key=lambda row:row[0])
                fresh=deepcopy(fresh);fresh['wall_escape_generic_role_fallback']=fallback
                return fresh,packet
        return None

    def _commit_replacement(self,chosen,reason):
        contact,packet=chosen
        self.contact=deepcopy(contact);self.target_revision+=1;packet=deepcopy(packet)
        if packet['target_revision']!=self.target_revision or packet['target_identity']!=target_identity(self.contact):
            raise ValueError('Target/route transaction identity mismatch')
        self.route_transaction=packet;self.established=False;self.awaiting_selected_contact=True
        self.lost_contact=2;self.target_dwell=0;self.no_progress=0;self.unconfirmed_routes=0;self.wall_stall=0
        self.escape_active='wall'in reason;self.eta_controls=None;self.last_replan_reason=reason
        self.wall_risk_mitigation_controls=0

    def _replace_if_needed(self,state,geometry,staging):
        if self.target_dwell<RULES['target_minimum_dwell_controls']:return None
        reason=None
        if self.wall_risk_mitigation_controls>=RULES['repeated_wall_risk_mitigation_before_reacquire']:reason='near_wall_risk2_reacquire'
        elif self.wall_stall>=RULES['wall_stall_controls']:reason='AO_OW_nearstatic3_wall_escape'
        elif self.no_progress>=RULES['no_progress_controls']:reason='route_no_progress3'
        elif self.unconfirmed_routes>=RULES['unconfirmed_route_controls']:reason='route_unconfirmed2'
        if not self.requested_initial_established and self.requested_fallback_reason is None:
            if reason and(reason.startswith('AO_OW')or reason.startswith('near_wall')):pass
            elif self.requested_unconfirmed>=RULES['requested_unconfirmed_controls_before_explicit_fallback']:
                reason='requested_arc_route_unconfirmed6'
            elif self.requested_incidental_controls>=RULES['requested_incidental_controls_before_explicit_fallback']:
                reason='requested_arc_incidental_contact3_reacquire'
            elif reason and not(reason.startswith('AO_OW')or reason.startswith('near_wall')):return None
        if reason is None:return None
        selected=self._choose_replacement(state,geometry,wall_escape=reason.startswith('AO_OW')or reason.startswith('near_wall'))
        if selected is not None:
            if not self.requested_initial_established and self.requested_fallback_reason is None:
                self.requested_fallback_reason=reason+'_initial_requested_arc_not_hit_or_not_stable'
            self._commit_replacement(selected,reason+'_committed_with_route')
        else:
            self.last_replan_reason=reason+'_no_confirmed_alternative';self.target_dwell=0
            self.no_progress=0;self.unconfirmed_routes=0
        return self.last_replan_reason

    def _current_route(self,state,geometry,staging):
        source=self.route_transaction['departure_source']if self.route_transaction is not None else self.departure_contact
        packet=self._route_packet(state,geometry,state[:2],staging[0],source)
        if packet is None:
            self.route_transaction=None;return None,'current_geometry_route_invalid_replan_required'
        if self.route_transaction is not None:
            packet['committed_route_identity']=self.route_transaction.get('committed_route_identity',self.route_transaction['route_identity'])
        else:packet['committed_route_identity']=packet['route_identity']
        packet['target_identity']=target_identity(self.contact);self.route_transaction=packet
        return packet,None

    def _safety_gate(self,state,geometry,action,mode):
        from .native_rgb_contact_pilot2_safety import assess_wall_risk
        context=self.contact;context_source='current_executable_target'
        # The first committed exit segment physically departs the old contact,
        # even though its destination target has already changed. This selects
        # the actual geometric contact context; it never exempts a route label.
        if self.route_transaction is not None and _departure_route(geometry,state[:2],action):
            source=self.route_transaction['departure_source']
            if validate_target_pair(self.slot['letter'],state,source)['valid']:
                context=source;context_source='committed_verified_departure_source'
        evidence_mode=('follow'if self.established else'entry')if mode=='contact_push_intent'else mode
        risk=assess_wall_risk(self.slot['letter'],state,action,contact=context,mode=evidence_mode)
        risk['assessment_contact_source']=context_source;risk['assessment_contact_pair']={k:deepcopy(context[k])for k in('arc_id','point_local','outward_normal_local')}
        original=action.copy();fallback=None
        if not risk['risky']:return action,mode,risk,fallback
        point,normal,tangent=contact_frame(state,context);start=state[:2];choices=[]
        for label,direction,distance in [('outward_retreat18',normal,18.),('outward_retreat33',normal,33.),
            ('tangent_positive',tangent,16.),('tangent_negative',-tangent,16.)]:
            candidate=_bounded(start+direction*distance,start,self.last_action)
            if not(_free(geometry).covers(_line(start,candidate))or _departure_route(geometry,start,candidate)):continue
            score=assess_wall_risk(self.slot['letter'],state,candidate,contact=context,mode='retreat'if label.startswith('outward')else'route')
            if score['risky']:continue
            choices.append((int(not label.startswith('outward')),float(score['risk_score']),label,candidate,score))
        if choices:
            _,_,label,action,score=min(choices,key=lambda row:row[:3]);mode='free_route'
            fallback=dict(reason=label,final_risk=score)
        else:
            action=_bounded(start,start,self.last_action);mode='hold';fallback=dict(reason='no_verified_retreat_or_tangent_current_state_hold')
        risk['original_candidate_action']=original.tolist();risk['final_action']=action.tolist()
        risk['normal_advance_and_feedforward_cancelled']=True
        return action,mode,risk,fallback

    def next_action(self,index):
        if index!=self.next_index or self.pending is not None:raise ValueError('Exactly one actual command/observation pair')
        state=np.asarray(self.env.state(),np.float64)
        if state.shape!=(15,)or not np.isfinite(state).all():raise ValueError('Invalid current state')
        geometry=geometry_at(self.slot['letter'],state);task=phase_for(index,self.slot['scenario']);start=state[:2]
        if task!=self.task_phase:
            if task=='withdraw':self.release_target=None;self.route_transaction=None
            if self.task_phase=='withdraw':
                self.established=False;self.awaiting_selected_contact=True;self.lost_contact=2;self.route_transaction=None
                self.escape_active=False;self.eta_controls=None
            self.task_phase=task
        validation=validate_target_pair(self.slot['letter'],state,self.contact);self.target_validation={k:v for k,v in validation.items()if k!='staging'}
        staging=validation.get('staging');reason=None
        if task not in('withdraw','static'):reason=self._replace_if_needed(state,geometry,staging)
        validation=validate_target_pair(self.slot['letter'],state,self.contact);staging=validation.get('staging')
        self.target_validation={k:v for k,v in validation.items()if k!='staging'}
        if not validation['valid']:
            self.established=False;self.awaiting_selected_contact=True
        point,normal,tangent=contact_frame(state,self.contact);raw=start.copy();mode='hold';actual='unconfirmed_hold'
        path=None;distance=None;beta=0.;route_reason=None
        if not validation['valid']:actual='reacquire_invalid_executable_pair';route_reason=validation['reason']
        if task=='static':actual='static'
        elif task=='withdraw':
            actual='withdraw'
            if self.release_target is None:
                for gap in RULES['withdraw_distances_world_px']:
                    target=point+normal*gap
                    if self._route_packet(state,geometry,start,target,self.departure_contact)is not None:self.release_target=target.copy();break
            if self.release_target is not None:
                packet=self._route_packet(state,geometry,start,self.release_target,self.departure_contact)
                if packet is not None:path=packet['waypoints'];raw=np.asarray(path[1]);mode='free_route'
        elif staging is not None:
            stage,center,_=staging
            acquire=not self.established
            normal_only=acquire
            if normal_only:drive=center-normal*RULES['entry_advance_world_px'];beta=0.
            else:drive,beta=self._drive_target(state,center,normal,tangent,RULES['current_follow_advance_world_px'])
            proposed=_bounded(drive,start,self.last_action)
            lateral=abs(float((start-center)@tangent))
            entry_ok=_contact_command_ok(geometry,start,proposed,center)and(not normal_only or lateral<=RULES['requested_entry_lateral_tolerance_world_px'])
            if entry_ok:
                raw=drive;mode='contact_push_intent';actual='contact_follow'if self.established else'normal_only_acquisition'
                distance=float(np.linalg.norm(start-center))
                if self.route_transaction is not None:
                    route_reason='entry_handoff_from_current_verified_geometry_not_assumed_waypoint_completion';self.route_transaction=None
            else:
                packet,route_reason=self._current_route(state,geometry,staging)
                if packet is not None:
                    path=packet['waypoints'];raw=np.asarray(path[1]);mode='free_route';distance=self.route_length(path)
                    actual='transactional_recovery_route'
        action=_bounded(raw,start,self.last_action);candidate_action=action.copy()
        verified=(mode=='hold'or _free(geometry).covers(_line(start,action))or _departure_route(geometry,start,action))if mode!='contact_push_intent'else staging is not None and _contact_command_ok(geometry,start,action,staging[1])
        if not verified:action=_bounded(start,start,self.last_action);mode='hold';actual='unconfirmed_hold';route_reason='bounded_next_segment_invalid_current_geometry'
        risk=None;fallback=None
        action,mode,risk,fallback=self._safety_gate(state,geometry,action,mode)
        if fallback is not None:actual=('withdraw_'if task=='withdraw'else'')+'wall_risk_mitigation_'+mode
        self.wall_risk_mitigation_controls=self.wall_risk_mitigation_controls+1 if fallback is not None else 0
        if task not in('withdraw','static'):
            self.unconfirmed_routes=self.unconfirmed_routes+1 if mode=='hold'else 0
            if distance is not None and self.eta_controls is None:self.eta_controls=math.ceil(distance/(RULES['route_eta_speed_world_px_s']*.1))+RULES['route_eta_extra_controls']
        signed,resultant=self._current_levers(beta)
        decision=dict(phase=PHASES.index(task),mode=MODES.index(mode),arc_id=int(self.contact['arc_id']),point_local=deepcopy(self.contact['point_local']),
            normal_local=deepcopy(self.contact['outward_normal_local']),normal_world=normal.tolist(),beta=beta,signed_lever=signed,resultant_signed_lever=resultant,
            raw_target=np.asarray(raw).tolist(),geometry_verified=bool(verified),state_before_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            planned_task_phase=task,controller_state=actual,controller_rules_sha256=RULES_SHA256,
            initial_requested_arc=self.initial_requested_arc,initial_requested_contact=deepcopy(self.initial_requested_contact),
            requested_arc_hit_substeps=self.requested_contact_substeps,requested_arc_positive_impulse_substeps=self.requested_positive_substeps,
            requested_initial_established=self.requested_initial_established,requested_fallback_reason=self.requested_fallback_reason,
            raw_observed_contact=deepcopy(self.raw_observed_contact),executable_target_validation=deepcopy(self.target_validation),observed_pair_rejection=self.last_pair_rejection,
            actual_contact_established_before_action=bool(self.established),target_revision=self.target_revision,replan_reason=reason,
            route_transaction=deepcopy(self.route_transaction),route_reason=route_reason,route=path,route_remaining_world_px=distance,
            route_eta_controls=self.eta_controls,route_elapsed_controls=self.recovery_elapsed,no_progress_controls=self.no_progress,
            unconfirmed_route_controls=self.unconfirmed_routes,wall_stall_controls=self.wall_stall,
            wall_risk_mitigation_controls=self.wall_risk_mitigation_controls,
            previous_observation=None if self.last_observation is None else{k:v for k,v in self.last_observation.items()if k not in('raw_points','representative')},
            candidate_action_before_wall_risk=candidate_action.tolist(),final_action=action.tolist(),wall_risk=risk,wall_risk_fallback=fallback,
            withdrawal_release_verified=self.withdrawal_release_verified,recovery_after_verified_release=self.recovery_observed_after_release,
            future_state_used=False,lookahead_physics_steps=0,task_budget_extended=False)
        self.pending=(state.copy(),action.copy(),decision);self.last_action=action.copy()
        return action,decision

    def _observed_pair(self,after,points):
        positives=[p for p in points if p['positive_impulse']]
        positives.sort(key=lambda p:(-p['local_substep'],float(np.linalg.norm(np.asarray(p['point_local'])-self.contact['point_local'])),p['event_index'],p['point_index']))
        for item in positives:
            normal=rot(after[4]).T@np.asarray(item['normal_world']);norm=np.linalg.norm(normal)
            if norm<1e-10:continue
            normal/=norm;arcs={int(a['id']):a for a in segments(self.slot['letter'])}
            eligible=[a for a in item['arc_ids']if a in arcs]
            eligible.sort(key=lambda a:(a!=int(self.contact['arc_id']),np.linalg.norm(np.asarray(arcs[a]['point'])-item['point_local'])))
            for arc in eligible:
                candidate=deepcopy(self.contact);candidate.update(arc_id=arc,point_local=deepcopy(item['point_local']),outward_normal_local=normal.tolist())
                validation=validate_target_pair(self.slot['letter'],after,candidate)
                if validation['valid']:return candidate,item,validation
                self.last_pair_rejection=validation['reason']
        return None,None,None

    def _track_actual_contact(self,after,observation):
        candidate,item,validation=self._observed_pair(after,observation['raw_points'])
        if candidate is None:return False
        if np.linalg.norm(np.asarray(candidate['point_local'])-self.contact['point_local'])>RULES['observed_pair_max_jump_world_px']:
            self.last_pair_rejection='observed_legal_pair_jump_rejected_without_componentwise_clipping';return False
        self.contact=candidate;self.departure_contact=deepcopy(candidate);self.last_pair_rejection=None
        signed,resultant=self._current_levers();self.contact['signed_lever_world_px']=signed;self.contact['resultant_signed_lever_world_px']=resultant
        self.route_transaction=None;return True

    def _requested_evidence(self,points):
        requested={p['local_substep']for p in points if self.initial_requested_arc in p['arc_ids']}
        positive={p['local_substep']for p in points if self.initial_requested_arc in p['arc_ids']and p['positive_impulse']}
        return requested,positive,len(positive)>=RULES['stable_contact_substeps']and 9 in positive

    def observe(self,before,action,after,info):
        if self.pending is None:raise ValueError('observe requires pending actual command')
        before,after=np.asarray(before,np.float64),np.asarray(after,np.float64)
        if not np.array_equal(before,self.pending[0])or not np.array_equal(action,self.pending[1])or after.shape!=(15,)or not np.isfinite(after).all():raise ValueError('Actual transition mismatch')
        decision=self.pending[2];obs=trace_observation(info,self.contact);points=_raw_points(info);obs['raw_points']=points
        self.raw_observed_contact=deepcopy(obs['representative'])
        requested,requested_positive,requested_stable=self._requested_evidence(points)
        self.requested_contact_substeps+=len(requested);self.requested_positive_substeps+=len(requested_positive)
        self.requested_initial_established=self.requested_initial_established or requested_stable
        candidate,item,validation=self._observed_pair(after,points)
        if candidate is not None:self.departure_contact=deepcopy(candidate)
        active_arc=int(self.contact['arc_id']);active={p['local_substep']for p in points if active_arc in p['arc_ids']and p['positive_impulse']}
        stable_selected=len(active)>=RULES['stable_contact_substeps']and 9 in active
        accepted=bool(obs['established']and candidate is not None and(stable_selected or self.established and not self.awaiting_selected_contact))
        if accepted:
            self.established=True;self.lost_contact=0;self.awaiting_selected_contact=False;self.recovery_elapsed=0;self.eta_controls=None
            if decision['planned_task_phase']!='withdraw':self._track_actual_contact(after,obs)
        else:
            self.lost_contact+=1
            if self.lost_contact>=RULES['lost_contact_controls']:self.established=False
        task=decision['planned_task_phase']
        if task=='withdraw'and obs['AO']==0:self.withdrawal_release_verified=True
        if self.next_index>=26 and self.withdrawal_release_verified and obs['AO']>0:self.recovery_observed_after_release=True
        if not self.requested_initial_established and task not in('withdraw','static'):
            self.requested_unconfirmed=self.requested_unconfirmed+1 if decision['mode']==MODES.index('hold')or obs['AO']>0 else 0
            self.requested_incidental_controls=self.requested_incidental_controls+1 if obs['AO']>0 and not requested_positive else 0
        translation=float(np.linalg.norm(area_center(self.slot['letter'],after)-area_center(self.slot['letter'],before))*96/512)
        rotation=float((after[4]-before[4])*180/math.pi);near=translation<.1 and abs(rotation)<.1
        self.wall_stall=self.wall_stall+1 if task not in('withdraw','static')and obs['AO']>0 and obs['OW']>0 and near else 0
        direction=np.asarray(action)-before[:2];length=float(np.linalg.norm(direction))
        progress=0. if length<1e-9 else float((after[:2]-before[:2])@(direction/length))
        if task not in('withdraw','static')and not accepted:
            self.recovery_elapsed+=1;self.no_progress=self.no_progress+1 if progress<RULES['no_progress_world_px']else 0
        else:self.no_progress=0
        obs.update(translation_image96_px=translation,signed_rotation_deg=rotation,nearstatic=bool(near),selected_contact_established=accepted,
            requested_arc_contact_substeps=len(requested),requested_arc_positive_substeps=len(requested_positive),requested_arc_stable_control=requested_stable,
            incidental_contact_without_requested_positive=bool(obs['AO']and not requested_positive),raw_observed_point_count=len(points),
            progress_along_command_world_px=progress,actual_controller_contact_established_after_observation=self.established)
        obs['progress_reference']='issued_final_action_not_cancelled_raw_candidate'
        self.last_observation=obs;self.history.append(dict(control=self.next_index,decision=decision,observed=obs))
        self.target_dwell+=1;self.next_index+=1;self.pending=None


ContactPilotPolicy=ContactPilot2Policy
