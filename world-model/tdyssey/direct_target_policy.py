"""Finite, geometry-only acceptance plans for the original absolute-XY action.

This module never constructs or steps an environment. Geometry is privileged
sampling information. It is not a model input. Every action is committed before
the trajectory starts, passed unchanged, and checked against the original
inclusive [0,512] action domain. There is no distance/change/velocity limiter.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, box

from .native_rgb_env import shape_geometry

LETTERS = tuple("TLOCXB")
SCENARIOS = ("translation", "rotation_cw", "rotation_ccw", "surface_change",
             "hold_turn_withdraw", "depart_recontact", "boundary", "negative")
VERSION = "direct-absolute-target-acceptance.v1"
SEED = 54781031
RULES = dict(version=VERSION, seed=SEED, letters=list(LETTERS),
    independent_families=96, maximum_attempts=120, controls_per_attempt=64,
    total_control_cap=7680, regular_substep_cap=76800,
    original_action_domain_inclusive=[0.,512.], pusher_radius=15.,
    history_real_controls=3, horizon_controls=[1,4,8,16],
    action_generation="complete pre-execution commitment; never reads future state",
    target_distance_cap=None, target_change_cap=None, target_inner_clip=None,
    initialization_contact_clearance_world_px=3.,
    initialization_boundary_gap_world_px=[3.,9.],
    pose_bins=8, exterior_arclength_bins=16,
    target_length_fraction_ranges=[[.18,.48],[.55,.95],[1.05,1.85]],
    target_switch_controls=[4,8,12],
    model_calls=0, optimizer_updates=0, training_eligible=False,
    no_retry_or_replacement=True)


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_targets(actions):
    """Reject invalid requests; never alter a target to make it admissible."""
    a=np.asarray(actions,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():
        raise ValueError("Targets must be a finite N by 2 array")
    if (a<0.).any() or (a>512.).any():
        raise ValueError("Raw target is outside original inclusive [0,512] action_space")
    return a.copy()


def _rotation(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.asarray(((c,-s),(s,c)),dtype=np.float64)


def geometry_at(letter,state):
    g=affinity.rotate(shape_geometry(letter),float(state[4]),origin=(0,0),use_radians=True)
    return affinity.translate(g,*np.asarray(state[2:4],float))


def _diameter(g):
    p=np.asarray(g.convex_hull.exterior.coords[:-1],float)
    return float(np.sqrt(np.max(np.sum((p[:,None,:]-p[None,:,:])**2,axis=-1))))


def _cross(a,b):
    return float(a[0]*b[1]-a[1]*b[0])


@lru_cache(maxsize=6)
def exterior_catalog(letter, samples_per_bin=9):
    """Continuous fixed-seed arc samples; holes are explicitly excluded.

    Circular reachability is checked against the entire union, so an outward
    normal on a narrow open notch is not automatically called reachable.
    """
    g=shape_geometry(letter); ring=g.exterior; length=float(ring.length)
    rng=np.random.default_rng(SEED+ord(letter)*1009)
    vertices=np.asarray(ring.coords[:-1],float)
    concave_vertices=[];convex_vertices=[]
    for vi,v in enumerate(vertices):
        quadrant_inside=sum(g.contains(Point(*(v+np.asarray((dx,dy))*.01)))
            for dx,dy in ((-1,-1),(-1,1),(1,-1),(1,1)))
        (concave_vertices if quadrant_inside==3 else convex_vertices).append(v)
    result=[]
    for bid in range(16):
        for j in range(samples_per_bin):
            frac=(bid+(j+float(rng.uniform(.05,.95)))/samples_per_bin)/16
            arc=frac*length; p=ring.interpolate(arc)
            lo=ring.interpolate((arc-.001)%length);hi=ring.interpolate((arc+.001)%length)
            tangent=np.asarray((hi.x-lo.x,hi.y-lo.y),float);tangent/=np.linalg.norm(tangent)
            normal=np.asarray((-tangent[1],tangent[0]))
            point=np.asarray((p.x,p.y),float)
            if g.contains(Point(*(point+normal*.01))):normal=-normal
            center=point+normal*18.
            clear=bool(g.distance(Point(*center))>=17.999999 and
                g.buffer(15.,resolution=32).intersection(LineString((center,point+normal*80.))).length<1e-7)
            nearest=float(np.min(np.linalg.norm(vertices-point,axis=1)))
            # Sign at nearest exact outer-ring vertex distinguishes convex and
            # concave corners independently of the ring winding.
            dconcave=min((float(np.linalg.norm(v-point)) for v in concave_vertices),default=None)
            dconvex=min((float(np.linalg.norm(v-point)) for v in convex_vertices),default=None)
            feature=("open_concave_side" if dconcave is not None and dconcave<=32.
                else "convex_corner_nearby" if dconvex is not None and dconvex<10. else "edge")
            result.append(dict(arc_id=bid,ring=0,arc_fraction=float(frac),
                point_local=point.tolist(),outward_normal_local=normal.tolist(),
                exterior_normal_ray_clear=clear,
                corner_distance_world_px=nearest,
                open_concave_corner_distance_world_px=dconcave,
                convex_corner_distance_world_px=dconvex,feature=feature,
                actual_contact_verified=False))
    return result


def _placed(letter,angle,center,boundary,gap):
    cog=np.asarray((0.,45.) if letter=="T" else (0.,0.))
    origin=np.asarray(center,float)-_rotation(angle)@cog
    state=np.zeros(15,float);state[2:4]=origin;state[4]=angle
    geom=geometry_at(letter,state)
    if boundary:
        x0,y0,x1,y1=geom.bounds;delta=np.zeros(2)
        if "left" in boundary:delta[0]=7.+gap-x0
        if "right" in boundary:delta[0]=504.-gap-x1
        if "top" in boundary:delta[1]=7.+gap-y0
        if "bottom" in boundary:delta[1]=504.-gap-y1
        state[2:4]+=delta
    return state,geometry_at(letter,state)


def _choose_contact(letter,state,geom,scenario,preferred_arc,rng,boundary):
    rotation=_rotation(state[4]);origin=state[2:4]
    cog=np.asarray((geom.centroid.x,geom.centroid.y),float)
    free=box(22.00001,22.00001,488.99999,488.99999).difference(geom.buffer(15.000001,resolution=32))
    pieces=list(free.geoms) if hasattr(free,"geoms") else [free]
    exterior=max((p for p in pieces if p.area>1e-6),key=lambda p:p.area)
    candidates=[]
    toward=np.zeros(2)
    if boundary:
        toward=np.asarray(((-1 if "left" in boundary else 1 if "right" in boundary else 0),
                           (-1 if "top" in boundary else 1 if "bottom" in boundary else 0)),float)
        toward/=np.linalg.norm(toward)
    for c in exterior_catalog(letter):
        if not c['exterior_normal_ray_clear']:continue
        point=origin+rotation@np.asarray(c['point_local']);normal=rotation@np.asarray(c['outward_normal_local'])
        pusher=point+normal*18.
        if not exterior.buffer(1e-7).covers(Point(*pusher)):continue
        tangent=np.asarray((-normal[1],normal[0]));direction=-normal
        lever=_cross(point-cog,direction)
        arc_delta=min((c['arc_id']-preferred_arc)%16,(preferred_arc-c['arc_id'])%16)
        if scenario=="translation":
            direct=cog-point;direct/=np.linalg.norm(direct)
            if direct@(-normal)<.75:continue
            direction=direct;lever=_cross(point-cog,direction)
            score=-abs(lever)-.5*arc_delta
        elif scenario=="rotation_cw":score=lever-.4*arc_delta
        elif scenario=="rotation_ccw":score=-lever-.4*arc_delta
        elif scenario=="surface_change":
            direction=(-normal+.6*tangent);direction/=np.linalg.norm(direction)
            score=-2.*arc_delta+(40. if c['feature']=="open_concave_side" else
                10. if c['feature']=="convex_corner_nearby" else 0.)
        elif scenario=="boundary":
            if (-normal)@toward<.35:continue
            direction=toward;score=15.*((-normal)@toward)-.4*arc_delta
        else:score=-3.*arc_delta
        candidates.append((score+float(rng.uniform(0,.01)),c,point,normal,pusher,direction,lever))
    if not candidates:raise ValueError(f"No finite exterior initialization for {letter}/{scenario}/{boundary}")
    _,contact,point,normal,pusher,direction,lever=max(candidates,key=lambda x:x[0])
    result=deepcopy(contact)
    result.update(point_world=point.tolist(),outward_normal_world=normal.tolist(),
        contact_center_world=(point+normal*15.).tolist(),staging_world=pusher.tolist(),
        pusher_radius_world_px=15.,clearance_world_px=3.,
        drive_direction_world=direction.tolist(),signed_normal_lever_world_px=float(lever),
        reachability="initial center in exterior-connected fixed-pose free component; future contact unverified",
        whole_contour_coverage="unknown until actual contacts are reported")
    return result,pusher,direction


def _endpoint(start,direction,length):
    goal=np.asarray(start,float)+np.asarray(direction,float)*float(length)
    return goal if np.isfinite(goal).all() and (goal>=0).all() and (goal<=512).all() else None


def _distance_endpoint(start,direction,D,bin_id,rng):
    lo,hi=RULES['target_length_fraction_ranges'][bin_id]
    # Finite geometric candidate enumeration is not a physics retry or action
    # clipping. No rejected endpoint is sent to the environment.
    fractions=rng.uniform(lo,hi,24)
    candidates=[(float(f),_endpoint(start,direction,D*f)) for f in fractions]
    available=[(f,g) for f,g in candidates if g is not None]
    if not available:
        return None,dict(requested_bin=bin_id,available=False,reason="all finite pre-execution geometric endpoints outside original action domain")
    f,goal=available[0]
    return goal,dict(requested_bin=bin_id,available=True,fraction_D=f,world_px=float(D*f))


def _line(start,goal,n):
    return np.asarray(start)[None,:]+np.arange(1,n+1)[:,None]/n*(np.asarray(goal)-np.asarray(start))[None,:]


def _actions(pusher,goal,direction,D,scenario,replicate,switch,rng,initial_geometry):
    """Generate the requested absolute path, rather than limiting another path."""
    tangent=np.asarray((-direction[1],direction[0]))
    # The control path may request a point in collision; original PD/collisions
    # determine motion. Only the input action_space is used for endpoint validity.
    away=_endpoint(pusher,-direction,.28*D)
    if away is None:away=np.asarray(pusher,float) # an explicitly chosen hold point
    turn=_endpoint(goal,tangent,.22*D)
    if turn is None:turn=_endpoint(goal,-tangent,.22*D)
    if turn is None:turn=np.asarray(pusher,float)
    mode="fixed" if replicate==0 else "moving"
    a=np.repeat(np.asarray(pusher,float)[None,:],64,axis=0)
    phases=["real_history_hold"]*3+["push_intent"]*61
    if mode=="fixed":a[3:]=goal
    else:
        a[3:19]=_line(pusher,goal,16);a[19:]=goal
    if scenario=="hold_turn_withdraw":
        k=3+switch;a[k:k+8]=turn;a[k+8:k+16]=away;a[k+16:]=away
        phases[k:k+8]=["turn_intent"]*8;phases[k+8:]=["withdraw_intent"]*(64-k-8)
    elif scenario=="depart_recontact":
        # This compact precommitted H16 contains push4, depart4, return8.
        # Recontact is an intended event, never an assumed success label.
        a[3:7]=goal;a[7:11]=away;a[11:19]=goal
        a[19:27]=away;a[27:43]=goal;a[43:51]=away;a[51:]=goal
        for l,r,label in ((7,11,"withdraw_intent"),(11,19,"recontact_intent"),
                          (19,27,"withdraw_intent"),(27,43,"recontact_intent"),
                          (43,51,"withdraw_intent"),(51,64,"recontact_intent")):
            phases[l:r]=[label]*(r-l)
    elif scenario=="surface_change":
        k=3+switch;a[k:k+16]=_line(a[k-1],turn,16) if mode=="moving" else turn
        a[k+16:]=turn;phases[k:]=["surface_change_intent"]*(64-k)
    elif scenario=="negative":
        if replicate==0:
            a[:]=pusher;phases[3:]=["stationary_control"]*61
        else:
            # Full inclusive-domain endpoint is intentional, including 512;
            # any illegal physical result is retained and terminated by observer.
            candidates=np.asarray(((0.,0.),(0.,512.),(512.,0.),(512.,512.)))
            free=[target for target in candidates
                if initial_geometry.distance(LineString((pusher,target)))>=15.000001]
            if not free:raise ValueError("No pre-execution full-domain negative-control free straightline")
            far=max(free,key=lambda target:float(np.linalg.norm(target-pusher)))
            a[3:19]=_line(pusher,far,16);a[19:]=far
            phases[3:]=["full_domain_empty_motion_intent"]*61
    return validate_targets(a),phases,mode


def build_plan(seed=SEED):
    """Freeze exactly 96 initial families and 120 maximum physical attempts."""
    if type(seed) is not int:raise ValueError("Seed must be an integer")
    families=[];attempts=[];catalog={}
    boundary_labels=("left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
    for li,letter in enumerate(LETTERS):
        D=_diameter(shape_geometry(letter));catalog[letter]=dict(geometry_diameter_world_px=D,
            exterior_arclength_world_px=float(shape_geometry(letter).exterior.length),
            closed_holes=len(shape_geometry(letter).interiors),
            closed_hole_contact_required=False,
            sample_catalog=exterior_catalog(letter))
        for si,scenario in enumerate(SCENARIOS):
            for ri in range(2):
                fi=si*2+ri;rng=np.random.default_rng(seed+li*10000+fi*101)
                family_id=f"DT-{letter}-{si:02d}-{ri}"
                angle=((fi%8)+float(rng.uniform(.08,.92)))*math.pi/4
                center=rng.uniform(190.,322.,2)
                boundary=boundary_labels[(li*2+ri)%8] if scenario=="boundary" else None
                state,geom=_placed(letter,angle,center,boundary,float(rng.uniform(3.,9.)))
                contact,pusher,direction=_choose_contact(letter,state,geom,scenario,fi,rng,boundary)
                state[:2]=pusher
                coglocal=np.asarray((0.,45.) if letter=="T" else (0.,0.))
                center=state[2:4]+_rotation(angle)@coglocal
                bin_id=(li+fi)%3
                if boundary:
                    # Construct a wall-directed original-domain goal. This is
                    # not the clipping of a different proposed coordinate.
                    goal=pusher.copy()
                    if "left" in boundary:goal[0]=0.
                    if "right" in boundary:goal[0]=512.
                    if "top" in boundary:goal[1]=0.
                    if "bottom" in boundary:goal[1]=512.
                    length=float(np.linalg.norm(goal-pusher));ratio=length/D
                    distance=dict(requested_bin=0 if ratio<=.5 else 1 if ratio<=1 else 2 if ratio<=2 else 3,
                        available=True,fraction_D=ratio,world_px=length,
                        selection="direct original-domain wall coordinate; not distance-bin limited")
                else:goal,distance=_distance_endpoint(pusher,direction,D,bin_id,rng)
                if goal is None:
                    # Keep this planned family as invalid with no physics call;
                    # no shorter substituted target or replacement family.
                    goal=pusher.copy()
                switch=(4,8,12)[(li+fi)%3]
                actions,phases,mode=_actions(pusher,goal,direction,D,scenario,ri,switch,rng,geom)
                actual_main_distance=float(np.linalg.norm(actions[18]-pusher))
                distance.update(actual_main_target_world_px=actual_main_distance,
                    actual_main_target_fraction_D=actual_main_distance/D,
                    actual_max_target_world_px=float(np.max(np.linalg.norm(actions-pusher,axis=1))),
                    actual_max_target_fraction_D=float(np.max(np.linalg.norm(actions-pusher,axis=1)))/D,
                    note="All distances are requested targets from initial pusher, not actual pusher/object travel")
                if scenario=="negative":
                    distance['unused_contact_distance_draw']=dict(requested_bin=distance['requested_bin'],
                        fraction_D=distance.get('fraction_D'),world_px=distance.get('world_px'))
                    distance.update(requested_bin="not_applicable_negative",fraction_D=actual_main_distance/D,
                        world_px=actual_main_distance,selection="stationary hold" if ri==0 else "direct inclusive-domain corner target")
                base=dict(family_id=family_id,letter=letter,scenario=scenario,
                    family_index=fi,seed=seed+li*10000+fi*101,split="acceptance_development",
                    initialization=dict(center=center.tolist(),pusher=pusher.tolist(),angle=float(angle)),
                    initial_state15=state.tolist(),initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    initial_pose_bin=fi%8,initial_space=boundary or "interior",
                    selected_contact=contact,geometry_diameter_world_px=D,
                    target_distance_design=distance,target_mode=mode,switch_control=switch,
                    plan_valid=bool(distance['available']),
                    plan_invalid_reason=None if distance['available'] else distance['reason'],
                    selected_boundary_intent=boundary,
                    action_causality="pre_execution_committed",all_actions_precommitted=True,
                    original_action_space=[0.,512.],old_model_action_space=[0.,511.],
                    model_normalization_changed=False,training_eligible=False)
                families.append(deepcopy(base))
                # One translation and one eccentric family per letter receive
                # two branches; every branch is a counted separate attempt.
                branches=("continue","turn","withdraw") if (scenario in ("translation","rotation_cw") and ri==0) else ("continue",)
                for bi,branch in enumerate(branches):
                    a=actions.copy();p=list(phases)
                    if bi:
                        if branch=="withdraw":
                            endpoint=_endpoint(pusher,-direction,.28*D)
                            if endpoint is None:endpoint=pusher.copy()
                            p[3:]=["branch_withdraw_intent"]*61
                        else:
                            tangent=np.asarray((-direction[1],direction[0]))
                            endpoint=_endpoint(goal,tangent,.25*D)
                            if endpoint is None:endpoint=_endpoint(goal,-tangent,.25*D)
                            if endpoint is None:endpoint=pusher.copy()
                            p[3:]=["branch_turn_intent"]*61
                        a[3:]=endpoint
                    validate_targets(a)
                    attempt=dict(deepcopy(base),attempt_id=f"{family_id}-b{bi}",branch_id=bi,branch=branch,
                        branch_family=len(branches)==3,branch_reference_attempt_id=f"{family_id}-b0",
                        common_prefix_controls=3,actions=a.tolist(),planned_phase=p,
                        planned_actions_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                        actual_planned_max_distance_from_initial_pusher_world_px=float(np.max(np.linalg.norm(a-pusher,axis=1))),
                        actual_planned_max_target_change_world_px=float(np.max(np.linalg.norm(np.diff(a,axis=0),axis=1))),
                        blocks=[dict(start=l,end=r,committed_before_control=0,causality="anchor_committed")
                                for l,r in ((3,19),(19,35),(35,51),(51,64))],
                        max_controls=64)
                    attempt['target_distance_design'].update(
                        branch_actual_main_target_world_px=float(np.linalg.norm(a[18]-pusher)),
                        branch_actual_main_target_fraction_D=float(np.linalg.norm(a[18]-pusher))/D,
                        branch_actual_max_target_world_px=float(np.max(np.linalg.norm(a-pusher,axis=1))),
                        branch_actual_max_target_fraction_D=float(np.max(np.linalg.norm(a-pusher,axis=1)))/D)
                    attempts.append(attempt)
    plan=dict(kind=VERSION,seed=seed,rules=RULES,rules_sha256=sha(RULES),
        families=families,attempts=attempts,geometry_catalog=catalog,
        physical_attempt_count=0,maximum_new_physical_attempts=120,
        no_new_simulator_validation_outside_attempts=True,
        action_commitment="All64 targets frozen before any reset/step; unchanged throughout each attempt",
        branch_protocol="Full original environment reconstruction and replay3 real prefix controls; compare4 RGB/state frames and3 targets",
        no_feedback_replanning=True,model_calls=0,optimizer_updates=0,training_eligible=False,
        qualification="Sampling intents only; actual motion/contact/recontact/wall coverage must be measured")
    if len(families)!=96 or len(attempts)!=120:raise AssertionError("Finite approved table changed")
    if len({f['initial_state_sha256'] for f in families})!=96:raise AssertionError("Independent initial states duplicated")
    plan['plan_sha256']=sha(plan)
    return plan


class DirectTargetPolicy:
    """Immutable index lookup with no state access or hidden target repair."""
    def __init__(self,attempt):
        self.attempt=deepcopy(attempt)
        self.actions=validate_targets(self.attempt['actions'])
        self.actions.setflags(write=False)
    def next_action(self,index):
        if type(index) is not int or not 0<=index<len(self.actions):raise IndexError(index)
        value=self.actions[index].copy()
        return value,dict(requested_action=value.tolist(),passed_action=value.tolist(),
            phase=self.attempt['planned_phase'][index],policy_version=VERSION,
            commitment=self.attempt['planned_actions_sha256'],future_state_read=False,
            target_clipped=False,execution_intervention=False)
