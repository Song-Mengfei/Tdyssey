"""R2B fixed-pose geometric planning and current-state-only contact following.

No environment is created or stepped here. Exact geometry is privileged
collection information; it is never a model input, prediction or loss.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, box

from .envs.official import official_shape
from .native_rgb_data import contour_segments
from .native_rgb_env import shape_geometry
from .native_rgb_supervised_sampling import _departure_route

LETTERS = tuple("TLOCXB")
LOCATIONS = ("center", "left", "right", "top", "bottom", "left-top", "right-top", "left-bottom", "right-bottom")
PROTOCOL_SHA256 = "58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6"
RULES = dict(
    kind="R2B-current-state-same-arc-feedback.v1", selected_protocol="D-half-life-0.1s",
    protocol_sha256=PROTOCOL_SHA256, damping=0.0009765625,
    max_controls=24, phases=[[0,4,"approach"],[4,14,"push"],[14,20,"withdraw"],[20,24,"recontact"]],
    pusher_radius_world_px=15., clearance_margin_world_px=.5,
    wall_inner_bounds_world_px=[7.,504.], safe_pusher_center_bounds_world_px=[22.5,488.5],
    arc_bin_world_px=8., normal_advance_world_px=48., approach_advance_world_px=24.,
    staging_distances_world_px=[33.,18.], withdraw_distances_world_px=[80.,64.,48.,33.,18.],
    max_target_distance_from_current_pusher_world_px=64., max_target_change_world_px=96.,
    maximum_translation_tangent_ratio=.5, mixed_tangent_ratio=.35,
    maximum_translation_residual_lever_world_px=5., rotation_minimum_signed_lever_world_px=12.,
    lost_contact_controls_before_recovery=2, maximum_same_arc_recovery_controls=4,
    staging_arrival_world_px=4., contact_entry_tangential_tolerance_world_px=18.,
    no_progress_world_px=.25, no_progress_controls=3,
    finite_pose_yaw_offsets_rad=[0.,math.pi/2,-math.pi/2,math.pi],
    finite_gap_world_px=[4.,18.,48.], signed_lever_numerical_zero_world_px=1e-9,
    route_scope="bounded geometric visibility graph, no dynamics lookahead; unproven is not proven unreachable",
    contact_scope="contact-push targets may enter the object; exterior prefix and intended entry are checked separately, never called a free route",
    recovery_scope="same local arc only; recovery consumes existing phase controls; no extra step/reset/replacement",
)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _rot(angle):
    c,s=math.cos(float(angle)),math.sin(float(angle))
    return np.array(((c,-s),(s,c)),dtype=np.float64)


@lru_cache(maxsize=6)
def _local(letter):
    if letter not in LETTERS:raise ValueError("R2B supports only the six authorized TRAIN letters")
    return shape_geometry(letter)


@lru_cache(maxsize=6)
def _segments(letter):
    return contour_segments(letter,8)


def geometry_at(letter,state):
    return affinity.translate(affinity.rotate(_local(letter),float(state[4]),origin=(0,0),use_radians=True),*map(float,state[2:4]))


def wall_gaps(geometry):
    x0,y0,x1,y1=geometry.bounds
    return dict(left=x0-7,right=504-x1,top=y0-7,bottom=504-y1)


def _free(geometry):
    return box(22.5,22.5,488.5,488.5).difference(geometry.buffer(15.5,resolution=32)).buffer(1e-7)


def _line(start,end):
    return Point(*start) if np.linalg.norm(np.asarray(end)-start)<1e-10 else LineString((start,end))


def free_route(geometry,start,end):
    """A finite visibility graph; None means unconfirmed, never teleport."""
    start,end=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    free=_free(geometry)
    if not free.covers(Point(*start)) or not free.covers(Point(*end)):return None
    if free.covers(_line(start,end)):return [start.tolist(),end.tolist()]
    hull=geometry.buffer(18.,resolution=4).convex_hull
    nodes=[start,end]
    for p in list(hull.exterior.coords)[:-1]+[(23.,23.),(488.,23.),(23.,488.),(488.,488.)]:
        p=np.asarray(p,dtype=float)
        if free.covers(Point(*p)) and not any(np.linalg.norm(p-q)<1e-8 for q in nodes):nodes.append(p)
    graph=[[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i):
            if free.covers(_line(nodes[i],nodes[j])):
                length=float(np.linalg.norm(nodes[i]-nodes[j]));graph[i].append((j,length));graph[j].append((i,length))
    queue=[(0.,0,())];seen=set()
    while queue:
        distance,i,path=heapq.heappop(queue)
        if i in seen:continue
        seen.add(i);path=(*path,i)
        if i==1:return [nodes[j].tolist() for j in path]
        for j,length in graph[i]:
            if j not in seen:heapq.heappush(queue,(distance+length,j,path))
    return None


def contact_frame(state,contact):
    rotation=_rot(state[4])
    point=np.asarray(state[2:4])+rotation@np.asarray(contact["point_local"])
    normal=rotation@np.asarray(contact["outward_normal_local"])
    tangent=np.asarray((-normal[1],normal[0]))
    return point,normal,tangent


def _stage(geometry,point,normal,free=None):
    free=_free(geometry) if free is None else free
    contact=point+normal*15.5
    pieces=list(free.geoms) if hasattr(free,"geoms") else [free]
    pieces=[p for p in pieces if p.area>1e-8]
    exterior=max(pieces,key=lambda p:p.area) if pieces else None
    for distance in RULES["staging_distances_world_px"]:
        stage=point+normal*distance
        if exterior is not None and exterior.buffer(1e-7).covers(_line(stage,contact)):
            return stage,contact,distance
    return None


def _room(geometry,direction):
    x0,y0,x1,y1=geometry.bounds;values=[]
    for d,lo,hi in ((direction[0],x0,x1),(direction[1],y0,y1)):
        if d>1e-8:values.append((504-hi)/d)
        elif d< -1e-8:values.append((7-lo)/d)
    return max(0.,min(values)) if values else 0.


def _cross(a,b):return float(a[0]*b[1]-a[1]*b[0])


def _role(letter,location):
    roles=("translation_1","translation_1","translation_2","rotation_positive","rotation_negative","mixed_positive","translation_2","mixed_negative","wall_along")
    role=roles[LOCATIONS.index(location)]
    if letter=="T" and location=="center":role="rotation_positive"
    if letter=="T" and location=="left":role="translation_1"
    if letter=="L" and location=="center":role="rotation_negative"
    if letter=="O" and location=="left":role="wall_along"
    if letter=="C" and location=="top":role="rotation_negative"
    if letter=="C" and location=="bottom":role="rotation_positive"
    return role


def _pose(letter,location,angle,gap):
    rotated=affinity.rotate(_local(letter),angle,origin=(0,0),use_radians=True)
    x0,y0,x1,y1=rotated.bounds
    origin=np.array((255.5-(x0+x1)/2,255.5-(y0+y1)/2),dtype=float)
    if "left" in location:origin[0]=7+gap-x0
    if "right" in location:origin[0]=504-gap-x1
    if "top" in location:origin[1]=7+gap-y0
    if "bottom" in location:origin[1]=504-gap-y1
    return origin,affinity.translate(rotated,*origin)


def _candidate(letter,state,geometry,segment,role,location,free=None):
    if segment["ring"]:return None,"closed_hole_not_exterior"
    contact=dict(point_local=segment["point"],outward_normal_local=segment["normal"])
    point,normal,tangent=contact_frame(state,contact)
    staging=_stage(geometry,point,normal,free)
    if staging is None:return None,"radius15_normal_staging_or_wall_blocked_unconfirmed"
    cog=np.array((0.,45.)) if letter=="T" else np.zeros(2)
    r=np.asarray(segment["point"])-cog
    local_n=np.asarray(segment["normal"]);local_t=np.array((-local_n[1],local_n[0]))
    normal_lever=_cross(r,-local_n);denom=_cross(r,local_t)
    beta=0.
    if role.startswith("translation"):
        if abs(denom)>1e-9:beta=-normal_lever/denom
        if abs(beta)>.5 or abs(normal_lever+beta*denom)>5:return None,"requested_translation_resultant_lever_not_supported"
    elif role=="rotation_positive" and normal_lever<12:return None,"positive_lever_under12"
    elif role=="rotation_negative" and normal_lever> -12:return None,"negative_lever_under12"
    elif role.startswith("mixed"):
        beta=.35 if role.endswith("positive") else -.35
    direction=-normal+beta*tangent;direction/=np.linalg.norm(direction)
    room=_room(geometry,direction)
    if role.startswith("translation") and room<24:return None,"translation_room_under24"
    if role.startswith("rotation") and room<8:return None,"rotation_drive_room_under8"
    process="interior_contact" if location=="center" else "leave_boundary" if role.startswith("translation") else "along_wall" if role=="wall_along" or role.startswith("mixed") else "boundary_rotation"
    if process in ("leave_boundary","along_wall"):
        wall=min(wall_gaps(geometry),key=wall_gaps(geometry).get)
        inward={"left":np.array((1.,0.)),"right":np.array((-1.,0.)),"top":np.array((0.,1.)),"bottom":np.array((0.,-1.))}[wall]
        dot=float(direction@inward)
        if process=="leave_boundary" and dot<.35:return None,"leave_direction_not_supported"
        if process=="along_wall" and abs(dot)>.65:return None,"along_direction_not_supported"
    stage,center,distance=staging
    return dict(arc_id=int(segment["id"]),segment_id=int(segment["id"]),ring=0,
        point_local=list(map(float,segment["point"])),outward_normal_local=list(map(float,segment["normal"])),
        point_world=point.tolist(),outward_normal_world=normal.tolist(),contact_center_world=center.tolist(),
        staging_world=stage.tolist(),staging_distance_world_px=distance,
        signed_lever_world_px=normal_lever,resultant_signed_lever_world_px=normal_lever+beta*denom,
        physical_local_CoG=cog.tolist(),drive_tangent_ratio=float(beta),drive_direction_world=direction.tolist(),
        translation_room_world_px=room,process_goal=process,
        free_approach_path=[stage.tolist(),center.tolist()],pusher_radius_world_px=15.,clearance_margin_world_px=.5,
        static_reachability="exterior_connected_conservative_fixed_pose_normal_path",actual_contact_verified=False),None


def build_plan(source_plan):
    """54 new requests from saved source poses, finite geometry only, zero step."""
    source_path=None
    if not isinstance(source_plan,dict):
        source_path=Path(source_plan);source_bytes=source_path.read_bytes();source_plan=json.loads(source_bytes)
    else:source_bytes=json.dumps(source_plan,sort_keys=True).encode()
    families={(f["letter"],f["requested_initial_space"]):f for f in source_plan["scene_families"] if f["phase"]=="R2B"}
    if set(families)!={(l,p) for l in LETTERS for p in LOCATIONS}:raise ValueError("Need exactly old54 R1B source poses, never dispatch old requests")
    requests=[];translation_arcs={letter:set() for letter in LETTERS};reject_counts=Counter()
    for li,letter in enumerate(LETTERS):
        for pi,location in enumerate(LOCATIONS):
            source=families[(letter,location)];role=_role(letter,location)
            requested_gap=None if location=="center" else RULES["finite_gap_world_px"][(li+pi-1)%3]
            gap_candidates=[0.] if requested_gap is None else [requested_gap,*[g for g in (4.,18.,48.) if g!=requested_gap]]
            candidates=[];rejections=Counter()
            for yi,offset in enumerate(RULES["finite_pose_yaw_offsets_rad"]):
                angle=source["initialization"]["angle_rad"]+offset
                for gi,gap in enumerate(gap_candidates):
                    origin,geometry=_pose(letter,location,angle,gap)
                    if min(wall_gaps(geometry).values())< -1e-7:continue
                    state=np.zeros(15);state[2:4]=origin;state[4]=angle
                    free=_free(geometry)
                    for segment in _segments(letter):
                        candidate,reason=_candidate(letter,state,geometry,segment,role,location,free)
                        if candidate is None:rejections[reason]+=1;continue
                        repeated=role.startswith("translation") and candidate["arc_id"] in translation_arcs[letter]
                        rolecost=abs(candidate["resultant_signed_lever_world_px"]) if role.startswith("translation") else -abs(candidate["signed_lever_world_px"])
                        priority=(int(repeated),gi,yi,rolecost,-min(candidate["translation_room_world_px"],96),candidate["arc_id"])
                        candidates.append((priority,state.copy(),candidate,gap,offset))
            reject_counts.update(rejections)
            chosen=min(candidates,key=lambda row:row[0]) if candidates else None
            request=dict(request_id=f"R2B-new-{li*9+pi:02d}-{letter}-{location}",phase="R2B",family_id=f"R2B-new-F{li*9+pi:02d}-{letter}-{location}",
                letter=letter,initial_space=location,spatial_class=location,role=role,protocol="D-half-life-0.1s",protocol_sha256=PROTOCOL_SHA256,
                max_controls=24,seed=202610300000+li*100+pi,source_family_id=source["family_id"],source=source["source"],
                source_old_plan_initialization_sha256=canonical_sha(source["initialization"]),requested_gap_world_px=requested_gap,
                dispatchable=chosen is not None,static_missing_reasons=[] if chosen else ["finite_geometry_candidates_do_not_support_fixed_role"],
                finite_candidate_count=len(candidates),rejected_candidate_reasons=dict(rejections),old_static_missing_goals=source.get("static_missing_goals",[]),
                actual_motion_or_contact_claim=False,policy_rules_sha256=canonical_sha(RULES))
            if chosen:
                priority,state,contact,gap,offset=chosen;state[:2]=contact["staging_world"]
                if role.startswith("translation"):translation_arcs[letter].add(contact["arc_id"])
                mass=state[2:4]+_rot(state[4])@np.asarray(contact["physical_local_CoG"])
                geom=geometry_at(letter,state);area=official_shape(letter).centroid
                request.update(selected_contact=contact,process_goal=contact["process_goal"],actual_initial_wall_gaps=wall_gaps(geom),
                    selected_gap_world_px=None if location=="center" else gap,gap_request_met=location=="center" or gap==requested_gap,
                    source_yaw_offset_rad=offset,source_object_origin_delta_world_px=(state[2:4]-source["initialization"]["object_origin"]).tolist(),selection_priority=list(priority),
                    initialization=dict(letter=letter,object_origin=state[2:4].tolist(),angle_rad=float(state[4]),
                        object_mass_center=mass.tolist(),object_area_centroid=(state[2:4]+_rot(state[4])@np.asarray(area)).tolist(),
                        pusher_start=state[:2].tolist(),reset_scene_arguments=dict(center=mass.tolist(),angle=float(state[4]),pusher=state[:2].tolist()),
                        expected_initial_state15=state.tolist(),expected_initial_state_json_sha256=canonical_sha(state.tolist()),all_initial_velocities_forces_torque_zero=True),
                    old_static_gap_resolution="new finite pose/gap/arc proves this fixed new role, not achievement of old dynamics" if source.get("static_missing_goals") else None)
            else:request["old_static_gap_resolution"]="unresolved; preserve slot unexecuted, no replacement"
            requests.append(request)
    first=[("T","center"),("C","center"),("X","center"),("B","center"),("L","center"),("O","left")]
    requests.sort(key=lambda q:(first.index((q["letter"],q["initial_space"])) if (q["letter"],q["initial_space"]) in first else 6+LETTERS.index(q["letter"])*9+LOCATIONS.index(q["initial_space"])))
    for order,q in enumerate(requests):q["execution_order"]=order;q["first_six_gate"]=order<6
    support={}
    for letter in LETTERS:
        chosen=[q for q in requests if q["letter"]==letter and q["dispatchable"]]
        trans={q["selected_contact"]["arc_id"] for q in chosen if q["role"].startswith("translation")}
        support[letter]=dict(dispatchable=len(chosen),translation_intent_distinct_arcs=sorted(trans),
            positive_rotation_intents=sum(q["role"]=="rotation_positive" for q in chosen),negative_rotation_intents=sum(q["role"]=="rotation_negative" for q in chosen),
            oblique_or_wall_intents=sum(q["role"].startswith("mixed") or q["role"]=="wall_along" for q in chosen),
            planned_minima_met=len(trans)>=2 and any(q["role"]=="rotation_positive" for q in chosen) and any(q["role"]=="rotation_negative" for q in chosen) and any(q["role"].startswith("mixed") or q["role"]=="wall_along" for q in chosen))
    return dict(kind="R2B-new-current-state-feedback-plan.v1",selected_protocol=dict(name="D-half-life-0.1s",damping=.0009765625,sha256=PROTOCOL_SHA256),
        source_plan=dict(path=None if source_path is None else str(source_path),sha256=hashlib.sha256(source_bytes).hexdigest(),use="saved initialpose/source only; old requests/actions not executed; no future motion read"),
        rules=RULES,rules_sha256=canonical_sha(RULES),requests=requests,scene_families=[dict(family_id=q["family_id"],phase="R2B",letter=q["letter"],requested_initial_space=q["initial_space"],initialization=q.get("initialization"),initial_contact_plan=q.get("selected_contact"),source=q["source"]) for q in requests],
        static_review=dict(planned_slots=54,planned_controls=1296,dispatchable_slots=sum(q["dispatchable"] for q in requests),
            first_six=[dict(request_id=q["request_id"],role=q["role"],dispatchable=q["dispatchable"]) for q in requests[:6]],
            per_letter_support=support,finite_rejection_counts=dict(reject_counts),
            old_four_static_gap_cases=[dict(source_family_id=q["source_family_id"],old_missing=q["old_static_missing_goals"],new_role=q["role"],new_dispatchable=q["dispatchable"],resolution=q["old_static_gap_resolution"]) for q in requests if q["old_static_missing_goals"]],
            new_physics_steps=0,new_environment_instances=0,new_renderer_calls=0,new_model_calls=0,
            scope="geometric feasibility and intent only; no claim of actual motion/contact/boundary achieved"))


def planned_phase(control_index):
    if type(control_index) is not int or not 0<=control_index<24:raise ValueError("Exactly24 authorized phase controls")
    return next(name for low,high,name in RULES["phases"] if low<=control_index<high)


def _bounded(raw,current,last):
    target=np.asarray(raw,dtype=float);current=np.asarray(current,dtype=float)
    # Project only the command; no body position/velocity is ever modified.
    for _ in range(16):
        delta=target-current;length=np.linalg.norm(delta)
        if length>64:target=current+delta*(64/length)
        if last is not None:
            delta=target-last;length=np.linalg.norm(delta)
            if length>96:target=last+delta*(96/length)
        target=np.clip(target,22.5,488.5)
    if np.linalg.norm(target-current)>64+1e-8 or last is not None and np.linalg.norm(target-last)>96+1e-8:
        target=np.clip(current,22.5,488.5)
        if np.linalg.norm(target-current)>64+1e-8 or last is not None and np.linalg.norm(target-last)>96+1e-8:
            raise ValueError("Current-state command bounds have no verified safe hold")
    return target


def _contact_command_ok(geometry,start,target,center):
    """Intentional contact: prove exterior prefix, not a free interior route."""
    start,target,center=map(lambda v:np.asarray(v,dtype=float),(start,target,center))
    if not box(22.5,22.5,488.5,488.5).covers(_line(start,target)):return False
    obstacle=geometry.buffer(15.5,resolution=32)
    if obstacle.covers(Point(*start)):
        return geometry.distance(Point(*start))>=13.-1e-6 and np.linalg.norm(start-center)<=24.
    route=_line(start,target)
    if _free(geometry).covers(route):return True
    intersection=route.intersection(obstacle.boundary)
    pieces=list(intersection.geoms) if hasattr(intersection,"geoms") else [intersection]
    points=[p for p in pieces if p.geom_type=="Point"]
    if not points:return False
    entry=min(points,key=lambda p:route.project(p))
    if entry.distance(Point(*center))>18:return False
    near=start+.999999*(np.asarray(entry.coords[0])-start)
    return _free(geometry).covers(_line(start,near))


class FeedbackPolicy:
    """Only env.state() is read. Root alone steps and records real contacts."""
    def __init__(self,env,request,rules=None):
        if not request.get("dispatchable") or request.get("protocol_sha256")!=PROTOCOL_SHA256:raise ValueError("Unapproved or static-blocked request")
        if rules is not None and rules!=RULES:raise ValueError("R2B rule mutation rejected")
        self.env,self.request=env,request;self.contact=request["selected_contact"]
        self.last_action=None;self.release_target=None;self.pending=None;self.next_index=0
        self.history=[];self.lost_contact=0;self.recovery_controls=0;self.no_progress=0

    def _route(self,geometry,start,stage):
        route=free_route(geometry,start,stage)
        if route is not None:return route,"free_exterior_route"
        if _departure_route(geometry,start,stage):return [start.tolist(),stage.tolist()],"contact_shell_departure_monotone_clearance"
        return None,"same_arc_exterior_route_unconfirmed"

    def next_action(self,control_index):
        if control_index!=self.next_index or self.pending is not None:raise ValueError("One next_action/observe pair per real control")
        phase=planned_phase(control_index);state=np.asarray(self.env.state(),dtype=np.float64)
        if state.shape!=(15,) or not np.isfinite(state).all():raise ValueError("Nonfinite/invalid current state")
        start=state[:2];geometry=geometry_at(self.request["letter"],state)
        point,normal,tangent=contact_frame(state,self.contact);stage=_stage(geometry,point,normal)
        raw=start.copy();route=None;reason="";mode="hold";actual_phase=phase+"_miss"
        need_recovery=phase=="push" and self.lost_contact>=2
        recovery_allowed=not need_recovery or self.recovery_controls<4
        if phase=="withdraw":
            if self.release_target is None:
                for distance in RULES["withdraw_distances_world_px"]:
                    desired=point+normal*distance
                    possible,why=self._route(geometry,start,desired)
                    if possible is not None:
                        self.release_target=desired.copy();break
            if self.release_target is not None:
                route,reason=self._route(geometry,start,self.release_target)
                if route is not None:
                    raw=np.asarray(route[1]);mode="free_route";actual_phase="withdraw_route" if control_index==14 else "decay_fixed_release_target"
            else:reason="no_same_arc_legal_withdrawal; hold, not reset"
        elif stage is None:
            reason="same_local_arc_now_staging_blocked; no alternative arc or scene"
        elif not recovery_allowed:
            reason="same_arc_recovery4_control_limit; hold for remaining scheduled controls"
        else:
            staging,center,distance=stage
            at_staging=np.linalg.norm(start-staging)<=4
            if need_recovery and not at_staging:
                self.recovery_controls+=1
                route,reason=self._route(geometry,start,staging)
                if route is not None:
                    raw=np.asarray(route[1]);mode="free_route";actual_phase="same_arc_recovery"
            elif phase in ("approach","recontact") and np.linalg.norm(start-center)>8 and not at_staging:
                route,reason=self._route(geometry,start,staging)
                if route is not None:
                    raw=np.asarray(route[1]);mode="free_route";actual_phase=phase+"_exterior_route"
            else:
                if need_recovery:self.recovery_controls+=1
                advance=48. if phase=="push" and not need_recovery else 24.
                raw=center-normal*advance+tangent*(advance*self.contact["drive_tangent_ratio"])
                mode="contact_push_intent";actual_phase="same_arc_recovery_contact_intent" if need_recovery else phase+"_contact_intent";reason="current_pose_fixed_local_arc_world_follow; actual contact only in observe"
        bounded=_bounded(raw,start,self.last_action)
        if mode=="free_route":
            verified=_free(geometry).covers(_line(start,bounded)) or _departure_route(geometry,start,bounded)
        elif mode=="contact_push_intent":
            verified=stage is not None and _contact_command_ok(geometry,start,bounded,stage[1])
        else:verified=True
        if not verified:
            bounded=_bounded(start,start,self.last_action);mode="hold";actual_phase=phase+"_unconfirmed_command_hold";reason+="; bounded actual command geometric route/entry rejected"
        bounded=np.asarray(bounded,dtype=np.float64)
        if not ((bounded>=0).all() and (bounded<=511).all()):raise ValueError("Illegal action after bounded geometry decision")
        decision=dict(control_index=control_index,planned_phase=phase,actual_phase=actual_phase,phase=phase,
            current_state=state.tolist(),state_before=state.tolist(),state_before_sha256=hashlib.sha256(state.tobytes(order="C")).hexdigest(),
            local_contact_point=self.contact["point_local"],local_contact_normal=self.contact["outward_normal_local"],
            contact_point_world=point.tolist(),outward_normal_world=normal.tolist(),fixed_arc_id=self.contact["arc_id"],
            raw_target=np.asarray(raw).tolist(),bounded_target=bounded.tolist(),proposed_action=bounded.tolist(),action_absolute=bounded.tolist(),
            target_distance_from_current_world_px=float(np.linalg.norm(bounded-start)),
            target_change_world_px=None if self.last_action is None else float(np.linalg.norm(bounded-self.last_action)),
            command_mode=mode,command_geometry_verified=bool(verified),route=route,reason=reason,
            recovery_controls_used=self.recovery_controls,lost_contact_controls=self.lost_contact,no_progress_controls=self.no_progress,
            policy_rules_sha256=canonical_sha(RULES),used_future_state=False,lookahead_physics_steps=0,actual_contact_claim=False)
        self.pending=decision;self.last_action=bounded.copy()
        return bounded,decision

    def observe(self,before,action,after,info):
        if self.pending is None:raise ValueError("observe without pending decision")
        before,after=np.asarray(before,dtype=float),np.asarray(after,dtype=float)
        if before.shape!=(15,) or after.shape!=(15,) or not np.isfinite(after).all():raise ValueError("Invalid observed states")
        if not np.array_equal(before,np.asarray(self.pending["current_state"])) or not np.array_equal(action,self.last_action):raise ValueError("Observed transition differs from issued decision")
        count=info.get("agent_object_contact_substeps",info.get("contact",0))
        contact=None if count is None else bool(count>0)
        self.lost_contact=0 if contact is True else self.lost_contact+1
        distance=float(np.linalg.norm(after[:2]-before[:2]));self.no_progress=self.no_progress+1 if distance<.25 else 0
        observed=dict(control_index=self.next_index,planned_phase=self.pending["planned_phase"],actual_contact=contact,
            actual_contact_substeps=info.get("agent_object_contact_substeps"),actual_contact_arc_ids=info.get("actual_contact_arc_ids"),
            contact_intent_missed=self.pending["command_mode"]=="contact_push_intent" and contact is not True,
            pusher_displacement_world_px=distance,lost_contact_controls=self.lost_contact,no_progress_controls=self.no_progress,
            same_arc_recovery_controls=self.recovery_controls,phase_budget_extended=False)
        self.history.append(dict(decision=self.pending,observed=observed));self.pending=None;self.next_index+=1
        return observed
