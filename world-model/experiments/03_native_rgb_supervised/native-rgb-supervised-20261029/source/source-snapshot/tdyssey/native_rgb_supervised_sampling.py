"""Stage-one continuous contact policy; only the root runner executes physics.

Geometry/state are privileged sampling diagnostics, never world-model inputs or
losses. Initial reset is the sole body-placement operation. The controller and
all physical parameters remain finite-force-pd-v1.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from .envs.official import official_shape
from .native_rgb_data import contour_segments, _reachable_segments
from .native_rgb_env import CONTROLLER, PROTOCOL_SHA256, shape_geometry

LETTERS=tuple("TLOCXB")
LOCATIONS=("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
PHASES=((0,4,"approach"),(4,40,"push"),(40,48,"sliding"),
        (48,52,"retreat"),(52,56,"recontact"),(56,64,"push_again"))
DEFAULT_SAMPLING=dict(wall_gap_bands_world_px=[[0,8],[8,32],[32,64]],
    wall_gap_representatives_world_px=[4,18,48],initial_legal_staging_world_px=18,
    action_distance_world_px=dict(near=32,mid=96,far=224))


def _sampling(settings=None):
    value={**DEFAULT_SAMPLING,**(settings or {})}
    if value["initial_legal_staging_world_px"]!=18:
        raise ValueError("This supervised stage fixes exterior staging at the reviewed 18 world pixels")
    distances=value["action_distance_world_px"]
    if not all(math.isfinite(distances[key]) and distances[key]>8 for key in ("near","mid","far")):
        raise ValueError("Near/mid/far targets must be finite and are not capped at 8px")
    return value


def plan_scenes(config):
    """Declare exactly six-by-nine independent scenes; perform zero physics."""
    if tuple(config.get("letters",LETTERS))!=LETTERS or tuple(config.get("locations",LOCATIONS))!=LOCATIONS:
        raise ValueError("Only the authorized six letters and nine spatial classes are supported")
    settings=_sampling(config.get("sampling"))
    requests=[]
    for location_index,location in enumerate(LOCATIONS):
        for letter_index,letter in enumerate(LETTERS):
            tier=(location_index-1+letter_index)%3
            requests.append(dict(letter=letter,spatial_class=location,
                seed=int(config.get("seed",2026102901))+location_index*1000+letter_index,
                gap_band=None if location=="center" else list(settings["wall_gap_bands_world_px"][tier]),
                gap_world_px=None if location=="center" else float(settings["wall_gap_representatives_world_px"][tier]),
                sampling=settings,controls=64))
    return requests


def _rotation(angle):
    import numpy as np
    c,s=math.cos(angle),math.sin(angle)
    return np.asarray(((c,-s),(s,c)),dtype=float)


def _count(counts,index):return int((counts or {}).get(index,(counts or {}).get(str(index),0)))


def wall_gaps(geometry):
    xmin,ymin,xmax,ymax=geometry.bounds
    return dict(left=float(xmin-7),right=float(504-xmax),top=float(ymin-7),bottom=float(504-ymax))


def _translation_room(geometry,direction):
    xmin,ymin,xmax,ymax=geometry.bounds
    limits=[]
    for component,low,high in ((direction[0],xmin,xmax),(direction[1],ymin,ymax)):
        if component>1e-8:limits.append(max(0,(504-high)/component))
        elif component< -1e-8:limits.append(max(0,(low-7)/(-component)))
    return float(min(limits)) if limits else 0.


def _candidate_priority(candidate,geometry,new_counts,historical_counts,rng):
    import numpy as np
    inward=-np.asarray(candidate["normal"]);room=_translation_room(geometry,inward)
    tier=0 if room>=96 else 1 if room>=32 else 2
    index=candidate["segment_id"]
    return (tier,int(_count(new_counts,index)>0),int(_count(historical_counts,index)>0),
        _count(new_counts,index),_count(historical_counts,index),-min(room,224),float(rng.random()))


def plan_scene(letter,spatial_class,seed,*,gap_band=None,gap_world_px=None,sampling=None,
               coverage_counts=None,historical_counts=None):
    """Pure geometric initial plan. No environment construction or physics step."""
    import numpy as np
    from shapely import affinity
    from shapely.geometry import box
    if letter not in LETTERS or spatial_class not in LOCATIONS:raise ValueError("Unauthorized stage-one scene")
    settings=_sampling(sampling);rng=np.random.default_rng(seed)
    angle=float(rng.uniform(-math.pi,math.pi));rotation=_rotation(angle)
    local=shape_geometry(letter);rotated=affinity.rotate(local,angle,origin=(0,0),use_radians=True)
    xmin,ymin,xmax,ymax=rotated.bounds
    origin=np.asarray((256-(xmin+xmax)/2,256-(ymin+ymax)/2),dtype=float)
    gap=float(48 if gap_world_px is None else gap_world_px)
    if spatial_class!="center" and (not math.isfinite(gap) or not 0<=gap<=64):raise ValueError("Invalid requested wall gap")
    if "left" in spatial_class:origin[0]=7+gap-xmin
    if "right" in spatial_class:origin[0]=504-gap-xmax
    if "top" in spatial_class:origin[1]=7+gap-ymin
    if "bottom" in spatial_class:origin[1]=504-gap-ymax
    geometry=affinity.translate(rotated,*origin)
    body=SimpleNamespace(angle=angle,local_to_world=lambda point:tuple(origin+rotation@np.asarray(point)))
    view=SimpleNamespace(block=body,world_geometry=lambda:geometry)
    segments=contour_segments(letter,8);reachability=_reachable_segments(view,segments)
    candidates=[row for row in reachability if row["status"]=="reachable_exterior_component"]
    candidates.sort(key=lambda row:_candidate_priority(row,geometry,coverage_counts,historical_counts,rng))
    chosen=candidates[0] if candidates else None
    free=box(22.5,22.5,488.5,488.5).difference(geometry.buffer(15.5,resolution=32))
    pieces=list(free.geoms) if hasattr(free,"geoms") else [free]
    pieces=[piece for piece in pieces if piece.area>0]
    fallback=max(pieces,key=lambda piece:piece.area).representative_point() if pieces else None
    pusher=chosen["staging"] if chosen else [fallback.x,fallback.y] if fallback else [256,400]
    cog=(0,45) if letter=="T" else (0,0)
    mass_center=origin+rotation@np.asarray(cog)
    gaps=wall_gaps(geometry);issues=[]
    if min(gaps.values())< -1e-7:issues.append("object_initial_wall_crossing")
    if chosen is None:issues.append("no_proven_exterior_connected_staging")
    if gap_band is not None and not gap_band[0]<=gap<=gap_band[1]:issues.append("requested_gap_outside_band")
    return dict(kind="supervised-continuous-scene-v1",letter=letter,spatial_class=spatial_class,seed=int(seed),
        sampling=settings,legal=not issues,issues=issues,gap_band=gap_band,gap_world_px=gap_world_px,
        object_origin=origin.tolist(),object_angle=angle,object_mass_center=mass_center.tolist(),
        object_area_centroid=[float(geometry.centroid.x),float(geometry.centroid.y)],
        pusher_start=list(map(float,pusher)),actual_wall_gaps=gaps,
        initial_min_wall_gap_world_px=min(gaps.values()),
        staging_segment=chosen["segment_id"] if chosen else None,staging=chosen,
        initial_translation_room_world_px=_translation_room(geometry,-np.asarray(chosen["normal"])) if chosen else 0,
        reachability=reachability,initializations_allowed=1,midtrajectory_placement_allowed=False,
        physics_protocol_sha256=PROTOCOL_SHA256,
        selection_scope="motion-room tier then new-stage unhit/lowhit and historical unhit/lowhit; actual contact counts only")


def initialize_scene(env,request,coverage_counts=None,historical_counts=None):
    """Exactly one trajectory initialization; never called during a rollout."""
    import numpy as np
    if getattr(env,"controller",CONTROLLER)!=CONTROLLER:raise ValueError("Wrong controller")
    scene=plan_scene(request["letter"],request["spatial_class"],request["seed"],
        gap_band=request.get("gap_band"),gap_world_px=request.get("gap_world_px"),sampling=request.get("sampling"),
        coverage_counts=coverage_counts,historical_counts=historical_counts)
    env.reset_scene(center=scene["object_mass_center"],angle=scene["object_angle"],pusher=scene["pusher_start"])
    state=np.asarray(env.state(),dtype=float);penetrations=np.asarray(env.penetrations(),dtype=float)
    if getattr(env,"control_step",0)!=0:scene["issues"].append("initialization_advanced_physics")
    if not np.isfinite(state).all():scene["issues"].append("initial_nonfinite_state")
    if not np.isfinite(penetrations).all() or penetrations.max()>.1:scene["issues"].append("initial_penetration")
    scene["legal"]=not scene["issues"];scene["initial_state"]=state.tolist()
    scene["initial_penetrations"]=penetrations.tolist();scene["actual_wall_gaps"]=wall_gaps(env.world_geometry())
    scene["initial_min_wall_gap_world_px"]=min(scene["actual_wall_gaps"].values())
    return scene


def _area_center(letter,state):
    import numpy as np
    shape=official_shape(letter)
    return np.asarray(state[2:4])+_rotation(float(state[4]))@np.asarray(shape.centroid)


def _free_route(geometry,start,end):
    from shapely.geometry import box,LineString
    free=box(22.5,22.5,488.5,488.5).difference(geometry.buffer(15.5,resolution=32)).buffer(1e-7)
    return bool(free.covers(LineString((start,end))))


def _departure_route(geometry,start,end):
    """Exit solver contact without deepening overlap, then keep full clearance."""
    import numpy as np
    from shapely.geometry import Point,box,LineString
    start=np.asarray(start,dtype=float);end=np.asarray(end,dtype=float)
    wall_overlap=max(0,22-start[0],22-start[1],start[0]-489,start[1]-489)
    if wall_overlap>2 or not box(22-wall_overlap-1e-7,22-wall_overlap-1e-7,
            489+wall_overlap+1e-7,489+wall_overlap+1e-7).covers(LineString((start,end))):return False
    distances=[]
    for alpha in np.linspace(0,1,max(3,int(np.linalg.norm(end-start))+2)):
        p=Point(*(start+alpha*(end-start)))
        if geometry.contains(p):return False
        distances.append(float(geometry.distance(p)))
    if distances[0]<13 or min(distances)<min(15.5,distances[0])-1e-5:return False
    first=next((i for i,value in enumerate(distances) if value>=15.5-1e-7),None)
    return first is not None and min(distances[first:])>=15.5-1e-7


class ContinuousContactSampler:
    """Propose/observe only: no step, body assignment, reset or history stitching."""
    def __init__(self,env,segments,coverage_counts,scene):
        if not scene.get("legal"):raise ValueError("Cannot roll out an illegal scene")
        self.env=env;self.segments=segments;self.counts=coverage_counts;self.scene=scene
        self.letter=scene["letter"];self.settings=_sampling(scene.get("sampling"))
        self.side_sign=1 if (LOCATIONS.index(scene["spatial_class"])+LETTERS.index(self.letter))%2 else -1
        self.exterior=shape_geometry(self.letter).exterior
        self.active=int(scene["staging_segment"]);self.pending=None;self.index=0
        self.blocked_streak=0;self.lost_streak=0;self.escape=None;self.escapes=0;self.cooldown=-1
        self.route_arc=None
        self.rng=__import__("numpy").random.default_rng(scene["seed"]+100000)

    def _normal(self,state):return _rotation(float(state[4]))@__import__("numpy").asarray(self.segments[self.active]["normal"])

    def _retreat(self,normal,distance):
        import numpy as np
        start=np.asarray(self.env.state()[:2]);tangent=np.asarray((-normal[1],normal[0]));geometry=self.env.world_geometry()
        for direction in (normal,(normal+tangent)/math.sqrt(2),(normal-tangent)/math.sqrt(2),tangent,-tangent):
            target=np.clip(start+direction*distance,22.5,488.5)
            if np.linalg.norm(target-start)>=2 and _departure_route(geometry,start,target):return target,"verified_contact_exit"
        return start,"retreat_path_unconfirmed_or_wall_blocked"

    def _recontact(self,prefer_alternate=False):
        import numpy as np
        start=np.asarray(self.env.state()[:2]);geometry=self.env.world_geometry()
        candidates=[row for row in _reachable_segments(self.env,self.segments) if row["status"]=="reachable_exterior_component"]
        candidates=[row for row in candidates if _free_route(geometry,start,row["staging"])]
        latched=next((row for row in candidates if row["segment_id"]==self.route_arc),None)
        if latched is not None:return np.asarray(latched["staging"]),"verified_latched_exterior_recontact"
        if prefer_alternate:candidates=[row for row in candidates if row["segment_id"]!=self.active]
        candidates.sort(key=lambda row:(0 if _translation_room(geometry,-np.asarray(row["normal"]))>=32 else 1,
            _count(self.counts,row["segment_id"]),float(np.linalg.norm(np.asarray(row["staging"])-start))))
        if candidates:
            chosen=candidates[0];self.active=int(chosen["segment_id"])
            self.route_arc=self.active
            return np.asarray(chosen["staging"]),"verified_exterior_straight_recontact"
        return start,"recontact_path_unconfirmed_no_straight_route"

    def next_action(self,step):
        import numpy as np
        if step!=self.index or self.pending is not None or not 0<=step<64:
            raise ValueError("Proposal/observation must follow the continuous 64-step timeline")
        state=np.asarray(self.env.state());normal=self._normal(state);tangent=np.asarray((-normal[1],normal[0]))
        phase=next(name for start,end,name in PHASES if start<=step<end)
        category="near" if phase=="approach" else "mid" if phase in ("sliding","retreat","recontact") else "far"
        distance=float(self.settings["action_distance_world_px"][category]);approach="normal";route="physical_push_target_may_be_inside_object"
        if phase=="push" and self.blocked_streak>=3 and step<32 and step>=self.cooldown and self.escapes<2 and self.escape is None:
            self.escape=dict(stage="retreat",remaining=3);self.escapes+=1;self.cooldown=step+12
        if phase not in ("push",):self.escape=None
        recovery=None
        if phase in ("push","push_again") and self.lost_streak>=3 and self.escape is None:
            waypoint,recovery_evidence=self._recontact()
            normal=self._normal(state);tangent=np.asarray((-normal[1],normal[0]))
            delta=waypoint-state[:2];length=float(np.linalg.norm(delta))
            if length>3 and recovery_evidence.startswith("verified"):
                direction=delta/length
                target=state[:2]+direction*float(self.settings["action_distance_world_px"]["mid"]) if direction.dot(-normal)>.2 else waypoint
                recovery=(target,recovery_evidence+"; extend into physical contact only when approach points inward")
        if recovery is not None:
            target,route=recovery;phase="contact_recovery_route";approach="path_following"
            category="mid";distance=float(self.settings["action_distance_world_px"]["mid"])
        elif self.escape is not None:
            distance=float(self.settings["action_distance_world_px"]["mid"])
            if self.escape["stage"]=="retreat":
                self.route_arc=None
                target,route=self._retreat(normal,float(self.settings["action_distance_world_px"]["mid"]))
                phase="blocked_escape_retreat";self.escape["remaining"]-=1
                if self.escape["remaining"]==0:self.escape=dict(stage="recontact",remaining=4)
            else:
                target,route=self._recontact(prefer_alternate=True);phase="blocked_escape_recontact"
                self.escape["remaining"]-=1
                if self.escape["remaining"]==0:self.escape=None
            approach="path_following";category="mid"
        elif phase=="retreat":
            self.route_arc=None
            target,route=self._retreat(normal,distance);approach="outward_exit"
        elif phase=="recontact":
            target,route=self._recontact();approach="path_following"
        else:
            inward=-normal
            if phase=="sliding":
                sign=self.side_sign
                direction=.5*inward+sign*math.sqrt(.75)*tangent;approach="tangent_with_inward_pressure"
            elif phase=="push" and 16<=step<28:
                sign=self.side_sign
                direction=math.cos(math.pi/9)*inward+sign*math.sin(math.pi/9)*tangent;approach="oblique_20deg"
            else:direction=inward
            target=state[:2]+direction*distance
        action=np.clip(np.asarray(target,dtype=float),0,511)
        normal=self._normal(state);tangent=np.asarray((-normal[1],normal[0]));delta=action-state[:2]
        metadata=dict(phase=phase,intent=phase,approach_kind=approach,active_arc=int(self.active),
            target_bin=int(self.active),target_distance_class=category,
            requested_distance_category=category,requested_distance_world_px=distance,
            approach_side_sign=int(self.side_sign),
            signed_action_angle_from_inward_normal_rad=float(math.atan2(delta.dot(tangent),delta.dot(-normal))),
            actual_target_distance_world_px=float(np.linalg.norm(action-state[:2])),route_evidence=route,
            blocked_contact_streak=int(self.blocked_streak),lost_contact_streak=int(self.lost_streak),
            escape_attempts=int(self.escapes),controller=CONTROLLER,midtrajectory_reset=False)
        self.pending=dict(before=state.copy(),action=action.copy(),metadata=metadata)
        return action,metadata

    def observe(self,before,action,after,info):
        import numpy as np
        if self.pending is None:raise ValueError("Observation without an outstanding proposal")
        before=np.asarray(before);after=np.asarray(after);action=np.asarray(action)
        if not np.array_equal(before,self.pending["before"]) or not np.array_equal(action,self.pending["action"]):
            raise ValueError("Observed step does not match its actual proposal and prior state")
        movement=float(np.linalg.norm(_area_center(self.letter,after)-_area_center(self.letter,before)))
        rotation=abs(float(after[4]-before[4]));contact=int(info.get("contact",0))>0
        near_static=movement<=.5 and rotation<=.005
        self.blocked_streak=self.blocked_streak+1 if contact and near_static else 0
        self.lost_streak=0 if contact else self.lost_streak+1
        points=np.asarray(info.get("contact_points",[]),dtype=float).reshape(-1,2)
        if len(points):
            from collections import Counter
            from shapely.geometry import Point
            votes=Counter()
            for xy in points:
                point=Point(*xy)
                if self.exterior.distance(point)>2:continue
                arc=float(self.exterior.project(point))
                match=next((i for i,row in enumerate(self.segments) if not row["ring"]
                    and row["arc_start"]<=arc<=row["arc_end"]+1e-6),None)
                if match is not None:votes[match]+=1
            if votes:self.active=min(votes,key=lambda i:(-votes[i],i!=self.active,i))
        result=dict(area_centroid_translation_world_px=movement,rotation_rad=rotation,
            moving=bool(movement>=2 or rotation>=math.pi/180),near_static=near_static,
            actual_contact=contact,physical_valid=bool(info.get("valid",False)),
            actual_wall_gaps=wall_gaps(self.env.world_geometry()),
            blocked_contact_streak=int(self.blocked_streak),lost_contact_streak=int(self.lost_streak))
        self.pending=None;self.index+=1
        return result
