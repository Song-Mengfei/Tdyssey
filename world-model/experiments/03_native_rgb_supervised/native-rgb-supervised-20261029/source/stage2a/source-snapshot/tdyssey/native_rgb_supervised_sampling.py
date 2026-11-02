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
ALLOWED_TRAINING_LETTERS=tuple("ABCEFHIJLMOPSTUVXY")
SIDE_SIGN_LETTERS=(*LETTERS,*sorted(set(ALLOWED_TRAINING_LETTERS)-set(LETTERS)))
SIDE_SIGN_INDEX={letter:index for index,letter in enumerate(SIDE_SIGN_LETTERS)}
LOCATIONS=("center","left","right","top","bottom","left-top","right-top","left-bottom","right-bottom")
PHASES=((0,4,"approach"),(4,40,"push"),(40,48,"sliding"),
        (48,52,"retreat"),(52,56,"recontact"),(56,64,"push_again"))
RECOVERY_LIMITS=dict(staging_arrival_tolerance_world_px=4.,maximum_free_lookahead_world_px=24.,
    progress_tolerance_world_px=.25,no_progress_controls=4,path_controls=12,
    final_contact_controls=8,episode_controls=20,alternative_switches=1,
    sustained_contact_controls=3,contact_moving_controls=2,maximum_episodes=4)
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
    if letter not in ALLOWED_TRAINING_LETTERS or spatial_class not in LOCATIONS:raise ValueError("Unauthorized stage-one scene")
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
    from shapely.geometry import box,LineString,Point
    free=box(22.5,22.5,488.5,488.5).difference(geometry.buffer(15.5,resolution=32)).buffer(1e-7)
    if math.hypot(end[0]-start[0],end[1]-start[1])<1e-12:return bool(free.covers(Point(*start)))
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
        self.side_sign=1 if (LOCATIONS.index(scene["spatial_class"])+SIDE_SIGN_INDEX[self.letter])%2 else -1
        self.exterior=shape_geometry(self.letter).exterior
        self.active=int(scene["staging_segment"]);self.pending=None;self.index=0
        self.blocked_streak=0;self.lost_streak=0;self.escape=None;self.escapes=0;self.cooldown=-1
        self.route_arc=None
        self.recovery=None;self.last_recovery=None;self.recovery_serial=0;self.recovery_cooldown=-1
        self.scheduled_recontact_started=False;self.recovery_events=[]
        self.failed_recovery_arcs=set()
        self.rng=__import__("numpy").random.default_rng(scene["seed"]+100000)

    def _normal(self,state):return _rotation(float(state[4]))@__import__("numpy").asarray(self.segments[self.active]["normal"])

    def _retreat(self,normal,distance):
        import numpy as np
        start=np.asarray(self.env.state()[:2]);tangent=np.asarray((-normal[1],normal[0]));geometry=self.env.world_geometry()
        for direction in (normal,(normal+tangent)/math.sqrt(2),(normal-tangent)/math.sqrt(2),tangent,-tangent):
            target=np.clip(start+direction*distance,22.5,488.5)
            if np.linalg.norm(target-start)>=2 and _departure_route(geometry,start,target):return target,"verified_contact_exit"
        return start,"retreat_path_unconfirmed_or_wall_blocked"

    def _recontact(self,prefer_alternate=False,excluded_arcs=()):
        import numpy as np
        start=np.asarray(self.env.state()[:2]);geometry=self.env.world_geometry()
        candidates=[row for row in _reachable_segments(self.env,self.segments) if row["status"]=="reachable_exterior_component"]
        candidates=[row for row in candidates if _free_route(geometry,start,row["staging"])]
        candidates=[row for row in candidates if row["segment_id"] not in excluded_arcs]
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

    def _recovery_snapshot(self):
        record=self.recovery or (self.last_recovery if self.last_recovery is not None
            and self.last_recovery["end_step"]==self.index else None)
        if record is None:return None
        return {key:value for key,value in dict(record,events=list(self.recovery_events),
            active=self.recovery is not None,maximum_episodes=RECOVERY_LIMITS["maximum_episodes"],
            failed_arc_attempts_disabled=sorted(self.failed_recovery_arcs)).items()
            if key not in ("before_distance","before_state")}

    def _begin_recovery(self,step,reason,prefer_alternate=False):
        import numpy as np
        self.recovery_serial+=1;self.route_arc=None
        waypoint,evidence=self._recontact(prefer_alternate,excluded_arcs=self.failed_recovery_arcs)
        distance=float(np.linalg.norm(waypoint-self.env.state()[:2]))
        self.recovery=dict(episode_id=self.recovery_serial,state="path",stage="path",status="running",
            reason=reason,start_step=int(step),end_step=None,failure_reason=None,
            arc=int(self.active),controls=0,path_controls=0,final_contact_controls=0,
            route_remaining_world_px=distance,progress_world_px=0.,no_progress_controls=0,
            route_completed=False,staging_arrived=False,staging_arrival_step=None,
            staging_controls=0,time_since_staging_controls=0,
            first_real_contact_step=None,consecutive_contact_controls=0,max_consecutive_contact_controls=0,
            contact_confirmed=False,
            contact_controls=0,post_contact_moving_controls=0,post_contact_contact_moving_controls=0,
            alternative_switches=0,route_evidence=evidence,initial_route_available=evidence.startswith("verified"))
        self.recovery_events.append("recovery_started")

    def _end_recovery(self,step,status,reason=None):
        if self.recovery is None:return
        record=self.recovery;record["status"]=status;record["end_step"]=int(step)
        record["failure_reason"]=reason
        record["state"]="sustained_moving" if status=="success" else "failed"
        record["stage"]=record["state"]
        if status!="success":self.failed_recovery_arcs.add(int(record["arc"]))
        self.recovery_events.append("sustained_moving_contact" if status=="success" else "recovery_failed:"+str(reason))
        self.last_recovery=record;self.recovery=None;self.recovery_cooldown=step+6

    def _recovery_waypoint(self):
        import numpy as np
        record=self.recovery;self.active=int(record["arc"])
        row=self.segments[self.active]
        normal=self._normal(self.env.state())
        point=np.asarray(self.env.block.local_to_world(row["point"]))
        return point+normal*18,point+normal*15.5,normal

    def _recovery_action(self,step):
        """Free path first; only a separately checked final approach pushes inward."""
        import numpy as np
        from shapely.geometry import Point
        record=self.recovery;start=np.asarray(self.env.state()[:2]);geometry=self.env.world_geometry()
        waypoint,contact_center,normal=self._recovery_waypoint()
        remaining=float(np.linalg.norm(waypoint-start));record["route_remaining_world_px"]=remaining
        record["before_distance"]=remaining
        if record["state"]=="path" and remaining<=RECOVERY_LIMITS["staging_arrival_tolerance_world_px"]:
            record.update(state="staging",stage="staging",staging_arrived=True,route_completed=True,
                staging_arrival_step=int(step))
            self.recovery_events.append("staging_arrived")
        if record["state"]=="path":
            if not record["initial_route_available"] or not _free_route(geometry,start,waypoint):
                record["pending_failure"]="unsafe_or_unavailable_free_path"
                return start,"recovery_path", "unverified_path_not_executed"
            direction=(waypoint-start)/max(remaining,1e-12)
            for lead in (24,16,8,4,2,0):
                target=waypoint+direction*min(float(lead),RECOVERY_LIMITS["maximum_free_lookahead_world_px"])
                if _free_route(geometry,start,target):
                    record["command_free_path_verified"]=True
                    record["lookahead_world_px"]=float(lead)
                    return target,"recovery_path","entire_bounded_lookahead_command_is_exterior_free"
            record["pending_failure"]="no_safe_lookahead_or_staging_path"
            return start,"recovery_path","unverified_path_not_executed"
        if record["state"] in ("staging","final_contact"):
            from shapely.geometry import LineString
            from shapely.ops import nearest_points
            # Always command inward, including when the pusher is already
            # closer than the conservative contact-center. Validate the true
            # clipped command's exterior prefix, not an unrelated two-leg path.
            target=np.clip(start-normal*float(self.settings["action_distance_world_px"]["mid"]),0,511)
            delta=target-start
            inward=bool(delta.dot(-normal)>0)
            start_point=Point(*start);inflated=geometry.buffer(15.5,resolution=32)
            approach_clear=False
            if inward and not inflated.contains(start_point):
                hit=LineString((start,target)).intersection(inflated)
                if not hit.is_empty:
                    first=nearest_points(start_point,hit)[1]
                    approach_clear=_free_route(geometry,start,(first.x,first.y))
            elif inward and not geometry.contains(start_point) and geometry.distance(start_point)>=13 and remaining<=4:
                # The pusher already lies in the <=2px solver-contact shell;
                # the exterior prefix is empty. This still does not establish
                # actual contact until the collision callback reports it.
                wall_overlap=max(0,22-start[0],22-start[1],start[0]-489,start[1]-489)
                approach_clear=wall_overlap<=2
            if not approach_clear:
                record["pending_failure"]="unsafe_final_contact_approach"
                return start,"recovery_final_contact","unverified_final_approach_not_executed"
            record["final_approach_verified"]=True
            return target,"recovery_final_contact",\
                "exterior_path_to_contact_verified_then_fixed_physics_handles_entity_contact"
        # Actual collision feedback, not arrival at a waypoint, enables this.
        normal=self._normal(self.env.state())
        return start-normal*float(self.settings["action_distance_world_px"]["far"]),"recovery_confirmed_contact",\
            "real_collision_confirmed_persistent_inward_push"

    def _observe_recovery(self,step,before,after,contact,moving,valid):
        import numpy as np
        from shapely.geometry import Point,LineString
        record=self.recovery
        if record is None:return
        record["controls"]+=1
        if not valid:
            self._end_recovery(step,"failed","invalid_physical_transition");return
        if record["state"]=="path":record["path_controls"]+=1
        if record["state"]=="staging":record["staging_controls"]+=1
        if record["staging_arrival_step"] is not None:
            record["time_since_staging_controls"]=max(0,step-record["staging_arrival_step"]+1)
        if record["state"] in ("staging","final_contact"):record["final_contact_controls"]+=1
        if contact:
            if record["first_real_contact_step"] is None:
                record["first_real_contact_step"]=int(step);self.recovery_events.append("first_real_contact")
            record["contact_confirmed"]=True
            record["contact_controls"]+=1;record["consecutive_contact_controls"]+=1
            record["max_consecutive_contact_controls"]=max(record["max_consecutive_contact_controls"],record["consecutive_contact_controls"])
            record["arc"]=int(self.active);record["state"]="confirmed_contact";record["stage"]="confirmed_contact"
        else:
            record["consecutive_contact_controls"]=0
            if record["state"]=="confirmed_contact":record["state"]="final_contact";record["stage"]="final_contact"
        if record["first_real_contact_step"] is not None and step>record["first_real_contact_step"] and moving:
            record["post_contact_moving_controls"]+=1
            record["post_contact_contact_moving_controls"]+=int(contact)
        if (record["consecutive_contact_controls"]>=RECOVERY_LIMITS["sustained_contact_controls"]
                and record["post_contact_contact_moving_controls"]>=RECOVERY_LIMITS["contact_moving_controls"]):
            self._end_recovery(step,"success");return
        if "pending_failure" in record:
            self._end_recovery(step,"failed",record.pop("pending_failure"));return
        if record["state"]=="path":
            waypoint,_,_=self._recovery_waypoint()
            remaining=float(np.linalg.norm(waypoint-after[:2]));progress=float(record["before_distance"]-remaining)
            record.update(route_remaining_world_px=remaining,progress_world_px=progress)
            record["no_progress_controls"]=record["no_progress_controls"]+1 if progress<RECOVERY_LIMITS["progress_tolerance_world_px"] else 0
            traveled=float(np.linalg.norm(after[:2]-before[:2]))
            crossed=(remaining if traveled<1e-12 else LineString((before[:2],after[:2])).distance(Point(*waypoint)))<=RECOVERY_LIMITS["staging_arrival_tolerance_world_px"]
            if remaining<=RECOVERY_LIMITS["staging_arrival_tolerance_world_px"] or crossed:
                record.update(state="staging",stage="staging",staging_arrived=True,route_completed=True,
                    staging_arrival_step=int(step));self.recovery_events.append("staging_arrived")
            elif record["no_progress_controls"]>=RECOVERY_LIMITS["no_progress_controls"]:
                if record["alternative_switches"]<RECOVERY_LIMITS["alternative_switches"]:
                    self.failed_recovery_arcs.add(int(record["arc"]))
                    self.route_arc=None;waypoint,evidence=self._recontact(prefer_alternate=True,excluded_arcs=self.failed_recovery_arcs)
                    record["alternative_switches"]+=1;record["arc"]=int(self.active)
                    record["initial_route_available"]=evidence.startswith("verified")
                    record["route_evidence"]=evidence
                    record["no_progress_controls"]=0;self.recovery_events.append("legal_alternative_requested")
                else:self._end_recovery(step,"failed","no_path_progress");return
        elif record["state"]=="staging":record["state"]="final_contact";record["stage"]="final_contact"
        if record["state"]=="path" and record["path_controls"]>=RECOVERY_LIMITS["path_controls"] and record["first_real_contact_step"] is None:
            self._end_recovery(step,"failed","path_timeout");return
        if record["final_contact_controls"]>=RECOVERY_LIMITS["final_contact_controls"] and record["first_real_contact_step"] is None:
            self._end_recovery(step,"failed","no_real_contact_after_staging");return
        if record["controls"]>=RECOVERY_LIMITS["episode_controls"]:
            self._end_recovery(step,"failed","episode_timeout");return
        if step==63:self._end_recovery(step,"failed","trajectory_budget_exhausted")

    def next_action(self,step):
        import numpy as np
        if step!=self.index or self.pending is not None or not 0<=step<64:
            raise ValueError("Proposal/observation must follow the continuous 64-step timeline")
        state=np.asarray(self.env.state());normal=self._normal(state);tangent=np.asarray((-normal[1],normal[0]))
        self.recovery_events=[]
        phase=next(name for start,end,name in PHASES if start<=step<end)
        category="near" if phase=="approach" else "mid" if phase in ("sliding","retreat","recontact") else "far"
        distance=float(self.settings["action_distance_world_px"][category]);approach="normal";route="physical_push_target_may_be_inside_object"
        if phase=="push" and self.recovery is None and self.blocked_streak>=3 and step<32 and step>=self.cooldown and self.escapes<2 and self.escape is None:
            self.escape=dict(stage="retreat",remaining=3);self.escapes+=1;self.cooldown=step+12
        if phase not in ("push",) and self.recovery is None:self.escape=None
        if self.recovery is None and self.recovery_serial<RECOVERY_LIMITS["maximum_episodes"] and step>=self.recovery_cooldown:
            if self.escape is not None and self.escape["stage"]=="recontact":
                self._begin_recovery(step,"blocked_escape",True);self.escape=None
            elif phase=="recontact" and not self.scheduled_recontact_started:
                self.scheduled_recontact_started=True;self._begin_recovery(step,"scheduled_recontact")
            elif phase in ("push","push_again") and self.lost_streak>=3 and self.escape is None:
                self._begin_recovery(step,"lost_contact")
        if self.recovery is not None:
            reason=self.recovery["reason"];target,approach,route=self._recovery_action(step)
            phase="recontact" if reason=="scheduled_recontact" else "blocked_escape_recontact" if reason=="blocked_escape" else "contact_recovery_route"
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
            target=state[:2];route="recovery_attempt_inactive_or_bounded_limit; no_waypoint_claimed_success";approach="recovery_disabled_hold"
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
        metadata["recovery"]=self._recovery_snapshot()
        metadata["recovery_disabled_reason"]=("maximum_episode_attempts" if self.recovery_serial>=RECOVERY_LIMITS["maximum_episodes"]
            else "failure_cooldown" if step<self.recovery_cooldown else None) if self.recovery is None else None
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
        self._observe_recovery(self.index,before,after,contact,
            bool(movement>=2 or rotation>=math.pi/180),bool(info.get("valid",False)))
        result["recovery"]=self._recovery_snapshot()
        self.pending=None;self.index+=1
        return result
