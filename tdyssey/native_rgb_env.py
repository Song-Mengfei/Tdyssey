"""Goal-free RGB observations and explicitly versioned, bounded-force PushT control.

Geometry is the existing official alphabet.  The finite-force controller is a
material physics change from the upstream kinematic pusher; it is not presented
as upstream simulator parity.  No geometry/state from this module is model input.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

from .envs.official import make_official_shape_env_class, official_shape

CONTROLLER = "finite-force-pd-v1"
CONTROL_PROTOCOL = dict(name=CONTROLLER, control_hz=10, physics_hz=100,
    target="absolute_xy_0_511", pusher_body="dynamic", pusher_mass=1.0,
    pusher_radius=15.0, pusher_moment="infinite_no_spin", kp=100.0, kv=20.0, force_limit=10000.0,
    pre_step_speed_limit=200.0, solver_iterations=50, collision_slop=0.1,
    object_geometry="unchanged_official_alphabet", object_inertia="unchanged",
    walls="unchanged_segments_inner_7_504", damping="upstream_0")
PROTOCOL_SHA256 = hashlib.sha256(json.dumps(CONTROL_PROTOCOL,sort_keys=True).encode()).hexdigest()
STATE_COLUMNS = ("pusher_x", "pusher_y", "object_x", "object_y", "angle",
    "pusher_vx", "pusher_vy", "object_vx", "object_vy", "omega",
    "pusher_fx", "pusher_fy", "object_fx", "object_fy", "object_torque")


def load_physics_source(source_root):
    """Verify just the two needed pinned files; avoid importing policy packages."""
    root = Path(source_root).resolve()
    spec = json.loads((Path(__file__).resolve().parents[1]/"manifests/official_dp.json").read_text())
    for relative in ("diffusion_policy/env/pusht/pusht_env.py",
                     "diffusion_policy/env/pusht/pymunk_override.py"):
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=spec["source_sha256"][relative]:
            raise ValueError(f"Physics source differs from pinned source: {relative}")
    existing = sys.modules.get("diffusion_policy")
    if existing and {Path(p).resolve() for p in existing.__path__}!={root/"diffusion_policy"}:
        raise ValueError("A different diffusion_policy is already imported")
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    return importlib.import_module("diffusion_policy.env.pusht.pusht_env").PushTEnv


def shape_geometry(letter):
    from shapely.geometry import box
    from shapely.ops import unary_union
    return unary_union([box(r.x0,r.y0,r.x1,r.y1) for r in official_shape(letter).components])


def rotational_split_audit(train_letters, unseen_letters):
    """Axis-aligned polygon edges restrict exact rotational matches to 90deg."""
    from shapely import affinity
    shapes = {}
    for letter in (*train_letters,*unseen_letters):
        geom=shape_geometry(letter)
        shapes[letter]=affinity.translate(geom,-geom.centroid.x,-geom.centroid.y)
    matches=[]
    for seen in train_letters:
        for unseen in unseen_letters:
            for degrees in (0,90,180,270):
                other=affinity.rotate(shapes[unseen],degrees,origin=(0,0))
                if shapes[seen].symmetric_difference(other).area < 1e-7:
                    matches.append(dict(seen=seen,unseen=unseen,degrees=degrees))
    return dict(passed=not matches,matches=matches,reflection_included=False,
                method="centroid-normalized exact polygon union; 0/90/180/270 rotations; no scaling")


def make_env(source_root, letter="T", controller=CONTROLLER, render_size=96):
    import numpy as np
    import pymunk
    from shapely import affinity
    Base=make_official_shape_env_class(load_physics_source(source_root))
    if letter not in tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise ValueError("Only the 26 existing fixed capital letters are supported")
    if controller not in (CONTROLLER,"upstream-kinematic"):
        raise ValueError("Unknown controller")

    class NativeRGBEnv(Base):
        def seed(self,seed=None):
            # Upstream's None default draws from global NumPy. Render-worker
            # construction must not consume training/resume RNG nondeterministically.
            return super().seed(0 if seed is None else seed)

        def __init__(self):
            self.controller=controller
            self.local_geometry=shape_geometry(letter)
            self.contact_points=[]
            self.contact_angles=[]
            self._pre_contact_angles={}
            self.substep_penetrations=[]
            super().__init__(shape=letter,render_size=render_size,render_action=False)

        def add_circle(self,position,radius):
            if self.controller=="upstream-kinematic":
                return super().add_circle(position,radius)
            import pygame
            body=pymunk.Body(1.0,float("inf"))
            body.position=position
            body.friction=1
            shape=pymunk.Circle(body,radius)
            shape.color=pygame.Color("RoyalBlue")
            self.space.add(body,shape)
            return body

        def _setup(self):
            super()._setup()
            if self.controller==CONTROLLER:
                self.space.iterations=CONTROL_PROTOCOL["solver_iterations"]
                self.space.collision_slop=CONTROL_PROTOCOL["collision_slop"]
            self.collision_handeler.post_solve=self._contacts
            self.collision_handeler.pre_solve=self._pre_contacts

        def _pre_contacts(self,arbiter,space,data):
            if {s.body for s in arbiter.shapes}=={self.agent,self.block}:
                normal=arbiter.contact_point_set.normal
                if arbiter.shapes[0].body!=self.block:normal=-normal
                tangent=pymunk.Vec2d(-normal.y,normal.x)
                angles=[]
                for contact in arbiter.contact_point_set.points:
                    p=contact.point_a if arbiter.shapes[0].body==self.block else contact.point_b
                    relative=self.agent.velocity_at_world_point(p)-self.block.velocity_at_world_point(p)
                    angles.append(math.atan2(relative.dot(tangent),-relative.dot(normal)))
                self._pre_contact_angles[tuple(arbiter.shapes)]=angles
            return True

        def _contacts(self,arbiter,space,data):
            bodies={s.body for s in arbiter.shapes}
            if self.agent in bodies and self.block in bodies:
                for index,contact in enumerate(arbiter.contact_point_set.points):
                    p=contact.point_a if arbiter.shapes[0].body==self.block else contact.point_b
                    self.contact_points.append(tuple(self.block.world_to_local(p)))
                    self.contact_angles.append(self._pre_contact_angles[tuple(arbiter.shapes)][index])

        def reset_scene(self,center=(256,256),angle=0.0,pusher=(256,400)):
            """Set a new independent initial scene, without hidden physics steps."""
            self._setup()
            self.block.angle=float(angle)
            offset=self.block.local_to_world(self.block.center_of_gravity)-self.block.position
            self.block.position=tuple(np.asarray(center,dtype=float)-np.asarray(offset))
            self.agent.position=tuple(pusher)
            self.space.reindex_shapes_for_body(self.agent)
            self.space.reindex_shapes_for_body(self.block)
            self.control_step=0
            self.latest_action=None
            self.contact_points=[]
            self.contact_angles=[]
            self.substep_penetrations=[]
            return self.render_clean()

        def state(self):
            return np.asarray((*self.agent.position,*self.block.position,self.block.angle,
                *self.agent.velocity,*self.block.velocity,self.block.angular_velocity,
                *self.agent.force,*self.block.force,self.block.torque),dtype=np.float64)

        def set_render_state(self,state):
            """Recorded-state renderer only; never used to continue a trajectory."""
            state=np.asarray(state,dtype=np.float64)
            if state.shape!=(15,) or not np.isfinite(state).all(): raise ValueError("Invalid recorded state")
            self.agent.position=tuple(state[:2])
            self.block.angle=state[4]
            self.block.position=tuple(state[2:4])
            self.agent.velocity=tuple(state[5:7])
            self.block.velocity=tuple(state[7:9])
            self.block.angular_velocity=state[9]
            self.agent.force=tuple(state[10:12])
            self.block.force=tuple(state[12:14])
            self.block.torque=state[14]

        def world_geometry(self):
            geom=affinity.rotate(self.local_geometry,float(self.block.angle),origin=(0,0),use_radians=True)
            return affinity.translate(geom,*self.block.position)

        def penetrations(self):
            from shapely.geometry import Point
            geom=self.world_geometry()
            minx,miny,maxx,maxy=geom.bounds
            wall=max(0,7-minx,7-miny,maxx-504,maxy-504)
            p=Point(*self.agent.position)
            signed=-geom.boundary.distance(p) if geom.contains(p) else geom.distance(p)
            object_pusher=max(0,15-signed)
            x,y=self.agent.position
            pusher_wall=max(0,22-x,22-y,x-489,y-489)
            return np.asarray((wall,object_pusher,pusher_wall),dtype=np.float64)

        def step(self,action):
            action=np.asarray(action,dtype=np.float64)
            if action.shape!=(2,) or not np.isfinite(action).all() or (action<0).any() or (action>511).any():
                raise ValueError("Absolute action must be finite xy in [0,511]")
            self.latest_action=action.copy()
            self.contact_points=[]
            self.contact_angles=[]
            self.substep_penetrations=[]
            for _ in range(10):
                velocity=np.asarray(self.agent.velocity)
                acceleration=self.k_p*(action-np.asarray(self.agent.position))-self.k_v*velocity
                if self.controller==CONTROLLER:
                    force=self.agent.mass*acceleration
                    norm=float(np.linalg.norm(force))
                    if norm>CONTROL_PROTOCOL["force_limit"]: force*=CONTROL_PROTOCOL["force_limit"]/norm
                    self.agent.force=tuple(force)
                    speed=float(np.linalg.norm(velocity))
                    if speed>CONTROL_PROTOCOL["pre_step_speed_limit"]:
                        self.agent.velocity=tuple(velocity*CONTROL_PROTOCOL["pre_step_speed_limit"]/speed)
                else:
                    self.agent.velocity=tuple(velocity+acceleration*.01)
                self.space.step(.01)
                self.substep_penetrations.append(self.penetrations())
            self.control_step+=1
            penetration=np.max(self.substep_penetrations,axis=0)
            return self.render_clean(),dict(contact=len(self.contact_points),
                contact_points=np.asarray(self.contact_points,dtype=np.float32).reshape(-1,2),
                contact_angles=np.asarray(self.contact_angles,dtype=np.float32),
                penetration=penetration,valid=bool(np.isfinite(self.state()).all() and np.max(penetration)<=2),
                physics_substeps=10,control_step=self.control_step)

        def render_layers(self,size=None):
            import cv2
            import pygame
            size=size or self.render_size
            mask=np.zeros((512,512),dtype=np.uint8)
            for polygon in self.block.shapes:
                vertices=np.asarray([self.block.local_to_world(v) for v in polygon.get_vertices()])
                cv2.fillPoly(mask,[np.rint(vertices).astype(np.int32)],255)
            rgb=np.full((512,512,3),255,dtype=np.uint8)
            rgb[mask>0]=(119,136,153)
            cv2.circle(rgb,tuple(np.rint(self.agent.position).astype(int)),15,(65,105,225),-1)
            small=cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA)
            object_mask=cv2.resize(mask,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)/255
            return dict(rgb=np.moveaxis(small,-1,0),object_mask=object_mask,full_object_mask=object_mask,
                        mask512=mask.astype(np.float32)/255)

        def render_clean(self): return self.render_layers()["rgb"]

    return NativeRGBEnv()
