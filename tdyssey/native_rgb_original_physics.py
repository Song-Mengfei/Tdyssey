"""Goal-free original PushT physics, with observation-only substep telemetry.

The pinned upstream PushTEnv.step executes the actual control/physics loop.
No finite-force controller, speed cap, half-life override or detached-body reset.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .native_rgb_env import make_env,load_physics_source,STATE_COLUMNS
from .native_rgb_r2a_env import _body_identity,_shape_geometry,_scalar,_xy,R2AStepAborted

PIN=json.loads((Path(__file__).resolve().parents[1]/'manifests/official_dp.json').read_text(encoding='utf-8'))
PROTOCOL=dict(name='original-kinematic-pusht-goalfree-rgb.v1',
    source_sha256={k:PIN['source_sha256'][k]for k in('diffusion_policy/env/pusht/pusht_env.py','diffusion_policy/env/pusht/pymunk_override.py')},
    pusher_body='KINEMATIC',pusher_radius=15,object_body='DYNAMIC',object_mass_inertia='unchanged official per-letter construction',
    space_damping=0.0,solver_iterations=10,physics_hz=100,control_hz=10,kp=100,kv=20,
    control_loop='Pinned PushTEnv.step: velocity += (kp*(action-position)+kv*(0-velocity))*dt; space.step(dt)',
    force_cap=None,pre_step_speed_cap=None,post_separation_velocity_reset=False,
    walls_materials='unchanged original source',renderer='unchanged native white/gray/blue96; no goal or UI overlay in RGB',
    step_guard='nonfinite only; penetration is observed, no added force/speed/position correction')
PROTOCOL_SHA256=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
LETTERS=tuple('TLOCXB')


class OriginalPhysicsEnv:
    def __init__(self,source_root,letter):
        if letter not in LETTERS:raise ValueError('Only the six previously trained letters are released')
        self.letter=letter;self.controller='original-upstream-kinematic-pd'
        self._base=make_env(source_root,letter=letter,controller='upstream-kinematic',render_size=96)
        self._upstream=load_physics_source(source_root);self._reset=False;self._terminal=False;self._in_step=False
        self.last_substeps=[];self._events=[];self._callback_error=None;self._action=None
    def __getattr__(self,name):return getattr(self._base,name)
    def __setattr__(self,name,value):
        if name in('latest_action','control_step')and'_base'in self.__dict__:setattr(self._base,name,value)
        else:object.__setattr__(self,name,value)
    def reset_scene(self,*args,**kwargs):
        if self._reset:raise RuntimeError('Start a new environment/session rather than switch physics mid-trajectory')
        self._reset=True;rgb=self._base.reset_scene(*args,**kwargs)
        import pymunk
        if self.agent.body_type!=pymunk.Body.KINEMATIC or self.block.body_type!=pymunk.Body.DYNAMIC:
            raise RuntimeError('Original body types changed')
        if self.space.damping!=0 or self.space.iterations!=10 or self.k_p!=100 or self.k_v!=20 or self.sim_hz!=100 or self.control_hz!=10:
            raise RuntimeError('Original controller/space configuration differs')
        self._native_step=self.space.step;self.space.step=self._observe_step
        handler=self.collision_handeler;self._native_pre=handler.pre_solve;self._native_post=handler.post_solve
        handler.pre_solve=self._pre;handler.post_solve=self._post
        self._initial=self._identity();return rgb
    def _role(self,body):return'agent'if body is self.agent else'object'if body is self.block else'wall'
    def _record(self,arbiter,phase):
        try:
            roles=[self._role(s.body)for s in arbiter.shapes];pair=frozenset(roles)
            kind={frozenset(('agent','object')):'agent_object',frozenset(('object','wall')):'object_wall',frozenset(('agent','wall')):'agent_wall'}.get(pair,'other')
            points=arbiter.contact_point_set
            value=dict(phase=phase,kind=kind,shapes_order=roles,normal=_xy(points.normal),points=[dict(point_a=_xy(p.point_a),point_b=_xy(p.point_b),distance=float(p.distance))for p in points.points])
            if phase=='post_solve':value.update(total_impulse=_xy(arbiter.total_impulse),total_ke=float(arbiter.total_ke))
            self._events.append(value)
        except Exception as exc:self._callback_error=repr(exc)
    def _pre(self,arbiter,space,data):
        result=self._native_pre(arbiter,space,data)if self._native_pre else True
        self._record(arbiter,'pre_solve');return result
    def _post(self,arbiter,space,data):
        if self._native_post:self._native_post(arbiter,space,data)
        self._record(arbiter,'post_solve')
    def signed_wall_gaps(self):
        x0,y0,x1,y1=self.world_geometry().bounds
        return dict(left=float(x0-7),right=float(504-x1),top=float(y0-7),bottom=float(504-y1))
    def _observe_step(self,dt):
        if not self._in_step or float(dt)!=.01:raise RuntimeError('Unexpected original physics entry')
        before=self.state().copy();self._events=[];self._callback_error=None
        row=dict(index=len(self.last_substeps),dt=float(dt),pre_state=before.tolist(),integrator_pre_state=before.tolist(),executed=False)
        try:
            self._native_step(dt);row['executed']=True
            row.update(post_state=self.state().tolist(),contacts=self._events,penetrations=self.penetrations().tolist(),signed_wall_gaps=self.signed_wall_gaps())
        finally:self.last_substeps.append(row)
        if self._callback_error:raise RuntimeError(self._callback_error)
    def step(self,action):
        import numpy as np
        if not self._reset or self._terminal or self._in_step:raise RuntimeError('A live original-physics session is required')
        action=np.asarray(action,dtype=np.float64)
        if action.shape!=(2,)or not np.isfinite(action).all()or(action<0).any()or(action>511).any():raise ValueError('Absolute finite targetxy in[0,511]required')
        self._action=action.copy();self.last_substeps=[];self._in_step=True
        self._base.contact_points=[];self._base.contact_angles=[];self._base.substep_penetrations=[]
        try:
            # Execute the pinned source method itself, not R2A/R2A2's integrator.
            self._upstream.step(self._base,action)
            self._base.control_step+=1
            if len(self.last_substeps)!=10 or not np.isfinite(self.state()).all():raise RuntimeError('Incomplete/nonfinite original control')
            info=dict(valid=True,physics_substeps=10,control_step=self.control_step,substeps=self.last_substeps,
                penetration=np.max([r['penetrations']for r in self.last_substeps],axis=0),
                protocol_sha256=PROTOCOL_SHA256,contact=len(self._base.contact_points),
                agent_object_positive_impulse_substeps=sum(any(e['kind']=='agent_object'and e['phase']=='post_solve'and np.linalg.norm(e['total_impulse'])>0 for e in r['contacts'])for r in self.last_substeps))
            for kind in('agent_object','object_wall','agent_wall'):
                info[kind+'_contact_substeps']=sum(any(e['kind']==kind and e['phase']=='post_solve'for e in r['contacts'])for r in self.last_substeps)
            return self.render_clean(),info
        except BaseException as exc:
            self._terminal=True
            if isinstance(exc,R2AStepAborted):raise
            raise R2AStepAborted(str(exc),dict(valid=False,aborted=True,substeps=self.last_substeps,control_step=self.control_step))from exc
        finally:self._in_step=False
    def _identity(self):
        space=self.space;shapes=[]
        for shape in space.shapes:
            filt=shape.filter
            shapes.append(dict(role=self._role(shape.body),body=_body_identity(shape.body),geometry=_shape_geometry(shape),friction=float(shape.friction),elasticity=float(shape.elasticity),sensor=bool(shape.sensor),collision_type=int(shape.collision_type),filter=dict(group=int(filt.group),categories=int(filt.categories),mask=int(filt.mask)),surface_velocity=_xy(shape.surface_velocity)))
        shapes.sort(key=lambda r:json.dumps(r,sort_keys=True))
        settings={k:_scalar(getattr(space,k))for k in('damping','iterations','collision_slop','collision_bias','collision_persistence','sleep_time_threshold','idle_speed_threshold')};settings['gravity']=_xy(space.gravity)
        physics=dict(controller=self.controller,kp=float(self.k_p),kv=float(self.k_v),control_dt=.1,substep_dt=.01,substeps_per_control=10,space=settings,agent=_body_identity(self.agent),object=_body_identity(self.block),shapes=shapes)
        return dict(kind='original-kinematic-native-RGB-identity.v1',letter=self.letter,protocol_sha256=PROTOCOL_SHA256,protocol=PROTOCOL,physics=physics,physics_sha256=hashlib.sha256(json.dumps(physics,sort_keys=True).encode()).hexdigest(),initial_state=self.state().tolist(),state_columns=list(STATE_COLUMNS),solver_cache_serialized=False)
    def initial_identity(self):
        if not hasattr(self,'_initial'):raise RuntimeError('Reset before identity')
        return json.loads(json.dumps(self._initial))
    def capture_state(self):return self._base.capture_state()
    def _physics_hash(self):return self._base._physics_hash()


def make_original_env(source_root,letter):return OriginalPhysicsEnv(source_root,letter)
