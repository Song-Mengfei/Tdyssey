"""Current-state wall-risk gate for CONTACT-PILOT2 collection actions.

This is a conservative kinematic heuristic, not a physics rollout, learned
dynamics component or mathematical safety guarantee. It never changes state,
forces, damping, limits, rendering or model inputs. Current union-boundary
velocities and the actual bounded pusher target are its only motion evidence.
"""
from __future__ import annotations

import math
import numpy as np
from shapely.geometry import Point

from .native_rgb_r2b_sampling import geometry_at
from .native_rgb_r3a_collection import rot,identity
from .native_rgb_selected_physics import SELECTED_DAMPING,SELECTED_PROTOCOL_SHA256


RULES=dict(kind='CONTACT-PILOT2-current-boundary-wall-risk.v1',
    protocol_sha256=SELECTED_PROTOCOL_SHA256,damping=SELECTED_DAMPING,
    control_dt=.1,physics_dt=.01,unchanged_pre_substep_speed_cap_world_px_s=200.,
    wall_inner_bounds_world_px=[7.,504.],pusher_radius_world_px=15.,
    pusher_contact_clearance_world_px=15.5,
    minimum_remaining_room_world_px=8.,
    boundary_velocity='bodyCOMvelocity + omega cross (union exterior vertex - physicalCOM)',
    damping_scope='analytic current unforced velocity integral; future contact forces are not predicted',
    rotation_remainder='0.5*maximum boundary radius*(abs(omega)*effective_dt)^2',
    active_travel='max(0,min(actual_target_distance,200*.1)-current_pusher_clearance)',
    active_rotation='current contact torque-sign and normalized lever allow conservative boundary sweep allowance',
    pushing_classification='actual action has inward contact-normal component and can reach current contact shell; mode label cannot bypass gate',
    retreat='an outward/no-new-push target can mitigate but does not certify inertial stopping',
    no_physics_rollout=True,no_state_edit=True,no_cap_or_guard_change=True,
    guarantee='heuristic only; finite sampling, unknown contact response and actuator force mean no mathematical safety guarantee')
RULES_SHA256=identity(RULES)

WALLS={
    'left':dict(toward=np.array((-1.,0.)),inward=[1.,0.]),
    'right':dict(toward=np.array((1.,0.)),inward=[-1.,0.]),
    'top':dict(toward=np.array((0.,-1.)),inward=[0.,1.]),
    'bottom':dict(toward=np.array((0.,1.)),inward=[0.,-1.]),
}


def _vertices(geometry):
    if geometry.geom_type=='Polygon':
        return np.asarray(geometry.exterior.coords[:-1],np.float64)
    return np.concatenate([np.asarray(poly.exterior.coords[:-1],np.float64)
                           for poly in geometry.geoms if poly.geom_type=='Polygon'])


def _contact_frame(state,contact):
    if contact is None:return None,None
    local=contact.get('point_local')
    normal=contact.get('outward_normal_local',contact.get('normal_local'))
    if local is None or normal is None:return None,None
    local,normal=np.asarray(local,np.float64),np.asarray(normal,np.float64)
    if local.shape!=(2,)or normal.shape!=(2,)or not np.isfinite(local).all()or not np.isfinite(normal).all():
        return None,None
    length=float(np.linalg.norm(normal))
    if length<=1e-9:return None,None
    rotation=rot(state[4])
    return state[2:4]+rotation@local,rotation@(normal/length)


def assess_wall_risk(letter,state,action,*,contact=None,mode='follow'):
    """Score the real bounded candidate; return compact JSON-safe evidence.

    A risky inward command should be cancelled/replaced by a separately geometry
    verified retreat/tangent route. A ``risky=False`` mitigation is not a claim
    that already accumulated body momentum can always be arrested in time.
    ``mode`` is recorded only; it cannot disguise an inward route as a retreat.
    """
    state=np.asarray(state,np.float64);action=np.asarray(action,np.float64)
    if state.shape!=(15,)or action.shape!=(2,)or not np.isfinite(state).all()or not np.isfinite(action).all():
        raise ValueError('Wall-risk gate requires finite current state15 and actual target2')
    if letter not in 'TLOCXB':raise ValueError('Only the six authorized pilot letters are exposed')
    if (action<0).any()or(action>511).any():raise ValueError('Assess the actual bounded absolute pusher target')
    geometry=geometry_at(letter,state);vertices=_vertices(geometry)
    local_cog=np.array((0.,45.))if letter=='T'else np.zeros(2)
    com=state[2:4]+rot(state[4])@local_cog
    offsets=vertices-com
    radii=np.linalg.norm(offsets,axis=1);radius=max(float(radii.max()),1e-9)
    velocity=state[7:9]+state[9]*np.column_stack((-offsets[:,1],offsets[:,0]))
    damping_rate=-math.log(RULES['damping'])
    effective_dt=-math.expm1(-damping_rate*RULES['control_dt'])/damping_rate
    angular_displacement=abs(float(state[9]))*effective_dt
    curvature_allowance=.5*radius*angular_displacement**2
    delta=action-state[:2];distance=float(np.linalg.norm(delta))
    direction=delta/distance if distance>1e-12 else np.zeros(2)
    free_clearance=max(0.,float(geometry.distance(Point(*state[:2])))-RULES['pusher_contact_clearance_world_px'])
    potential_travel=min(distance,RULES['unchanged_pre_substep_speed_cap_world_px_s']*RULES['control_dt'])
    active_travel=max(0.,potential_travel-free_clearance)
    contact_point,normal=_contact_frame(state,contact)
    if normal is None:
        inward_distance=distance
        pushing=active_travel>1e-9 and distance>1e-9
        lever_fraction=0.;torque_sign=0.
    else:
        inward_distance=float(-delta@normal)
        pushing=active_travel>1e-9 and inward_distance>.05
        lever=float(np.cross(contact_point-com,direction))
        lever_fraction=min(1.,abs(lever)/radius)
        torque_sign=float(np.sign(lever))
    wall_risks={};hazardous=[];state_hazardous=[];maximum_pressure=0.;state_pressure=0.
    for name,spec in WALLS.items():
        toward=spec['toward']
        if name=='left':gaps=vertices[:,0]-7.
        elif name=='right':gaps=504.-vertices[:,0]
        elif name=='top':gaps=vertices[:,1]-7.
        else:gaps=504.-vertices[:,1]
        closing=np.maximum(0.,velocity@toward)
        inertial=closing*effective_dt+curvature_allowance
        translational_drive=active_travel*max(0.,float(direction@toward))if pushing else 0.
        rotational_projection=torque_sign*(offsets[:,0]*toward[1]-offsets[:,1]*toward[0])/radius
        rotational_drive=active_travel*lever_fraction*np.maximum(0.,rotational_projection)if pushing else np.zeros(len(vertices))
        # Summing the two geometric allowances errs conservatively. It is not a
        # claimed estimate of physical impulse transfer, force or future velocity.
        active=translational_drive+rotational_drive
        state_remaining=gaps-inertial
        remaining=state_remaining-active
        critical=int(np.argmin(remaining));state_critical=int(np.argmin(state_remaining))
        min_remaining=float(remaining[critical]);min_state_remaining=float(state_remaining[state_critical])
        clearance=float(gaps.min());closing_max=float(closing.max())
        state_bad=min_state_remaining<RULES['minimum_remaining_room_world_px']
        command_bad=pushing and min_remaining<RULES['minimum_remaining_room_world_px']
        if state_bad:state_hazardous.append(name)
        if command_bad:hazardous.append(name)
        pressure=max(0.,(RULES['minimum_remaining_room_world_px']-min_remaining)/RULES['minimum_remaining_room_world_px'])if pushing else 0.
        maximum_pressure=max(maximum_pressure,pressure)
        state_pressure=max(state_pressure,max(0.,(RULES['minimum_remaining_room_world_px']-min_state_remaining)/RULES['minimum_remaining_room_world_px']))
        critical_speed=float(closing[critical])
        wall_risks[name]=dict(gap_world_px=clearance,
            critical_vertex_world=vertices[critical].tolist(),critical_vertex_gap_world_px=float(gaps[critical]),
            critical_vertex_velocity_world_px_s=velocity[critical].tolist(),
            closing_boundary_speed_world_px_s=closing_max,
            critical_closing_speed_world_px_s=critical_speed,
            inertial_closing_displacement_world_px=float(inertial[critical]),
            active_drive_displacement_world_px=float(active[critical]),
            active_translation_displacement_world_px=float(translational_drive),
            active_rotation_displacement_world_px=float(rotational_drive[critical]),
            projected_remaining_room_world_px=min_remaining,
            inertial_remaining_room_world_px=min_state_remaining,
            time_to_wall_seconds=None if critical_speed<=1e-12 else max(0.,float(gaps[critical])/critical_speed),
            time_to_wall_note='no current closing velocity'if critical_speed<=1e-12 else'current velocity ratio only, not collision forecast',
            command_risky=bool(command_bad),state_risky=bool(state_bad),
            toward_wall=toward.tolist(),preferred_inward_direction=spec['inward'])
    preferred=[]
    for name in hazardous:
        inward=WALLS[name]['inward'];tangent=[-inward[1],inward[0]]
        preferred.append(dict(wall=name,inward_direction=inward,tangent_directions=[tangent,[-tangent[0],-tangent[1]]]))
    return dict(kind='CONTACT-PILOT2-wall-risk-observation.v1',rules_sha256=RULES_SHA256,
        risky=bool(hazardous),cancel_push=bool(hazardous),state_risky=bool(state_hazardous),
        pushing_command=bool(pushing),hazardous_walls=hazardous,state_hazardous_walls=state_hazardous,
        risk_score=float(maximum_pressure),state_risk_score=float(state_pressure),wall_risks=wall_risks,
        preferred_escape_directions=preferred,mode_label=str(mode),
        original_candidate_action=action.tolist(),pusher_position=state[:2].tolist(),
        pusher_velocity=state[5:7].tolist(),object_COM_velocity=state[7:9].tolist(),
        object_angular_velocity=float(state[9]),physical_COM_world=com.tolist(),
        actual_target_distance_world_px=distance,current_pusher_free_clearance_world_px=free_clearance,
        active_contact_travel_allowance_world_px=active_travel,
        actual_target_inward_normal_distance_world_px=float(inward_distance),
        damping_effective_seconds=float(effective_dt),rotation_curvature_allowance_world_px=float(curvature_allowance),
        union_exterior_vertices_checked=len(vertices),minimum_remaining_room_world_px=RULES['minimum_remaining_room_world_px'],
        contact_available=normal is not None,mitigation_does_not_prove_safety=True,
        heuristic_not_mathematical_guarantee=True,physics_rollout_steps=0,state_modified=False)


def action_risk_rank(risk):
    """Deterministic ranking helper; geometry validity remains the caller's job."""
    return (bool(risk['risky']),float(risk['risk_score']))
