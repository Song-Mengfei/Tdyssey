"""Requested-action entry latch; no environment/control/physics replacement.

Calibration-v1 saved records show repeated staging returns before contact under
the original slower PD. Once a valid entry request begins, this policy keeps
requesting that tracked entry target until real selected contact is confirmed.
It preserves the original controller and every requested/executed action record.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

from . import original_contact_pilot as saved
from .strict_original_contact_pilot import RequestedTargetPolicy as V1Policy
from .native_rgb_r2b_sampling import geometry_at, contact_frame, _stage, _room

POLICY_VERSION = 'strict-original-requested-target-policy.v2'


class RequestedTargetPolicyV2(V1Policy):
    def __init__(self, env, slot, rules=None):
        super().__init__(env, slot, rules)
        self.entry_started = False

    def next_action(self, index):
        self.last_route_kind = None
        state = np.asarray(self.env.state(), np.float64)
        geometry = geometry_at(self.slot['letter'], state)
        phase = saved.phase_for(index, self.slot['scenario'])
        start = state[:2]
        if phase != self.last_phase:
            self.release_target = None
            if phase in ('approach', 'withdraw', 'recontact'):
                self.awaiting_acquisition = True
                self.acquisition_no_route = 0
                self.entry_started = False
        self.last_phase = phase
        _, normal, tangent = contact_frame(state, self.contact)
        direction = -normal+self.contact['drive_tangent_ratio']*tangent
        direction /= np.linalg.norm(direction)
        prior_room = _room(geometry, direction)
        reason = ('geometric_drive_room_under24' if phase == 'push' and prior_room < self.rules['wall_room_refresh_world_px'] else
            'wall_contact_stall3' if self.stall >= 3 else
            'acquisition_no_route3' if self.awaiting_acquisition and self.acquisition_no_route >= 3 else
            'lost_contact3' if not self.awaiting_acquisition and self.lost >= 3 else None)
        changed = None
        if phase in ('push', 'recontact') and reason is not None:
            fresh = saved.select_contact(self.slot['letter'], state, self.slot['scenario'],
                self.slot['arc_start']+index*3, self.slot['initial_space'], exclude_arc=int(self.contact['arc_id']))
            if fresh is not None:
                self.contact = fresh
                self.refresh_count += 1
                self.awaiting_acquisition = True
                self.acquisition_no_route = 0
                self.lost = self.stall = 0
                self.entry_started = False
                changed = reason
                # departure_contact deliberately remains the last actually
                # confirmed contact, including during target refresh/routing.
        point, normal, tangent = contact_frame(state, self.contact)
        direction = -normal+self.contact['drive_tangent_ratio']*tangent
        direction /= np.linalg.norm(direction)
        selected_room = _room(geometry, direction)
        staging = _stage(geometry, point, normal)
        raw, mode = start.copy(), 'hold'
        if phase in ('static', 'stop'):
            pass
        elif phase == 'withdraw':
            if self.release_target is None:
                for distance in self.rules['departure_distances_world_px']:
                    candidate = point+normal*distance
                    if self.route(geometry, start, candidate) is not None:
                        self.release_target = candidate
                        break
            if self.release_target is not None:
                route = self.route(geometry, start, self.release_target)
                if route is not None:
                    raw, mode = np.asarray(route[1]), 'free_route'
        elif staging is not None:
            stage, center, _ = staging
            # A contact approach has two ordered states. Returning to staging is
            # useful only before entry begins, not while the PD moves inward.
            route_to_stage = (self.awaiting_acquisition and not self.entry_started
                and np.linalg.norm(start-center) > 8 and np.linalg.norm(start-stage) > 4)
            if route_to_stage:
                route = self.route(geometry, start, stage)
                if route is not None:
                    raw, mode = np.asarray(route[1]), 'free_route'
                    self.acquisition_no_route = 0
                else:
                    self.acquisition_no_route += 1
            else:
                entering = self.awaiting_acquisition
                advance = self.rules['entry_advance_world_px'] if entering or phase != 'push' else self.rules['push_advance_world_px']
                beta = 0. if entering else self.contact['drive_tangent_ratio']
                raw = center-normal*advance+tangent*advance*beta
                mode = 'contact_entry_intent' if entering else 'contact_push_intent'
                self.acquisition_no_route = 0
                if entering:
                    self.entry_started = True
        elif phase in ('approach', 'push', 'recontact'):
            self.acquisition_no_route += 1
        action = np.asarray(raw, float)
        displacement = action-start
        length = float(np.linalg.norm(displacement))
        if length > self.rules['target_distance_world_px']:
            action = start+displacement*self.rules['target_distance_world_px']/length
        if self.last_action is not None:
            change = action-self.last_action
            length = float(np.linalg.norm(change))
            if length > self.rules['target_change_world_px']:
                action = self.last_action+change*self.rules['target_change_world_px']/length
        action = np.clip(action, *self.rules['requested_target_inner_bounds'])
        self.last_action = action.copy()
        return action, dict(phase=phase, mode=mode, selected_arc_id=int(self.contact['arc_id']),
            selected_point_local=self.contact['point_local'], selected_normal_local=self.contact['outward_normal_local'],
            drive_tangent_ratio=float(self.contact['drive_tangent_ratio']), raw_sampler_target=np.asarray(raw).tolist(),
            requested_action=action.tolist(), target_refresh_reason=changed, target_refresh_count=self.refresh_count,
            acquisition_required=bool(self.awaiting_acquisition), acquisition_no_route_controls=self.acquisition_no_route,
            prior_selected_contact_seconds=self.last_selected_contact_seconds, route_kind=self.last_route_kind,
            departure_source_point_local=deepcopy(self.departure_contact['point_local']),
            entry_started=bool(self.entry_started), policy_execution_version=POLICY_VERSION,
            policy_implementation_source_sha256=saved.file_sha(__file__),
            geometric_drive_room_world_px=float(selected_room), prior_contact_drive_room_world_px=float(prior_room),
            privileged_current_state=True, original_control_and_physics_unchanged=True, execution_intervention=False)


def install_frozen_policy(policy_path, gate_path):
    """Explicit process-local policy substitution, with exact source gate checks.

    No file or original environment is patched. The frozen v1 collector remains
    the executor; metadata names v2 and decisions bind this implementation SHA.
    """
    import json
    from . import strict_original_contact_pilot as collector
    policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    gate = json.loads(Path(gate_path).read_text(encoding='utf-8'))
    source = str(Path(__file__).resolve())
    digest = saved.file_sha(source)
    if (policy.get('policy_execution_version') != POLICY_VERSION
            or policy.get('implementation_module') != 'tdyssey.strict_original_target_policy_v2'
            or str(Path(policy.get('implementation_source', '')).resolve()) != source
            or policy.get('implementation_source_sha256') != digest
            or gate.get('bindings', {}).get(source) != digest):
        raise ValueError('Actual requested-target-v2 implementation is not frozen in policy/gate')
    collector.RequestedTargetPolicy = RequestedTargetPolicyV2
    collector.POLICY_VERSION = POLICY_VERSION
    return dict(policy_execution_version=POLICY_VERSION, implementation_module=__name__,
        implementation_source_path=source, implementation_source_sha256=digest,
        mutation_scope='Process-local requested-action planner selection only; no source or environment writes')
