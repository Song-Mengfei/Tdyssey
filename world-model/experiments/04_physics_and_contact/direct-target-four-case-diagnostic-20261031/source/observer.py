"""Isolated C2/C3 observations for four explicitly approved diagnostic cases.

No integration loop, physical value, requested target or legacy file is changed.
Construction/stepping is performed only by the separately bounded owner runner.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from tdyssey.direct_target_collection import DirectTargetOriginalEnv
from tdyssey.native_rgb_original_physics import OriginalPhysicsEnv
from tdyssey.native_rgb_r2a_env import R2AStepAborted
from tdyssey.strict_original_contact_pilot import state_invalidity

STATE_ATOL = 1e-8
LEGACY_DEPTH = 1.0
DEPTH_EPSILON = 1e-8
OUTER_WALL_MIN = 3.0
OUTER_WALL_MAX = 508.0


class PrefixMismatch(R2AStepAborted):
    """Engineering stop: the saved public prefix was not reproduced."""


class DiagnosticHardStop(R2AStepAborted):
    """An approved retained or additional geometric hard stop was observed."""


class BudgetExceeded(R2AStepAborted):
    """Engineering stop before an unapproved extra integration call."""


class FourCaseEnv(DirectTargetOriginalEnv):
    """Original pinned step with only the approved diagnostic observer delta.

    ``expected_rows`` is the old NPZ telemetry flattened in actual microstep
    order. Its final row is the original early-stop location, not a full control
    boundary. The pure renderer is also compared at that exact microstep.
    """
    def __init__(self, source_root, letter, *, expected_rows=(),
                 expected_terminal_rgb=None, expected_actions=None,
                 max_substeps=80):
        self._configure_observer(expected_rows, expected_terminal_rgb,
                                 expected_actions, max_substeps)
        super().__init__(source_root, letter)

    def _configure_observer(self, expected_rows=(), expected_terminal_rgb=None,
                            expected_actions=None, max_substeps=80):
        """Pure configuration, also used by fixtures without constructing physics."""
        if type(max_substeps) is not int or not 0 < max_substeps <= 110:
            raise ValueError('A finite approved case substep cap, at most110, is required')
        self.expected_rows = deepcopy(list(expected_rows))
        self.expected_terminal_rgb = (None if expected_terminal_rgb is None else
                                      np.asarray(expected_terminal_rgb).copy())
        self.expected_actions = (None if expected_actions is None else
                                 np.asarray(expected_actions, np.float64).copy())
        if self.expected_actions is not None:
            if self.expected_actions.ndim != 2 or self.expected_actions.shape[1:] != (2,):
                raise ValueError('Expected original absolute actions must have shape[N,2]')
        self.max_diagnostic_substeps = max_substeps
        self.diagnostic_substeps = 0
        self.prefix_rows_checked = 0
        self.prefix_max_abs_error = 0.0
        self.prefix_terminal_rgb_matched = None
        self.prefix_terminal_rgb_snapshot = None
        self.first_legacy_event = None
        self.first_legacy_match = None
        self.hard_stop_event = None
        self.prefix_mismatch_event = None

    @property
    def prefix_complete(self):
        return (bool(self.expected_rows)
                and self.prefix_rows_checked == len(self.expected_rows)
                and (self.expected_terminal_rgb is None or
                     self.prefix_terminal_rgb_matched is True))

    @property
    def legacy_terminal_rgb_snapshot(self):
        """Actual pure render at the old partial-control termination microstep."""
        return self.prefix_terminal_rgb_snapshot

    @property
    def prefix_terminal_rgb_actual(self):
        return self.prefix_terminal_rgb_snapshot

    def prefix_report(self):
        return dict(expected_microsteps=len(self.expected_rows),
                    checked_microsteps=self.prefix_rows_checked,
                    max_state_abs_error=self.prefix_max_abs_error,
                    state_absolute_tolerance=STATE_ATOL,
                    terminal_rgb_matched=self.prefix_terminal_rgb_matched,
                    complete=self.prefix_complete,
                    first_legacy_match=self.first_legacy_match,
                    mismatch=self.prefix_mismatch_event)

    def _stop_info(self, **extra):
        return dict(valid=False, diagnostic_only=True, training_eligible=False,
                    substeps=self.last_substeps,
                    control_step=int(self.control_step),
                    diagnostic_substeps=self.diagnostic_substeps, **extra)

    def _object_vertex_bounds(self):
        """Read actual existing collision-shape vertices; create no physics objects."""
        vertices = []
        for shape in self.block.shapes:
            if not hasattr(shape, 'get_vertices') or float(shape.radius) != 0.0:
                raise RuntimeError('Approved object geometry requires zero-radius polygons')
            vertices.extend(tuple(self.block.local_to_world(v))
                            for v in shape.get_vertices())
        values = np.asarray(vertices, np.float64).reshape(-1, 2)
        if not len(values):
            raise RuntimeError('Actual object collision vertices are unavailable')
        if not np.isfinite(values).all():
            return None
        return [float(values[:, 0].min()), float(values[:, 1].min()),
                float(values[:, 0].max()), float(values[:, 1].max())]

    def _verify_prefix(self, row, index):
        if index >= len(self.expected_rows):
            return
        old = self.expected_rows[index]
        failures = []
        if float(old.get('dt', float('nan'))) != float(row.get('dt', float('nan'))):
            failures.append('dt')
        for key in ('pre_state', 'post_state'):
            actual = np.asarray(row.get(key, []), np.float64)
            expected = np.asarray(old.get(key, []), np.float64)
            if actual.shape != expected.shape or actual.shape != (15,):
                failures.append(key+'_shape')
                continue
            error = (float(np.max(np.abs(actual-expected)))
                     if np.isfinite(actual).all() and np.isfinite(expected).all()
                     else float('inf'))
            self.prefix_max_abs_error = max(self.prefix_max_abs_error, error)
            if not np.allclose(actual, expected, rtol=0, atol=STATE_ATOL):
                failures.append(key)
        depths = np.asarray(row.get('penetrations', []), float)
        old_depths = np.asarray(old.get('penetrations', []), float)
        if (depths.shape != (3,) or old_depths.shape != (3,)
                or not np.allclose(depths, old_depths, rtol=0, atol=STATE_ATOL)):
            failures.append('penetrations')
        if old.get('pusher_region') is not None and row['pusher_region'] != old['pusher_region']:
            failures.append('pusher_region')
        if self.expected_actions is not None:
            control = int(self.control_step)
            if (control >= len(self.expected_actions)
                    or not np.array_equal(self._action, self.expected_actions[control])):
                failures.append('requested_action')
        expected_legacy_invalid = not bool(old.get('valid', True))
        if bool(row['would_stop_legacy']) != expected_legacy_invalid:
            failures.append('legacy_stop_location')
        if index == len(self.expected_rows)-1 and self.expected_terminal_rgb is not None:
            rgb = np.asarray(self.render_clean()).copy()
            self.prefix_terminal_rgb_snapshot = rgb
            self.prefix_terminal_rgb_matched = bool(np.array_equal(rgb, self.expected_terminal_rgb))
            row['prefix_terminal_rgb_matched'] = self.prefix_terminal_rgb_matched
            if not self.prefix_terminal_rgb_matched:
                failures.append('terminal_partial_RGB')
        row['prefix_state_match'] = not failures
        if failures:
            self.prefix_mismatch_event = dict(global_microstep=index,
                control=int(self.control_step), substep=int(row['index']),
                fields=failures, max_state_abs_error=self.prefix_max_abs_error)
            row['prefix_mismatch'] = self.prefix_mismatch_event
            raise PrefixMismatch('Original saved prefix mismatch',
                self._stop_info(prefix_mismatch=self.prefix_mismatch_event,
                                engineering_stop=True))
        self.prefix_rows_checked += 1
        if row['would_stop_legacy'] and self.first_legacy_match is None:
            self.first_legacy_match = expected_legacy_invalid

    def _observe_step(self, dt):
        # Check before the native call, so an accidental extra loop cannot spend
        # a 111th/81st substep. This check introduces no physics call.
        if self.diagnostic_substeps >= self.max_diagnostic_substeps:
            raise BudgetExceeded('Approved case integration cap reached',
                self._stop_info(budget_exceeded=True, engineering_stop=True))
        index = self.diagnostic_substeps
        before_rows = len(self.last_substeps)
        try:
            # Deliberately bypass StrictObservedOriginalEnv._observe_step, whose
            # single-depth abort is the only observer behavior being diagnosed.
            OriginalPhysicsEnv._observe_step(self, dt)
        finally:
            # The original observer appends the row in its own finally clause.
            # Preserve actual successful calls even if its later telemetry fails.
            if len(self.last_substeps) > before_rows and self.last_substeps[-1].get('executed'):
                self.diagnostic_substeps += 1
        row = self.last_substeps[-1]
        row['global_microstep'] = index
        row['control_index'] = int(self.control_step)
        state = np.asarray(row.get('post_state', []), np.float64)
        depths = np.asarray(row.get('penetrations', []), np.float64)
        hard = []
        state_finite = state.shape == (15,) and np.isfinite(state).all()
        depths_finite = depths.shape == (3,) and np.isfinite(depths).all()
        if not state_finite:
            hard.append('nonfinite_state')
        if not depths_finite:
            hard.append('nonfinite_penetration')
        region = self._point_region() if state_finite else 'unknown_nonfinite'
        row['pusher_region'] = region
        if region == 'solid' or region.startswith('hole-'):
            hard.append('illegal_pusher_center_region')
        if depths_finite and depths[2] > LEGACY_DEPTH+DEPTH_EPSILON:
            hard.append('AW_overlap_above_1_world_px')
        bounds = self._object_vertex_bounds() if state_finite else None
        if bounds is None:
            hard.append('nonfinite_object_collision_vertices')
        elif (min(bounds[:2]) < OUTER_WALL_MIN or max(bounds[2:]) > OUTER_WALL_MAX):
            hard.append('object_outer_wall_envelope_crossed')
        legacy_reasons = state_invalidity(state, depths, region)
        row.update(legacy_invalid_reasons=legacy_reasons,
                   legacy_valid=not legacy_reasons,
                   would_stop_legacy=bool(legacy_reasons),
                   would_stop_legacy_AO=bool(depths_finite and depths[1] > 1.+DEPTH_EPSILON),
                   would_stop_legacy_OW=bool(depths_finite and depths[0] > 1.+DEPTH_EPSILON),
                   diagnostic_hard_reasons=hard,
                   object_collision_vertex_bounds=bounds,
                   diagnostic_only=True, training_eligible=False,
                   valid=not hard)
        if legacy_reasons and self.first_legacy_event is None:
            self.first_legacy_event = dict(global_microstep=index,
                control=int(self.control_step), substep=int(row['index']),
                reasons=legacy_reasons, penetrations=depths.tolist())
        # A mismatched prefix is an engineering stop for the whole batch. This
        # comparison occurs before any following substep can be requested.
        self._verify_prefix(row, index)
        if hard:
            self.hard_stop_event = dict(global_microstep=index,
                control=int(self.control_step), substep=int(row['index']),
                reasons=hard, penetrations=depths.tolist(),
                object_collision_vertex_bounds=bounds)
            raise DiagnosticHardStop('Approved diagnostic hard stop',
                self._stop_info(hard_stop=self.hard_stop_event,
                                first_invalid_substep=int(row['index'])))
