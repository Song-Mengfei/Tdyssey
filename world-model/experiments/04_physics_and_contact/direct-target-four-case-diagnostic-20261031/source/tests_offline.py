"""Pure fixtures/mocks: no environment construction or real native step."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from observer import FourCaseEnv, PrefixMismatch, DiagnosticHardStop, BudgetExceeded
from tdyssey.native_rgb_original_physics import OriginalPhysicsEnv


def fixture(depth=(0., 0., 0.), region='outside', vertices=None, expected=(), rgb=None,
            actions=None, max_substeps=8):
    env = FourCaseEnv.__new__(FourCaseEnv)
    env._configure_observer(expected, rgb, actions, max_substeps)
    env._base = SimpleNamespace(control_step=0)
    env._action = np.asarray([0., 512.])
    env._reset = True; env._terminal = False; env._in_step = False
    env.last_substeps = []
    env._point_region = lambda: region
    vv = vertices if vertices is not None else [(100., 100.), (110., 100.), (100., 110.)]
    shape = SimpleNamespace(radius=0., get_vertices=lambda: vv)
    env.block = SimpleNamespace(shapes=[shape], local_to_world=lambda p: p)
    env.state = lambda: np.zeros(15)
    env.render_clean = lambda: np.zeros((3, 96, 96), np.uint8)
    template = dict(dt=.01, pre_state=np.zeros(15).tolist(),
        post_state=np.zeros(15).tolist(), penetrations=list(depth),
        pusher_region=region, executed=True, valid=True)
    def fake_native(instance, dt):
        row=deepcopy(template); row['index']=len(instance.last_substeps)
        instance.last_substeps.append(row)
    return env, fake_native, deepcopy(template)


class FourCaseObservationFixtures(unittest.TestCase):
    def test_AO_OW_above_one_are_marks_and_do_not_abort(self):
        env, native, _ = fixture(depth=(1.4, 4.7, 0.))
        with patch.object(OriginalPhysicsEnv, '_observe_step', native):
            for _ in range(3): env._observe_step(.01)
        self.assertEqual(env.diagnostic_substeps, 3)
        self.assertTrue(env.last_substeps[-1]['would_stop_legacy_AO'])
        self.assertTrue(env.last_substeps[-1]['would_stop_legacy_OW'])
        self.assertFalse(env.last_substeps[-1]['legacy_valid'])
        self.assertTrue(env.last_substeps[-1]['valid'])

    def test_AW_region_nonfinite_and_outerwall_halt_immediately(self):
        cases=[dict(depth=(0,0,1.01)), dict(region='solid'), dict(region='hole-0'),
               dict(depth=(float('nan'),0,0)), dict(vertices=[(509.,100.),(500.,100.),(500.,110.)])]
        for params in cases:
            with self.subTest(params=params):
                env,native,_=fixture(**params)
                with patch.object(OriginalPhysicsEnv,'_observe_step',native):
                    with self.assertRaises(DiagnosticHardStop):
                        for _ in range(5): env._observe_step(.01)
                self.assertEqual(env.diagnostic_substeps,1)
                self.assertFalse(env.last_substeps[-1]['valid'])
                self.assertTrue(env.last_substeps[-1]['diagnostic_hard_reasons'])

    def test_prefix_mismatch_stops_at_first_actual_row(self):
        env,native,old=fixture()
        old['post_state'][2]=.00001
        env._configure_observer([old],None,[[0,512]],8)
        with patch.object(OriginalPhysicsEnv,'_observe_step',native):
            with self.assertRaises(PrefixMismatch):
                for _ in range(3):env._observe_step(.01)
        self.assertEqual(env.diagnostic_substeps,1)
        self.assertEqual(len(env.last_substeps),1)

    def test_terminal_partial_RGB_and_exact_action_are_checked(self):
        env,native,old=fixture(depth=(0,1.1,0))
        old.update(valid=False,invalid_reasons=['overlap_above_1_world_px'])
        env._configure_observer([old],np.zeros((3,96,96),np.uint8),[[0,512]],8)
        with patch.object(OriginalPhysicsEnv,'_observe_step',native):env._observe_step(.01)
        self.assertTrue(env.prefix_complete)
        self.assertTrue(env.prefix_terminal_rgb_matched)
        self.assertTrue(env.first_legacy_match)
        env,native,old=fixture()
        env._configure_observer([old],None,[[0,511]],8)
        with patch.object(OriginalPhysicsEnv,'_observe_step',native):
            with self.assertRaises(PrefixMismatch):env._observe_step(.01)
        self.assertIn('requested_action',env.prefix_mismatch_event['fields'])

    def test_cap_is_checked_before_extra_native_call(self):
        env,native,_=fixture(max_substeps=1)
        with patch.object(OriginalPhysicsEnv,'_observe_step',native):
            env._observe_step(.01)
            with self.assertRaises(BudgetExceeded):env._observe_step(.01)
        self.assertEqual(len(env.last_substeps),1)
        self.assertEqual(env.diagnostic_substeps,1)

    def test_inherited_step_preserves_engineering_exception(self):
        env,native,old=fixture()
        old['post_state'][3]=1.
        env._configure_observer([old],None,[[0,512]],8)
        env._upstream=SimpleNamespace(step=lambda base,target:[env._observe_step(.01) for _ in range(10)])
        with patch.object(OriginalPhysicsEnv,'_observe_step',native):
            with self.assertRaises(PrefixMismatch):env.step([0,512])
        self.assertEqual(env.diagnostic_substeps,1)


if __name__=='__main__':unittest.main(verbosity=2)
