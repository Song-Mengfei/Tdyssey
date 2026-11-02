"""Budget and actual-start accounting without entering a real simulator."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts import native_rgb_stage2b as stage


class Stage2BTests(unittest.TestCase):
    def test_slot_cap_and_unresolved_slot_are_not_replayed(self):
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as d:
            ledger=stage.RemainingTrainingLedger(d,cap=14976)
            ledger.value['trajectories']=[dict(status='completed') for _ in range(233)]
            index=ledger.begin_trajectory('last')
            with self.assertRaises(RuntimeError):ledger.begin_trajectory('replacement')
            ledger.finish(index)
            with self.assertRaises(RuntimeError):ledger.begin_trajectory('extra')

    def _failed_slot(self,charge_failure=False):
        class FakeEnv:
            control_step=0
            substep_penetrations=[]
            def state(self):return np.zeros(15,dtype=np.float64)
            def penetrations(self):return np.zeros(3)
            def step(self,action):raise RuntimeError('entered mocked step; no simulator')
        class FakePolicy:
            def __init__(self,*args):pass
            def next_action(self,t):return [1.,1.],{}
        with tempfile.TemporaryDirectory(dir=stage.ROOT/'artifacts') as d:
            out=Path(d);ledger=stage.RemainingTrainingLedger(out,cap=14976);attempts={55:4}
            q=dict(request_id='mock',stage='2B',letter='O',spatial_class='center',seed=1,gap_world_px=None,gap_band=None)
            with patch('tdyssey.native_rgb_env.make_env',return_value=FakeEnv()),\
                patch('tdyssey.native_rgb_supervised_sampling.initialize_scene',return_value=dict(legal=True,staging_segment=55,selection_version='motion-room-planned-attempts-v2')),\
                patch('tdyssey.native_rgb_supervised_sampling.ContinuousContactSampler',FakePolicy):
                if charge_failure:
                    with patch.object(ledger,'charge',side_effect=OSError('mock durable charge failure')):
                        e,error=stage.run_slot(out,q,'unused',ledger,[],{}, {},set(),planned_attempts=attempts)
                else:e,error=stage.run_slot(out,q,'unused',ledger,[],{}, {},set(),planned_attempts=attempts)
            meta=json.loads((out/'data'/e['metadata_file']).read_text())
            self.assertIsNotNone(error);self.assertEqual(e['stage'],'2B');self.assertEqual(e['controls'],0)
            self.assertEqual(attempts[55],4 if charge_failure else 5)
            self.assertEqual(meta['selection_inputs']['planned_initial_attempts_before'],{'55':4})
            self.assertEqual(meta['selection_inputs']['planned_attempt_counted'],not charge_failure)
            self.assertEqual(ledger.value['charged_controls'],0 if charge_failure else 1)
            self.assertEqual(ledger.value['completed_controls'],0)
            self.assertEqual(ledger.value['trajectories'][0]['status'],'failed')

    def test_entered_first_step_failure_counts_once_and_keeps_cost(self):self._failed_slot()
    def test_durable_charge_failure_does_not_mark_initial_arc_started(self):self._failed_slot(True)


if __name__=='__main__':unittest.main()
