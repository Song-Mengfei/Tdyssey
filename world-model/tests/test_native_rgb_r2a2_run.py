"""Pure R2A2 accounting/identity checks: no simulator calls."""
import unittest
import numpy as np
from scripts.native_rgb_r2a2_run import compare_pair,guard_charge,DAMPINGS,ARMS


class R2A2Guards(unittest.TestCase):
    def pair(self):
        return [dict(state=np.zeros(15),rgb=np.zeros((3,96,96),dtype=np.uint8),
            identity={'physics':{'space':{'damping':d},'mass':1.,'friction':0.}}) for d in DAMPINGS]

    def test_exact_new_damping_pair_and_nominal_retention(self):
        result=compare_pair(self.pair())
        self.assertEqual(result['damping'],[.0009765625,.09921256574801246])
        self.assertAlmostEqual(DAMPINGS[0]**.1,.5,places=15)
        self.assertAlmostEqual(DAMPINGS[1]**.1,2**(-1/3),places=15)

    def test_other_parameter_or_initial_state_difference_is_terminal(self):
        pair=self.pair();pair[1]['identity']['physics']['friction']=1.
        with self.assertRaises(RuntimeError):compare_pair(pair)
        pair=self.pair();pair[1]['state'][0]=1e-12
        with self.assertRaises(RuntimeError):compare_pair(pair)

    def test_old_damping_not_a_new_arm(self):
        pair=self.pair();pair[0]['identity']['physics']['space']['damping']=0.
        with self.assertRaises(RuntimeError):compare_pair(pair)

    def test_precharged_failure_and_cap_cannot_be_replayed(self):
        ledger=dict(closed=False,pending_control=None,charged_controls=511,max_controls=512)
        row=dict(slot_id='synthetic-not-a-run',charged_controls=0)
        guard_charge(ledger,row,0)
        self.assertEqual(ledger['charged_controls'],512)
        with self.assertRaises(RuntimeError):guard_charge(ledger,row,1)
        ledger['pending_control']=None
        with self.assertRaises(RuntimeError):guard_charge(ledger,row,1)
        self.assertEqual(ledger['charged_controls'],512)


if __name__=='__main__':unittest.main()
