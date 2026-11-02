"""Pure accounting/pair guards; never construct a physics environment."""
import copy
import unittest
import numpy as np
from scripts.native_rgb_r2a_run import guard_charge, compare_pair, json_safe


class R2AGuards(unittest.TestCase):
    def ledger(self):
        return dict(closed=False,pending_control=None,charged_controls=0,max_controls=512)

    def row(self):
        return dict(slot_id="synthetic-not-an-executed-slot",charged_controls=0)

    def test_pending_failed_attempt_cannot_repeat_or_refund(self):
        ledger=self.ledger();row=self.row();guard_charge(ledger,row,0)
        self.assertEqual((ledger['charged_controls'],row['charged_controls']),(1,1))
        with self.assertRaises(RuntimeError):guard_charge(ledger,row,1)
        self.assertEqual(ledger['charged_controls'],1)

    def test_slot_and_global_caps_apply_before_next_attempt(self):
        for total,slot,index in [(512,0,0),(8,8,8)]:
            ledger=self.ledger();row=self.row();ledger['charged_controls']=total;row['charged_controls']=slot
            with self.assertRaises(RuntimeError):guard_charge(ledger,row,index)
            self.assertEqual((ledger['charged_controls'],row['charged_controls']),(total,slot))

    def pair(self):
        return [dict(state=np.zeros(15),rgb=np.zeros((3,96,96),dtype=np.uint8),
            identity={'physics':{'space':{'damping':float(i)},'mass':1.,'friction':0.}}) for i in (0,1)]

    def test_only_damping_can_differ(self):
        self.assertTrue(compare_pair(self.pair())['non_damping_physics_equal'])
        for key in ['mass','friction']:
            pair=self.pair();pair[1]['identity']['physics'][key]=9.
            with self.assertRaises(RuntimeError):compare_pair(pair)

    def test_actual_state_and_pixels_must_match_before_control(self):
        pair=self.pair();pair[1]['state'][0]=1e-12
        with self.assertRaises(RuntimeError):compare_pair(pair)
        pair=self.pair();pair[1]['rgb'][0,0,0]=1
        with self.assertRaises(RuntimeError):compare_pair(pair)

    def test_nonfinite_failure_metadata_is_explicit(self):
        self.assertEqual(json_safe({'state':[float('nan'),float('inf')]}),
                         {'state':['NONFINITE:nan','NONFINITE:inf']})


if __name__=='__main__':unittest.main()
