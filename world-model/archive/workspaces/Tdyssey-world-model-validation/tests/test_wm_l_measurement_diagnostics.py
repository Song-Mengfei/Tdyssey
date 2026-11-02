"""Residual attribution properties; synthetic arrays, no predictor or physics."""
import unittest
import numpy as np
from tdyssey.wm_l_measurement_diagnostics import partition_residual


class ResidualAttributionTests(unittest.TestCase):
    def test_local_pusher_disagreement_is_not_reclassified_as_valid(self):
        template=np.zeros((96,96));template[40:55,40:55]=1
        alpha=template.copy();alpha[47:49,47:49]=0
        before=alpha.copy()
        position=[(47.5+.5)*512/96-.5]*2
        result=partition_residual(alpha,template,position)
        self.assertAlmostEqual(result['discs'][0]['residual_inside_fraction'],1.)
        self.assertAlmostEqual(result['discs'][0]['outside_soft_iou'],1.)
        self.assertFalse(result['eligibility_changed'])
        self.assertNotIn('valid',result)
        np.testing.assert_array_equal(alpha,before)

    def test_missing_arm_outside_pusher_remains_visible_in_residual(self):
        template=np.zeros((96,96));template[40:55,40:55]=1
        alpha=template.copy();alpha[40:45,40:55]=0
        result=partition_residual(alpha,template,[400.,400.])
        self.assertEqual(result['discs'][0]['residual_inside_fraction'],0.)
        self.assertLess(result['discs'][0]['outside_soft_iou'],.85)

    def test_absent_pusher_and_zero_residual_are_explicit(self):
        array=np.ones((96,96))
        self.assertEqual(partition_residual(array,array,None)['status'],'pusher_unavailable')
        result=partition_residual(array,array,[256.,256.])
        self.assertIsNone(result['discs'][0]['residual_inside_fraction'])
        self.assertEqual(result['discs'][0]['outside_soft_iou'],1.)
        self.assertFalse(result['truth_used'])

    def test_truth_cannot_guide_partition_and_invalid_maps_fail(self):
        array=np.zeros((96,96))
        with self.assertRaises(TypeError):partition_residual(array,array,[256.,256.],true_pose=[1,2,3])
        with self.assertRaises(ValueError):partition_residual(array[:1],array,[256.,256.])
        with self.assertRaises(ValueError):partition_residual(array,array,[float('nan'),0])


if __name__=='__main__':unittest.main()
