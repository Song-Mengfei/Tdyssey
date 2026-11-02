"""T1 hand-built arrays/math only; no saved images, model or decoder imports."""

import hashlib
import json
import math
import random
import unittest

import numpy as np

from tdyssey.wm_compat_transform import (T1_SHA256, T1_SPEC, _winner_indices,
    transform_for_arm, transform_t1)


W=np.array([255,255,255],dtype=np.float64)
G0=np.array([119,136,153],dtype=np.float64)
G1=np.array([143,163,184],dtype=np.float64)
R0=np.array([165,28,48],dtype=np.float64)
R1=np.array([198,34,58],dtype=np.float64)
V=np.array([144,238,144],dtype=np.float64)


def solid(color):
    pixel=(np.asarray(color,dtype=np.float64)/255).astype(np.float32)
    return np.broadcast_to(pixel[:,None,None],(3,96,96)).copy()


class T1FixedTransformChecks(unittest.TestCase):
    def test_spec_pins_colors_order_precision_and_hash(self):
        payload=json.dumps(T1_SPEC,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        self.assertEqual(T1_SHA256,hashlib.sha256(payload).hexdigest())
        self.assertEqual(len(T1_SPEC["segments"]),10)
        self.assertEqual([r["order"] for r in T1_SPEC["segments"]],list(range(1,11)))
        self.assertEqual(T1_SPEC["swatches_rgb_255"]["G0"],[119,136,153])
        self.assertEqual(T1_SPEC["swatches_rgb_255"]["R1"],[198,34,58])
        self.assertEqual(T1_SPEC["arithmetic"]["tie_squared_distance"],1e-9)
        self.assertEqual(T1_SPEC["arithmetic"]["rms_channel_residual_max_rgb_255"],8)

    def test_identity_and_unmatched_pixels_are_bitwise_preserved(self):
        for color in ((255,255,255),(211,211,211),(65,105,225),(78,126,255),(255,0,255),(0,0,0)):
            with self.subTest(color=color):
                image=solid(color)
                before=image.tobytes()
                output,diagnostics=transform_t1(image)
                self.assertEqual(output.tobytes(),before)
                self.assertEqual(image.tobytes(),before)
                self.assertFalse(np.shares_memory(output,image))
                self.assertEqual(diagnostics["changed_pixels"],0)
                if color in ((255,0,255),(0,0,0)):
                    self.assertEqual(diagnostics["unmatched_pixels"],96*96)
                else:
                    self.assertEqual(diagnostics["identity_preserved_pixels"],96*96)

    def test_known_endpoints_and_fractional_color_lines(self):
        cases=((G0,R0,"05:W--G0"),(G1,R1,"06:G0--G1"),(V,np.zeros(3),"08:W--V"),
               (W*.25+G0*.75,W*.25+R0*.75,"05:W--G0"),
               ((G0+G1)/2,(R0+R1)/2,"06:G0--G1"),
               ((V+G0)/2,R0/2,"09:V--G0"),
               ((V+G1)/2,R1/2,"10:V--G1"))
        for source,expected,winner in cases:
            with self.subTest(source=source.tolist()):
                output,diagnostics=transform_t1(solid(source))
                np.testing.assert_allclose(output[:,0,0],expected.astype(np.float32)*np.float32(1/255),rtol=0,atol=2e-7)
                self.assertEqual(diagnostics["winning_segment_counts"][winner],96*96)
                self.assertEqual(diagnostics["converted_pixels"],96*96)
                self.assertEqual(output.dtype,np.float32)
                self.assertTrue(diagnostics["output_in_range"])
                json.dumps(diagnostics,allow_nan=False)

    def test_global_minimum_tie_order_is_not_a_chained_pairwise_tie(self):
        distances=np.full((10,4),100.,dtype=np.float64)
        distances[0,:]=[2.+.5e-9,2.+2e-9,2.+1.8e-9,0]
        distances[1,:]=[2.,2.,2.+.9e-9,0]
        distances[2,:]=[9.,9.,2.,0]
        np.testing.assert_array_equal(_winner_indices(distances),[0,1,1,0])
        # White lies on many segments; the declared first identity line wins.
        _,diagnostics=transform_t1(solid(W))
        self.assertEqual(diagnostics["winning_segment_counts"]["01:W--Q"],96*96)

    def test_rms_eight_boundary_uses_stored_float32_pixel_without_clipping(self):
        # G0-(8,8,8) projects beyond the G0 endpoint on W--G0.
        # The two adjacent representable input values straddle RMS=8.
        inside=solid(G0-8)
        outside=np.nextafter(inside,np.float32(0))
        def endpoint_rms(image):
            difference=image[:,0,0].astype(np.float64)*255-G0
            return math.sqrt(float(np.dot(difference,difference))/3)
        self.assertLessEqual(endpoint_rms(inside),8)
        self.assertGreater(endpoint_rms(outside),8)
        converted,accepted=transform_t1(inside)
        preserved,rejected=transform_t1(outside)
        self.assertEqual(accepted["accepted_pixels"],96*96)
        self.assertEqual(accepted["converted_pixels"],96*96)
        self.assertEqual(rejected["unmatched_pixels"],96*96)
        self.assertNotEqual(converted.tobytes(),inside.tobytes())
        self.assertEqual(preserved.tobytes(),outside.tobytes())
        self.assertAlmostEqual(accepted["residual_rms_rgb_255"]["mean"],endpoint_rms(inside),places=12)

    def test_pixel_locations_immutability_and_external_rng_are_preserved(self):
        image=solid(W)
        image[:,11,27]=(G0/255).astype(np.float32)
        image[:,12,27]=(V/255).astype(np.float32)
        # A read-only, non-contiguous input must work without modification.
        image=image[:,:,::-1]
        image.setflags(write=False)
        before=image.tobytes()
        python_rng=random.getstate()
        numpy_rng=np.random.get_state()
        output,diagnostics=transform_t1(image)
        changed=np.any(output!=image,axis=0)
        self.assertEqual(set(map(tuple,np.argwhere(changed))),{(11,68),(12,68)})
        self.assertEqual(diagnostics["changed_pixels"],2)
        self.assertEqual(image.tobytes(),before)
        self.assertEqual(random.getstate(),python_rng)
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),numpy_rng)))
        self.assertEqual(sum(diagnostics["winning_segment_counts"].values()),96*96)
        self.assertEqual(diagnostics["converted_pixels"]+diagnostics["identity_preserved_pixels"]+diagnostics["unmatched_pixels"],96*96)

    def test_arm_a_is_exact_copy_and_unknown_arm_is_rejected(self):
        image=solid(G0)
        identity,diagnostics=transform_for_arm(image,"A")
        self.assertEqual(identity.tobytes(),image.tobytes())
        self.assertFalse(np.shares_memory(identity,image))
        self.assertFalse(diagnostics["classification_performed"])
        transformed,other=transform_for_arm(image,"B")
        direct,direct_diagnostics=transform_t1(image)
        self.assertEqual(transformed.tobytes(),direct.tobytes())
        self.assertEqual(other,direct_diagnostics)
        for arm in ("a","T1","identity",None,1):
            with self.assertRaises(ValueError):
                transform_for_arm(image,arm)

    def test_malformed_input_is_rejected_without_coercion_or_repair(self):
        correct=solid(W)
        bad=[correct.tolist(),correct.astype(np.float64),correct.astype(np.uint8),
             np.moveaxis(correct,0,-1),correct[:,:95]]
        for value in (np.nan,np.inf,-1e-7,1.+np.finfo(np.float32).eps):
            array=correct.copy();array[0,0,0]=value;bad.append(array)
        for image in bad:
            for arm in ("A","B"):
                with self.assertRaises(ValueError):
                    transform_for_arm(image,arm)


if __name__=="__main__":
    unittest.main()
