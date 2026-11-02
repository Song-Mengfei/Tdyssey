"""CPU-only tests of an offline measurement instrument, never new controls."""

import hashlib
import json
from importlib.util import find_spec
from pathlib import Path
import unittest
from unittest.mock import patch

if any(find_spec(name) is None for name in ('numpy', 'scipy', 'PIL', 'shapely')):
    raise unittest.SkipTest('Install the numerical profile for physical metric tests')
import numpy as np

from tdyssey.native_rgb_physical_metrics import (
    PROTOCOL, PredictionTrace, _raw_addresses, _transform, equivalent_angle_degrees,
    fit_physical_pose, score_from_trace,
)
from tdyssey.native_rgb_evaluation import WORLD_SCALE, _stable_seed


PALETTE=dict(object_rgb=[119/255,136/255,153/255],
    palette=dict(minimum_alpha=.4,maximum_alpha=1.25,colour_residual=.12))


def mask():
    value=np.zeros((96,96),dtype=np.float32)
    value[29:57,31:37]=1
    value[51:57,31:53]=1
    return value


def rgb(full,blue=None):
    value=1-np.asarray(full)[None]*(1-np.asarray(PALETTE["object_rgb"]))[:,None,None]
    if blue is not None:
        from scipy import ndimage
        y,x=np.mgrid[:96,:96]
        circle=(x-blue[0])**2+(y-blue[1])**2<=3**2
        value[:,circle]=np.asarray([65,105,225])[:,None]/255
    return value.astype(np.float32)


def physical_calibration():
    import tdyssey.native_rgb_physical_metrics as module
    value=dict(protocol=dict(PROTOCOL),physical_source_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
               palette_calibration=PALETTE,physical_measurement_calibrated=True,model_outputs_used=False)
    value["sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    return value


class Echo:
    def __call__(self,history,actions,seed):
        self.last=history[-1].copy()
        return self.last
    def batch_predict(self,histories,actions,*,batch_seed):
        self.last=histories[:,-1].copy()
        return self.last
    def close(self):pass


class Dataset:
    split="seen_val"
    entries=[dict(trajectory_id="t",shape="L",split="seen_val",controls=10)]
    windows=[(0,3),(0,4)]
    def __init__(self):
        self.arrays=dict(actions=np.arange(20,dtype=np.float32).reshape(10,2),states=np.zeros((11,15)),
                         contact=np.ones(10),valid_transition=np.ones(10,dtype=bool))
    def trajectory(self,index):return self.entries[index],self.arrays
    def render_layers(self,entry,state):return dict(rgb=rgb(mask(),(12,12)),full_object_mask=mask())
    def render_state(self,entry,state):return rgb(mask(),(12,12))


class PhysicalMeasurementTests(unittest.TestCase):
    def test_translation_rotation_and_occlusion_fit_complete_centroid(self):
        from tdyssey.native_rgb_physical_metrics import _centroid
        reference=mask()
        for dx,dy,angle,blue in ((4,-3,0,None),(0,0,15,(34,43)),(3,1,-20,(42,47))):
            with self.subTest(angle=angle,translation=(dx,dy)):
                target=_transform(reference,angle,(dx,dy),_centroid(reference>.4))
                score=fit_physical_pose(rgb(target,blue),reference,PALETTE)
                self.assertEqual(score["status"],"ok",score)
                np.testing.assert_allclose(score["translation_delta_px"],np.asarray([dx,dy])*WORLD_SCALE,atol=3.)
                self.assertLessEqual(abs(score["angle_delta_deg"]-angle),2.)
                self.assertTrue(score["scoring_only"])
                self.assertFalse(score["output_repaired"])

    def test_blue_motion_alone_cannot_look_like_object_motion_improvement(self):
        from tdyssey.native_rgb_physical_metrics import _centroid
        original=mask()
        future=_transform(original,0,(5,0),_centroid(original>.4))
        held=fit_physical_pose(rgb(original,(32,35)),future,PALETTE)
        moved_blue=fit_physical_pose(rgb(original,(35,47)),future,PALETTE)
        self.assertEqual(held["status"],"ok",held)
        self.assertEqual(moved_blue["status"],"ok",moved_blue)
        self.assertGreater(moved_blue["translation_error_px"],20.)
        apparent_gain=(held["physical_motion_error_px"]-moved_blue["physical_motion_error_px"])/held["physical_motion_error_px"]
        self.assertLess(apparent_gain,.20)

    def test_symmetry_uses_declared_geometry_equivalence(self):
        self.assertEqual(equivalent_angle_degrees(180,2),0)
        self.assertEqual(equivalent_angle_degrees(90,4),0)
        self.assertEqual(equivalent_angle_degrees(180,1),-180)
        symmetric=np.zeros((96,96),dtype=np.float32)
        symmetric[30:60,42:48]=1
        result=fit_physical_pose(rgb(np.rot90(symmetric,2)),symmetric,PALETTE,symmetry_order=2)
        self.assertEqual(result["status"],"ok")
        self.assertLessEqual(result["angle_error_deg"],1.)

    def test_missing_shape_and_clipping_are_unmeasurable(self):
        reference=mask()
        broken=reference.copy();broken[29:48]=0
        score=fit_physical_pose(rgb(broken),reference,PALETTE)
        self.assertEqual(score["status"],"unmeasurable")
        edge=np.roll(reference,-31,axis=1)
        self.assertEqual(fit_physical_pose(rgb(edge),reference,PALETTE)["reason"],"viewport_clipping")

    def test_trace_passes_same_result_object_and_cap_only_affects_measurement(self):
        predictor=Echo();trace=PredictionTrace(predictor,max_bytes=3*96*96*4)
        history=np.repeat(rgb(mask())[None],4,axis=0);actions=np.zeros((4,2),dtype=np.float32)
        first=trace(history,actions,10)
        self.assertIs(first,predictor.last)
        np.testing.assert_array_equal(trace.raw(0),first)
        second=trace(history,actions,11)
        self.assertIs(second,predictor.last)
        self.assertIsNone(trace.raw(1))
        self.assertEqual(trace.snapshot()["dropped_raw_outputs"],1)
        self.assertLessEqual(trace.retained_bytes,trace.max_bytes)
        with patch.object(trace,"_record",side_effect=MemoryError("instrument full")):
            self.assertIs(trace(history,actions,12),predictor.last)
        self.assertEqual(trace.instrumentation_errors,1)
        trace.clear();self.assertEqual(trace.retained_bytes,0)

    def test_batch_trace_maps_original_indices_through_invalid_item_shrink(self):
        dataset=Dataset()
        trace=PredictionTrace(Echo())
        histories=np.repeat(np.repeat(rgb(mask())[None],4,axis=0)[None],2,axis=0)
        histories[0,-1]=np.nan
        actions=np.stack([dataset.arrays["actions"][0:4],dataset.arrays["actions"][1:5]])
        result=trace.batch_predict(histories,actions,batch_seed=99)
        self.assertIs(result,trace.predictor.last)
        trace.batch_predict(histories[1:],dataset.arrays["actions"][2:6][None],batch_seed=100)
        report=dict(inference_batch_size=2,horizons=[1,2],rows=[dict(window="t:3"),dict(window="t:4")],
                    batches=[dict(windows=["t:3","t:4"],batch_seed=99)])
        addresses=_raw_addresses(report,dataset,trace,5478)
        self.assertEqual(addresses["t:4",1],(0,1))
        self.assertEqual(addresses["t:4",2],(1,0))
        self.assertNotIn(("t:3",2),addresses)

    def test_trace_cache_limit_keeps_all_physical_denominators(self):
        dataset=Dataset();trace=PredictionTrace(Echo(),max_bytes=0)
        history=np.repeat(rgb(mask())[None],4,axis=0)
        trace(history,dataset.arrays["actions"][0:4],_stable_seed("t:3",5478))
        rows=[dict(window="t:3",method=name,horizon=1,letter="L",anchor=3,trajectory_id="t",status="ok",
                   contact=True,rotation=True,boundary=True,stationary=False) for name in ("hold","fine_tuned")]
        report=dict(inference_batch_size=1,horizons=[1],methods=["hold","fine_tuned"],rows=rows)
        result=score_from_trace(report,dataset,{"fine_tuned":trace},physical_calibration())
        self.assertEqual(len(result["rows"]),2)
        self.assertEqual(result["summary"]["fine_tuned"]["1"]["overall"]["unmeasurable"],1)
        self.assertFalse(result["physical_dynamics_verified"])
        self.assertEqual(result["qualification"]["fine_tuned"]["1"]["subsets"]["contact"]["total"],1)

    def test_same_core_outputs_are_reused_without_any_predictor_rerun(self):
        from tdyssey.native_rgb_evaluation import evaluate_windows
        core_cal=dict(PALETTE,minimum_component_pixels=2,angle_grid_degrees=20,
            contour_tolerance_px=5.,missing_extra_fraction_tolerance=.2,aligned_iou_floor=.7,
            per_letter={"L":dict(max_area_fraction_tolerance=.2,components_min=1,components_max=1,holes_min=0,holes_max=0)})
        for batch_size in (1,2):
            with self.subTest(batch_size=batch_size):
                dataset=Dataset();trace=PredictionTrace(Echo())
                core=evaluate_windows(dataset,{"fine_tuned":trace},calibration=core_cal,
                    horizons=(1,4),windows_per_letter=2,gallery_limit=0,batch_size=batch_size)
                calls=trace.total_calls
                result=score_from_trace(core,dataset,{"fine_tuned":trace},physical_calibration())
                self.assertEqual(trace.total_calls,calls)
                self.assertEqual(len(result["rows"]),len(core["rows"]))
                self.assertTrue(all(row["status"]=="ok" for row in result["rows"]),result["rows"])
                self.assertTrue(all(row["same_raw_core_rgb_mse"] for row in result["rows"]))

    def test_relative_gain_below_calibrated_pose_precision_cannot_verify_dynamics(self):
        dataset=Dataset();trace=PredictionTrace(Echo())
        history=np.repeat(rgb(mask(),(12,12))[None],4,axis=0)
        for step in (0,3):
            trace(history,dataset.arrays["actions"][step:step+4],(_stable_seed("t:3",5478)+step)%2**32)
        rows=[dict(window="t:3",method=name,horizon=h,letter="L",anchor=3,trajectory_id="t",status="ok",
                   contact=True,rotation=True,boundary=True,stationary=False)
              for h in (1,4) for name in ("hold","fine_tuned")]
        core=dict(inference_batch_size=1,horizons=[1,4],methods=["hold","fine_tuned"],rows=rows,
                  all_letter_coverage_complete=True)
        calibration=physical_calibration();calibration.pop("sha256")
        calibration.update(maximum_observed_translation_error_px=2.,maximum_observed_angle_error_deg=0.)
        calibration["sha256"]=hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest()
        def measurement(error):
            return dict(status="ok",reason="",translation_error_px=error,angle_error_deg=0.,
                        physical_motion_error_px=error,fit_iou=1.,object_radius_px=10.)
        with patch("tdyssey.native_rgb_physical_metrics.fit_physical_pose",
                   side_effect=[measurement(10),measurement(7.9),measurement(10),measurement(7.9)]):
            result=score_from_trace(core,dataset,{"fine_tuned":trace},calibration)
        check=result["qualification"]["fine_tuned"]["1"]["subsets"]["contact"]
        self.assertGreater(check["relative_improvement"],.20)
        self.assertLess(check["absolute_improvement_px"],check["finite_calibration_gain_uncertainty_px"])
        self.assertFalse(check["passed"])
        self.assertFalse(result["physical_dynamics_verified"])


if __name__=="__main__":unittest.main()
