"""Recorded-state statistics only: no simulator controls, models, or CUDA."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey.envs.official import T_REFERENCE_COG, official_shape
from tdyssey.native_rgb_motion_statistics import (
    area_centroids, continuous_distribution, geometric_wall_gaps, gap_band,
    motion_components, proposed_index_feasibility, read_records,
    render_motion_contact_sheet, summarize_records, transition_rows,
)


def record(identifier="r",letter="L",x=None,location="center",contact=None):
    x=np.asarray(x if x is not None else [200,200,200,200,202,202.6,202.6,202.6,202.6],float)
    n=len(x)-1
    states=np.zeros((n+1,15),float);states[:,:2]=(100,100);states[:,2]=x;states[:,3]=220
    arrays=dict(states=states,actions=np.tile([256.,400.],(n,1)),contact=np.ones(n) if contact is None else np.asarray(contact),
        valid_transition=np.ones(n,bool),penetrations=np.zeros((n+1,3)),substep_penetrations=np.zeros((n,10,3)))
    entry=dict(trajectory_id=identifier,shape=letter,file=identifier+".npz",location=location,split="stage1",
               gap_band_requested="0-8",gap_requested_world_px=4)
    steps=[dict(phase="push",intent="push",target_bin=0,approach_kind="normal",target_distance_class="near") for _ in range(n)]
    return dict(entry=entry,arrays=arrays,steps=steps)


class MotionStatisticsTests(unittest.TestCase):
    def test_t_area_centroid_is_not_legacy_physical_cog(self):
        local=np.asarray(official_shape("T").centroid)
        self.assertFalse(np.allclose(local,T_REFERENCE_COG))
        states=np.zeros((2,15));states[:,4]=(0,math.pi/2)
        physical=np.asarray(T_REFERENCE_COG);center=np.asarray([256,256])
        states[0,2:4]=center-physical
        states[1,2:4]=center-np.asarray([-physical[1],physical[0]])
        expected=np.stack([center+local-physical,center+np.asarray([-(local-physical)[1],(local-physical)[0]])])
        np.testing.assert_allclose(area_centroids("T",states),expected)
        self.assertGreater(motion_components("T",states)["translation_world_px"][0],6)
        # Rotation about the area centroid has zero centroid translation.
        states[0,2:4]=center-local;states[1,2:4]=center-np.asarray([-local[1],local[0]])
        result=motion_components("T",states)
        self.assertLess(result["translation_world_px"][0],1e-10)
        self.assertEqual(result["motion_class"],["moving"])

    def test_fixed_thresholds_and_zero_inclusive_continuous_quantiles(self):
        states=record(x=[200,200,200.5,201.5,203.5])["arrays"]["states"]
        result=motion_components("L",states)
        self.assertEqual(result["motion_class"],["near_static","near_static","intermediate","moving"])
        continuous=continuous_distribution(result["translation_world_px"])
        self.assertEqual(continuous["total"],4);self.assertEqual(continuous["zeros"],1)
        self.assertEqual(continuous["quantiles"]["0"],0)
        self.assertAlmostEqual(continuous["quantiles"]["50"],.75)
        states=record(x=[200,200,200])["arrays"]["states"]
        states[:,4]=(0,.005,.005+math.radians(1))
        self.assertEqual(motion_components("L",states)["motion_class"],["near_static","moving"])

    def test_unwrapped_full_rotation_cannot_be_misclassified_as_static(self):
        states=record(x=[200,200])["arrays"]["states"];states[:,4]=(0,2*math.pi)
        result=motion_components("L",states)
        self.assertEqual(result["motion_class"],["moving"])
        self.assertAlmostEqual(result["rotation_abs_rad"][0],2*math.pi)
        states[:,4]=(math.pi-.002,-math.pi+.002)
        self.assertEqual(motion_components("L",states)["motion_class"],["moving"])
        self.assertEqual(motion_components("L",states,"wrapped")["motion_class"],["near_static"])

    def test_raw_controls_and_h1_candidate_denominators_keep_static_and_invalid(self):
        source=record();result=summarize_records([source],expected_letters=("L",))
        self.assertEqual(result["all_real_transitions"]["total"],8)
        self.assertEqual(result["raw_h1_anchor_universe"]["total"],5)
        self.assertEqual(result["valid_h1_candidates"]["total"],5)
        self.assertEqual(result["valid_h1_candidates"]["class_counts"],dict(moving=1,intermediate=1,near_static=3))
        self.assertFalse(result["per_letter_gate"]["L"]["passed"])
        source["arrays"]["substep_penetrations"][4,5,1]=3
        result=summarize_records([source],expected_letters=("L",))
        self.assertEqual(result["all_real_transitions"]["total"],8)
        self.assertEqual(result["all_real_transitions"]["invalid"],1)
        self.assertEqual(result["raw_h1_anchor_universe"]["total"],5)
        self.assertEqual(result["valid_h1_candidates"]["total"],1)
        self.assertEqual(result["invalid_history_windows"],4)
        self.assertTrue(result["per_letter_gate"]["L"]["moving_target_passed"])
        self.assertFalse(result["per_letter_gate"]["L"]["passed"])

    def test_missing_substep_and_underreported_wall_evidence_do_not_pass(self):
        source=record(x=[36,36]);source["arrays"].pop("substep_penetrations")
        rows=transition_rows(source["entry"],source["arrays"],source["steps"])
        self.assertFalse(rows[0]["independent_valid"])
        self.assertTrue(rows[0]["geometric_wall_penetration_underreported"])
        self.assertEqual(rows[0]["max_penetration_world_px"],1)
        source=record();source["arrays"]["states"][5,2]=float("nan")
        result=summarize_records([source],expected_letters=("L",))
        self.assertEqual(result["all_real_transitions"]["total"],8)
        self.assertEqual(result["all_real_transitions"]["continuous"]["translation_world_px"]["missing_or_nonfinite"],2)

    def test_signed_gaps_and_both_corner_walls_are_retained(self):
        source=record(x=[36.5,37,45,69,101]);states=source["arrays"]["states"]
        gaps=geometric_wall_gaps("L",states)
        np.testing.assert_allclose(gaps[:,0],[-.5,0,8,32,64])
        self.assertEqual([gap_band(value) for value in gaps[:,0]],["negative","0-8","0-8","8-32","32-64"])
        source=record(x=[45,45],location="left-top");source["arrays"]["states"][:,3]=114
        row=transition_rows(source["entry"],source["arrays"],source["steps"])[0]
        self.assertEqual(row["left_gap_before"],8);self.assertEqual(row["top_gap_before"],32)
        self.assertEqual(row["gap_band_before"],"0-8")
        self.assertEqual(row["requested_corner_max_gap_before"],32)
        self.assertEqual(row["requested_corner_gap_band_before"],"8-32")

    def test_boundary_behavior_is_gap_observation_or_explicit_contact_proxy(self):
        examples=[("along_wall_moving",[37,37],[220,223],False),
                  ("release_from_boundary",[37,40],[220,220],False),
                  ("approaching_wall",[40,37],[220,220],False),
                  ("likely_wall_blocked_proxy",[37,37],[220,220],True),
                  ("interior",[200,203],[220,220],False)]
        for expected,x,y,contact in examples:
            source=record(x=x,contact=[int(contact)]);source["arrays"]["states"][:,3]=y
            source["arrays"]["actions"][:]=(0,100)
            row=transition_rows(source["entry"],source["arrays"],source["steps"])[0]
            self.assertEqual(row["boundary_behavior"],expected)
        source=record(x=[200,200]);row=transition_rows(source["entry"],source["arrays"],source["steps"])[0]
        self.assertEqual(row["boundary_behavior"],"near_static_contact")
        self.assertFalse(row["likely_wall_blocked_proxy"])

    def test_proposed_index_requires_real_support_and_exact_integer_ratio(self):
        rows=[dict(index_stratum=name) for name,n in (("moving",7),("intermediate",1),("near_static_contact",1),("near_static_no_contact",1)) for _ in range(n)]
        result=proposed_index_feasibility(rows,10)
        self.assertTrue(result["feasible_without_replacement"])
        self.assertEqual(result["max_exact_without_replacement_count"],10)
        self.assertFalse(proposed_index_feasibility(rows,19)["exact_ratio_integer_count"])
        result=proposed_index_feasibility(rows[:-1],10)
        self.assertFalse(result["feasible_with_replacement"])
        self.assertEqual(result["missing_strata"],["near_static_no_contact"])

    def test_missing_letter_stays_failed_and_files_preserve_raw_rows(self):
        source=record(x=np.arange(9)*2+200)
        with tempfile.TemporaryDirectory() as folder:
            result=summarize_records([source],expected_letters=("L","T"),output=folder)
            self.assertTrue(result["all_present_letters_raw_gate_passed"])
            self.assertFalse(result["all_expected_letters_raw_gate_passed"])
            self.assertEqual(result["missing_letters"],["T"])
            saved=json.loads((Path(folder)/"motion-statistics.json").read_text())
            self.assertNotIn("rows",saved)
            self.assertEqual(len((Path(folder)/"real-transition-statistics.csv").read_text().splitlines()),9)
            with self.assertRaises(FileExistsError):summarize_records([source],expected_letters=("L",),output=folder)

    def test_unseen_or_escaping_record_rejected_before_file_open(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,"sealed"):
                read_records(folder,dict(entries=[dict(shape="D",file="not-created.npz")]))
            with self.assertRaisesRegex(ValueError,"escapes"):
                read_records(folder,dict(entries=[dict(shape="L",file="../not-created.npz")]))

    def test_contact_sheet_uses_saved_consecutive_frames_and_prefers_blocked_proxy(self):
        moving=record("moving",x=np.arange(9)*2+200)
        blocked=record("blocked",x=np.repeat(37.,9),location="left")
        blocked["arrays"]["actions"][:]=(0,100)
        static=record("static",x=np.repeat(200.,9))
        calls=[]
        class RenderOnly:
            def reset_scene(self):calls.append("reset")
            def set_render_state(self,state):calls.append(float(state[2]))
            def render_clean(self):return np.full((3,96,96),119,np.uint8)
            def step(self,*args):raise AssertionError("No physics control is permitted")
        with tempfile.TemporaryDirectory() as folder,patch("tdyssey.native_rgb_env.make_env",return_value=RenderOnly()):
            result=render_motion_contact_sheet([static,moving,blocked],"unused",folder,max_cases=2)
            self.assertEqual(result["physics_steps_executed"],0)
            self.assertEqual([case["trajectory_id"] for case in result["cases"]],["moving","blocked"])
            for case in result["cases"]:self.assertEqual(case["frames"],list(range(case["anchor"]-3,case["anchor"]+2)))
            self.assertEqual(len(calls),11)
            self.assertTrue((Path(folder)/"motion-vs-near-static-contact.png").is_file())
            self.assertTrue(result["cases"][1]["likely_wall_blocked_proxy"])


if __name__=="__main__":unittest.main()
