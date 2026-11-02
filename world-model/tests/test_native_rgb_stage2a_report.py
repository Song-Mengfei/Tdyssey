"""Formal TRAIN report checks over analytic saved arrays; zero real controls."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.native_rgb_stage2a_report import (
    CATEGORIES,NEW_LETTERS,analyze_records,build_report,horizon_audit,render_examples,select_examples,
)
from tdyssey.native_rgb_data import TRAIN_LETTERS
from tdyssey.native_rgb_motion_statistics import SPATIAL_CLASSES


def record(letter="A",location="center",category=None):
    n=64;states=np.zeros((65,15));states[:,:2]=(100,100);states[:,2:4]=(220,220)
    if category=="moving":states[:,2]+=np.arange(65)*2
    if category=="rotation":states[:,4]=np.arange(65)*.02
    arrays=dict(states=states,actions=np.tile([256.,400.],(n,1)),contact=np.ones(n) if category=="near_static_contact" else np.zeros(n),
        valid_transition=np.ones(n,bool),penetrations=np.zeros((65,3)),substep_penetrations=np.zeros((64,10,3)))
    identifier=letter+"-"+location
    steps=[dict(phase="push",intent="push") for _ in range(64)]
    if category=="failure":steps[63]["observed"]=dict(recovery=dict(episode_id=1,status="failed",failure_reason="path_timeout"))
    return dict(entry=dict(trajectory_id=identifier,shape=letter,location=location,split="train",file=identifier+".npz"),
                arrays=arrays,steps=steps)


def requests():
    return dict(requests=[dict(letter=letter,spatial_class=location,split="train",controls=64,seed=index)
        for index,(letter,location) in enumerate((l,p) for l in TRAIN_LETTERS for p in SPATIAL_CLASSES)])


class Stage2AReportTests(unittest.TestCase):
    def test_exact_complete_window_counts_and_exclusive_boundary_exclusions(self):
        source=record()
        h1,h4=horizon_audit(source,1),horizon_audit(source,4)
        self.assertEqual((h1["candidate_universe"],h1["valid_candidates"]),(61,61))
        self.assertEqual((h4["candidate_universe"],h4["valid_candidates"]),(58,58))
        self.assertEqual(h4["valid_anchors"],list(range(3,61)))
        self.assertEqual(h4["exclusive_anchor_outcomes"],dict(insufficient_recorded_history=3,valid=58,insufficient_recorded_future=3))
        self.assertEqual(h4["elapsed_seconds"],.4)

    def test_invalid_substep_excludes_all_affected_history_and_future_without_hiding_denominator(self):
        source=record();source["arrays"]["substep_penetrations"][42,5,0]=3
        h1,h4=horizon_audit(source,1),horizon_audit(source,4)
        self.assertEqual(h1["candidate_universe"],61);self.assertEqual(h1["valid_candidates"],57)
        self.assertEqual(h4["candidate_universe"],58);self.assertEqual(h4["valid_candidates"],51)
        self.assertEqual(h4["exclusive_anchor_outcomes"]["invalid_history"],3)
        self.assertEqual(h4["exclusive_anchor_outcomes"]["invalid_future"],4)
        self.assertEqual(h4["invalid_source_reasons_nonexclusive"]["penetration_over_fixed_2px"],7)
        source["arrays"]["substep_penetrations"][44,5,0]=3
        self.assertGreater(horizon_audit(source,4)["exclusive_anchor_outcomes"]["invalid_history_and_future"],0)

    def test_h4_endpoint_and_cumulative_path_are_different_and_do_not_change_h1_gate(self):
        source=record("L");source["arrays"]["states"][4:8,2]+=[2,4,2,0]
        source["arrays"]["states"][4:8,4]=np.deg2rad([1,2,1,0])
        result=horizon_audit(source,4)
        self.assertAlmostEqual(result["continuous_values"]["endpoint_translation_world_px"][0],0)
        self.assertAlmostEqual(result["continuous_values"]["path_translation_world_px"][0],8)
        self.assertAlmostEqual(result["continuous_values"]["endpoint_rotation_abs_deg"][0],0)
        self.assertAlmostEqual(result["continuous_values"]["path_rotation_abs_rad"][0],np.deg2rad(4))
        report=analyze_records([source],planned_requests=requests())
        self.assertFalse(report["horizons"]["4"]["h1_gate_changed"])
        self.assertFalse(report["per_letter"]["L"]["h1_gate"]["passed"])

    def test_missing_slots_strata_support_and_static_windows_stay_visible(self):
        source=record();before=source["arrays"]["states"].copy();report=analyze_records([source],planned_requests=requests())
        self.assertEqual(report["planned_controls"],10368);self.assertEqual(report["recorded_controls"],64)
        self.assertEqual(report["missing_planned_controls"],10304)
        self.assertEqual(len(report["missing_slots"]),161)
        self.assertEqual(report["missing_unrecorded_h1_candidate_windows"],9882-61)
        self.assertEqual(report["missing_unrecorded_h4_candidate_windows"],9396-58)
        support=report["per_letter"]["A"]["strata_support"]
        self.assertEqual(support["strata"]["near_static_no_contact"],dict(unique_windows=61,independent_trajectories=1))
        self.assertFalse(support["feasibility_analysis_only"]["feasible_with_replacement"])
        self.assertEqual(report["index_draws"],0);self.assertEqual(len(report["no_contact_trajectories"]),1)
        np.testing.assert_array_equal(source["arrays"]["states"],before)

    def test_formal_split_duplicate_slots_and_incomplete_request_plan_rejected(self):
        source=record();source["entry"]["split"]="seen_val"
        with self.assertRaisesRegex(ValueError,"formal train"):analyze_records([source])
        source=record()
        with self.assertRaisesRegex(ValueError,"Repeated"):analyze_records([source,source])
        with self.assertRaisesRegex(ValueError,"162"):analyze_records([source],planned_requests=requests()["requests"][:-1])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/"manifest.json").write_text(json.dumps(dict(entries=[dict(shape="D",split="train",file="not-opened.npz")])))
            with self.assertRaisesRegex(ValueError,"sealed"):build_report(path,path/"output",planned_requests=requests())

    def test_fixed_example_selection_covers_twelve_new_letters_and_discloses_missing_category(self):
        records=[record(letter,category=CATEGORIES[index%4]) for index,letter in enumerate(NEW_LETTERS)]
        first=select_examples(records);second=select_examples(list(reversed(records)))
        self.assertEqual([(r["row_id"],r["selected_category"]) for r in first["cases"]],[(r["row_id"],r["selected_category"]) for r in second["cases"]])
        self.assertEqual({r["letter"] for r in first["cases"]},set(NEW_LETTERS))
        self.assertEqual(len(first["cases"]),12)
        self.assertTrue(all(not r["desired_category_unavailable"] for r in first["cases"]))
        for source in records:
            if source["steps"][63].get("observed"):source["steps"][63].pop("observed")
        changed=select_examples(records)
        self.assertTrue(any(r["desired_category_unavailable"] for r in changed["cases"]))
        self.assertTrue(any(r.get("desired_category")=="failure" for r in changed["missing"]))

    def test_saved_frame_examples_bounded_without_physics_or_padding(self):
        records=[record(letter,category=CATEGORIES[index%4]) for index,letter in enumerate(NEW_LETTERS)]
        calls=[]
        class Renderer:
            def reset_scene(self):calls.append("reset")
            def set_render_state(self,state):calls.append(float(state[2]))
            def render_clean(self):return np.full((3,96,96),119,np.uint8)
            def step(self,*args):raise AssertionError("No physics controls allowed")
        with tempfile.TemporaryDirectory() as folder,patch("tdyssey.native_rgb_env.make_env",return_value=Renderer()):
            output=Path(folder)/"examples";result=render_examples(records,"unused",output)
            self.assertEqual(len(result["sequences"]),12)
            self.assertEqual(len(list(output.glob("*.png"))),2)
            self.assertLessEqual(len(list(output.glob("*.gif"))),4)
            for sequence in result["sequences"]:
                self.assertEqual(len(sequence["frames"]),8)
                self.assertEqual(sequence["frames"],list(range(sequence["frames"][0],sequence["frames"][0]+8)))
                self.assertLessEqual(sequence["frames"][-1],64)
            self.assertEqual(result["physics_controls_executed"],0)
            self.assertEqual(len(calls),12+12*8)

    def test_report_writes_new_outputs_and_preserves_frozen_statistics_source(self):
        source=record();entry=source["entry"].copy();entry["metadata_file"]="metadata.json"
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);data=path/"data";data.mkdir()
            np.savez_compressed(data/entry["file"],**source["arrays"])
            (data/entry["metadata_file"]).write_text(json.dumps(dict(steps=source["steps"])))
            (data/"manifest.json").write_text(json.dumps(dict(entries=[entry])))
            report=build_report(data,path/"report",planned_requests=requests())
            self.assertEqual(report["model_calls"],0)
            self.assertTrue((path/"report/h1/motion-statistics.json").is_file())
            self.assertTrue((path/"report/stage2a-report.json").is_file())
            self.assertFalse((path/"report/examples").exists())
            with self.assertRaises(FileExistsError):build_report(data,path/"report",planned_requests=requests())


if __name__=="__main__":unittest.main()
