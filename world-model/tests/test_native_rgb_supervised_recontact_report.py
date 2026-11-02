"""CPU mocks of saved-data reporting; no physical controls or model calls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.native_rgb_supervised_recontact_report import (
    build_report,compare_records,recovery_episodes,render_pairs,summarize_record,
)


def record(identifier,moving=False):
    n=64;states=np.zeros((n+1,15));states[:,:2]=np.column_stack((100+np.arange(n+1),np.repeat(100,n+1)))
    states[:,2:4]=(200,220)
    if moving:states[13:,2]+=np.minimum(np.arange(n-12),3)*2
    arrays=dict(states=states,actions=np.tile((300.,100.),(n,1)),contact=np.zeros(n),valid_transition=np.ones(n,bool),
                penetrations=np.zeros((n+1,3)),substep_penetrations=np.zeros((n,10,3)))
    steps=[dict(phase="push",intent="push",lost_contact_streak=0) for _ in range(n)]
    for t in range(10,15):steps[t]["phase"]="contact_recovery_route"
    return dict(entry=dict(trajectory_id=identifier,shape="L",location="center",file=identifier+".npz"),arrays=arrays,steps=steps)


def fixtures():
    old=[record("old"+str(i)) for i in range(6)]
    new=[record("new"+str(i),moving=True) for i in range(8)]
    slots=[dict(slot_id="slot"+str(i),role="paired" if i<6 else "new",new_trajectory_id="new"+str(i),
                baseline_trajectory_id="old"+str(i) if i<6 else None,initial_state=new[i]["arrays"]["states"][0].tolist()) for i in range(8)]
    return old,new,slots


class RecontactReportTests(unittest.TestCase):
    def test_staging_is_not_contact_and_physical_metrics_use_recorded_arrays(self):
        source=record("new",moving=True)
        for t in range(10,15):
            source["steps"][t]["recovery"]=dict(episode_id=1,state="path",status="active",route_completed=False,
                staging_arrived=False,route_remaining_world_px=15-(t-10)*3,no_progress_controls=0)
            source["steps"][t]["observed"]=dict(recovery=dict(episode_id=1,state="final_contact",status="active",
                route_completed=True,staging_arrived=True,staging_arrival_step=10,
                first_real_contact_step=None,max_consecutive_contact_controls=0))
        no_contact=recovery_episodes(source,version="new")[0]
        self.assertTrue(no_contact["route_completed"]);self.assertTrue(no_contact["staging_arrived"])
        self.assertFalse(no_contact["physically_recontacted"])
        self.assertIsNone(no_contact["first_real_contact_step"])
        source["arrays"]["contact"][12:15]=1
        for t in range(12,15):
            source["steps"][t]["observed"]["recovery"].update(first_real_contact_step=12,max_consecutive_contact_controls=t-11)
        episode=recovery_episodes(source,version="new")[0]
        self.assertEqual(episode["first_real_contact_step"],12)
        self.assertAlmostEqual(episode["first_real_contact_seconds_from_episode_start"],.3)
        self.assertEqual(episode["max_consecutive_contact_controls"],3)
        self.assertEqual(episode["post_contact_contact_moving_controls"],2)
        self.assertEqual(episode["logger_discrepancies"],[])
        source["steps"][14]["observed"]["recovery"]["first_real_contact_step"]=11
        self.assertIn("first_real_contact_step",recovery_episodes(source,version="new")[0]["logger_discrepancies"])

    def test_logged_failure_and_no_progress_are_retained_without_fake_contact(self):
        source=record("failed")
        for t in range(60,64):
            source["steps"][t]["recovery"]=dict(episode_id="failed-episode",state="failed" if t==63 else "path",
                status="failed" if t==63 else "active",failure_reason="budget_exhausted" if t==63 else None,
                no_progress_controls=t-59,route_remaining_world_px=40,staging_arrived=False,route_completed=False)
        episode=recovery_episodes(source,version="new")[0]
        self.assertEqual(episode["failure_reason"],"budget_exhausted")
        self.assertEqual(episode["reported_no_progress_controls_max"],4)
        self.assertTrue(episode["end_at_trajectory_boundary"])
        self.assertEqual(episode["control_seconds"],.4)
        self.assertFalse(episode["physically_recontacted"])

    def test_old_episodes_are_explicit_approximations_and_cannot_claim_staging(self):
        source=record("old");episode=recovery_episodes(source,version="old")[0]
        self.assertEqual(episode["semantics"],"old_phase_approximation")
        self.assertEqual((episode["first_step"],episode["last_step"]),(10,14))
        self.assertIsNone(episode["route_completed"]);self.assertIsNone(episode["staging_arrived"])
        self.assertEqual(episode["absolute_target_path_world_px"],0)
        self.assertEqual(episode["actual_pusher_path_world_px"],5)
        self.assertGreater(episode["actual_target_distance_first_world_px"],episode["actual_target_distance_last_world_px"])

    def test_all64_and_valid61_kept_separate_including_invalid_history(self):
        source=record("invalid");source["arrays"]["substep_penetrations"][42,0,0]=3
        report=summarize_record(source,version="old")
        self.assertEqual(report["all_real"]["total"],64)
        self.assertEqual(report["all_real"]["invalid"],1)
        self.assertEqual(report["raw_h1_universe"]["total"],61)
        self.assertEqual(report["valid_h1"]["total"],57)
        self.assertEqual(report["invalid_history_windows"],4)
        self.assertEqual(report["valid_h1"]["class_counts"]["near_static"],57)

    def test_pairing_failure_and_missing_slots_preserve_planned_denominators(self):
        old,new,slots=fixtures();new[0]["arrays"]["states"][0,0]+=1
        report=compare_records(old,new[:-2],slots)
        self.assertFalse(report["all_eight_slots_recorded_complete"])
        self.assertFalse(report["all_six_pair_initial_states_exact"])
        self.assertFalse(report["all_recorded_frozen_initial_states_exact"])
        self.assertEqual(report["stage1b_all8"]["planned_controls"],512)
        self.assertEqual(report["stage1b_all8"]["recorded_controls"],384)
        self.assertEqual(report["stage1b_all8"]["missing_planned_controls"],128)
        self.assertEqual(report["new_two_x"]["planned_h1_candidates"],122)
        self.assertEqual(report["new_two_x"]["recorded_slots"],0)
        self.assertTrue(report["slots"][0]["delta_is_unmatched_diagnostic"])

    def test_two_new_and_paired_six_remain_separate_and_no_extra_slot_allowed(self):
        old,new,slots=fixtures();report=compare_records(old,new,slots)
        self.assertTrue(report["all_eight_slots_recorded_complete"])
        self.assertTrue(report["all_six_pair_initial_states_exact"])
        self.assertEqual(report["paired_six_old"]["recorded_controls"],384)
        self.assertEqual(report["paired_six_new"]["recorded_controls"],384)
        self.assertEqual(report["new_two_x"]["recorded_controls"],128)
        self.assertTrue(report["no_phase1_and_phase1b_pooled_gate"])
        with self.assertRaisesRegex(ValueError,"Unexpected"):compare_records(old,new+[record("extra")],slots)
        duplicated=copy.deepcopy(slots);duplicated[1]["slot_id"]=duplicated[0]["slot_id"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):compare_records(old,new,duplicated)

    def test_missing_episode_metadata_cannot_stitch_contact_streak(self):
        source=record("gapped")
        for t in (10,14):
            source["steps"][t]["recovery"]=dict(episode_id=1,state="final_contact",status="active")
            source["arrays"]["contact"][t]=1
        episode=recovery_episodes(source,version="new")[0]
        self.assertTrue(episode["nonconsecutive_episode_metadata"])
        self.assertEqual(episode["max_consecutive_contact_controls"],1)

    def test_report_files_are_bounded_and_do_not_overwrite(self):
        old,new,slots=fixtures()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,records in (("baseline",old),("variant",new)):
                data=root/name;data.mkdir();entries=[]
                for item in records:
                    entry=copy.deepcopy(item["entry"]);entry["metadata_file"]=entry["trajectory_id"]+".json"
                    np.savez_compressed(data/entry["file"],**item["arrays"])
                    (data/entry["metadata_file"]).write_text(json.dumps(dict(steps=item["steps"])))
                    entries.append(entry)
                (data/"manifest.json").write_text(json.dumps(dict(entries=entries)))
            report=build_report(root/"baseline",root/"variant",root/"output",slots=slots)
            self.assertEqual(report["model_calls"],0)
            self.assertTrue((root/"output/recontact-report.md").is_file())
            self.assertNotIn("initial_state",report["slots"][0])
            with self.assertRaises(FileExistsError):build_report(root/"baseline",root/"variant",root/"output",slots=slots)

    def test_paired_renderer_only_reads_five_consecutive_saved_states(self):
        old,new,slots=fixtures();report=compare_records(old,new,slots);calls=[]
        class Renderer:
            def reset_scene(self):calls.append("reset")
            def set_render_state(self,state):calls.append(float(state[0]))
            def render_clean(self):return np.zeros((3,96,96),np.uint8)
            def step(self,*args):raise AssertionError("No real control permitted")
        with tempfile.TemporaryDirectory() as folder,patch("tdyssey.native_rgb_env.make_env",return_value=Renderer()):
            result=render_pairs(old,new,report,"unused",folder,max_pairs=2)
            self.assertEqual(len(result["sequences"]),4)
            for row in result["sequences"]:self.assertEqual(row["frames"],list(range(row["frames"][0],row["frames"][0]+5)))
            self.assertEqual(len(calls),21)
            self.assertEqual(result["physics_controls_executed"],0)


if __name__=="__main__":unittest.main()
