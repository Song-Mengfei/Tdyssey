"""CPU-only process/accounting tests. Mock work executes zero physics controls."""

import os
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

from tdyssey.native_rgb_data import BudgetLedger,SPLIT_CAPS
from tdyssey.native_rgb_parallel import (allocate_jobs,run_budgeted_jobs,collect_letter_job,
    sampling_policy,staging_candidates,AdaptiveArcSampler)


def metadata_worker(job):
    time.sleep(job.get("delay",.05))
    return dict(actual_controls=job["reserved_controls"],rejected_resets=0,
        seed_value=random.Random(job["seed"]).random(),worker_pid=os.getpid(),real_physics_controls=0)


def failing_metadata_worker(job):
    raise RuntimeError("deliberate zero-physics mock failure")


def overclaim_metadata_worker(job):
    return dict(actual_controls=job["reserved_controls"]+1)


def temporary_directory():
    parent=Path(__file__).resolve().parents[1]/"artifacts/native-rgb-parallel-test-tmp"
    parent.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


class AllocationTests(unittest.TestCase):
    def test_remaining_local_budget_balanced_and_no_new_scenes_executed(self):
        jobs=allocate_jobs({},SPLIT_CAPS,{"train":1416,"seen_val":0},("train","seen_val"))
        train=[job for job in jobs if job["split"]=="train"]
        self.assertEqual(sum(job["reserved_controls"] for job in train),26232)
        counts=[len(job["scenes"]) for job in train]
        self.assertEqual((min(counts),max(counts)),(22,23))
        self.assertEqual(sum(job["reserved_controls"] for job in jobs),33144)
        single=allocate_jobs({},SPLIT_CAPS,{},("seen_val",))
        all_splits=allocate_jobs({},SPLIT_CAPS,{},("train","seen_val"))
        self.assertEqual(single[0]["scenes"],next(job for job in all_splits if job["id"]==single[0]["id"])["scenes"])


class GeometryAndSamplerTests(unittest.TestCase):
    def env(self,geometry):
        from types import SimpleNamespace
        return SimpleNamespace(block=SimpleNamespace(angle=0,local_to_world=lambda point:point),world_geometry=lambda:geometry)

    def segments(self,count=3):
        return [dict(id=i,ring=0,contact_count=0 if i==0 else 100,features=["thin"] if i==0 else []) for i in range(count)]

    def candidates(self,segments,distances=(18.,16.)):
        return [dict(segment_id=row["id"],staging_distance=distance,staging=[200,200])
            for row in segments for distance in distances]

    def test_sampler_policy_is_predeclared_and_does_not_change_physics(self):
        from tdyssey.native_rgb_env import PROTOCOL_SHA256
        policy=sampling_policy({})
        self.assertEqual(policy["staging_distances"],[18.,16.])
        self.assertEqual(policy["max_same_approach_misses"],2)
        self.assertEqual(policy["physics_protocol_sha256"],PROTOCOL_SHA256)
        for collection in ({"max_same_approach_misses":3},{"staging_distances":[14]},
                           {"staging_distances":[18,18]},{"staging_distances":[float("nan")]}):
            with self.assertRaises(ValueError):sampling_policy(dict(collection=collection))

    def test_shorter_staging_can_be_geometry_legal_near_a_wall_without_physics_steps(self):
        from shapely.geometry import box,Point,LineString
        env=self.env(box(450,200,472,240));segment=dict(id=0,ring=0,point=[472,220],normal=[1,0])
        rows=staging_candidates(env,[segment],[18,16])
        self.assertEqual([row["staging_distance"] for row in rows],[16.])
        self.assertEqual(rows[0]["staging"],[488.,220.])
        self.assertGreaterEqual(env.world_geometry().distance(Point(*rows[0]["staging"])),15.5)
        self.assertTrue(box(22.5,22.5,488.5,488.5).covers(LineString(rows[0]["normal_contact_path"])))

    def test_shorter_staging_cannot_cross_a_thin_wall_or_enter_a_closed_hole(self):
        from shapely.geometry import box
        from shapely.ops import unary_union
        env=self.env(unary_union((box(240,200,250,240),box(251,200,252,240))))
        segment=dict(id=0,ring=0,point=[250,220],normal=[1,0])
        self.assertEqual(staging_candidates(env,[segment],[18,16]),[])
        segment["ring"]=1
        self.assertEqual(staging_candidates(self.env(box(200,200,220,240)),[segment],[18,16]),[])

    def test_persistent_target_miss_rotates_bins_and_approaches_without_adding_actual_hits(self):
        import numpy as np
        segments=self.segments();sampler=AdaptiveArcSampler(segments,sampling_policy({}));rng=np.random.default_rng(7)
        choices=[]
        for attempt in range(120):
            chosen=sampler.choose(self.candidates(segments),rng,f"independent-pose-{attempt}")
            token=chosen["sampler_choice"];choices.append(token)
            sampler.record_outcome(token["choice_id"],False)
        planned=[row["planned_attempts"] for row in sampler.snapshot()]
        self.assertLessEqual(max(planned)-min(planned),2)
        runs=[];last=None;length=0
        for token in choices:
            if token["segment_id"]==last:length+=1
            else:runs.append(length);last=token["segment_id"];length=1
        self.assertLessEqual(max(runs+[length]),2)
        for segment_id in range(3):
            per_bin=[token for token in choices if token["segment_id"]==segment_id]
            for i in range(0,len(per_bin)-2,2):
                self.assertNotEqual((per_bin[i]["approach_cycle"],per_bin[i]["variant"]),
                    (per_bin[i+2]["approach_cycle"],per_bin[i+2]["variant"]))
        self.assertEqual([row["contact_count"] for row in segments],[0,100,100])
        self.assertTrue(all(row["actual_target_hits"]==0 for row in sampler.snapshot()))

    def test_exhausted_approaches_require_independent_pose_and_do_not_mark_unreachable(self):
        import numpy as np
        segments=self.segments(1);sampler=AdaptiveArcSampler(segments,sampling_policy({}));rng=np.random.default_rng(4)
        for _ in range(12):
            chosen=sampler.choose(self.candidates(segments),rng,"same-pose")
            sampler.record_outcome(chosen["sampler_choice"]["choice_id"],False)
        with self.assertRaises(RuntimeError):sampler.choose(self.candidates(segments),rng,"same-pose")
        chosen=sampler.choose(self.candidates(segments),rng,"new-independent-pose")
        self.assertEqual(chosen["sampler_choice"]["approach_cycle"],1)
        self.assertNotIn("unreachable",sampler.snapshot()[0])


class ProcessTests(unittest.TestCase):
    def jobs(self,count=4):
        return [dict(id=f"mock-{i}",split="train",reserved_controls=2,seed=100+i) for i in range(count)]

    def test_real_spawn_workers_are_bounded_seed_repeatable_and_parent_accounted(self):
        with temporary_directory() as first,temporary_directory() as second:
            serial=run_budgeted_jobs(self.jobs(),metadata_worker,BudgetLedger(first,{"train":8}),workers=1,job_timeout_seconds=15)
            parallel=run_budgeted_jobs(self.jobs(),metadata_worker,BudgetLedger(second,{"train":8}),workers=2,job_timeout_seconds=15)
            if serial["status"]!="completed" or parallel["status"]!="completed":
                log=Path(__file__).resolve().parents[1]/"artifacts/native-rgb-parallel-test-failure.json"
                log.write_text(json.dumps(dict(serial=serial,parallel=parallel),indent=2)+"\n",encoding="utf8")
            self.assertEqual(serial["status"],"completed",json.dumps(serial,sort_keys=True))
            self.assertEqual(parallel["status"],"completed",json.dumps(parallel,sort_keys=True))
            self.assertEqual(parallel["max_active_workers"],2)
            self.assertTrue(all(row["pid"]!=os.getpid() for row in parallel["results"]))
            self.assertEqual(len({row["pid"] for row in parallel["results"]}),4)
            values=lambda report:{row["job_id"]:row["result"]["seed_value"] for row in report["results"]}
            self.assertEqual(values(serial),values(parallel))
            self.assertEqual(parallel["budget"]["charged_controls"]["train"],8)
            self.assertEqual(parallel["budget"]["actual_controls"],8)
            self.assertTrue(all(row["result"]["real_physics_controls"]==0 for row in parallel["results"]))

    def test_failure_preserves_reservation_and_does_not_launch_pending_job(self):
        with temporary_directory() as directory:
            report=run_budgeted_jobs(self.jobs(),failing_metadata_worker,BudgetLedger(directory,{"train":8}),workers=1,job_timeout_seconds=15)
            self.assertEqual(report["status"],"failed")
            self.assertEqual(report["budget"]["charged_controls"]["train"],2)
            self.assertEqual(report["budget"]["actual_controls"],0)
            self.assertEqual(len(report["pending_job_ids"]),3)
            self.assertEqual(report["results"][0]["status"],"failed")

    def test_overclaim_is_a_failed_whole_run_and_cancels_pending_dispatch(self):
        with temporary_directory() as directory:
            report=run_budgeted_jobs(self.jobs(),overclaim_metadata_worker,BudgetLedger(directory,{"train":8}),workers=1,job_timeout_seconds=15)
            self.assertEqual(report["status"],"failed")
            self.assertEqual(report["error"]["type"],"InvalidWorkerAccounting")
            self.assertEqual(report["budget"]["actual_controls"],0)
            self.assertEqual(len(report["pending_job_ids"]),3)

    def test_transient_ledger_sink_denials_never_repeat_budget_mutations(self):
        from tdyssey import native_rgb_data
        original=native_rgb_data.write_json;calls=[]
        def sharing_denial(path,value):
            calls.append(value["attempts"][-1]["status"])
            if len(calls) in (1,2):raise PermissionError("injected transient ledger sharing denial")
            return original(path,value)
        with temporary_directory() as directory:
            ledger=BudgetLedger(directory,{"train":2})
            with patch("tdyssey.native_rgb_data.write_json",side_effect=sharing_denial):
                report=run_budgeted_jobs(self.jobs(1),metadata_worker,ledger,workers=1,job_timeout_seconds=15)
            self.assertEqual(report["status"],"completed",json.dumps(report,sort_keys=True))
            self.assertEqual(calls,["reserved","completed"])
            self.assertEqual(len(ledger.value["attempts"]),1)
            self.assertEqual(ledger.value["charged_controls"]["train"],2)
            self.assertEqual(ledger.value["actual_controls"],2)
            self.assertEqual(json.loads(ledger.path.read_text()),ledger.value)

    def test_duplicate_ids_fractional_workers_and_nonfinite_deadlines_are_rejected_before_reservation(self):
        with temporary_directory() as directory:
            ledger=BudgetLedger(directory,{"train":8});jobs=self.jobs(2);jobs[1]["id"]=jobs[0]["id"]
            with self.assertRaises(ValueError):run_budgeted_jobs(jobs,metadata_worker,ledger,workers=1)
            with self.assertRaises(ValueError):run_budgeted_jobs(self.jobs(),metadata_worker,ledger,workers=1.5)
            with self.assertRaises(ValueError):run_budgeted_jobs(self.jobs(),metadata_worker,ledger,workers=1,job_timeout_seconds=float("nan"))
            self.assertEqual(ledger.value["attempts"],[])

    def test_real_worker_import_and_scene_setup_execute_zero_physics_controls(self):
        source=Path(__file__).resolve().parents[2]/"Tdyssey/third_party/diffusion_policy"
        if not (source/"diffusion_policy/env/pusht/pusht_env.py").exists():
            self.skipTest("Existing pinned physics source is not available")
        with temporary_directory() as directory:
            root=Path(directory);job=dict(id="train-L",shape="L",split="train",reserved_controls=0,
                scenes=[],source_root=str(source),config={},output=str(root/"data"),
                job_dir=str(root/"worker"),progress_file=str(root/"worker/progress.json"))
            report=run_budgeted_jobs([job],collect_letter_job,BudgetLedger(root,{"train":0}),workers=1,job_timeout_seconds=20)
            self.assertEqual(report["status"],"completed")
            self.assertEqual(report["budget"]["actual_controls"],0)
            self.assertFalse(report["results"][0]["result"]["torch_imported"])
            self.assertEqual(list(root.rglob("*.npz")),[])

    def test_timeout_stops_only_owned_child_without_running_physics(self):
        with temporary_directory() as directory:
            jobs=self.jobs(1);jobs[0]["delay"]=2
            report=run_budgeted_jobs(jobs,metadata_worker,BudgetLedger(directory,{"train":2}),workers=1,job_timeout_seconds=.1)
            self.assertEqual(report["status"],"failed")
            self.assertEqual(report["error"]["type"],"WorkerTimeout")
            self.assertTrue(report["only_owned_processes_terminated"])
            self.assertEqual(report["budget"]["charged_controls"]["train"],2)


if __name__=="__main__":unittest.main()
