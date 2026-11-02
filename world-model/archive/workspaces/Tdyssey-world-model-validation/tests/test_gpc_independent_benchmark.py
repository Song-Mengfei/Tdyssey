"""Pure geometry and constructed NumPy simulator substitutes, never physics."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import gpc_independent_benchmark as benchmark
from tdyssey.collection import reset_issues
from tdyssey.dataset import json_bytes
from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.state import BodyState,Snapshot


class FakeBudget:
    def __init__(self,output):
        self.output=Path(output);self.pending=None;self.counts=Counter();self.completed=Counter()
        self.records={};self.failure=None;self.ordinal=0;self.caps=deepcopy(benchmark.LIMITS)
    def check(self):
        if self.failure:raise RuntimeError(self.failure)
    def reserve(self,**costs):
        self.check()
        if self.pending is not None:raise AssertionError("Nested fake operation")
        if any(self.counts[k]+v>self.caps[k] for k,v in costs.items()):raise OSError("fixed fake cap")
        self.ordinal+=1;self.pending=(self.ordinal,costs);self.counts.update(costs)
        return self.ordinal
    def complete(self,operation,detail=None):
        if operation!=self.pending[0]:raise AssertionError("Completion mismatch")
        self.completed.update(self.pending[1]);self.pending=None
    def fail(self,reason):
        if self.failure is None:self.failure=reason
    def write_bytes(self,name,payload,run_id=None,replace=False):
        self.check()
        if run_id!="B" or replace:raise AssertionError("Wrong account or overwrite")
        if name in self.records:raise AssertionError("Duplicate immutable output")
        if sum(self.records.values())+len(payload)>self.caps["output_bytes"]:raise OSError("fake bytes cap")
        path=self.output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:stream.write(payload)
        self.records[name]=len(payload)
        return dict(name=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())


class FakeEnv:
    def __init__(self,shape,budget,factory):
        self.project_shape=shape;self.sim_hz=100;self.control_hz=10
        self.space=None;self.budget=budget;self.factory=factory;self.closed=False;self.state=None
        factory.active+=1;factory.max_active=max(factory.max_active,factory.active)
    def seed(self,value):self.seed_value=value
    def assert_cost(self,name):
        if self.budget.pending is None or self.budget.pending[1].get(name)!=1:
            raise AssertionError("Physical substitute was called without reservation")
    def reset(self):
        self.assert_cost("resets");self.factory.calls["reset"]+=1
        if self.factory.fail_reset:raise RuntimeError("injected fake reset failure")
        self.space=object();self.success_threshold=self.factory.reset_threshold
        self.state=Snapshot(self.project_shape,official_shape(self.project_shape).geometry_hash,
            hashlib.sha256(self.project_shape.encode()).hexdigest(),BodyState(Pose((180+self.seed_value%9,180),.1)),
            BodyState(Pose((55,55),0)),benchmark.goal_for(self.project_shape),0)
        return self.observation()
    def observation(self):
        import numpy as np
        value=np.zeros((3,96,96),dtype=np.float32)
        value[0,0,0]=self.state.control_step/20
        value[1,0,0]=self.state.object_body.pose.position[0]/512
        value[2,0,0]=self.state.pusher_body.pose.position[0]/512
        return dict(image=value,agent_pos=np.asarray(self.state.pusher_body.pose.position))
    def capture_state(self):return self.state
    def restore_state(self,state):
        self.assert_cost("restores");self.factory.calls["restore"]+=1;self.state=state
        if self.factory.corrupt_restore and state.control_step==3:
            self.state=replace(state,pusher_body=replace(state.pusher_body,velocity=(99,0)))
        return self.observation()
    def step(self,action):
        self.assert_cost("controls");self.factory.calls["step"]+=1;index=self.state.control_step+1
        if self.factory.fail_at==index:raise RuntimeError("injected fake step failure")
        body=replace(self.state.object_body,velocity=(float(index),.5),angular_velocity=.01,force=(.1,.2),torque=.3)
        if self.factory.terminal_at==index:body=replace(body,pose=self.state.goal)
        self.state=replace(self.state,object_body=body,pusher_body=BodyState(Pose(action,0),velocity=(.2,.3),force=(.4,.5)),
                           control_step=index,pusher_target=tuple(action))
        coverage=benchmark._label(self.state)["coverage"]
        return self.observation(),min(coverage/.95,1),coverage>.95,dict(n_contacts=0)
    def close(self):
        if not self.closed:self.factory.active-=1;self.factory.calls["close"]+=1;self.closed=True


class Factory:
    def __init__(self,budget,*,terminal_at=None,fail_at=None,corrupt_restore=False,
                 reset_threshold=.95,fail_reset=False):
        self.budget=budget;self.terminal_at=terminal_at;self.fail_at=fail_at;self.corrupt_restore=corrupt_restore
        self.reset_threshold=reset_threshold;self.fail_reset=fail_reset
        self.calls=Counter();self.active=self.max_active=0;self.created=[];self.constructor_records=[]
    def __call__(self,shape):
        self.calls["create"]+=1;self.created.append(shape)
        env=FakeEnv(shape,self.budget,self)
        self.constructor_records.append(dict(threshold_absent=not hasattr(env,"success_threshold"),
            counts=dict(self.budget.counts),pending=self.budget.pending))
        return env


class FixedGeometryChecks(unittest.TestCase):
    def test_fixed_geometry_categories_clearance_and_absolute_actions(self):
        specs=benchmark.fixed_anchors();self.assertEqual(len(specs),12)
        self.assertEqual([s["id"] for s in specs],[f"I{s}{i:02d}" for s in ("T","L") for i in range(6)])
        self.assertEqual([s["seed"] for s in specs],list(range(930000,930012)))
        for spec in specs:
            shape=official_shape(spec["shape"])
            state=Snapshot(shape.name,shape.geometry_hash,"0"*64,BodyState(spec["pose"]),
                BodyState(Pose(spec["pusher"],0)),spec["goal"],0)
            self.assertEqual([v for v in reset_issues(shape,state) if v!="already_near_goal"],[])
            self.assertLessEqual(benchmark._label(state)["coverage"],.95)
            for candidate in benchmark.CANDIDATES:
                actions=benchmark.candidate_actions(spec,candidate)
                self.assertEqual(len(actions),4);self.assertTrue(all(0<=v<=511 for a in actions for v in a))
        self.assertEqual(benchmark.fixed_plan(),benchmark.fixed_plan())


@unittest.skipUnless(importlib.util.find_spec("numpy"),"Constructed array fixtures require NumPy")
class IndependentBenchmarkChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.output=self.root/"output";self.output.mkdir();self.budget=FakeBudget(self.output)
        reviewed=[hashlib.sha256(f"synthetic-reviewed-{i}".encode()).hexdigest() for i in range(316)]
        source=self.root/"reviewed.json";payload=json_bytes(dict(reviewed_scene_sha256=reviewed));source.write_bytes(payload)
        self.patchers=[patch.object(benchmark,"REVIEWED_PATH",source),
            patch.object(benchmark,"REVIEWED_FILE_SHA256",hashlib.sha256(payload).hexdigest()),
            patch.object(benchmark,"REVIEWED_LIST_SHA256",hashlib.sha256(json.dumps(sorted(reviewed),separators=(",",":")).encode()).hexdigest())]
        for item in self.patchers:item.start();self.addCleanup(item.stop)
        self.plan=self.root/"frozen.json";self.plan.write_bytes(json_bytes(benchmark.fixed_plan()))
        self.scenes=set(reviewed);self.seeds=set(benchmark.DEVELOPMENT_SEEDS)
    def run_fake(self,factory,**kwargs):
        return benchmark.run_benchmark(self.output,self.budget,env_factory=factory,excluded_scene_sha256=self.scenes,
            excluded_reset_seeds=self.seeds,plan_path=self.plan,**kwargs)

    def test_official_lifecycle_has_no_threshold_or_operations_before_charged_reset(self):
        factory=Factory(self.budget);env=factory("T");self.addCleanup(env.close)
        self.assertFalse(hasattr(env,"success_threshold"));self.assertIsNone(env.space)
        self.assertEqual(dict(self.budget.counts),{});self.assertEqual(factory.calls["reset"],0)
        self.assertIsNone(self.budget.pending)
        env.seed(930000)
        benchmark._invoke(self.budget,dict(resets=1,renders=1,reset_substeps=1),env.reset)
        self.assertEqual(env.success_threshold,.95)
        self.assertEqual(dict(self.budget.counts),dict(resets=1,renders=1,reset_substeps=1))
        self.assertEqual(self.budget.counts,self.budget.completed)

    def test_threshold_mismatch_after_charged_reset_stops_before_restore_or_history(self):
        factory=Factory(self.budget,reset_threshold=.90)
        with self.assertRaisesRegex(ValueError,"threshold.*charged reset"):self.run_fake(factory)
        self.assertEqual(factory.calls["reset"],1);self.assertEqual(factory.calls["restore"],0)
        self.assertEqual(factory.calls["step"],0);self.assertEqual(factory.active,0)
        self.assertEqual(dict(self.budget.counts),dict(resets=1,renders=1,reset_substeps=1))
        self.assertEqual(self.budget.counts,self.budget.completed);self.assertIsNone(self.budget.pending)
        self.assertEqual(list(self.budget.records),["fixed-plan.json"])

    def test_reset_exception_preserves_charges_without_threshold_read_or_retry(self):
        factory=Factory(self.budget,fail_reset=True)
        with self.assertRaisesRegex(RuntimeError,"injected fake reset failure"):self.run_fake(factory)
        self.assertEqual((factory.calls["reset"],factory.calls["restore"],factory.calls["step"]),(1,0,0))
        self.assertEqual(dict(self.budget.counts),dict(resets=1,renders=1,reset_substeps=1))
        self.assertEqual(dict(self.budget.completed),{});self.assertIsNotNone(self.budget.pending)
        self.assertIn("injected fake reset failure",self.budget.failure);self.assertEqual(factory.active,0)

    def test_complete_grid_exact_charges_and_identical_full_history(self):
        import numpy as np
        factory=Factory(self.budget);report=self.run_fake(factory)
        self.assertEqual((report["status"],len(report["cases"]),len(report["anchors"])),("complete",48,12))
        self.assertEqual(factory.created,["T","L"]);self.assertEqual((factory.max_active,factory.active),(1,0))
        self.assertTrue(all(row["threshold_absent"] for row in factory.constructor_records))
        self.assertEqual(factory.constructor_records[0]["counts"],{})
        self.assertIsNone(factory.constructor_records[0]["pending"])
        self.assertEqual(dict(self.budget.counts),dict(resets=12,restores=60,controls=228,renders=300,control_substeps=2280,reset_substeps=12))
        self.assertEqual(self.budget.counts,self.budget.completed)
        self.assertEqual(report["executed_controls"],228);self.assertFalse(report["coverage_ready"])
        self.assertEqual(len([n for n in self.budget.records if n.endswith(".npz")]),48)
        for anchor in report["anchors"]:
            cases=[c for c in report["cases"] if c["anchor"]==anchor["anchor"]]
            histories=[]
            for case in cases:
                self.assertEqual(case["history_states"],anchor["history_states"])
                self.assertEqual(case["latest_state"],anchor["latest_state"])
                self.assertEqual(case["history_sha256"],anchor["history_sha256"])
                data=(self.output/case["payload"]["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(),case["payload"]["sha256"])
                with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
                    self.assertEqual((arrays["history"].shape,arrays["actual"].shape),((4,3,96,96),(4,3,96,96)))
                    self.assertEqual(arrays["history"].dtype,np.float32);histories.append(arrays["history"].copy())
            self.assertTrue(all(np.array_equal(histories[0],v) for v in histories[1:]))
        self.assertLess(sum(self.budget.records.values()),64*1024**2)

    def test_success_absorbs_future_without_padded_controls_or_renders(self):
        import numpy as np
        factory=Factory(self.budget,terminal_at=4);report=self.run_fake(factory)
        self.assertEqual((self.budget.counts["controls"],self.budget.counts["renders"]),(84,156))
        for case in report["cases"]:
            self.assertEqual((case["executed_future_controls"],case["absorbing_future_frames"]),(1,3))
            self.assertEqual([s["action"] is None for s in case["future"]],[False,True,True,True])
            self.assertTrue(all(s["truth"]["coverage"]==1 for s in case["future"]))
            with np.load(self.output/case["payload"]["path"],allow_pickle=False) as arrays:
                self.assertTrue(all(np.array_equal(arrays["actual"][0],frame) for frame in arrays["actual"][1:]))
        self.assertEqual(factory.active,0)

    def test_reviewed_or_held_scene_conflict_stops_before_control_without_replacement(self):
        spec=benchmark.fixed_anchors()[0];shape=official_shape(spec["shape"])
        state=Snapshot(shape.name,shape.geometry_hash,hashlib.sha256(shape.name.encode()).hexdigest(),
            BodyState(spec["pose"]),BodyState(Pose(spec["pusher"],0)),spec["goal"],0)
        # A held-role manifest can supply this identity without opening a payload.
        held_metadata=dict(role="held_model_test",initial_state_sha256=benchmark.scene_hash(state))
        self.scenes.add(held_metadata["initial_state_sha256"]);factory=Factory(self.budget)
        with self.assertRaisesRegex(ValueError,"Excluded"):self.run_fake(factory)
        self.assertEqual((factory.calls["reset"],factory.calls["step"],factory.calls["restore"]),(1,0,0))
        self.assertEqual(factory.created,["T"]);self.assertEqual(factory.active,0)
        self.assertFalse((self.output/"benchmark.json").exists())

    def test_excluded_seed_missing_reviewed_set_or_changed_frozen_plan_refuse_before_factory(self):
        factory=Factory(self.budget);self.seeds.add(930000)
        with self.assertRaisesRegex(ValueError,"Fixed seed"):self.run_fake(factory)
        self.seeds.remove(930000);self.scenes.pop()
        with self.assertRaisesRegex(ValueError,"Missing reviewed"):self.run_fake(factory)
        self.plan.write_bytes(self.plan.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"frozen"):self.run_fake(factory)
        self.assertEqual(factory.calls["create"],0);self.assertFalse(self.budget.records)

    def test_first_future_exception_retains_spent_reservation_and_never_retries(self):
        factory=Factory(self.budget,fail_at=4)
        with self.assertRaisesRegex(RuntimeError,"injected fake step"):self.run_fake(factory)
        self.assertEqual((factory.calls["step"],self.budget.counts["controls"],self.budget.completed["controls"]),(4,4,3))
        self.assertIsNotNone(self.budget.pending);self.assertIn("injected fake step",self.budget.failure)
        self.assertEqual(factory.active,0);self.assertFalse((self.output/"benchmark.json").exists())

    def test_success_during_short_history_cannot_be_padded_into_independent_case(self):
        factory=Factory(self.budget,terminal_at=2)
        with self.assertRaisesRegex(ValueError,"three real history"):self.run_fake(factory)
        self.assertEqual(self.budget.counts["controls"],2);self.assertEqual(factory.calls["step"],2)
        self.assertFalse(any(name.endswith(".npz") for name in self.budget.records));self.assertEqual(factory.active,0)

    def test_restore_mismatch_and_cap_both_stop_before_any_future_control(self):
        factory=Factory(self.budget,corrupt_restore=True)
        with self.assertRaisesRegex(ValueError,"identical full latest"):self.run_fake(factory)
        self.assertEqual(factory.calls["step"],3);self.assertEqual(self.budget.counts["restores"],2)
        self.assertEqual(factory.active,0)

    def test_admission_cap_rejects_control_before_substitute_is_called(self):
        self.budget.caps["controls"]=3;factory=Factory(self.budget)
        with self.assertRaisesRegex(OSError,"cap"):self.run_fake(factory)
        self.assertEqual((factory.calls["step"],self.budget.counts["controls"]),(3,3))
        self.assertEqual(factory.active,0)


if __name__=="__main__":unittest.main()
