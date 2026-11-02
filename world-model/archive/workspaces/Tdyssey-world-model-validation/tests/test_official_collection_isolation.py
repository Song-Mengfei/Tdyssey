"""Synthetic collection boundaries; never import upstream or run physics."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tdyssey.dataset import digest, json_bytes
from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.gpc_collection_runtime import CollectionBudget
from tdyssey.gpc_data_collection import collect_fixed_runs
from tdyssey.official_collection import (ENVIRONMENT_FAMILY, VERSION,
    OfficialCollectionConfig, collect_official_episode, validate_official_dataset)
from tdyssey.state import BodyState, Snapshot
from tdyssey.training_data import write_catalog, read_catalog

class FakeEnvironment:
    sim_hz, control_hz = 100, 10
    def __init__(self, shape="L", stop=6, fail_reset=False):
        self.project_shape, self.stop, self.fail_reset = shape, stop, fail_reset
        self.actions=[]; self.closed=False
    def seed(self, seed): self.seed_value=seed
    def observation(self):
        import numpy as np
        return dict(image=np.full((3,96,96),self.state.control_step/255,dtype=np.float32),
            agent_pos=np.array(self.state.pusher_body.pose.position))
    def reset(self):
        if self.fail_reset: raise RuntimeError("injected reset failure")
        self.state=Snapshot(self.project_shape,official_shape(self.project_shape).geometry_hash,
            "a"*64,BodyState(Pose((220+(self.seed_value%100000)/10000,220),0)),
            BodyState(Pose((100,100),0)),Pose((350,350),0),0)
        return self.observation()
    def capture_state(self): return self.state
    def restore_state(self,state): self.state=state; return self.observation()
    def step(self,action):
        action=tuple(float(x) for x in action); self.actions.append(action)
        self.state=replace(self.state,control_step=self.state.control_step+1,
            pusher_target=action,pusher_body=BodyState(Pose(action,0)))
        return self.observation(),0.,self.state.control_step==self.stop,dict(n_contacts=0)
    def close(self): self.closed=True

class CollectionIsolationChecks(unittest.TestCase):
    def setUp(self):
        fixtures=Path(__file__).resolve().parents[1]/"artifacts/gpc-data-stage-engineering-20261026/isolation"
        fixtures.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=fixtures); self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)
        self.addCleanup(patch.stopall)
        patch("tdyssey.gpc_collection_runtime.stage_paths",side_effect=lambda stage:
            (self.root/stage,self.root/(stage+"-claim.json"))).start()
        patch("tdyssey.official_collection.reset_issues",return_value=[]).start()
    def budget(self):
        budget=CollectionBudget.create("A",dict(synthetic=True)); self.addCleanup(budget.close); return budget
    def provenance(self,config):
        return dict(environment_family=ENVIRONMENT_FAMILY,collector_version=VERSION,
            geometry_hash=official_shape(config.shape).geometry_hash)
    def test_full_fake_four_runs_and_held_open_trap(self):
        budget=self.budget(); envs=[]
        def factory(shape):
            env=FakeEnvironment(shape); envs.append(env); return env
        with patch("tdyssey.training_data.load_episode",side_effect=AssertionError("payload reopened")):
            result=collect_fixed_runs(budget,factory,exclusions=dict(reviewed_scene_sha256=[],
                excluded_reset_seeds=[],sha256="b"*64),provenance_factory=self.provenance)
        self.assertEqual(result["role_counts"],dict(train=24,development=2,independent_validation=2,held_model_test=4))
        self.assertEqual(budget.counts["controls"],192); self.assertEqual(budget.counts["resets"],32)
        self.assertEqual(budget.counts["renders"],256); self.assertTrue(all(e.closed for e in envs))
        self.assertEqual(budget.finish()["status"],"complete")
        catalog,roots,manifests=read_catalog(budget.output/"catalog.json")
        forbidden={root/f["name"] for root,m in zip(roots,manifests) for e in m["episodes"]
            if e["trajectory_id"] in m["splits"]["model_test"] for f in e["files"]}
        original_open=Path.open
        def guarded(path,*args,**kwargs):
            if path in forbidden: raise AssertionError("held payload opened")
            return original_open(path,*args,**kwargs)
        with patch.object(Path,"open",guarded):
            for root in roots: self.assertEqual(len(validate_official_dataset(root)[1]),7)
            write_catalog(roots,self.root/"second-catalog.json")
    def test_failed_reset_precharged_without_retry(self):
        budget=self.budget(); budget.begin_run("T-A")
        counts=dict(reset_attempts=0,reset_physics_substeps=0,control_steps=0,control_physics_substeps=0)
        with self.assertRaisesRegex(RuntimeError,"injected reset"):
            collect_official_episode(FakeEnvironment("T",fail_reset=True),
                OfficialCollectionConfig(shape="T"),0,counts,budget=budget)
        self.assertEqual(counts["reset_attempts"],1); self.assertEqual(counts["reset_physics_substeps"],1)
        self.assertEqual(budget.counts["resets"],1); self.assertEqual(budget.completed["resets"],0)
        self.assertEqual(budget.counts["controls"],0)
    def test_reviewed_scene_filtered_before_control_and_early_success_not_padded(self):
        from tdyssey.collection import derived_seed
        config=OfficialCollectionConfig(shape="L"); env=FakeEnvironment("L",stop=1)
        env.seed(derived_seed(config.seed,VERSION,"L",0,"reset",0)); env.reset()
        excluded=digest(json_bytes(asdict(env.capture_state())))
        counts=dict(reset_attempts=0,reset_physics_substeps=0,control_steps=0,control_physics_substeps=0)
        episode=collect_official_episode(env,config,0,counts,excluded_initial_state_sha256=[excluded])
        self.assertEqual(counts["reset_attempts"],2); self.assertEqual(len(env.actions),1)
        self.assertEqual(episode.resets[0]["issues"],["excluded_full_initial_state"])
        self.assertEqual(episode.stop_reason,"upstream_termination"); self.assertEqual(len(episode.images),2)
        self.assertEqual(episode.trajectory.transitions[0].action,env.actions[0])

if __name__=="__main__": unittest.main()
