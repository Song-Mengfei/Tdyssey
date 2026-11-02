"""Tiny synthetic official-format episodes; no torch/model/simulator execution."""

from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey.gpc_adaptation_data import (
    ENVIRONMENT_FAMILY, KIND, T1_SPEC_SHA256, GPCAdaptationDataset,
    fit_train_stats, read_adaptation_manifest,
)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def empty_manifest():
    return dict(schema_version=1, kind=KIND, environment_family=ENVIRONMENT_FAMILY,
        t1_spec_sha256=T1_SPEC_SHA256, catalogs=[], episodes=[],
        reviewed_scene_sha256=[], previously_used_episode_ids=[])


class AdaptationMetadataChecks(unittest.TestCase):
    def test_empty_manifest_exposes_both_data_and_validation_gaps_without_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"adaptation.json"
            write_json(path,empty_manifest())
            with patch("tdyssey.gpc_adaptation_data.load_episode",side_effect=AssertionError("payload read")):
                result=read_adaptation_manifest(path)
                self.assertFalse(result["readiness"]["ready"])
                self.assertEqual(len(result["readiness"]["gaps"]),4)
                with self.assertRaisesRegex(ValueError,"not ready"):
                    read_adaptation_manifest(path,require_ready=True)
            with self.assertRaisesRegex(ValueError,"model_test"):
                GPCAdaptationDataset(path,"model_test")


@unittest.skipUnless(importlib.util.find_spec("numpy"),"NumPy required for tiny synthetic arrays")
class AdaptationDataChecks(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from tdyssey.collection import CollectedEpisode
        from tdyssey.dataset import DatasetWriter
        from tdyssey.envs.official import official_shape
        from tdyssey.geometry import Pose
        from tdyssey.official_collection import OfficialCollectionConfig, VERSION, DEVELOPMENT_SEEDS
        from tdyssey.state import BodyState, Snapshot
        from tdyssey.training_data import initial_state_hash
        from tdyssey.trajectories import Trajectory, Transition
        self.np=np
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.manifest=empty_manifest()
        self.in_memory={}
        class FixtureWriter(DatasetWriter):
            def _manifest(self):
                # This tests the reader, not atomic collector publication.
                write_json(self.root/"manifest.json",self.manifest)
        for shape_number,shape in enumerate(("T","L")):
            root=self.root/shape
            config=OfficialCollectionConfig(shape=shape,episodes=3,steps=6)
            provenance=dict(environment_family=ENVIRONMENT_FAMILY,collector_version=VERSION,
                            geometry_hash=official_shape(shape).geometry_hash)
            writer=FixtureWriter(root,config,provenance)
            writer.manifest["rates"]=dict(simulation_hz=100,control_hz=10)
            catalog_rows=[]
            for episode_number,split in enumerate(("train","validation","model_test")):
                identifier=shape+"-"+split
                initial=Snapshot(shape,official_shape(shape).geometry_hash,"a"*64,
                    BodyState(Pose((210+episode_number*8,220),.1*shape_number)),
                    BodyState(Pose((100,100),0)),Pose((350,350),0),0)
                current=initial; transitions=[]; diagnostics=[]; states=[initial]
                for step in range(6):
                    action=(100+step*11,200+step*7)
                    after=replace(current,control_step=step+1,pusher_target=action,
                        pusher_body=BodyState(Pose(action,0)),
                        object_body=BodyState(Pose((current.object_body.pose.position[0]+1,220),.05*(step+1)+.1*shape_number)))
                    transitions.append(Transition(current,action,after,0,False,step==5))
                    diagnostics.append(dict(n_contacts=int(step%2),upstream_terminated=False,
                        upstream_truncated=False,collector_truncated=step==5))
                    states.append(after); current=after
                trajectory=Trajectory(identifier,"object-directed",VERSION,5478+episode_number,.1,tuple(transitions))
                images=np.empty((7,3,96,96),dtype=np.uint8)
                images[:]=np.array([119,136,153],dtype=np.uint8)[None,:,None,None]
                for frame in range(7):
                    images[frame,:,0,0]=[5,frame*17,9]
                    images[frame,:,0,1]=[7,shape_number*60+episode_number*11,13]
                reset_seed=200000+episode_number+shape_number*10
                resets=[dict(simulated=True,issues=[],seed=reset_seed,accepted_state=json.loads(initial.to_json()))]
                episode=CollectedEpisode(trajectory,images,np.asarray([s.pusher_body.pose.position for s in states]),
                    np.asarray([(*s.goal.position,s.goal.angle) for s in states]),diagnostics,resets,reset_seed,5478,"horizon")
                writer.add(episode)
                catalog_rows.append(dict(trajectory_id=identifier,split=split,initial_state_sha256=initial_state_hash(trajectory),
                    control_steps=6,contact_steps=3))
                self.manifest["episodes"].append(dict(trajectory_id=identifier,
                    role={"train":"train","validation":"independent_validation","model_test":"held_model_test"}[split],prior_uses=[]))
                self.in_memory[identifier]=(trajectory,images,catalog_rows[-1])
            writer.manifest["counts"].update(control_steps=18,control_physics_substeps=180,reset_attempts=3,reset_physics_substeps=3)
            writer.finish({split:[shape+"-"+split] for split in ("train","validation","model_test")})
            catalog=dict(schema_version=1,environment_family=ENVIRONMENT_FAMILY,rates=writer.manifest["rates"],
                runs=[dict(path=shape,manifest_sha256=hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),episodes=catalog_rows)])
            catalog_path=self.root/(shape+".catalog.json"); write_json(catalog_path,catalog)
            self.manifest["catalogs"].append(dict(path=catalog_path.name,sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest()))
        self.path=self.root/"adaptation.json"
        self.save()

    def save(self):
        write_json(self.path,self.manifest)

    def test_exact_four_control_window_next_target_t1_both_sides_and_shape_indices(self):
        from tdyssey.wm_compat_transform import transform_t1
        dataset=GPCAdaptationDataset(self.path,require_ready=True)
        self.assertEqual(len(dataset),6)
        self.assertEqual(dataset.shape_indices("T"),(0,1,2))
        self.assertEqual(dataset.shape_indices("L"),(3,4,5))
        sample=dataset[0]
        trajectory,images,_=self.in_memory["T-train"]
        expected=self.np.stack([transform_t1(frame.astype(self.np.float32)/self.np.float32(255))[0] for frame in images[:5]])
        self.np.testing.assert_array_equal(sample["image"],expected)
        self.assertEqual(sample["image"].shape,(5,3,96,96))
        self.assertEqual(sample["anchor"],3)
        self.np.testing.assert_allclose(sample["image"][:, :,5,5],self.np.tile(self.np.array([165,28,48],dtype=self.np.float32)/255,(5,1)),rtol=0,atol=1e-7)
        actions=self.np.asarray([t.action for t in trajectory.transitions[:4]],dtype=self.np.float32)/self.np.float32(511)*self.np.float32(2)-self.np.float32(1)
        self.np.testing.assert_array_equal(sample["action"],actions)
        last=dataset[2]
        self.assertEqual(last["anchor"],5)
        self.assertEqual(dataset[3]["shape"],"L")
        self.assertEqual(dataset[-1]["episode_id"],"L-train")
        self.assertEqual(images.dtype,self.np.uint8)
        with self.assertRaises(IndexError): dataset[6]

    def test_metadata_and_train_windows_never_open_held_record_or_archive(self):
        original=Path.open
        def checked_open(path,*args,**kwargs):
            if path.name in ("episode-0002.json","episode-0002.npz"):
                raise AssertionError("Held model_test payload opened")
            return original(path,*args,**kwargs)
        with patch.object(Path,"open",checked_open):
            self.assertTrue(read_adaptation_manifest(self.path,require_ready=True)["readiness"]["ready"])
            self.assertEqual(GPCAdaptationDataset(self.path)[0]["shape"],"T")
            with self.assertRaisesRegex(ValueError,"model_test"):
                GPCAdaptationDataset(self.path,"model_test")

    def test_statistics_are_unique_train_controls_not_overlapping_windows_or_validation(self):
        dataset=GPCAdaptationDataset(self.path)
        with patch("tdyssey.wm_compat_transform.transform_t1",side_effect=AssertionError("Statistics transformed images")):
            statistics=fit_train_stats(dataset)
        self.assertEqual(statistics["unique_controls"],12)
        self.assertEqual(statistics["controls_by_shape"],{"T":6,"L":6})
        self.assertEqual(statistics["contact_steps"],6)
        self.assertEqual(statistics["episode_ids"],["T-train","L-train"])
        with self.assertRaisesRegex(ValueError,"training episodes only"):
            fit_train_stats(GPCAdaptationDataset(self.path,"independent_validation"))

    def test_prior_use_reviewed_scene_and_model_test_reassignment_fail_closed(self):
        self.manifest["episodes"][1]["prior_uses"]=["customCNN validation selection"]
        self.save()
        with self.assertRaisesRegex(ValueError,"Previously used"):
            read_adaptation_manifest(self.path)
        self.manifest["episodes"][1]["prior_uses"]=[]
        self.manifest["reviewed_scene_sha256"]=[self.in_memory["T-validation"][2]["initial_state_sha256"]]
        self.save()
        with self.assertRaisesRegex(ValueError,"Previously used"):
            read_adaptation_manifest(self.path)
        self.manifest["reviewed_scene_sha256"]=[]
        self.manifest["episodes"][2]["role"]="independent_validation"
        self.save()
        with self.assertRaisesRegex(ValueError,"cannot be reassigned"):
            read_adaptation_manifest(self.path)

    def test_development_is_readable_but_missing_independent_shape_blocks_start(self):
        self.manifest["episodes"][1]["role"]="development"
        self.manifest["episodes"][1]["prior_uses"]=["reviewed"]
        self.save()
        result=read_adaptation_manifest(self.path)
        self.assertFalse(result["readiness"]["ready"])
        self.assertEqual(GPCAdaptationDataset(self.path,"development")[0]["shape"],"T")
        with self.assertRaisesRegex(ValueError,"independent validation"):
            read_adaptation_manifest(self.path,require_ready=True)

    def test_missing_payload_stat_blocks_readiness_without_open_and_manifest_tamper_blocks_load(self):
        (self.root/"T/episode-0000.npz").unlink()
        with patch("tdyssey.gpc_adaptation_data.load_episode",side_effect=AssertionError("payload read")):
            self.assertFalse(read_adaptation_manifest(self.path)["readiness"]["ready"])
        dataset=GPCAdaptationDataset(self.path)
        self.manifest["note"]="changed after freeze"; self.save()
        with self.assertRaisesRegex(ValueError,"manifest changed"):
            dataset[0]

    def test_duplicate_scene_and_missing_declaration_are_rejected_before_loading(self):
        catalog_path=self.root/"T.catalog.json"
        catalog=json.loads(catalog_path.read_bytes())
        catalog["runs"][0]["episodes"][1]["initial_state_sha256"]=catalog["runs"][0]["episodes"][0]["initial_state_sha256"]
        write_json(catalog_path,catalog)
        self.manifest["catalogs"][0]["sha256"]=hashlib.sha256(catalog_path.read_bytes()).hexdigest(); self.save()
        with self.assertRaisesRegex(ValueError,"initial state crosses|Duplicate initial scene"):
            read_adaptation_manifest(self.path)

    def test_exhaustive_source_membership_and_held_row_mutation_are_rejected(self):
        self.manifest["episodes"].pop()
        self.save()
        with self.assertRaisesRegex(ValueError,"Declare every source episode"):
            read_adaptation_manifest(self.path)
        self.manifest["episodes"].append(dict(trajectory_id="L-model_test",role="held_model_test",prior_uses=[]))
        self.save()
        dataset=GPCAdaptationDataset(self.path)
        dataset.episodes[0]["role"]="held_model_test"
        with patch("tdyssey.gpc_adaptation_data.load_episode",side_effect=AssertionError("Held payload read")):
            with self.assertRaisesRegex(ValueError,"model_test"):
                dataset[0]

    def test_cached_episode_does_not_hide_changed_frozen_manifest(self):
        dataset=GPCAdaptationDataset(self.path)
        dataset[0]
        self.manifest["note"]="changed after caching"
        self.save()
        with self.assertRaisesRegex(ValueError,"manifest changed"):
            dataset[1]

    def test_development_and_independent_roles_have_disjoint_explicit_loaders(self):
        self.manifest["episodes"][1]["role"]="development"
        self.manifest["episodes"][1]["prior_uses"]=["development review"]
        self.save()
        dev=GPCAdaptationDataset(self.path,"development")
        independent=GPCAdaptationDataset(self.path,"independent_validation")
        self.assertEqual((len(dev),len(independent)),(3,3))
        self.assertEqual(dev.shape_indices("T"),(0,1,2))
        self.assertEqual(dev.shape_indices("L"),())
        self.assertEqual(independent.shape_indices("T"),())
        self.assertEqual(independent.shape_indices("L"),(0,1,2))
        self.assertEqual(dev[0]["role"],"development")
        self.assertEqual(independent[0]["split"],"independent_validation")
        self.assertEqual(independent[0]["source_split"],"validation")
        with self.assertRaisesRegex(ValueError,"ambiguous validation"):
            GPCAdaptationDataset(self.path,"validation")


if __name__ == "__main__":
    unittest.main()
