"""CPU-only joint references/timelines. All source arrays are synthetic mocks."""

import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey import native_rgb_reference as ref
from tdyssey.native_rgb_data import NativeRGBDataset,verify_dataset
from tdyssey.native_rgb_env import CONTROLLER,PROTOCOL_SHA256,STATE_COLUMNS


def write(path,value):path.write_text(json.dumps(value),encoding="utf-8")


def fixture(base,*,duplicate=False,failed=False):
    physics=base/"physics";marker=physics/"diffusion_policy/env/pusht/pusht_env.py";marker.parent.mkdir(parents=True);marker.write_text("# mock source\n")
    entries=[]
    for index,cohort in enumerate(("2A","2B","2C"),1):
        folder=base/ref.PREFIX[cohort].split("/")[0];data=folder/"data";data.mkdir(parents=True)
        snapshot=folder/"source-snapshot/collector.py";snapshot.parent.mkdir();snapshot.write_text("# historical collector\n")
        physics_snapshot=snapshot.parent/"pusht_env.py";physics_snapshot.write_bytes(marker.read_bytes())
        identity=dict(files=[dict(source=str(base/"mutable-old-collector.py"),snapshot="source-snapshot/collector.py",
            bytes=snapshot.stat().st_size,sha256=ref.sha(snapshot)),dict(source=str(marker),snapshot="source-snapshot/pusht_env.py",
            bytes=physics_snapshot.stat().st_size,sha256=ref.sha(physics_snapshot))])
        write(folder/"source-identity.json",identity)
        states=np.zeros((65,15),np.float64);states[:,0]=index*20+np.arange(65);states[:,1]=100;states[:,2:4]=(220,220)
        actions=np.arange(128,dtype=np.float32).reshape(64,2);split=ref.SPLIT[cohort];identifier=cohort+"-L"
        arrays=dict(states=states,actions=actions,contact=np.zeros(64),valid_transition=np.ones(64,bool),
            substep_penetrations=np.zeros((64,10,3)),penetrations=np.zeros((65,3)))
        np.savez_compressed(data/(identifier+".npz"),**arrays)
        metadata=dict(request=dict(split=split,seed=1 if duplicate and cohort=="2C" else index),
            scene=dict(initial_state=states[0].tolist(),selection_version="motion-room-actual-hit-prior-v1" if cohort=="2A" else "motion-room-planned-attempts-v2"),
            steps=[dict(step=t) for t in range(64)])
        write(data/(identifier+".json"),metadata)
        original=dict(trajectory_id=identifier,shape="L",split=split,stage=cohort,seed=metadata["request"]["seed"],controls=64,
            file=identifier+".npz",metadata_file=identifier+".json",initial_scene_hash=hashlib.sha256(states[0].tobytes()).hexdigest(),
            sha256=ref.sha(data/(identifier+".npz")),metadata_sha256=ref.sha(data/(identifier+".json")))
        manifest=dict(split=split,stage=cohort,entries=[original],source_root=str(physics),controller=CONTROLLER,
            protocol_sha256=PROTOCOL_SHA256,state_columns=list(STATE_COLUMNS),source_identity_sha256=ref.sha(folder/"source-identity.json"))
        write(data/"manifest.json",manifest)
        write(folder/"physics-audit.json",dict(manifest_sha256=ref.sha(data/"manifest.json"),passed=not failed,
            physics_collection_gate_passed=not failed,invalid_controls=0,controls=64,audited_substeps=640,
            valid_h1_candidates=61,valid_h4_candidates=58,threshold_world_px=2,threshold_relaxed=False))
        if cohort!="2C":
            entry=dict(original,cohort=cohort,selector_version=metadata["scene"]["selection_version"])
            for key in ("file","metadata_file"):entry[key]=(Path(ref.PREFIX[cohort])/entry[key]).as_posix()
            entries.append(entry)
    view=base/"stage2b/report/merged-train-manifest.json";view.parent.mkdir()
    write(view,dict(kind="native-rgb-formal-train-reference-view.v1",base_root=str(base),entries=entries))
    return physics


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.stack=ExitStack();self.stack.enter_context(patch.object(ref,"EXPECTED_COHORT_COUNTS",{"2A":1,"2B":1,"2C":1}))
        self.stack.enter_context(patch.object(ref,"REFERENCE_LETTERS",("L",)))
        self.stack.enter_context(patch.object(ref,"TRAIN_VIEW_SHA256",None))
    def tearDown(self):self.stack.close()

    def test_joint_reference_identity_and_direct_rgb_timeline_uint8_lru(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);physics=fixture(base);result=ref.build_joint_reference(base);joint=Path(result["root"])
            manifest,audit=verify_dataset(joint)
            self.assertEqual(len(manifest["entries"]),3);self.assertEqual(len(list(base.rglob("*.npz"))),3)
            self.assertEqual(audit["valid_h1_by_split"],dict(train=122,seen_val=61))
            self.assertEqual(audit["valid_h4_by_split"],dict(train=116,seen_val=58))
            dataset=NativeRGBDataset(joint,"seen_val",cache_bytes=3*96*96,physics_source_root=physics)
            dataset.render_state=lambda entry,state:np.full((3,96,96),state[0],np.uint8)
            item=dataset[0]
            self.assertEqual(item["image"].shape,(5,3,96,96));self.assertEqual(item["action"].shape,(4,2))
            np.testing.assert_array_equal(item["image"][:,0,0,0],np.arange(60,65,dtype=np.float32)/255)
            np.testing.assert_allclose(item["action"],np.arange(8,dtype=np.float32).reshape(4,2)/511*2-1)
            self.assertEqual(len(dataset.windows),61);self.assertEqual(len(dataset.h4_windows),58)
            self.assertEqual(dataset.h4_windows[-1],(0,60))
            self.assertLessEqual(dataset._image_bytes,dataset.cache_bytes)
            self.assertTrue(all(value.dtype==np.uint8 for value in dataset._images.values()))
            self.assertNotIn("states",item)

    def test_source_metadata_snapshot_audit_and_split_tampering_cannot_bypass_gate(self):
        for kind in ("metadata","snapshot","audit","split"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                base=Path(directory);fixture(base);joint=Path(ref.build_joint_reference(base)["root"])
                if kind=="metadata":target=base/"stage2c/data/2C-L.json";target.write_text(target.read_text()+" ")
                elif kind=="snapshot":(base/"stage2a/source-snapshot/collector.py").write_text("changed")
                elif kind=="audit":(base/"stage2b/physics-audit.json").write_text("{}")
                else:
                    m=ref._read(joint/"manifest.json");m["entries"][-1]["split"]="train";write(joint/"manifest.json",m)
                    a=ref._read(joint/"physics-audit.json");a["dataset_manifest_sha256"]=ref.sha(joint/"manifest.json");write(joint/"physics-audit.json",a)
                with self.assertRaises(ValueError):verify_dataset(joint,require_gate=False)

    def test_escape_and_duplicate_full_scene_seed_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);fixture(base,duplicate=True)
            with self.assertRaisesRegex(ValueError,"duplicate trajectory,seed"):ref.build_joint_reference(base)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);fixture(base);joint=Path(ref.build_joint_reference(base)["root"]);m=ref._read(joint/"manifest.json")
            m["entries"][-1]["file"]="../../outside.npz"
            with self.assertRaisesRegex(ValueError,"escapes"):ref.resolve_reference_entry(joint,m,m["entries"][-1])

    def test_wrong_physics_alias_gpc_path_and_large_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);physics=fixture(base);joint=Path(ref.build_joint_reference(base)["root"])
            with self.assertRaisesRegex(ValueError,"must match"):NativeRGBDataset(joint,source_root="gpc",physics_source_root=physics)
            with self.assertRaisesRegex(ValueError,"GPC source"):NativeRGBDataset(joint,source_root="gpc")
            with self.assertRaisesRegex(ValueError,"128MiB"):NativeRGBDataset(joint,cache_bytes=129*1024**2)
            with self.assertRaisesRegex(ValueError,"only formal"):NativeRGBDataset(joint,"unseen_test")

    def test_failed_source_physical_gate_and_hot_collection_block_joint_build(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);fixture(base,failed=True)
            with self.assertRaisesRegex(RuntimeError,"physical gate failed"):ref.build_joint_reference(base)
            self.assertFalse((base/"stage2c/joint-data").exists())
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);fixture(base);(base/"stage2c/COLLECTION_RUNNING.lock").write_text("1")
            with self.assertRaisesRegex(RuntimeError,"still active"):ref.build_joint_reference(base)

    def test_historical_current_collector_can_change_but_preserved_snapshot_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);fixture(base);(base/"mutable-old-collector.py").write_text("new adapter allowed")
            joint=Path(ref.build_joint_reference(base)["root"]);self.assertTrue(verify_dataset(joint)[1]["source_snapshots_verified"])
            (base/"physics/diffusion_policy/env/pusht/pusht_env.py").write_text("changed actual physics")
            with self.assertRaisesRegex(ValueError,"Active official physics"):verify_dataset(joint)

    def test_cpu_probe_direct_samples_no_model_iterator_or_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);fixture(base);joint=Path(ref.build_joint_reference(base)["root"])
            samples=[dict(split="train",trajectory_id="2A-L",anchor=3),dict(split="seen_val",trajectory_id="2C-L",anchor=60)]
            with patch.object(NativeRGBDataset,"render_state",lambda self,entry,state:np.full((3,96,96),state[0],np.uint8)):
                result=ref.validate_cpu_samples(joint,base/"cpu.json",samples)
            self.assertEqual(result["batches"]["train"],dict(image=[1,5,3,96,96],action=[1,4,2]))
            self.assertEqual(result["samples"][1]["target_state_index"],61)
            self.assertEqual(result["physics_controls"],0);self.assertEqual(result["index_draws"],0)

    def test_original_getitem_render_and_normalization_are_source_identical(self):
        repo=Path(__file__).resolve().parents[1];baseline=repo/"artifacts/native-rgb-supervised-20261029/stage2c/baseline-source/native_rgb_data.py"
        if not baseline.exists():self.skipTest("Local2C preserved source snapshot unavailable")
        old=baseline.read_text(encoding="utf-8")
        current=(repo/"tdyssey/native_rgb_data.py").read_text(encoding="utf-8")
        def methods(source):
            cls=next(node for node in ast.parse(source).body if isinstance(node,ast.ClassDef) and node.name=="NativeRGBDataset")
            return {node.name:ast.get_source_segment(source,node) for node in cls.body if isinstance(node,ast.FunctionDef)}
        before,after=methods(old),methods(current)
        for name in ("__getitem__","render_state","render_layers","_image","_env"):
            self.assertEqual(before[name],after[name])


if __name__=="__main__":unittest.main()
