"""Focused CPU checks for split integrity, bounded data and physical gates."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey.native_rgb_data import (BudgetLedger, NativeRGBDataset, TRAIN_LETTERS,
    UNSEEN_LETTERS, collection_caps, file_sha256, verify_dataset, write_json,scene_seed)


def temporary_directory():
    parent=Path(__file__).resolve().parents[1]/"artifacts/native-rgb-test-tmp"
    parent.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


class BudgetTests(unittest.TestCase):
    def test_reservations_survive_restart_and_include_failed_attempts(self):
        with temporary_directory() as directory:
            ledger=BudgetLedger(directory,{"train":8,"seen_val":4})
            attempt=ledger.reserve("train",8,"far-click failed probe")
            ledger.finish(attempt,3,status="invalid",invalid_controls=1)
            ledger.reject("closed-hole staging rejected")
            restored=BudgetLedger(directory,{"train":8,"seen_val":4})
            self.assertEqual(restored.value["actual_controls"],3)
            self.assertEqual(restored.value["charged_controls"]["train"],8)
            self.assertEqual(restored.value["rejected_resets"],1)
            with self.assertRaises(RuntimeError):restored.reserve("train",1,"cannot reset budget")
            with self.assertRaises(ValueError):BudgetLedger(directory,{"train":16,"seen_val":4})

    def test_server_budget_is_independent_and_finite(self):
        config=dict(collection=dict(controls=64,train_trajectories=400,val_trajectories=40,
            seen_test_trajectories=40,unseen_test_trajectories=80,max_attempted_controls=593920))
        caps=collection_caps(config)
        self.assertEqual(caps["unseen_test"],40960)
        self.assertEqual(sum(caps.values()),593920)
        config["collection"]["max_attempted_controls"]=34560
        with self.assertRaises(ValueError):collection_caps(config)


class DatasetTests(unittest.TestCase):
    def _fixture(self,directory,invalid=False):
        import numpy as np
        root=Path(directory);path=root/"episode.npz"
        states=np.zeros((8,15),dtype=np.float64)
        states[:,0]=np.arange(8)
        actions=np.asarray([[i,511-i] for i in range(7)],dtype=np.float32)
        valid=np.ones(7,dtype=np.bool_)
        if invalid:valid[3]=False
        np.savez_compressed(path,states=states,actions=actions,valid_transition=valid)
        entry=dict(file="episode.npz",sha256=file_sha256(path),shape="L",split="train",trajectory_id="scene-one",controls=7)
        write_json(root/"manifest.json",dict(entries=[entry],source_root="not-used",controller="finite-force-pd-v1"))
        write_json(root/"physics-audit.json",dict(dataset_manifest_sha256=file_sha256(root/"manifest.json"),
            dataset_npz_sha256_verified=True,physics_collection_gate_passed=not invalid))
        return root

    def test_aligned_five_frame_window_four_absolute_actions_and_lru(self):
        import numpy as np
        with temporary_directory() as directory:
            root=self._fixture(directory)
            dataset=NativeRGBDataset(root,cache_bytes=3*96*96)
            dataset.render_state=lambda entry,state:np.full((3,96,96),state[0],dtype=np.uint8)
            result=dataset[0]
            np.testing.assert_array_equal(result["image"][:,0,0,0],np.arange(5,dtype=np.float32)/255)
            np.testing.assert_allclose(result["action"][:,0],np.arange(4,dtype=np.float32)/511*2-1)
            self.assertEqual(result["image"].shape,(5,3,96,96))
            self.assertLessEqual(dataset._image_bytes,dataset.cache_bytes)
            self.assertNotIn("states",result)
            self.assertEqual(dataset.windows,[(0,3),(0,4),(0,5),(0,6)])

    def test_invalid_transition_is_retained_and_excluded_but_gate_still_fails(self):
        with temporary_directory() as directory:
            root=self._fixture(directory,invalid=True)
            with self.assertRaises(RuntimeError):NativeRGBDataset(root)
            diagnostic=NativeRGBDataset(root,require_gate=False)
            self.assertEqual(diagnostic.windows,[])
            self.assertTrue((root/"episode.npz").exists())

    def test_manifest_binding_and_npz_hash_prevent_stale_audit(self):
        with temporary_directory() as directory:
            root=self._fixture(directory)
            with (root/"episode.npz").open("ab") as stream:stream.write(b"changed")
            with self.assertRaises(ValueError):verify_dataset(root)


class GeometryTests(unittest.TestCase):
    def test_fixed_split_seed_mapping_is_independent_of_request_order(self):
        self.assertEqual(scene_seed(20261028,"train",0,0),20261028)
        self.assertEqual(scene_seed(20261028,"seen_val",0,0),21261028)
        self.assertEqual(scene_seed(20261028,"seen_test",0,0),22261028)
        self.assertEqual(scene_seed(20261028,"unseen_test",0,0),23261028)
        self.assertEqual(len({scene_seed(1,split,repeat,letter) for split in
            ("train","seen_val","seen_test","unseen_test") for repeat in range(20) for letter in range(18)}),1440)

    def test_environment_default_seed_does_not_touch_global_numpy(self):
        import numpy as np
        from tdyssey.native_rgb_env import make_env
        class FakeBase:
            def __init__(self,**kwargs):self.seed()
            def seed(self,seed=None):
                if seed is None:np.random.randint(0,12345)
                self.observed_seed=seed
        with patch("tdyssey.native_rgb_env.load_physics_source",return_value=FakeBase), \
                patch.object(np.random,"randint",side_effect=AssertionError("global RNG used")):
            env=make_env("not-read","T")
            self.assertEqual(env.observed_seed,0)
            env.seed(19)
            self.assertEqual(env.observed_seed,19)

    def test_closed_holes_and_normal_path_blocked_by_a_thin_wall(self):
        from types import SimpleNamespace
        from shapely.geometry import box
        from shapely.ops import unary_union
        from tdyssey.native_rgb_data import _reachable_segments
        class FakeEnv:
            block=SimpleNamespace(angle=0,local_to_world=lambda point:point)
            def world_geometry(self):return unary_union((box(240,200,250,240),box(251,200,252,240)))
        segment=dict(id=0,ring=0,point=[250,220],normal=[1,0])
        row=_reachable_segments(FakeEnv(),[segment])[0]
        self.assertEqual(row["status"],"normal_contact_path_blocked_reachability_unconfirmed")
        segment["ring"]=1
        self.assertEqual(_reachable_segments(FakeEnv(),[segment])[0]["status"],"proven_closed_hole_from_exterior")

    def test_current_split_has_no_exact_rotated_duplicate(self):
        from tdyssey.native_rgb_env import rotational_split_audit
        self.assertTrue(rotational_split_audit(TRAIN_LETTERS,UNSEEN_LETTERS)["passed"])
        self.assertFalse(rotational_split_audit(("L",),("L",))["passed"])

    def test_arclength_bins_include_all_hole_perimeters_without_overlap(self):
        from tdyssey.native_rgb_data import contour_segments
        from tdyssey.native_rgb_env import shape_geometry
        rows=contour_segments("B")
        self.assertGreater(max(row["ring"] for row in rows),0)
        self.assertAlmostEqual(sum(row["arc_end"]-row["arc_start"] for row in rows),shape_geometry("B").length)
        self.assertTrue(all(0<row["arc_end"]-row["arc_start"]<=8.000001 for row in rows))


if __name__=="__main__":unittest.main()
