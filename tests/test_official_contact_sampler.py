"""Source checks for contact headings, moved-T labels and replay mix."""

from dataclasses import replace
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from tdyssey.collection import CollectionConfig, WaypointPlay
from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.motion_replay import object_moved, planned_replay_take, sample_replay_batch
from tdyssey.official_collection import OfficialCollectionConfig
from tdyssey.official_collector import OfficialShapeCollector
from tdyssey.official_contact_sampler import (ContactBalancedPlay, classify_headings, fallback_heading,
                                              preserve_init_checkpoint)
from tdyssey.gpc_denoiser_training import load_official_windows
from tdyssey.official_collection import ENVIRONMENT_FAMILY
from tdyssey.state import BodyState, Snapshot

ROOT = Path(__file__).resolve().parents[1]


def official_state(position=(250, 250), angle=0.0, pusher=(100, 265)):
    pose = Pose(position, angle)
    return Snapshot("T", official_shape("T").geometry_hash, "a" * 64,
                    BodyState(pose), BodyState(Pose(pusher, 0)), Pose((256, 256), math.pi / 4), 0)


class OfficialContactSamplerChecks(unittest.TestCase):
    def test_hit_headings_include_the_side_facing_the_t_and_exclude_the_opposite(self):
        pose = Pose((250, 250), 0)
        rows = classify_headings((100, 265), pose, official_shape("T"))
        east = next(row for row in rows if abs(row["heading"]) < 1e-9)
        west = next(row for row in rows if abs(row["heading"] - math.pi) < 1e-9)
        self.assertTrue(east["hit"])
        self.assertFalse(west["hit"])
        self.assertGreater(east["t_enter"], 0)

    def test_fallback_offsets_stay_inside_the_declared_spread(self):
        pose = Pose((250, 250), 0)
        pusher = (100, 295)
        axis = math.atan2(295 - pusher[1], 250 - pusher[0])
        spread = math.radians(150)
        for seed in range(40):
            heading = fallback_heading(pusher, pose, random.Random(seed))
            offset = (heading - axis + math.pi) % (2 * math.pi) - math.pi
            self.assertLessEqual(abs(offset), spread + 1e-9)

    def test_object_moved_uses_translation_or_wrapped_rotation(self):
        start = official_state()
        self.assertFalse(object_moved(start, replace(start, control_step=1)))
        shifted = replace(start, control_step=1,
                          object_body=BodyState(Pose((251.0, 250.0), 0)))
        self.assertTrue(object_moved(start, shifted))
        turned = replace(start, control_step=1,
                         object_body=BodyState(Pose((250.0, 250.0), 0.02)))
        self.assertTrue(object_moved(start, turned))
        tiny = replace(start, control_step=1,
                       object_body=BodyState(Pose((250.2, 250.0), 0.001)))
        self.assertFalse(object_moved(start, tiny))

    def test_replay_batch_keeps_the_requested_mix_when_both_pools_are_large(self):
        moved = [{"moved": True, "i": index} for index in range(50)]
        still = [{"moved": False, "i": index} for index in range(50)]
        batch, mix = sample_replay_batch(moved, still, 10, 0.6, random.Random(0))
        self.assertEqual(len(batch), 10)
        self.assertEqual((mix["moved"], mix["still"]), (6, 4))
        self.assertEqual(sum(item["moved"] for item in batch), 6)
        short = planned_replay_take(1, 100, 4, 0.6)
        self.assertEqual((short["moved"], short["still"]), (1, 3))

    def test_contact_balanced_is_accepted_and_model_test_stays_closed(self):
        config = OfficialCollectionConfig(shape="T", mode="contact-balanced")
        self.assertEqual(config.max_target_delta, 200)
        collector = OfficialShapeCollector(config, ROOT)
        self.assertEqual(collector.config.mode, "contact-balanced")
        with self.assertRaisesRegex(ValueError, "ContactBalancedPlay"):
            WaypointPlay(CollectionConfig(mode="contact-balanced", max_target_delta=200), 1).action(
                {"agent_pos": (100, 100)}, None, 0)
        first = ContactBalancedPlay(config, 3)
        second = ContactBalancedPlay(config, 3)
        record = official_state()
        action = first.action({"agent_pos": (100, 100)}, record, 0)
        self.assertEqual(action, second.action({"agent_pos": (100, 100)}, record, 0))
        self.assertTrue(all(25 <= value <= 486 for value in action))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for index, identifier in enumerate(("a", "b", "c")):
                entries.append({"trajectory_id": identifier, "control_steps": 4,
                                "observation_count": 5, "contact_steps": 0, "files": [
                                    {"name": f"episode-{index:04d}.npz", "bytes": 1, "sha256": "a" * 64},
                                    {"name": f"episode-{index:04d}.json", "bytes": 1, "sha256": "b" * 64}]})
            payload = {"schema_version": 1, "status": "complete",
                       "config": {"episodes": 3, "shape": "T", "mode": "contact-balanced"},
                       "provenance": {"environment_family": ENVIRONMENT_FAMILY},
                       "episodes": entries,
                       "splits": {"train": ["a"], "validation": ["b"], "model_test": ["c"]},
                       "counts": {"reset_attempts": 3, "reset_physics_substeps": 3,
                                  "control_steps": 12, "control_physics_substeps": 120},
                       "rates": {"simulation_hz": 100, "control_hz": 10}}
            (root / "manifest.json").write_text(json.dumps(payload))
            opened = []
            with self.assertRaisesRegex(ValueError, "model_test"):
                load_official_windows(root, ("model_test",), opened)
            self.assertEqual(opened, [])

    def test_best_init_checkpoint_is_copied_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "angles"
            destination = Path(temporary) / "init"
            source.mkdir()
            (source / "denoiser.pt").write_bytes(b"weights")
            (source / "denoiser.json").write_text("{}\n")
            first = preserve_init_checkpoint(source / "denoiser.pt", destination)
            self.assertEqual(first["copied"], ["denoiser.pt", "denoiser.json"])
            (source / "denoiser.pt").write_bytes(b"changed")
            second = preserve_init_checkpoint(source / "denoiser.pt", destination)
            self.assertEqual(second["copied"], [])
            self.assertEqual((destination / "denoiser.pt").read_bytes(), b"weights")


if __name__ == "__main__":
    unittest.main()
