"""Offline tests of event isolation, action alignment, dedup and family splits."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("accelerated_selection", Path(__file__).with_name("selection.py"))
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def fixture(root, name, *, split="train", family=None, n=35, clean=None,
            moving=True, high_action=None):
    directory = Path(root) / "collection" / "attempts"
    directory.mkdir(parents=True, exist_ok=True)
    states = np.zeros((n + 1, 15), np.float64)
    rgb = np.full((n + 1, 3, 96, 96), 255, np.uint8)
    if moving:
        states[:, 0] = np.arange(n + 1)
        states[:, 2] = np.arange(n + 1) * .8
        states[:, 4] = np.arange(n + 1) * .01
        for i in range(n + 1):
            rgb[i, :, 0, i % 96] = 0
    actions = np.tile([250., 250.], (n, 1))
    if high_action is not None:
        actions[high_action, 0] = 512.
    clean = n if clean is None else clean
    valid = np.arange(n) < clean
    np.savez_compressed(directory / (name + ".npz"), states=states, rgb=rgb,
                        requested_actions=actions, passed_actions=actions.copy(),
                        control_valid=valid, contact_positive=np.zeros(n, bool),
                        control_time_seconds=np.arange(n + 1) * .1)
    meta = dict(attempt_id=name, family_id=family or name, letter="O", task="translation",
                split=split, clean_control_count=clean,
                segments=[dict(start_control=3,end_control=19),dict(start_control=19,end_control=35)],
                events=[])
    if clean < n:
        meta.update(first_warning_control=clean,
                    events=[dict(status="RECOVERED_CONTACT_CANDIDATE", first_control=clean)])
    (directory / (name + ".json")).write_text(json.dumps(meta), encoding="utf-8")
    return directory / (name + ".npz")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="offline-selection-")
        self.root = Path(self.temporary.name)
        self.storage = patch.object(selection, "_storage", lambda *args: None)
        self.storage.start()

    def tearDown(self):
        self.storage.stop()
        self.temporary.cleanup()

    def test_recovery_suffix_is_never_reintroduced_and_eval_is_committed(self):
        fixture(self.root,"warning",clean=12)
        fixture(self.root,"validation",split="seen_val")
        report = selection.select(self.root,{"run_id":"mock"})
        train = selection.read(self.root / "selection/train-index.json")
        val = selection.read(self.root / "selection/seen-val-index.json")
        self.assertEqual([row["anchor"] for row in train],list(range(3,12)))
        self.assertTrue(all(row["anchor"] + row["horizon"] <= 12 for row in train))
        self.assertEqual([row["anchor"] for row in val],[3,19])
        self.assertTrue(all(row["committed_future"] for row in val))
        self.assertTrue(report["ready_for_training"])
        self.assertFalse(set(report["train_families"]) & set(report["seen_val_families"]))

    def test_original_512_action_is_preserved_but_incompatible_windows_excluded(self):
        raw = fixture(self.root,"action",high_action=7)
        selection.select(self.root,{})
        rows = selection.read(self.root / "selection/train-index.json")
        self.assertTrue(all(not (row["anchor"] - 3 <= 7 <= row["anchor"]) for row in rows))
        summary = selection.read(self.root / "selection/summary.json")
        self.assertEqual(summary["action_contract"]["excluded_clean_windows_by_horizon"]["1"],4)
        with np.load(raw,allow_pickle=False) as archive:
            self.assertEqual(archive["passed_actions"][7,0],512.)

    def test_quiet_exact_contexts_dedup_but_blue_pusher_motion_is_retained(self):
        fixture(self.root,"quiet",moving=False)
        raw = fixture(self.root,"blue",moving=True)
        with np.load(raw,allow_pickle=False) as archive:
            values = {key:archive[key] for key in archive.files}
        values["states"][:,2:5] = 0.  # Only the blue pusher moves.
        np.savez_compressed(raw,**values)
        # Some clean object motion elsewhere is present; no need to repeat it.
        fixture(self.root,"object",moving=True)
        summary = selection.select(self.root,{})
        rows = selection.read(self.root / "selection/train-index.json")
        self.assertEqual(sum(row["family_id"]=="quiet" for row in rows),1)
        self.assertEqual(sum(row["family_id"]=="blue" for row in rows),32)
        self.assertEqual(summary["exact_duplicate_train_windows_removed"],31)

    def test_family_leakage_and_incomplete_claimed_clean_control_are_errors(self):
        fixture(self.root,"train",family="same")
        fixture(self.root,"val",split="seen_val",family="same")
        with self.assertRaisesRegex(ValueError,"crosses"):
            selection.select(self.root,{})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="offline-selection-") as other:
            raw = fixture(other,"bad")
            with np.load(raw,allow_pickle=False) as archive:
                values = {key:archive[key] for key in archive.files}
            values["control_time_seconds"][-1] -= .01
            np.savez_compressed(raw,**values)
            with self.assertRaisesRegex(ValueError,"10Hz"):
                selection.select(Path(other),{})


if __name__ == "__main__":
    unittest.main(verbosity=2)
