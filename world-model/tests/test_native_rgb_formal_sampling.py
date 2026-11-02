"""Formal sampling CPU tests using only synthetic metadata, never real draws."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey import native_rgb_formal_sampling as sampling

ROOT = Path(__file__).resolve().parents[1]


def synthetic_rows():
    rows = []
    for letter in sampling.LETTERS:
        for trajectory in range(2):
            for stratum_index, name in enumerate(sampling.STRATA):
                for window in range(3):
                    rows.append(dict(dataset_index=len(rows), letter=letter,
                        trajectory_id=f"{letter}-trajectory{trajectory}", anchor=3+stratum_index*3+window,
                        index_stratum=name, split="train", valid_h1_candidate=True))
    return rows


def data_identity():
    return dict(manifest_sha256="1"*64, audit_sha256="2"*64, split_sha256="3"*64,
                train_reference_manifest_sha256=sampling.TRAIN_REFERENCE_SHA256)


def manifest():
    return sampling.build_pool_manifest(synthetic_rows(), data_identity=data_identity())


def reseal(state):
    state.pop("state_sha256", None)
    state["state_sha256"] = sampling.canonical_hash(state)
    return state


class FormalSamplingTests(unittest.TestCase):
    def test_two_complete_cycles_exact_quotas_trajectory_balance_and_window_no_replacement(self):
        order = sampling.FormalWindowOrder(manifest())
        indices = order.take(360)
        for start in (0, 180):
            rows = [order.window_metadata(i) for i in indices[start:start+180]]
            self.assertEqual(Counter(r["index_stratum"] for r in rows),
                             Counter(moving=126, intermediate=18, near_static_contact=18, near_static_no_contact=18))
            for letter in sampling.LETTERS:
                self.assertEqual(Counter(r["index_stratum"] for r in rows if r["letter"] == letter), sampling.QUOTAS)
        for letter in sampling.LETTERS:
            moving = [order.window_metadata(i) for i in indices if order.window_metadata(i)["letter"] == letter
                      and order.window_metadata(i)["index_stratum"] == "moving"]
            self.assertEqual(len({row["dataset_index"] for row in moving[:6]}), 6)
            for end in range(1, len(moving)+1):
                counts = Counter(row["trajectory_id"] for row in moving[:end])
                values = [counts[f"{letter}-trajectory{i}"] for i in range(2)]
                self.assertLessEqual(max(values)-min(values), 1)
        self.assertEqual(order.cursor, 360)

    def test_continuous_batch16_cross_cycle_and_json_restore_preserve_all_independent_rngs(self):
        pools = manifest()
        reference = sampling.FormalWindowOrder(pools).take(720)
        a = sampling.FormalWindowOrder(pools)
        prefix = a.take(176)
        state = json.loads(json.dumps(a.state_dict()))
        restored = sampling.FormalWindowOrder(pools)
        restored.load_state_dict(state)
        batches = [restored.take(16) for _ in range(34)]
        self.assertEqual(prefix+[i for batch in batches for i in batch], reference)
        self.assertEqual(restored.cursor, 720)
        self.assertEqual(a.cursor, 176)
        before = np.random.get_state()
        try:
            sampling.FormalWindowOrder(pools).take(180)
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            self.assertTrue(np.array_equal(before[1], after[1]))
            self.assertEqual(before[2:], after[2:])
        finally:
            np.random.set_state(before)

    def test_restore_rejects_identity_rng_queue_and_checksum_tampering_without_mutation(self):
        order = sampling.FormalWindowOrder(manifest())
        order.take(176)
        original = order.state_dict()
        for identity_key in ("policy_sha256", "pools_sha256", "data_sha256", "catalogue_sha256", "seed"):
            state = deepcopy(original)
            state["identity"][identity_key] = "tampered"
            with self.subTest(identity=identity_key), self.assertRaises(ValueError):
                order.load_state_dict(reseal(state))
        state = deepcopy(original)
        state["slot_rng"]["bit_generator"] = "Philox"
        with self.assertRaises(ValueError):
            order.load_state_dict(reseal(state))
        state = deepcopy(original)
        key = next(k for k, value in state["trajectory_queues"].items() if value["position"] > 0)
        state["trajectory_queues"][key]["position"] -= 1
        with self.assertRaisesRegex(ValueError, "cursor/exposure"):
            order.load_state_dict(reseal(state))
        state = deepcopy(original)
        state["cursor"] += 1
        with self.assertRaises(ValueError):
            order.load_state_dict(state)
        self.assertEqual(order.state_dict(), original)

    def test_empty_stratum_duplicate_anchor_nontrain_and_pool_tampering_fail_closed(self):
        rows = synthetic_rows()
        missing = [row for row in rows if not (row["letter"] == "A" and row["index_stratum"] == "intermediate")]
        for i, row in enumerate(missing):
            row["dataset_index"] = i
        with self.assertRaisesRegex(ValueError, "Empty letter-stratum"):
            sampling.build_pool_manifest(missing, data_identity=data_identity())
        for mode in ("duplicate", "split"):
            rows = synthetic_rows()
            if mode == "duplicate":
                rows[1]["anchor"] = rows[0]["anchor"]
            else:
                rows[0]["split"] = "seen_val"
            with self.assertRaises(ValueError):
                sampling.build_pool_manifest(rows, data_identity=data_identity())
        pools = manifest()
        pools["policy"]["per_letter_quotas"]["moving"] = 6
        with self.assertRaises(ValueError):
            sampling.FormalWindowOrder(pools)

    def test_dataset_factory_checks_only_metadata_and_rejects_subset_identity_or_anchor_changes(self):
        rows = synthetic_rows()
        entries, by_id, windows = [], {}, []
        for row in rows:
            trajectory = row["trajectory_id"]
            if trajectory not in by_id:
                by_id[trajectory] = len(entries)
                entries.append(dict(trajectory_id=trajectory, shape=row["letter"], split="train"))
            windows.append((by_id[trajectory], row["anchor"]))
        with tempfile.TemporaryDirectory(dir=ROOT/"artifacts") as directory:
            root = Path(directory)
            splits = dict(train=[entry["trajectory_id"] for entry in entries], seen_val=[], unseen_test=[])
            data = dict(entries=entries, splits=splits)
            audit = dict(physics_collection_gate_passed=True)
            (root/"manifest.json").write_text(json.dumps(data), encoding="utf-8")
            (root/"physics-audit.json").write_text(json.dumps(audit), encoding="utf-8")
            identity = dict(data_identity(), manifest_sha256=sampling._file_hash(root/"manifest.json"),
                audit_sha256=sampling._file_hash(root/"physics-audit.json"), split_sha256=sampling.canonical_hash(splits))
            pools = sampling.build_pool_manifest(rows, data_identity=identity)
            dataset = SimpleNamespace(root=root, split="train", manifest=data, audit=audit, entries=entries, windows=windows)
            order = sampling.FormalWindowOrder.from_dataset(dataset, pools)
            self.assertEqual(order.cursor, 0)
            self.assertEqual(order.identity(), pools["identity"])
            with self.assertRaises(ValueError):
                sampling.FormalWindowOrder.from_dataset(dataset, pools, indices=[0])
            dataset.windows[0] = (0, 4)
            with self.assertRaisesRegex(ValueError, "anchor"):
                sampling.FormalWindowOrder.from_dataset(dataset, pools)
            dataset.windows[0] = (0, 3)
            dataset.audit["physics_collection_gate_passed"] = False
            with self.assertRaises(ValueError):
                sampling.FormalWindowOrder.from_dataset(dataset, pools)

    def test_record_catalogue_uses_original_declared_history_and_fails_independent_validity_mismatch(self):
        valid = [True]*64
        valid[4] = False
        rows = [dict(valid_h1_candidate=bool(t>=3 and all(valid[t-3:t+1]))) for t in range(64)]
        record = dict(entry=dict(split="train", stage="2A", shape="A"), arrays=dict(valid_transition=valid))
        with patch("tdyssey.native_rgb_motion_statistics.transition_rows", return_value=rows):
            catalogue = sampling.candidate_rows_from_records([record])
            self.assertEqual(len(catalogue), 57)
            self.assertEqual([row["anchor"] for row in catalogue[:2]], [3, 8])
            self.assertEqual([row["dataset_index"] for row in catalogue], list(range(57)))
            rows[8]["valid_h1_candidate"] = False
            with self.assertRaisesRegex(ValueError, "validity differs"):
                sampling.candidate_rows_from_records([record])


if __name__ == "__main__":
    unittest.main()
