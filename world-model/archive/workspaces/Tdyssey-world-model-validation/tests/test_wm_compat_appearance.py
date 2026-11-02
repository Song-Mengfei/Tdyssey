"""Temporary RGB/record fixtures only; no saved inputs, decoder or model."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
import zlib

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("Appearance array fixtures require optional NumPy")

import numpy as np

from tdyssey.wm_compat_appearance import (
    CATEGORIES, MANIFEST_LIMIT, PNG_LIMIT, appearance_metrics, export_case_plates,
)
from tdyssey.wm_compat_transform import transform_for_arm


class FakeBudget:
    def __init__(self, root, failure=None):
        self.output_dir = root
        self.chunks = {}
        self.writes = []
        self.failure = failure

    def write_chunk(self, name, payload, *, max_bytes, final=False):
        self.writes.append((name, max_bytes, final))
        if self.failure:
            raise self.failure
        if name in self.chunks or len(payload) > max_bytes:
            raise AssertionError("Immutable bounded chunks required")
        self.chunks[name] = payload


class FakeSink:
    def __init__(self, root, failure=None):
        self.budget = FakeBudget(root, failure)

    def record(self, name, record, *, maximum, final=False):
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
        self.budget.write_chunk(name, payload, max_bytes=maximum, final=final)


def branch_fixture():
    # The goal-green/gray colors are deliberately changed by B's fixed RGB map.
    history = np.empty((4, 3, 96, 96), dtype=np.float32)
    history[:] = np.array([144, 238, 144], dtype=np.float32)[None, :, None, None]/np.float32(255)
    actual = np.empty_like(history)
    actual[:] = np.array([119, 136, 153], dtype=np.float32)[None, :, None, None]/np.float32(255)
    return SimpleNamespace(anchor="T12", candidate="hold", actual=actual, history=history)


def row_fixture(root, seed=5478, horizon=1, arm="A", mode="AR", *, valid=False, pusher=False):
    key = f"T12/hold/{seed}/{mode}/{arm}/H{horizon}"
    path = "predictions/"+key.replace("/", "-")+".npy"
    # Local deterministic random arrays are mathematical fixtures, not a sampler.
    value = np.random.default_rng(seed+horizon+(arm=="B")*10).uniform(-0.2, 1.2, (3,96,96)).astype(np.float32)
    target = root/path
    target.parent.mkdir(exist_ok=True)
    with target.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    return dict(key=key, anchor="T12", candidate="hold", seed=seed, mode=mode, arm=arm,
                horizon=horizon, status="completed", raw_path=path,
                measurement=dict(state=dict(valid=valid, reason="fixture state"),
                                 shape=dict(valid=valid, reason="fixture shape"),
                                 pusher=dict(valid=pusher, reason="fixture pusher")),
                truth=dict(pose=[250.0, 260.0, 0.2], coverage=0.8), contact_by_h4=True)


def parse_png(payload):
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("PNG signature")
    chunks, offset = [], 8
    while offset < len(payload):
        length = struct.unpack(">I", payload[offset:offset+4])[0]
        kind = payload[offset+4:offset+8]
        data = payload[offset+8:offset+8+length]
        crc = struct.unpack(">I", payload[offset+8+length:offset+12+length])[0]
        if crc != zlib.crc32(kind+data) & 0xffffffff:
            raise AssertionError("PNG CRC")
        chunks.append((kind, data))
        offset += length+12
    if offset != len(payload):
        raise AssertionError("Trailing PNG data")
    return dict(chunks)


class AppearanceTests(unittest.TestCase):
    def test_raw_mse_and_persistence_use_each_arms_real_rgb_without_modification(self):
        branch = branch_fixture()
        raw = np.full((3,96,96), -0.5, dtype=np.float32)
        raw_before, actual_before, history_before = raw.copy(), branch.actual.copy(), branch.history.copy()
        results = {}
        for arm in ("A", "B"):
            actual = transform_for_arm(branch.actual[1], arm)[0].astype(np.float64)
            latest = transform_for_arm(branch.history[-1], arm)[0].astype(np.float64)
            results[arm] = appearance_metrics(raw, branch, arm, 2)
            self.assertEqual(results[arm]["forecast_rgb_mse"], float(np.square(raw.astype(np.float64)-actual).mean()))
            self.assertEqual(results[arm]["persistence_rgb_mse"], float(np.square(latest-actual).mean()))
        self.assertNotEqual(results["A"], results["B"])
        np.testing.assert_array_equal(raw, raw_before)
        np.testing.assert_array_equal(branch.actual, actual_before)
        np.testing.assert_array_equal(branch.history, history_before)

    def test_empty_cases_have_explicit_manifest_without_an_image(self):
        with tempfile.TemporaryDirectory() as directory:
            sink = FakeSink(Path(directory))
            result = export_case_plates([], [], sink)
            self.assertEqual(result, dict(image=None, manifest="case-plates.json", selected_cases=0, display_only=True))
            manifest = json.loads(sink.budget.chunks["case-plates.json"])
            self.assertEqual(manifest["categories"], list(CATEGORIES))
            self.assertFalse(manifest["feedback_or_measurement_use"])
            self.assertEqual(sink.budget.writes, [("case-plates.json", MANIFEST_LIMIT, True)])

    def test_four_categories_are_canonical_bounded_and_truth_only_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root, branch = Path(directory), branch_fixture()
            rows = []
            for seed in (5478,5479,5480):
                for horizon in (1,2,3,4):
                    rows.extend((row_fixture(root,seed,horizon),
                                 row_fixture(root,seed,horizon,"B",valid=True,pusher=True)))
                    if horizon in (2,4):
                        rows.append(row_fixture(root,seed,horizon,"A","TF",valid=True,pusher=True))
            paths = tuple((root/"predictions").iterdir())
            before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
            actual_before, history_before = branch.actual.copy(), branch.history.copy()
            sink = FakeSink(root)
            result = export_case_plates(list(reversed(rows)), [branch], sink)
            self.assertEqual(result["selected_cases"], 24)
            self.assertEqual(sink.budget.writes, [("case-plates.png",PNG_LIMIT,True),
                                                  ("case-plates.json",MANIFEST_LIMIT,True)])
            png, encoded = sink.budget.chunks["case-plates.png"], sink.budget.chunks["case-plates.json"]
            self.assertLessEqual(len(png), PNG_LIMIT)
            self.assertLessEqual(len(encoded), MANIFEST_LIMIT)
            chunks = parse_png(png)
            self.assertEqual(struct.unpack(">IIBBBBB",chunks[b"IHDR"]), (96,864,8,3,0,0,0))
            self.assertEqual(len(chunks[b"PLTE"]), 256*3)
            self.assertEqual(len(zlib.decompress(chunks[b"IDAT"])), 864*97)
            manifest = json.loads(encoded)
            for index, category in enumerate(CATEGORIES):
                cases = manifest["cases"][index*6:index*6+6]
                self.assertTrue(all(case["category"]==category for case in cases))
                self.assertEqual(cases[0]["keys"][0], "T12/hold/5478/AR/A/H2" if category==CATEGORIES[3]
                                 else "T12/hold/5478/AR/A/H1")
            tf_case = manifest["cases"][-1]
            self.assertEqual(tf_case["panels"], ["original actual","T1 actual","A AR","A TF"])
            self.assertTrue(tf_case["physical_truth_separate"]["contact_by_h4"])
            self.assertNotIn("truth", tf_case["model_validity"][0])
            # Physical labels cannot change selection or display image bytes.
            for row in rows:
                row["truth"] = dict(pose=[-123.0,999.0,-2.0],coverage=0.0)
                row["contact_by_h4"] = False
            second = FakeSink(root)
            export_case_plates(rows,[branch],second)
            self.assertEqual(second.budget.chunks["case-plates.png"],png)
            self.assertEqual([case["keys"] for case in json.loads(second.budget.chunks["case-plates.json"])["cases"]],
                             [case["keys"] for case in manifest["cases"]])
            self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},before)
            np.testing.assert_array_equal(branch.actual,actual_before)
            np.testing.assert_array_equal(branch.history,history_before)

    def test_selected_raw_path_and_array_are_validated_before_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = row_fixture(root), row_fixture(root,arm="B",valid=True,pusher=True)
            for alteration in (dict(raw_path="../outside.npy"),dict(raw_path="predictions/other.npy")):
                sink = FakeSink(root)
                with self.assertRaises(ValueError):
                    export_case_plates([dict(first,**alteration),second],[branch_fixture()],sink)
                self.assertEqual(sink.budget.writes,[])
            for value in (np.zeros((3,96,96),dtype=np.float64),
                          np.full((3,96,96),np.nan,dtype=np.float32),
                          np.zeros((96,96,3),dtype=np.float32)):
                with (root/first["raw_path"]).open("wb") as stream:
                    np.save(stream,value,allow_pickle=False)
                sink = FakeSink(root)
                with self.assertRaises(ValueError):
                    export_case_plates([first,second],[branch_fixture()],sink)
                self.assertEqual(sink.budget.writes,[])

    def test_final_chunk_failure_propagates_without_a_retry_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [row_fixture(root),row_fixture(root,arm="B",valid=True,pusher=True)]
            sink = FakeSink(root, OSError("fixture final write failure"))
            with self.assertRaisesRegex(OSError,"fixture final write failure"):
                export_case_plates(rows,[branch_fixture()],sink)
            self.assertEqual(sink.budget.writes,[("case-plates.png",PNG_LIMIT,True)])
            self.assertEqual(sink.budget.chunks,{})


if __name__ == "__main__":
    unittest.main()
