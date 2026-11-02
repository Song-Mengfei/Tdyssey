"""Synthetic temporary TRAIN files only; root owns execution of these checks."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy unavailable for synthetic conditioning-data checks")
import numpy as np

from tdyssey import gpc_conditioning_data as data
from scripts import gpc_checkpoint500_offline_pixels as pixels


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


class FixedTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest_path = "artifacts/fixed.json"
        self.records, self.windows, self.roles, entries = {}, [], [], [None]*34
        identifiers = ("T0", "T1", "T0", "T1", "L0", "L1", "L2", "L3")
        for identifier in dict.fromkeys(identifiers):
            shape = identifier[0]
            parent = "T-B" if shape == "T" else "L-A" if identifier in ("L0", "L1") else "L-B"
            ordinal = int(identifier[1:])
            record_path = f"artifacts/gpc-data-stage-a-20261026/{parent}/episode-{ordinal:04d}.json"
            payload_path = record_path[:-5]+".npz"
            actions = [[float(i*50), float(511-i*40)] for i in range(8)]
            def state(index):
                return dict(shape=shape, control_step=index, geometry_hash="g", physics_hash="p",
                    goal=dict(position=[256.,256.], angle=0.),
                    object_body=dict(pose=dict(position=[100.+ordinal,200.], angle=0.)),
                    pusher_body=dict(pose=dict(position=[float(index),float(ordinal)], angle=0.)),
                    pusher_target=actions[index-1] if index else None)
            transitions = [dict(before=state(i), after=state(i+1), action=actions[i],
                                reward=0., terminated=False, truncated=False) for i in range(8)]
            images = np.stack([np.full((3,96,96), i+ordinal, np.uint8) for i in range(9)])
            buffer = io.BytesIO()
            np.savez_compressed(buffer, images=images)
            payload = self.write(payload_path, buffer.getvalue())
            diagnostics = [dict(n_contacts=int((shape == "T" and i == 4) or
                (shape == "L" and identifier in ("L2", "L3") and i == 3)), coverage=.1) for i in range(8)]
            record = dict(trajectory=dict(trajectory_id=identifier,
                action_semantics="world_frame_position_target_pixels", control_dt=.1, transitions=transitions),
                image_file=Path(payload_path).name, image_sha256=payload["sha256"], step_diagnostics=diagnostics)
            self.records[identifier] = (record_path, record, payload)
            self.roles.append(dict(trajectory_id=identifier, role="train"))
        for i, (cursor, identifier) in enumerate(zip(data.CURSORS, identifiers)):
            path, record, payload = self.records[identifier]
            anchor = 4 if i in (2,3) else 3
            entry = dict(cursor=cursor, step=cursor+1, shape=data.SHAPES[i], trajectory_id=identifier,
                initial_state_sha256=hashlib.sha256(encoded(record["trajectory"]["transitions"][0]["before"])).hexdigest(),
                anchor=anchor, dataset_index=1000+i, stratum="contact_only" if data.CONTACTS[i] else "other",
                contact=data.CONTACTS[i], rotation_degrees=0.)
            entries[cursor] = deepcopy(entry)
            self.windows.append(dict(entry, record=self.write(path, encoded(record)), payload=payload))
        self.metadata = dict(episodes=self.roles)
        self.schedule = dict(entries=entries)
        self.manifest = dict(train_windows=self.windows, bindings={
            "data": {}, "schedule": {},
            "benchmark": dict(path="forbidden-B2.json", bytes=1, sha256="0"*64)})
        self.publish()
        self.identity_patch = patch.object(data, "FIXED_SELECTION", self.fixed_identity)
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)

    def write(self, relative, payload):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return dict(path=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def publish(self):
        self.manifest["bindings"]["data"] = self.write("artifacts/data.json", encoded(self.metadata))
        self.manifest["bindings"]["schedule"] = self.write("artifacts/schedule.json", encoded(self.schedule))
        self.fixed_identity = self.write(self.manifest_path, encoded(self.manifest))
        if hasattr(self, "identity_patch"):
            data.FIXED_SELECTION = self.fixed_identity

    def no_payload_open(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.suffix == ".npz" or "forbidden" in str(path):
                raise AssertionError("metadata path opened forbidden/payload bytes")
            return original(path, *args, **kwargs)
        return patch.object(Path, "open", guarded)

    def test_metadata_only_exact_order_and_real_scene_denominators(self):
        with self.no_payload_open():
            prepared = data.prepare_training_windows(self.root)
        self.assertEqual([w["cursor"] for w in prepared["windows"]], list(data.CURSORS))
        self.assertEqual([w["window_id"] for w in prepared["windows"]],
                         [f"train-cursor-{cursor:03d}" for cursor in data.CURSORS])
        self.assertEqual(prepared["denominators"]["total"],
            dict(windows=8, trajectories=6, scenes=6, contact_windows=4, noncontact_windows=4))
        self.assertEqual(prepared["denominators"]["by_shape"]["T"]["scenes"], 2)
        self.assertEqual(prepared["denominators"]["by_shape"]["L"]["scenes"], 4)
        self.assertEqual(len(prepared["identities"]), 9)
        self.assertEqual(prepared["scope"]["payload_opens"], 0)
        json.dumps(prepared, allow_nan=False)

    def test_wrong_role_cursor_source_and_schedule_rejected_before_payload_open(self):
        original = deepcopy((self.metadata, self.schedule, self.manifest))
        for issue in ("held_model_test", "development", "cursor", "source", "index"):
            self.metadata, self.schedule, self.manifest = deepcopy(original)
            if issue in ("held_model_test", "development"):
                self.metadata["episodes"][0]["role"] = issue
            elif issue == "cursor":
                self.manifest["train_windows"][0]["cursor"] = 25
            elif issue == "source":
                self.manifest["train_windows"][0]["record"]["path"] = "artifacts/held/episode.json"
            else:
                self.schedule["entries"][24]["dataset_index"] += 1
            self.publish()
            with self.subTest(issue=issue), self.no_payload_open(), self.assertRaises(ValueError):
                data.prepare_training_windows(self.root)

    def test_five_selective_rows_t1_once_and_four_aligned_normalized_actions(self):
        from tdyssey import wm_compat_transform
        spec = data.prepare_training_windows(self.root)["windows"][0]
        transformed, reads = [], []
        original = pixels.read_npy_window
        def read(*args, **kwargs):
            reads.append((args[1], args[2], args[3]))
            return original(*args, **kwargs)
        def t1(frame):
            transformed.append(frame.copy())
            return frame + np.float32(.01), {}
        with patch.object(pixels, "read_npy_window", side_effect=read), \
                patch.object(wm_compat_transform, "transform_t1", side_effect=t1):
            loaded = data.load_training_window(spec)
        self.assertEqual(set(loaded), {"images", "raw_images", "actions", "read_receipt"})
        self.assertEqual(reads, [("images.npy", spec["anchor"]-3, 5)])
        self.assertEqual(len(transformed), 5)
        np.testing.assert_array_equal(loaded["images"], loaded["raw_images"]+np.float32(.01))
        expected = np.asarray(spec["actions_absolute"],np.float32)/np.float32(511)*np.float32(2)-np.float32(1)
        np.testing.assert_array_equal(loaded["actions"], expected)
        self.assertEqual(loaded["read_receipt"]["selected_member"]["rows"], 5)
        self.assertEqual(loaded["read_receipt"]["unselected_numeric_frames_decoded"], 0)
        self.assertEqual(spec["latest_state"]["control_step"]+1, spec["target_state"]["control_step"])

    def test_mutated_prepared_spec_and_changed_train_record_stop_before_image_open(self):
        spec = data.prepare_training_windows(self.root)["windows"][0]
        bad = deepcopy(spec)
        bad["actions_absolute"][0][0] += 1
        with self.no_payload_open(), self.assertRaises(ValueError):
            data.load_training_window(bad)
        path = self.root/spec["record"]["path"]
        path.write_bytes(path.read_bytes()+b" ")
        with self.no_payload_open(), self.assertRaises(ValueError):
            data.load_training_window(spec)

    def test_h1_measurement_keeps_invalid_nulls_components_and_raw_rgb(self):
        from tdyssey import wm_image_measurement_v2 as measurement
        spec = data.prepare_training_windows(self.root)["windows"][0]
        raw = np.full((3,96,96), -.2, np.float32)
        target = np.full_like(raw, .5)
        components = [dict(position=[1.,2.],mass=12.),dict(position=[4.,5.],mass=13.)]
        full = dict(state=dict(valid=False,pose=None,candidate_pose=dict(position=[3.,4.],angle=0.)),
            shape=dict(valid=False,reason="fragmented_object",full_soft_iou=.2,roi_soft_iou=.8),
            coverage=dict(valid=False,estimate=None),termination=dict(status="unknown"),
            pusher=dict(valid=False,position=None,reason="ambiguous_multiple_pushers",meaningful_components=components))
        chromatic = {key: None for key in ("roi","object_alpha_mass","alpha_mass_in_roi","alpha_mass_outside_roi",
            "alpha_fraction_outside_roi","seed_component_count","largest_component_mass_fraction",
            "raw_min","raw_max","raw_channel_means","raw_out_of_range_channel_fraction")}
        before = raw.copy()
        with patch.object(measurement, "decode", return_value=full) as decode, \
                patch.object(pixels,"pixel_diagnostics",return_value=(chromatic,{})):
            result = data.measure_h1(raw,spec,target)
        self.assertIs(decode.call_args.args[0],raw)
        self.assertEqual(decode.call_args.args[1:], (spec["shape"],spec["goal"]))
        self.assertIsNone(result["measurement"]["state"]["pose"])
        self.assertIsNone(result["pusher_position_error_pixels"])
        self.assertEqual(result["pusher_diagnostics"]["meaningful_components"],components)
        self.assertEqual(result["measurement"]["shape"]["full_soft_iou"],.2)
        self.assertAlmostEqual(result["target_mae"],.7,places=6)
        np.testing.assert_array_equal(raw,before)
        json.dumps(result,allow_nan=False)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow unavailable")
    def test_panel_is_bounded_display_only_and_keeps_both_weight_roles(self):
        from PIL import Image
        loaded = dict(images=np.zeros((5,3,96,96),np.float32))
        predicted = {role: {arm: np.full((3,96,96), -.1 if arm == "A" else 1.1,np.float32)
                            for arm in ("A","B")} for role in ("frozen","adapted")}
        before = deepcopy(predicted)
        payload = data.training_panel_png(loaded,predicted)
        self.assertLessEqual(len(payload),data.PANEL_MAX_BYTES)
        with Image.open(io.BytesIO(payload)) as image:
            self.assertEqual(image.size,(384,240))
        for role in predicted:
            for arm in predicted[role]:np.testing.assert_array_equal(predicted[role][arm],before[role][arm])


if __name__ == "__main__":
    unittest.main()
