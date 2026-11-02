"""Pure saved-evidence round trips; no simulator, model, GPU or optimizer."""
import hashlib
import io
import json
import math
import unittest
import zlib

import numpy as np

from tdyssey.original_contact_pilot_storage import (
    FORMAT, STATE_FIELDS, TELEMETRY_STORAGE_KEYS, canonical, safe_json,
    pack_telemetry, unpack_telemetry, load_attempt_archive,
)


def complete_row(values):
    return {**{field: list(values) for field in STATE_FIELDS},
            "dt": .0005, "executed": True,
            "contacts": [{"kind": "agent_object", "phase": "post_solve",
                          "points": [{"point_a": [-0.0, 1.25], "distance": -.05}],
                          "total_impulse": [123.45678901234567, -0.0]}],
            # Reserved-looking user data must stay ordinary data.
            "__state_float64_refs_v2__": {"post_state": "literal user data"}}


def legacy(telemetry, decisions):
    payload = canonical(safe_json(dict(telemetry=telemetry, decisions=decisions)))
    return dict(telemetry_zlib=np.frombuffer(zlib.compress(payload), np.uint8),
                telemetry_sha256=np.frombuffer(hashlib.sha256(payload).digest(), np.uint8))


class TelemetryStorageTest(unittest.TestCase):
    def test_exact_finite_bits_canonical_hash_and_independent_aliases(self):
        values = [0.0, -0.0, 1.0, -1.0, np.nextafter(0., 1.),
                  float(np.finfo(np.float64).max), 1e-300, 1e300,
                  math.pi, 1.2345678901234567, 5e-324, -5e-324,
                  1000000000000000.0, 0.000000123456789, -.05]
        telemetry = [{"substeps": [complete_row(values), complete_row(values)]}]
        decisions = [{"target": [1, 1.0], "phase": "接触"}]
        before = canonical(safe_json(dict(telemetry=telemetry, decisions=decisions)))
        packed = pack_telemetry(telemetry, decisions)
        after = unpack_telemetry(packed)
        self.assertEqual(canonical(after), before)
        self.assertEqual(packed['telemetry_sha256'].tobytes(), hashlib.sha256(before).digest())
        self.assertEqual(packed['telemetry_microstates_float64'].shape, (1, 15))
        restored = after['telemetry'][0]['substeps'][0]
        for field in STATE_FIELDS:
            self.assertEqual(np.asarray(restored[field], '<f8').tobytes(), np.asarray(values, '<f8').tobytes())
        self.assertIsNot(restored['pre_state'], restored['integrator_pre_state'])
        restored['integrator_pre_state'][0] = 2.
        self.assertEqual(restored['pre_state'][0], 0.)
        self.assertEqual(canonical(safe_json(dict(telemetry=telemetry, decisions=decisions))), before)

    def test_partial_mixed_types_nonfinite_and_ragged_rows_remain_json(self):
        rows = []
        for special in ([0]*15, [0.]*14, [True]+[0.]*14,
                        [2**90]+[0.]*14, None,
                        [float('nan'), float('inf'), -float('inf')]+[0.]*12):
            row = complete_row([0.]*15)
            row['post_state'] = special
            rows.append(row)
        missing = complete_row([0.]*15)
        del missing['post_state']
        rows.extend([missing, {"executed": False}, None])
        telemetry = [{"substeps": rows}, {"substeps": []}, "opaque legacy row"]
        packed = pack_telemetry(telemetry, [])
        self.assertEqual(packed['telemetry_microstates_float64'].shape, (0, 15))
        self.assertEqual(canonical(unpack_telemetry(packed)),
                         canonical(safe_json(dict(telemetry=telemetry, decisions=[]))))

    def test_signed_zero_vectors_are_not_deduplicated_together(self):
        positive = [0.]*15
        negative = positive.copy(); negative[0] = -0.0
        data = pack_telemetry([{"substeps": [complete_row(positive), complete_row(negative)]}], [])
        self.assertEqual(len(data['telemetry_microstates_float64']), 2)
        restored = unpack_telemetry(data)['telemetry'][0]['substeps']
        self.assertFalse(np.signbit(restored[0]['pre_state'][0]))
        self.assertTrue(np.signbit(restored[1]['pre_state'][0]))

    def test_legacy_and_v2_npz_allow_pickle_false_and_public_keys(self):
        import tempfile
        from pathlib import Path
        telemetry = [{"substeps": [complete_row([.5]*15)]}]
        decisions = [{"phase": "push"}]
        with tempfile.TemporaryDirectory() as temp:
            for name, storage in (("legacy", legacy(telemetry, decisions)),
                                  ("v2", pack_telemetry(telemetry, decisions))):
                path = Path(temp)/(name+'.npz')
                images = np.zeros((2, 3, 96, 96), np.uint8)
                actions = np.asarray([[1., -0.]], np.float64)
                np.savez_compressed(path, rgb=images, requested_actions=actions, **storage)
                with np.load(path, allow_pickle=False) as saved:
                    self.assertTrue(all(saved[k].dtype.kind != 'O' for k in saved.files))
                loaded = load_attempt_archive(path)
                self.assertEqual(set(loaded), {'rgb', 'requested_actions', 'telemetry', 'decisions'})
                self.assertFalse(set(loaded)&set(TELEMETRY_STORAGE_KEYS))
                self.assertEqual(canonical({k: loaded[k] for k in ('telemetry', 'decisions')}),
                                 canonical(dict(telemetry=telemetry, decisions=decisions)))
                self.assertEqual(loaded['rgb'].tobytes(), images.tobytes())
                self.assertEqual(loaded['requested_actions'].tobytes(), actions.tobytes())

    def test_corrupt_states_refs_shape_and_skeleton_are_rejected(self):
        packed = pack_telemetry([{"substeps": [complete_row([.5]*15)]}], [])
        variants = []
        def copy(): return {k: v.copy() for k, v in packed.items()}
        changed = copy(); changed['telemetry_microstates_float64'][0, 1] += 1.; variants.append(changed)
        changed = copy(); changed['telemetry_microstates_float64'][0, 1] = np.nan; variants.append(changed)
        changed = copy(); changed['telemetry_microstates_float64'] = np.zeros((1, 14)); variants.append(changed)
        changed = copy(); changed['telemetry_microstates_float64'] = np.zeros((1, 15), np.float32); variants.append(changed)
        changed = copy(); changed['telemetry_microstate_refs'][0, 0] = 4; variants.append(changed)
        changed = copy(); changed['telemetry_microstate_refs'][0, 3] = 9; variants.append(changed)
        changed = copy(); changed['telemetry_microstate_refs'] = np.vstack([changed['telemetry_microstate_refs'], changed['telemetry_microstate_refs'][0]]); variants.append(changed)
        changed = copy(); changed['telemetry_microstate_refs'] = changed['telemetry_microstate_refs'].astype(np.float64); variants.append(changed)
        changed = copy(); changed['telemetry_microstate_refs'] = changed['telemetry_microstate_refs'][:-1]; variants.append(changed)
        changed = copy(); changed['telemetry_microstates_float64'] = np.vstack([changed['telemetry_microstates_float64'], np.ones(15)]); variants.append(changed)
        changed = copy(); skeleton = json.loads(zlib.decompress(changed['compact_telemetry_zlib']).decode())
        skeleton['decisions'] = ['corrupt']
        payload = canonical(skeleton)
        changed['compact_telemetry_zlib'] = np.frombuffer(zlib.compress(payload), np.uint8)
        changed['compact_telemetry_sha256'] = np.frombuffer(hashlib.sha256(payload).digest(), np.uint8)
        variants.append(changed)
        for index, variant in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                unpack_telemetry(variant)

    def test_empty_and_legacy_digest_checks(self):
        for telemetry in ([], None, 'opaque'):
            self.assertEqual(unpack_telemetry(pack_telemetry(telemetry, [])), dict(telemetry=telemetry, decisions=[]))
        corrupt = legacy([], [])
        corrupt['telemetry_sha256'] = np.zeros(32, np.uint8)
        with self.assertRaises(ValueError):unpack_telemetry(corrupt)
        missing = pack_telemetry([], []); del missing['telemetry_format']
        with self.assertRaises(ValueError):unpack_telemetry(missing)
        mixed = pack_telemetry([], []); mixed['telemetry_zlib'] = legacy([], [])['telemetry_zlib']
        with self.assertRaises(ValueError):unpack_telemetry(mixed)


if __name__ == '__main__':
    unittest.main()
