"""CPU callback fixtures for paired provenance, raw feedback and denominators."""
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey import native_rgb_r3b_evaluation as evaluation


class FakeModel:
    training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)


class FakeStore:
    def __init__(self, spec):
        self.renderer = type('Renderer', (), {'stats': {'physics_steps': 0}})()

    def inputs(self, row):
        return np.zeros((4, 3, 96, 96), dtype=np.float32), np.arange(14).reshape(7, 2).astype(np.float64)

    def frame(self, row, index):
        # This distinct future canary must never enter the model history.
        return np.full((3, 96, 96), 255 if index <= row['anchor'] else 99, dtype=np.uint8)

    def full_mask(self, row, index):
        return np.zeros((96, 96), dtype=np.float32)

    def close(self):
        pass


def fixture_spec():
    ids = ['family-T:3', 'family-C:3']
    rows = [dict(window_id=key, anchor=3, letter=letter, cohorts=['natural'],
        truth_motion={'1': dict(translation_image96_px=0., signed_rotation_deg=0.),
                      '4': dict(translation_image96_px=3., signed_rotation_deg=6.)}) for key, letter in zip(ids, 'TC')]
    return dict(rows=rows, batches=[dict(window_ids=ids, context='new_seen_union', batch_index=0, seeds=[11, 22, 33, 44])])


class R3BEvaluationTests(unittest.TestCase):
    def test_family_anchor_is_legal_repeatable_and_does_not_consume_global_rng(self):
        before = random.getstate(); np_before = np.random.get_state()
        anchors = [evaluation._uniform_anchor(f'family-{i}', range(3, 45)) for i in range(1000)]
        self.assertEqual(random.getstate(), before)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_before[1]))
        self.assertEqual(set(anchors), set(range(3, 45)))
        self.assertEqual(anchors, [evaluation._uniform_anchor(f'family-{i}', range(3, 45)) for i in range(1000)])

    def _run(self, root, callback):
        spec = fixture_spec(); evaluation._json(root / 'evaluation/spec.json', spec)
        model = FakeModel()
        with patch.object(evaluation, '_spec', return_value=spec), patch.object(evaluation, 'WindowStore', FakeStore), \
                patch('tdyssey.native_rgb_r2c1r1_training.torch_native_sample', return_value=callback):
            result = evaluation.evaluate(root, model, object(), 't0')
        self.assertTrue(model.training)
        return result

    def test_raw_float32_feedback_and_h1_h4_references_share_calls(self):
        calls = []

        def callback(history, actions, *, seed):
            step = len(calls)
            self.assertEqual(seed, [11, 22, 33, 44][step])
            self.assertEqual(history.dtype, np.float32)
            self.assertEqual(actions.dtype, np.float32)
            if step:
                np.testing.assert_array_equal(history[:, -1], calls[-1])
            self.assertFalse(np.any(history == np.float32(99 / 255)))
            raw = np.full((2, 3, 96, 96), np.float32(1.2 + step), dtype=np.float32)
            calls.append(raw.copy())
            return raw

        with tempfile.TemporaryDirectory(dir=evaluation.ROOT / 'artifacts') as temporary:
            root = Path(temporary); result = self._run(root, callback)
            self.assertEqual(len(calls), 4)
            self.assertEqual(result['stats']['native_sampler_batch_attempts'], 4)
            index = evaluation._read(root / 'evaluation/t0/index.json')
            self.assertEqual(len(index['raw_files']), 4)
            refs = index['endpoint_refs']['family-T:3']
            self.assertEqual(refs['1']['step'], 0)
            self.assertEqual(refs['4']['step'], 3)
            np.testing.assert_array_equal(evaluation._load_ref(refs['1']), calls[0][0])
            np.testing.assert_array_equal(evaluation._load_ref(refs['4']), calls[3][0])
            self.assertEqual(len(result['rows']), 8)
            failed = [r for r in result['rows'] if r['method'] == 't0']
            self.assertTrue(all(r['metrics']['hard_generation_failure'] for r in failed))
            self.assertTrue(all(r['metrics']['rotation_error_deg'] is None for r in failed))

    def test_nonfinite_stops_once_and_retains_all_planned_rows(self):
        calls = []

        def callback(history, actions, *, seed):
            calls.append(seed)
            return np.full((2, 3, 96, 96), np.nan, dtype=np.float32)

        with tempfile.TemporaryDirectory(dir=evaluation.ROOT / 'artifacts') as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, 'Nonfinite raw output'):
                self._run(root, callback)
            result = evaluation._read(root / 'evaluation/t0/metrics.json')
            self.assertFalse(result['evaluation_complete'])
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(result['rows']), 8)
            self.assertTrue(all(r['metrics']['shape_status'] == 'unknown' for r in result['rows']))
            self.assertTrue(all(r['metrics']['rotation_error_deg'] is None for r in result['rows']))
            groups = [r for r in result['summaries'] if r['cohort'] == 'natural' and r['horizon'] == 4 and r['method'] == 't0' and r['subgroup'] == 'all']
            self.assertEqual(groups[0]['planned_denominator'], 2)
            self.assertEqual(groups[0]['shape_unknown'], 2)
            self.assertEqual(groups[0]['shape_pass_fraction_full_denominator'], 0.)
            # Retrying an uncertain attempt is explicitly blocked before callback.
            with self.assertRaisesRegex(RuntimeError, 'incomplete metrics'):
                self._run(root, callback)
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
