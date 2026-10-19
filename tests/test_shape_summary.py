"""Portable report presentation preserves outcomes and escapes external text."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('summarize_shapes', Path(__file__).resolve().parents[1] / 'scripts/summarize_shapes.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ShapeSummaryChecks(unittest.TestCase):
    def test_peak_diagnostics_record_earliest_tie_and_lost_progress(self):
        episode = {"steps": [{"metrics": {"iou": iou, "coverage": coverage}}
                             for iou, coverage in ((.2, .3), (.8, .89), (.8, .89), (.1, .18))]}
        result = module.peak_diagnostics(episode)
        self.assertEqual(result["best_step"], 2)
        self.assertEqual(result["best_coverage"], .89)
        self.assertEqual(result["final_coverage"], .18)
        self.assertAlmostEqual(result["best_to_final_iou_drop"], .7)
        self.assertTrue(all(v is None for v in module.peak_diagnostics({"steps": []}).values()))

    def test_budget_failure_and_partial_cutoff_stay_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {'status': 'stopped_early', 'episodes': []}
            for shape, success, reason in [('T', True, 'upstream_success'), ('L', False, 'control_budget'), ('<Q>', False, 'wall_time_budget')]:
                report['episodes'].append(dict(shape=shape, scene_seed=100000, success=success, stop_reason=reason,
                                               control_steps=128, max_iou=.925, final_iou=.1, max_reward=1, contact_steps=70))
            path = root / 'report.json'
            path.write_text(json.dumps(report))
            (root / 'T-100000.mp4').touch()
            output = root / 'summary.html'
            module.write_summary(path, output)
            text = output.read_text()
            self.assertIn('Not solved in budget', text)
            self.assertIn('wall_time_budget', text)
            self.assertIn('&lt;Q&gt;', text)
            self.assertNotIn('<Q>', text)
            self.assertIn('src="T-100000.mp4"', text)
            with output.with_suffix('.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['success'] for row in rows], ['True', 'False', 'False'])
            self.assertEqual(path.read_text(), json.dumps(report))


if __name__ == '__main__':
    unittest.main()
