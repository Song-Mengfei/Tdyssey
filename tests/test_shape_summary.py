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
