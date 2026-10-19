"""Peak diagnostics preserve the timing and size of lost progress."""

import importlib.util
from pathlib import Path
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


if __name__ == '__main__':
    unittest.main()
