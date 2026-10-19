"""Check saved peak-frame indexing without inference or physical stepping."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('TDYSSEY_OFFICIAL_ROOT'), 'Enable the installed official profile for OpenCV checks')
class PeakFrameChecks(unittest.TestCase):
    def test_best_control_uses_post_action_frame_after_initial_frame(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'T-100000.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64,64))
            self.assertTrue(writer.isOpened())
            for color in ((220,0,0), (0,220,0), (0,0,220)):
                writer.write(np.full((64,64,3), color, dtype=np.uint8))
            writer.release()
            report = {'episodes': [dict(shape='T', scene_seed=100000, steps=[
                {'metrics': {'iou': .8, 'coverage': .89}},
                {'metrics': {'iou': .1, 'coverage': .18}}])]}
            path = root/'report.json'
            path.write_text(json.dumps(report))
            command = Path(__file__).resolve().parents[1]/'scripts/extract_peak_frames.py'
            result = subprocess.run([sys.executable, '-B', str(command), '--report', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['saved'][0]['control_step'], 1)
            decoded = cv2.VideoCapture(str(video))
            decoded.read()  # initial state
            ok, expected = decoded.read()  # after control 1: the peak
            decoded.release()
            self.assertTrue(ok)
            np.testing.assert_array_equal(cv2.imread(str(root/'T-100000-best.png')), expected)
            self.assertEqual(json.loads(path.read_text()), report)


if __name__ == '__main__':
    unittest.main()
