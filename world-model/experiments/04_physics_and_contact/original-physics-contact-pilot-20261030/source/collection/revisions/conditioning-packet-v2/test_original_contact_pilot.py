"""Synthetic saved-label checks; no simulator/model construction."""
import unittest
import numpy as np

from tdyssey.original_contact_pilot_report import trajectory_rows, _micro_summary


def state(x=256., angle=0.):
    result = np.zeros(15)
    result[:4] = [180., 256., x, 256.]
    result[4] = angle
    return result


class SavedLabelMetricsTest(unittest.TestCase):
    def test_contact_duration_uses_dt_not_microstep_count(self):
        before, middle, end = state(), state(266.), state(266.)
        rows = [dict(dt=.02, executed=True, pre_state=before.tolist(), post_state=middle.tolist(),
                     contacts=[dict(kind='agent_object', phase='post_solve', shapes_order=['object', 'agent'], points=[])], penetrations=[0, 0, 0]),
                dict(dt=.08, executed=True, pre_state=middle.tolist(), post_state=end.tolist(), contacts=[], penetrations=[0, 0, 0])]
        value = _micro_summary('O', dict(substeps=rows))
        self.assertAlmostEqual(value['seconds'], .1)
        self.assertAlmostEqual(value['AO_seconds']/value['seconds'], .2)

    def test_return_path_is_not_erased_by_zero_endpoint_motion(self):
        before, middle, end = state(), state(266.), state()
        telemetry = dict(valid=True, substeps=[
            dict(dt=.025, executed=True, pre_state=before.tolist(), post_state=middle.tolist(), contacts=[], penetrations=[0, 0, 0]),
            dict(dt=.075, executed=True, pre_state=middle.tolist(), post_state=end.tolist(), contacts=[], penetrations=[0, 0, 0])])
        metadata = dict(slot=dict(letter='O', slot_id='synthetic', family_id='F-synthetic', split='calibration', phase='calibration-v1'), complete=False)
        data = dict(states=np.asarray([before, end]), requested_actions=np.zeros((1, 2)),
                    control_valid=np.ones(1, bool), rgb=np.zeros((2, 3, 96, 96), np.uint8),
                    telemetry=[telemetry], decisions=[dict(phase='push', selected_arc_id=0)])
        _, windows, _ = trajectory_rows(metadata, data)
        self.assertAlmostEqual(windows[0]['translation_world_px'], 0.)
        self.assertAlmostEqual(windows[0]['path_world_px'], 20.)
        self.assertAlmostEqual(windows[0]['seconds'], .1)
        self.assertFalse(windows[0]['candidate'])

    def test_partial_control_cannot_be_called_h1(self):
        before, end = state(), state(266.)
        metadata = dict(slot=dict(letter='O', slot_id='synthetic', family_id='F-synthetic', split='train_candidate', phase='formal'), complete=False)
        data = dict(states=np.asarray([before, end]), requested_actions=np.zeros((1, 2)),
                    control_valid=np.zeros(1, bool), rgb=np.zeros((2, 3, 96, 96), np.uint8),
                    telemetry=[dict(substeps=[dict(dt=.07, executed=True, pre_state=before.tolist(),
                    post_state=end.tolist(), contacts=[], penetrations=[0, 0, 0])])],
                    decisions=[dict(phase='push', selected_arc_id=0)])
        controls, windows, _ = trajectory_rows(metadata, data)
        self.assertEqual(controls[0]['category'], 'invalid')
        self.assertFalse(windows[0]['valid'])
        self.assertFalse(windows[0]['candidate'])


if __name__ == '__main__':
    unittest.main()
