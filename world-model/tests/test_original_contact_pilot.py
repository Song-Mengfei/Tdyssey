"""Synthetic saved-label checks; no simulator/model construction."""
import unittest
import numpy as np

from tdyssey.original_contact_pilot_report import trajectory_rows, _micro_summary, conditioning_packet_sha256, arc_cohort_rows, aggregate_actual_arc_hits


def state(x=256., angle=0.):
    result = np.zeros(15)
    result[:4] = [180., 256., x, 256.]
    result[4] = angle
    return result


class SavedLabelMetricsTest(unittest.TestCase):
    def test_arc_compaction_preserves_dt_and_distinct_microcounts_per_cohort(self):
        all_rows = []
        for phase, complete, valid, durations in (
                ('formal', True, True, [.01, .02, .03]),
                ('calibration-v1', True, True, [.02, .03]),
                ('formal', False, False, [.015, .025])):
            identity = dict(trajectory_id=phase+str(complete), phase_group=phase,
                            split='calibration' if phase.startswith('cal') else 'train_candidate',
                            trajectory_complete=complete, control_valid=valid)
            for micro, seconds in enumerate(durations):
                all_rows.append(dict(identity, ring=0, arc_bin=7, microstep_index=micro, seconds=seconds))
        compact = []
        for tid in {row['trajectory_id'] for row in all_rows}:
            source = [row for row in all_rows if row['trajectory_id'] == tid]
            header = {key: source[0][key] for key in ('trajectory_id', 'phase_group', 'split', 'trajectory_complete', 'control_valid')}
            compact.extend(dict(header, **row) for row in aggregate_actual_arc_hits(source))
        for cohort in ('all_stages_all_attempts', 'formal_primary_candidates'):
            before = arc_cohort_rows(all_rows, cohort)
            after = arc_cohort_rows(compact, cohort)
            self.assertAlmostEqual(sum(row['seconds'] for row in before), sum(row['seconds'] for row in after))
            self.assertEqual(len(before), sum(row['microstep_count'] for row in after))
        duplicate = all_rows[0]
        result = aggregate_actual_arc_hits([duplicate, duplicate])
        self.assertEqual(result[0]['microstep_count'], 1)
        self.assertAlmostEqual(result[0]['seconds'], duplicate['seconds'])

    def test_calibration_or_invalid_arcs_cannot_enter_formal_candidate_coverage(self):
        primary = dict(trajectory_id='formal-good', phase_group='formal', split='train_candidate',
                       trajectory_complete=True, control_valid=True)
        rows = [primary,
                dict(primary, trajectory_id='cal', phase_group='calibration-v1', split='calibration'),
                dict(primary, trajectory_id='extra', phase_group='supplement'),
                dict(primary, trajectory_id='invalid-control', control_valid=False),
                dict(primary, trajectory_id='partial-trajectory', trajectory_complete=False)]
        self.assertEqual([row['trajectory_id'] for row in arc_cohort_rows(rows, 'formal_primary_candidates')], ['formal-good'])
        self.assertEqual(len(arc_cohort_rows(rows, 'all_stages_all_attempts')), 5)

    def test_past_action_difference_is_not_deduplicated(self):
        # Same actual RGB history and future actions, different recorded past
        # action: the model's action conditioning must remain distinguishable.
        history = [bytes([i])*32 for i in range(10)]
        original = np.zeros((9, 2), np.float64)
        altered = original.copy()
        altered[2] = [11., 19.]
        for horizon in (1, 4):
            self.assertNotEqual(conditioning_packet_sha256(history, original, 3, horizon),
                                conditioning_packet_sha256(history, altered, 3, horizon))
        # Unrelated actions outside history3+H do not alter this packet.
        altered = original.copy()
        altered[8] = [123., 456.]
        self.assertEqual(conditioning_packet_sha256(history, original, 3, 4),
                         conditioning_packet_sha256(history, altered, 3, 4))

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
