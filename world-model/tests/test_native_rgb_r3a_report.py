"""Geometric/array fixtures only: no simulator, render, model or new rollout."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tdyssey.native_rgb_r3a_report import area_centers, motion, motion_summary, audit
from tdyssey.envs.official import official_shape

ROOT = Path(__file__).resolve().parents[1]


class ReportContracts(unittest.TestCase):
    def states(self, letter='L', increment_degrees=7., local=None):
        # Rotate an offset local centroid about a fixed world area center.
        # Body origin must move, so measuring state[:,2:4] would be wrong.
        local = np.asarray(official_shape(letter).centroid if local is None else local)
        angle = np.deg2rad(np.arange(49)*increment_degrees)
        rotated = np.column_stack((np.cos(angle)*local[0]-np.sin(angle)*local[1],
            np.sin(angle)*local[0]+np.cos(angle)*local[1]))
        state = np.zeros((49, 15), np.float64)
        state[:, 0:2] = (120., 120.)
        state[:, 2:4] = np.asarray((256., 256.))-rotated
        state[:, 4] = angle
        return state

    def test_offset_origin_pure_rotation_has_zero_area_center_translation(self):
        local = (12., -9.)
        states = self.states(local=local)
        self.assertGreater(float(np.linalg.norm(states[1, 2:4]-states[0, 2:4])), 1.)
        # Existing centered L/O have centroid0; explicit offset fixture guards
        # the general coordinate contract instead of relying on that accident.
        with patch('tdyssey.envs.official.official_shape', return_value=SimpleNamespace(centroid=local)):
            np.testing.assert_allclose(area_centers(states, 'L'), np.full((49, 2), 256.), atol=1e-12)
            for horizon, count in ((1, 45), (4, 42)):
                translation, rotation = motion(states, horizon, 'L')
                self.assertEqual(len(translation), count)
                np.testing.assert_allclose(translation, 0., atol=1e-12)
                np.testing.assert_allclose(rotation, 7.*horizon, atol=1e-10)

    def test_signed_unwrapped_rotation_is_not_folded_by_shape_symmetry(self):
        for sign in (-1, 1):
            states = self.states(letter='O', increment_degrees=sign*120.)
            for horizon, count in ((1, 45), (4, 42)):
                translation, rotation = motion(states, horizon, 'O')
                np.testing.assert_allclose(rotation, sign*120.*horizon, atol=1e-9)
                record = motion_summary(translation, rotation)
                self.assertEqual(record['count'], count)
                self.assertEqual(record['obvious_count'], count)
                self.assertEqual(record['positive_rotation_ge5'], count if sign > 0 else 0)
                self.assertEqual(record['negative_rotation_le_minus5'], count if sign < 0 else 0)

    def test_actual_substep_contact_gate_excludes_wall_and_precontact_arcs(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts', prefix='r3a-report-contract-') as temp:
            base = Path(temp); states = self.states(increment_degrees=60.)
            states[:, 4] += np.deg2rad(29.)
            actions = np.full((48, 2), 256., np.float32)
            substep_control = np.repeat(np.arange(48, dtype=np.int16), 10)
            flags = np.zeros((480, 3), np.bool_)
            flags[:10, 0] = True; flags[20:40, 0] = True  # AO at controls0,2,3.
            flags[10:20, 1] = True                       # Wall-only control1.
            # tag columns are substep, contact-kind(0AO/1OW/2AW), phase(0pre/1post).
            tags = np.array(((0, 0, 0, 1, 31), (0, 0, 1, 1, 31),
                (10, 1, 1, 1, 31), (10, 2, 1, 0, 31)), np.int16)
            substates = np.vstack([states[0]]+[states[ci]+(states[ci+1]-states[ci])*(j+1)/10
                for ci in range(48) for j in range(10)])
            substates[1, :2] = (200., 200.)  # post pusher differs from pre120/120
            contacts = np.zeros((4, 11), np.float32)
            contacts[:, 4] = 1.  # actual contact normal x
            contacts[1, :4] = (256., 256., 3., 4.)
            contacts[1, 10] = 4.  # actual normal lever
            contacts[2, :2] = (7., 120.)  # actual object-wall point near left wall
            path = base/'fixture.npz'
            np.savez_compressed(path, states=states, actions=actions,
                control_valid=np.ones(48, np.bool_), substep_penetrations=np.zeros((480, 3), np.float32),
                substep_control_index=substep_control, contact_flags=flags, contact_tags=tags,
                substep_states=substates,
                contact_values=contacts, contact_arc_offsets=np.arange(5, dtype=np.int32),
                contact_arc_ids=np.array((91, 7, 80, 81), np.int16))
            metadata = base/'fixture.json'; metadata.write_text(json.dumps({'scenario': 'CPU-fixture'}), encoding='utf-8')
            entry = dict(trajectory_id='CPU-fixture', family_id='CPU-family', slot_id='CPU-slot',
                letter='L', split='train', seed=1, file=path.name, metadata_file=metadata.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
            manifest = base/'manifest.json'; manifest.write_text(json.dumps({'entries': [entry]}), encoding='utf-8')
            result, rows, _ = audit(manifest, base/'report')
            self.assertEqual(result['completed_controls'], 48); self.assertEqual(result['known_substeps'], 480)
            self.assertEqual(result['invalid_successful_entries'], [])
            self.assertEqual(rows[0]['H1_count'], 45); self.assertEqual(rows[0]['H4_count'], 42)
            self.assertLess(rows[0]['total_translation_image96px'], 1e-10)
            self.assertEqual(rows[0]['contact_control_count'], 3)
            self.assertEqual(rows[0]['max_contact_run_controls'], 2)
            self.assertEqual(result['actual_post_agent_object_contact_arc_counts']['L'], {7: 1})
            self.assertEqual(result['actual_contact_rows_by_kind'], {0: 2, 1: 1, 2: 1})
            self.assertEqual(result['actual_post_object_wall_contacts_by_wall']['L'], {'left': 1})
            coverage = result['actual_boundary_arc_pose_action_coverage'][0]
            self.assertEqual((coverage['letter'], coverage['arc'], coverage['post_contact_records']), ('L', 7, 1))
            self.assertEqual(coverage['actual_signed_normal_lever_worldpx']['median'], 4.)
            self.assertEqual(coverage['yaw_bins12'], [1])  # post35deg, not pre29deg bin0
            self.assertEqual(coverage['executed_target_minus_pre_substep_pusher_on_post_normal_worldpx']['median'], 136.)
            self.assertEqual(coverage['executed_target_minus_pre_substep_pusher_on_post_tangent_worldpx']['median'], 136.)
            self.assertEqual(result['physics_replay_controls'], 0); self.assertEqual(result['model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
