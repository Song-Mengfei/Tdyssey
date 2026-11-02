"""Synthetic storage/IO tests; no simulator/model/data collection."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from tdyssey.original_contact_pilot_resources import (
    ResourceReservation, ResourceLimitError, load_resource_policy,
    SHARED_CAP, ATOMIC_RESERVE, DEFAULT_ATTEMPT, POSTPROCESS_RESERVE,
)

REPO = Path(__file__).resolve().parents[1]
TEMP_PARENT = (REPO/'artifacts/original-physics-contact-pilot-20261030/collection/revisions/resource-write-guard-v4/test-tmp').resolve()
if not TEMP_PARENT.is_relative_to(REPO/'artifacts'):
    raise RuntimeError('Synthetic recursive temporary cleanup must stay in project artifacts')
TEMP_PARENT.mkdir(parents=True, exist_ok=True)


def tempdir():
    return tempfile.TemporaryDirectory(dir=TEMP_PARENT)


def policy_file(directory):
    value = dict(collection_cap_bytes=1536*1024**2, shared_cap_bytes=SHARED_CAP,
                 atomic_reserve_bytes=ATOMIC_RESERVE,
                 per_attempt_atomic_reserve_bytes=DEFAULT_ATTEMPT,
                 maximum_concurrent_collectors=2,
                 concurrent_attempt_reserve_bytes=2*DEFAULT_ATTEMPT,
                 postprocess_reserve_bytes=POSTPROCESS_RESERVE,
                 training_authorized=False)
    path = Path(directory)/'policy.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


class ResourceCoordinationTest(unittest.TestCase):
    def test_joint_headroom_cannot_be_expanded_by_both_peers(self):
        with tempdir() as temporary:
            root = Path(temporary)/'output'
            root.mkdir()
            policy = policy_file(temporary)
            base = SHARED_CAP-ATOMIC_RESERVE-POSTPROCESS_RESERVE-2*DEFAULT_ATTEMPT-5*1024**2
            def synthetic(roots):
                return dict(bytes=0 if len(roots) == 1 else base, files=0, missing_roots=[])
            options = dict(_snapshot_fn=synthetic, _disk_free_fn=lambda: SHARED_CAP)
            with ResourceReservation(root, [Path(temporary)/'other'], policy, 'A', **options) as first:
                with ResourceReservation(root, [Path(temporary)/'other'], policy, 'B', **options) as second:
                    first.ensure_capacity(19*1024**2)  # plus1MiB metadata =20MiB
                    with self.assertRaises(ResourceLimitError):
                        second.ensure_capacity(19*1024**2)
                    self.assertEqual(second.remaining_bytes, DEFAULT_ATTEMPT)
                    with self.assertRaises(ResourceLimitError):
                        with ResourceReservation(root, [Path(temporary)/'other'], policy, 'third', **options):
                            pass
            ledger = json.loads((root/'resource-reservations.json').read_text())
            self.assertEqual([row['status'] for row in ledger['leases']], ['released', 'released'])

    def test_checkpoint_uses_existing_lease_without_another_full_scan(self):
        with tempdir() as temporary:
            root = Path(temporary)/'output'
            root.mkdir()
            calls = []
            def synthetic(roots):
                calls.append(len(roots))
                return dict(bytes=0, files=0, missing_roots=[])
            with ResourceReservation(root, [Path(temporary)/'other'], policy_file(temporary), 'A',
                                     _snapshot_fn=synthetic, _disk_free_fn=lambda: SHARED_CAP) as lease:
                baseline = len(calls)
                lease.write_bytes(root/'raw.npz', b'checkpoint')
                self.assertEqual(len(calls), baseline)
                lease.snapshot()
                self.assertEqual(len(calls), baseline+2)

    def test_report_guard_precedes_file_mutation(self):
        with tempdir() as temporary:
            root = Path(temporary)/'output'
            free = [SHARED_CAP]
            with ResourceReservation(root, [], policy_file(temporary), 'report', role='postprocess',
                                     _disk_free_fn=lambda: free[0]) as lease:
                first = root/'report'/'first.txt'
                lease.write_bytes(first, b'first')
                free[0] = ATOMIC_RESERVE
                with self.assertRaises(ResourceLimitError):
                    lease.write_bytes(root/'report'/'blocked.txt', b'must not write')
                self.assertFalse((root/'report'/'blocked.txt').exists())
                self.assertFalse(list(root.rglob('*.resource-tmp-*')))

    def test_report_budget_includes_preexisting_outputs_and_replacements(self):
        with tempdir() as temporary:
            root = Path(temporary)/'output'
            (root/'report').mkdir(parents=True)
            old = root/'report'/'old.txt'
            old.write_bytes(b'a'*20)
            with ResourceReservation(root, [], policy_file(temporary), 'report', role='postprocess') as lease:
                self.assertEqual(lease.remaining_bytes, POSTPROCESS_RESERVE-20)
                lease.write_bytes(root/'report'/'new.txt', b'new')
                self.assertEqual(lease.remaining_bytes, POSTPROCESS_RESERVE-23)
                lease.write_bytes(old, b'ab')
                self.assertEqual(lease.remaining_bytes, POSTPROCESS_RESERVE-5)

    def test_fixed_human_caps_cannot_be_silently_raised(self):
        with tempdir() as temporary:
            policy = policy_file(temporary)
            value = json.loads(policy.read_text())
            value['shared_cap_bytes'] += 1
            policy.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_resource_policy(policy)

    def test_two_actual_processes_share_one_reservation_ledger(self):
        with tempdir() as temporary:
            root = Path(temporary)/'output'
            root.mkdir()
            policy = policy_file(temporary)
            worker = (
                'import sys\nfrom pathlib import Path\n'
                'from tdyssey.original_contact_pilot_resources import ResourceReservation\n'
                'with ResourceReservation(sys.argv[1],[],sys.argv[2],sys.argv[3]):\n'
                ' Path(sys.argv[4]).write_text("ready")\n'
                ' sys.stdin.readline()\n')
            processes = []
            try:
                for owner in ('A', 'B'):
                    ready = Path(temporary)/(owner+'.ready')
                    process = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', '-c', worker,
                                                str(root), str(policy), owner, str(ready)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    processes.append(process)
                    deadline = time.monotonic()+30
                    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(ready.exists(), 'Small synthetic writer did not acquire its lease')
                ledger = json.loads((root/'resource-reservations.json').read_text())
                active = [row for row in ledger['leases'] if row['status'] == 'active']
                self.assertEqual(len(active), 2)
                self.assertEqual(len({row['pid'] for row in active}), 2)
                with self.assertRaises(ResourceLimitError):
                    with ResourceReservation(root, [], policy, 'third'):
                        pass
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.communicate(input=b'\n', timeout=30)
                    else:
                        process.communicate(timeout=30)
                for process in processes:
                    self.assertEqual(process.returncode, 0)
            ledger = json.loads((root/'resource-reservations.json').read_text())
            self.assertFalse(any(row['status'] == 'active' for row in ledger['leases']))

    def test_report_helper_failure_cannot_create_output(self):
        from tdyssey.original_contact_pilot_report import _write_output
        class RejectingLease:
            def write_bytes(self, path, payload):
                raise ResourceLimitError('synthetic output refusal')
        with tempdir() as temporary:
            output = Path(temporary)/'blocked.png'
            with self.assertRaises(ResourceLimitError):
                _write_output(output, b'png bytes', RejectingLease())
            self.assertFalse(output.exists())

    def test_video_budget_failure_occurs_before_encoder_start(self):
        import numpy as np
        import imageio_ffmpeg
        from tdyssey.original_contact_pilot_report import _video
        class EmptyLease:
            remaining_bytes = 0
        with tempdir() as temporary:
            with mock.patch('tdyssey.original_contact_pilot_report.subprocess.Popen') as encoder:
                with self.assertRaises(RuntimeError):
                    _video(Path(temporary)/'video.mp4', {}, {}, [], EmptyLease())
                encoder.assert_not_called()

    def test_default_unlabelled_resource_policy_remains_available(self):
        policy, digest = load_resource_policy(None)
        self.assertIsNone(digest)
        self.assertEqual(policy['collection_cap_bytes'], 512*1024**2)
        with tempdir() as temporary:
            with ResourceReservation(Path(temporary)/'output', [], None, 'default') as lease:
                lease.write_bytes(Path(temporary)/'output'/'small.json', b'{}')

    def test_guarded_real_encoder_and_exact_csv_outputs(self):
        import numpy as np
        from tdyssey.original_contact_pilot_report import _video, csv_write
        with tempdir() as temporary:
            root = Path(temporary)/'output'
            with ResourceReservation(root, [], policy_file(temporary), 'report', role='postprocess') as lease:
                csv_write(root/'report'/'small.csv', [dict(label='synthetic,fixture', value=1.25)], lease)
                self.assertIn('"synthetic,fixture"', (root/'report'/'small.csv').read_text())
                states = np.zeros((3, 15), np.float64)
                states[:, 2:4] = [256., 256.]
                images = np.full((3, 3, 96, 96), 255, np.uint8)
                images[:, :, 42:54, 42:54] = 100
                data = dict(states=states, rgb=images)
                metadata = dict(slot=dict(letter='O', slot_id='synthetic-resource-video'))
                controls = [dict(seconds=.1, translation_image_px=0., rotation_deg=0.,
                                 category='noncontact_static', planned_phase='static', valid=True)]*2
                result = _video(root/'report'/'videos'/'synthetic.mp4', metadata, data, controls, lease)
                self.assertEqual(result['frames'], 3)
                self.assertEqual(result['fps'], 10.)
                self.assertLess(result['bytes'], 8*1024**2)
                self.assertLess(lease.remaining_bytes, POSTPROCESS_RESERVE)


if __name__ == '__main__':
    unittest.main()
