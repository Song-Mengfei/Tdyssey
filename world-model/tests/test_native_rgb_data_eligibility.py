"""Admission/continuity checks on synthetic arrays; no physics/model calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tdyssey import native_rgb_data_eligibility as data


class EligibilityTest(unittest.TestCase):
    def setUp(self):
        self.repo = data.ROOT
        self.protocol, self.protocol_sha = data._protocol()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(data, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.upstream = self.repo.parent/'Tdyssey/third_party/diffusion_policy'
        self.sources = []
        for relative in ('tdyssey/native_rgb_original_physics.py',
                         'tdyssey/strict_original_contact_pilot.py',
                         'tdyssey/native_rgb_data_eligibility.py', 'manifests/official_dp.json'):
            target = self.root/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.repo/relative).read_bytes())
            self.sources.append(dict(path=str(target), sha256=data.sha256(target)))
        for relative, digest in self.protocol['source_sha256'].items():
            self.sources.append(dict(path=str(self.upstream/relative), sha256=digest))
        self.output = self.root/'output'
        self.output.mkdir()
        self.raw = self.output/'raw.npz'
        self.meta_path = self.output/'metadata.json'
        self.manifest = self.output/'manifest.json'
        self.valid = np.array([True]*5+[False])
        self.frames = np.arange(7, dtype=np.int64)
        self.segments = np.zeros(7, dtype=np.int64)
        self.times = np.array([0., .1, .2, .3, .4, .5, .52])
        np.savez_compressed(self.raw, rgb=np.zeros((7,3,96,96), np.uint8),
            requested_actions=np.zeros((6,2), np.float64), control_valid=self.valid,
            frame_indices=self.frames, segment_ids=self.segments, control_time_seconds=self.times)
        self.meta = dict(kind=data.ATTEMPT_KIND,
            slot=dict(slot_id='strict-original-T-0', letter='T', split='train_candidate', phase='formal'),
            protocol_sha256=self.protocol_sha, old_physics_data=False,
            physics_source_sha256=data.sha256(self.root/'tdyssey/native_rgb_original_physics.py'),
            collector_source_sha256=data.sha256(self.root/'tdyssey/strict_original_contact_pilot.py'),
            initial_identity=dict(protocol=self.protocol, protocol_sha256=self.protocol_sha,
                physics=dict(controller='original-upstream-kinematic-pd', kp=100,kv=20,
                    control_dt=.1,substep_dt=.01,substeps_per_control=10,
                    space=dict(iterations=10,damping=0),agent=dict(body_type=1),object=dict(body_type=0))),
            raw=dict(sha256=data.sha256(self.raw)), actual_valid_controls=5,
            valid_contiguous_control_ranges=[[0,5]], contiguous_frames=True,
            training_eligible=False,pending_user_approval=True)
        self.value = dict(kind=data.MANIFEST_KIND, root='.', control_dt=.1,
            protocol_sha256=self.protocol_sha, old_physics_data=False,
            upstream_source_root=str(self.upstream), source_bindings=self.sources,
            training_eligible=False,training_authorized=False,pending_user_approval=True,
            entries=[dict(file='raw.npz',metadata_file='metadata.json',sha256=data.sha256(self.raw),
                metadata_sha256='',trajectory_id='strict-original-T-0',letter='T',split='train_candidate',phase='formal')])
        self.freeze()

    def freeze(self):
        self.meta_path.write_text(json.dumps(self.meta), encoding='utf8')
        self.value['entries'][0]['metadata_sha256'] = data.sha256(self.meta_path)
        self.manifest.write_text(json.dumps(self.value), encoding='utf8')
        return data.sha256(self.manifest)

    def admitted(self):
        return data.verify_strict_original_dataset(self.manifest,
            expected_manifest_sha256=data.sha256(self.manifest),require_training_authorization=False)

    def guard(self, verified):
        return data.guard_trajectory(self.raw, self.meta, verified,frame_indices=self.frames,
            valid_controls=self.valid,control_time_seconds=self.times,segment_ids=self.segments)

    def test_candidate_audit_does_not_authorize_training(self):
        verified = self.admitted()
        self.guard(verified)
        with self.assertRaisesRegex(ValueError, 'human training approval'):
            data.StrictOriginalRGBDataset(self.manifest, expected_manifest_sha256=data.sha256(self.manifest))

    def test_relabelled_modified_physics_is_rejected(self):
        # A new filename and original protocol label do not conceal solver50.
        self.meta['initial_identity']['physics']['space']['iterations'] = 50
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'Actual construction'):
            self.guard(self.admitted())
        self.meta['initial_identity']['physics']['space']['iterations'] = 10
        self.meta['initial_identity']['physics']['agent']['body_type'] = 0
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'Actual construction'):
            self.guard(self.admitted())

    def test_externally_frozen_manifest_and_executed_sources_are_required(self):
        digest = data.sha256(self.manifest)
        self.value['control_dt'] = .4
        self.freeze()
        with self.assertRaisesRegex(ValueError, 'manifest SHA'):
            data.verify_strict_original_dataset(self.manifest,expected_manifest_sha256=digest)
        self.value['control_dt'] = .1
        self.freeze()
        verified = self.admitted()
        (self.root/'tdyssey/strict_original_contact_pilot.py').write_text('changed', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.guard(verified)

    def test_partial_terminal_duration_is_preserved_but_cannot_enter_window(self):
        self.guard(self.admitted())
        self.assertTrue(data.validate_window(self.frames,self.valid,0,5,
            control_time_seconds=self.times,segment_ids=self.segments))
        with self.assertRaisesRegex(ValueError, 'crosses invalid'):
            data.validate_window(self.frames,self.valid,2,6,
                control_time_seconds=self.times,segment_ids=self.segments)

    def test_actual_strict_writer_schema_and_partial_time_pass_candidate_guard(self):
        from tdyssey.strict_original_contact_pilot import save_attempt
        class FixtureReservation:
            def write_bytes(self, path, payload):
                Path(path).write_bytes(payload)
        telemetry = [dict(substeps=[dict(dt=.01,executed=True)]*n) for n in (10,10,10,10,10,2)]
        raw = save_attempt(self.raw, np.zeros((7,15)), np.zeros((6,2)), self.valid,
            np.zeros((7,3,96,96),np.uint8), telemetry, [], reservation=FixtureReservation())
        self.meta['raw'] = raw
        self.value['entries'][0]['sha256'] = raw['sha256']
        self.freeze()
        with np.load(self.raw, allow_pickle=False) as saved:
            self.frames=saved['frame_indices'].copy()
            self.times=saved['control_time_seconds'].copy()
            self.segments=saved['segment_ids'].copy()
            self.valid=saved['control_valid'].copy()
        self.assertAlmostEqual(self.times[-1], .52, places=12)
        self.guard(self.admitted())

    def test_gap_reset_and_time_skips_are_not_legal_windows(self):
        for key in ('frames','segments','times'):
            values = {name:getattr(self,name).copy() for name in ('frames','segments','times')}
            values[key][3:] += 1 if key!='times' else .1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError, 'timeline gap'):
                data.validate_window(values['frames'],self.valid,0,5,
                    control_time_seconds=values['times'],segment_ids=values['segments'])

    def test_frozen_membership_prevents_copied_or_changed_raw(self):
        verified=self.admitted()
        copied=self.output/'renamed.npz'
        copied.write_bytes(self.raw.read_bytes())
        with self.assertRaisesRegex(ValueError, 'frozen member'):
            data.guard_trajectory(copied,self.meta,verified)
        with self.raw.open('ab') as handle:handle.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'frozen member'):
            self.guard(verified)

    def test_in_place_quarantine_rejects_even_forged_original_manifest(self):
        with patch.object(data,'LEGACY_ROOT',self.output):
            with self.assertRaisesRegex(ValueError, 'excluded'):
                self.admitted()

    def test_loader_keeps_only_continuous_h1_and_exposes_only_rgb_action(self):
        # Synthetic authorization exercises interface only; it authorizes no run.
        self.value.update(training_eligible=True,training_authorized=True,
            pending_user_approval=False,human_approval_evidence='synthetic test fixture')
        digest=self.freeze()
        dataset=data.StrictOriginalRGBDataset(self.manifest,expected_manifest_sha256=digest)
        self.assertEqual(len(dataset),2)
        item=dataset[0]
        self.assertEqual(set(item),{'image','action'})
        self.assertEqual(item['image'].shape,(5,3,96,96))
        self.assertEqual(item['action'].shape,(4,2))
        self.assertEqual(item['image'].dtype,np.float32)


class RevocationTest(unittest.TestCase):
    def test_old_worker_and_direct_collector_cannot_restart(self):
        from tdyssey.original_contact_pilot import collect
        with self.assertRaisesRegex(RuntimeError,'authorization revoked'):
            collect(None,None,None,None,None,None)
        path=data.LEGACY_ROOT/'background/common.py'
        spec=importlib.util.spec_from_file_location('revoked_contact_background_common',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaisesRegex(RuntimeError,'authorization revoked'):
            module.validate_config(path,'unused',require_release=True)


if __name__=='__main__':
    unittest.main()
