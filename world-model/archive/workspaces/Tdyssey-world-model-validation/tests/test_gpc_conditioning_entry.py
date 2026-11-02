"""Fixed-authority stdlib fixtures; no native, tensor, array or process call.

Root owns execution. All writes stay in new workspace temporary fixtures;
model provenance and training-window preparation are ordinary marker objects.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation_guard as guard
from tdyssey import gpc_conditioning_entry as entry

EVIDENCE = entry.ROOT/'artifacts/gpc-conditioning-contract-preparation-20261026/entry-fixtures'


class ConditioningEntryTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='entry-', dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.root_patch = patch.object(entry, 'ROOT', self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.cfg = entry.configuration_template()
        self.config = self.root/entry.CONFIG_RELATIVE
        self.config.parent.mkdir(); self.config.write_text(json.dumps(self.cfg), encoding='utf-8')
        self.script = self.root/entry.SCRIPT
        self.script.parent.mkdir(); self.script.write_text('# marker; never executed\n', encoding='utf-8')
        self.sha = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.prepared = dict(ready=True, config=deepcopy(self.cfg), config_path=str(self.config),
            config_sha256=self.sha, inputs_sha256='i'*64, assets_sha256='a'*64)

    def binding(self):
        return dict(candidate_id=entry.CANDIDATE, authorization_id=entry.AUTHORIZATION,
            output_dir=entry.OUTPUT, config_path=str(self.config), source_commit='b'*40,
            config_sha256=self.sha)

    def test_fixed_json_matches_template_and_scientific_knobs_cannot_change(self):
        entry.validate_config(self.cfg)
        actual = json.loads((Path(__file__).resolve().parents[1]/entry.CONFIG_RELATIVE).read_bytes())
        self.assertEqual(actual, self.cfg)
        self.assertEqual(self.cfg['limits']['forecasts'], 32)
        self.assertEqual(self.cfg['limits']['euler_steps'], 96)
        self.assertEqual(self.cfg['limits']['model_loads'], 2)
        self.assertEqual(self.cfg['forecast'], dict(input_windows=8, seed=5478, mode='AR', horizon=1))
        for key, value in (('wall_seconds', 301), ('forecasts', 33), ('eval_batches', 1),
                           ('max_updates', 1), ('output_bytes', 65*1024**2)):
            wrong = deepcopy(self.cfg); wrong['limits'][key] = value
            with self.assertRaises(ValueError):entry.validate_config(wrong)
        for key, value in (('history_noise', True), ('arms', {'A': {'sigma_cond': None}, 'B': {'sigma_cond': .003}})):
            wrong = deepcopy(self.cfg); wrong['conditioning'][key] = value
            with self.assertRaises(ValueError):entry.validate_config(wrong)

    def test_identity_rejects_old_or_crossed_authority_config_and_output(self):
        result = entry.identity(self.config, self.cfg, self.binding())
        self.assertTrue(result['output_dir'].endswith('/'+entry.AUTHORIZATION))
        for cfg in (dict(self.cfg, candidate_id='gpc-checkpoint500-evaluation-01'),
                    dict(self.cfg, output_dir=str(Path(entry.OUTPUT).parent))):
            with self.assertRaises(PermissionError):entry.identity(self.config, cfg, self.binding())
        with self.assertRaises(PermissionError):entry.identity(self.config.with_name('other.json'), self.cfg)
        with self.assertRaises(PermissionError):entry.identity(self.config, self.cfg, dict(authorization_id='old'))

    def test_plan_uses_only_metadata_helper_and_marked_provenance_without_claim(self):
        from tdyssey import gpc_conditioning_data as data
        from tdyssey import gpc_evaluation_assets as assets
        inputs = dict(windows=[dict(window_id='marker')], scope='metadata-only')
        verified = dict(sources=deepcopy(self.cfg['sources']))
        with patch.object(data, 'prepare_training_windows', return_value=inputs) as prepare, \
                patch.object(assets, 'verify_evaluation_assets', return_value=verified) as verify, \
                patch.object(assets, 'check_prepared_assets') as check, \
                patch.object(guard, 'ResourceProvider', side_effect=AssertionError('No native query')), \
                patch.object(guard, 'supervise', side_effect=AssertionError('No process')):
            result = entry.plan(self.config)
        self.assertTrue(result['ready']); self.assertEqual(result['inputs'], inputs)
        self.assertEqual(result['inputs_sha256'], hashlib.sha256(entry.original.encoded(inputs)).hexdigest())
        self.assertEqual((result['real_operations'], result['native_resource_queries']), (0, 0))
        self.assertFalse((self.root/entry.OUTPUT).exists())
        prepare.assert_called_once_with(self.root); verify.assert_called_once_with(repo_root=self.root)
        check.assert_called_once_with(verified)

    def test_execution_requires_exact_clean_commit_config_and_explicit_authority(self):
        with patch.object(entry.subprocess, 'check_output', side_effect=AssertionError('No actual git')) as git:
            with self.assertRaises(PermissionError):
                entry.verify_execution(self.prepared, authorization_id='old', reviewed_commit='b'*40,
                    authorized_config_sha256=self.sha)
            with self.assertRaises(PermissionError):
                entry.verify_execution(self.prepared, authorization_id=entry.AUTHORIZATION, reviewed_commit='b'*40,
                    authorized_config_sha256='0'*64)
        git.assert_not_called()
        with patch.object(entry.subprocess, 'check_output', side_effect=['b'*40+'\n', '']) as git:
            result = entry.verify_execution(self.prepared, authorization_id=entry.AUTHORIZATION,
                reviewed_commit='b'*40, authorized_config_sha256=self.sha)
        self.assertEqual((result['mode'], result['max_updates'], result['source_checkpoint_update']), ('conditioning_only', 0, 500))
        self.assertEqual(result['conditioning_inputs_sha256'], self.prepared['inputs_sha256'])
        self.assertTrue(all(call.kwargs['timeout'] == 5 for call in git.call_args_list))
        with patch.object(entry.subprocess, 'check_output', side_effect=['b'*40, ' M marker.py']):
            with self.assertRaises(PermissionError):
                entry.verify_execution(self.prepared, authorization_id=entry.AUTHORIZATION,
                    reviewed_commit='b'*40, authorized_config_sha256=self.sha)

    def test_spent_fixed_account_refuses_before_source_audit(self):
        output = self.root/entry.OUTPUT; output.mkdir(parents=True)
        checkpoint = self.root/self.cfg['sources']['frozen']['path']
        checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'not-weights')
        with patch.object(entry, 'plan', return_value=self.prepared), \
                patch.object(entry, 'verify_execution', return_value=self.binding()), \
                patch.object(guard, '_assets', side_effect=AssertionError('No weight/source audit')) as audit:
            with self.assertRaises(FileExistsError):
                entry.preclaim(self.prepared, self.binding(), sys.executable, self.root,
                               checkpoint, self.script, self.config)
        audit.assert_not_called()

    def test_conditioning_clocks_caps_and_freshness_tighten_without_changing_old_defaults(self):
        binding = guard._clocks({}, 1000., 10., conditioning_only=True)
        self.assertEqual(binding['deadline_unix'], 1300.)
        self.assertEqual(binding['deadline_monotonic'], 310.)
        self.assertIsNone(binding['training_deadline_unix'])
        old = guard._clocks({}, 1000., 10.)
        self.assertEqual((old['deadline_unix'], old['training_deadline_unix']), (4600., 1900.))
        with self.assertRaises(ValueError):guard._clocks(binding, 1001., 11., eval_only=True)
        value = guard.status_template(conditioning_only=True)
        value.update(phase='evaluation', actual_worker_pid=303, nonce='marker', at_unix=1000., at_monotonic=10.,
            cuda_sample_state='fresh', cuda_sample_at_unix=1000., cuda_sample_at_monotonic=10.,
            cuda_allocated_bytes=0, cuda_reserved_bytes=0, cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0)
        def reason(v, wall=1000., mono=10.):
            return guard.monitor_reason(v, binding, pid=303, nonce='marker', now_unix=wall,
                                        now_monotonic=mono, conditioning_only=True)
        self.assertIsNone(reason(value)); entry.validate_status_record(value)
        for field, invalid in (('forecasts', 33), ('euler_steps', 97), ('eval_batches', 1),
                               ('eval_examples', 1), ('optimizer_updates', 1), ('model_loads', 3)):
            changed = dict(value, **{field: invalid})
            self.assertEqual(reason(changed), 'worker_counter_invalid:'+field)
            with self.assertRaises(ValueError):entry.validate_status_record(changed)
        self.assertEqual(reason(dict(value, conditioning_only=False)), 'conditioning_only_contract_invalid')
        self.assertEqual(reason(dict(value, at_unix=994.99)), 'worker_status_stale_or_invalid')
        self.assertEqual(reason(value, wall=1300., mono=310.), 'total_deadline_exceeded')

    def test_worker_requires_exact_nonce_ack_script_and_original_three_hundred_seconds(self):
        directory = self.root/entry.OUTPUT/'guardian'; directory.mkdir(parents=True)
        handoff_path = directory/'handoff.json'; nonce = 'c'*32
        binding = guard._clocks(self.binding(), 1000., 10., conditioning_only=True)
        handoff = dict(kind='gpc_conditioning_guard_handoff.v1', prepared_ready=True,
            ownership='native_job_before_ack', actual_worker_pid=303, launcher_pid=202, guard_pid=101,
            nonce=nonce, job_memory_bytes=3*1024**3, kill_on_job_close=True,
            output_dir=str(directory.parent), status_path=str(directory/'evaluation-status.journal'),
            ack_path=str(directory/'worker-ack.json'), script_path=str(self.script),
            script_sha256=hashlib.sha256(self.script.read_bytes()).hexdigest(),
            config_path=str(self.config), binding=binding)
        (directory/'worker.json').write_text(json.dumps(dict(pid=303, parent_pid=202, nonce=nonce)))
        (directory/'worker-ack.json').write_text(json.dumps(dict(pid=303, nonce=nonce)))
        def save(value):handoff_path.write_text(json.dumps(value), encoding='utf-8')
        save(handoff)
        env = {guard.ENV_HANDOFF: str(handoff_path), guard.ENV_NONCE: nonce, guard.ENV_PARENT: '101'}
        with patch.dict(os.environ, env), patch.object(entry.os, 'getpid', return_value=303), \
                patch.object(entry.os, 'getppid', return_value=202), \
                patch.object(entry.time, 'time', return_value=1000.25), \
                patch.object(entry.time, 'monotonic', return_value=10.25), \
                patch.object(guard, '_read', side_effect=lambda p: json.loads(Path(p).read_bytes())), \
                patch.object(entry, 'plan', return_value=self.prepared) as prepare, \
                patch.object(entry, 'verify_execution', return_value=self.binding()), \
                patch.object(entry, '_verify_native_job', side_effect=AssertionError('No native call')) as native:
            result, prepared = entry.verify_worker(handoff_path, native_check=False)
            self.assertEqual(result, handoff); self.assertEqual(prepared, self.prepared)
            prepare.assert_called_once_with(str(self.config), claimed=True)
            # Base CPython executes runpy in the launcher itself; a venv
            # redirector may instead be the worker's immediate parent.
            save(dict(handoff, launcher_pid=303))
            base, _ = entry.verify_worker(handoff_path, native_check=False)
            self.assertEqual(base['launcher_pid'], base['actual_worker_pid'])
            for changes in (dict(nonce='d'*32), dict(script_sha256='0'*64),
                            dict(kind='gpc_evaluation_guard_handoff.v1'), dict(launcher_pid=999)):
                save(dict(handoff, **changes))
                with self.assertRaises(PermissionError):entry.verify_worker(handoff_path, native_check=False)
            save(handoff)
            wrong = deepcopy(handoff); wrong['binding']['deadline_monotonic'] += 1; save(wrong)
            with self.assertRaises(ValueError):entry.verify_worker(handoff_path, native_check=False)
        native.assert_not_called()


if __name__ == '__main__':unittest.main()
