"""Stdlib entry and guardian composition fixtures; root owns execution.

No inherited BridgeTests: discovery must not repeat the older test family.
Only current engineering fixtures may be written; no real model/native call.
"""
from contextlib import ExitStack
from copy import deepcopy
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation_guard as guard
from tdyssey import gpc_checkpoint_evaluation as engine
from tdyssey import gpc_evaluation_assets as asset_api
from tdyssey import gpc_evaluation_entry as entry
from tdyssey import gpc_evaluation_status as journal

EVIDENCE = Path(__file__).resolve().parents[1]/'artifacts/gpc-checkpoint-evaluation-engineering-20261026/entry-guard-fixtures'
_spec = importlib.util.spec_from_file_location('_gpc_eval_bridge_fixture', Path(__file__).with_name('test_gpc_adaptation_guard.py'))
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)


class EvaluationEntryTests(unittest.TestCase):
    def test_eval_factory_without_verified_state_refuses_before_source_or_sdk(self):
        from tdyssey import gpc_supervised_adaptation as factory
        with patch.object(factory,'audit_upstream_objective',side_effect=AssertionError('Source/runtime access forbidden')) as audit:
            with self.assertRaisesRegex(ValueError,'model-only state'):
                factory.make_real_denoiser('absent-source','absent-weights','cuda:0',
                    dict(execute=True,evaluation_only=True))
        audit.assert_not_called()

    def fixture(self, *, first_error=False, partial_tail=False):
        # Composition retains all ownership/resource injection from the old suite.
        with patch.object(bridge, 'EVIDENCE', EVIDENCE):
            base = bridge.BridgeTests(methodName='test_owned_ack_success_and_durable_report')
            base.setUp()
        self.addCleanup(base.doCleanups)
        base.config = base.root/entry.CONFIG_RELATIVE
        base.script = base.root/entry.SCRIPT
        base.script.write_text('# marker worker; never executed\n')
        base.prepared['config'].update(candidate_id=entry.CANDIDATE, output_dir=entry.OUTPUT)
        base.binding.update(authorization_id=entry.AUTHORIZATION, candidate_id=entry.CANDIDATE,
                            mode='eval_only', source_checkpoint_update=500, max_updates=0, max_train_examples=0)
        base.sync_identity()
        base.prepared.update(data=dict(sha256='d'*64), benchmark=dict(sha256='c'*64), assets_sha256='e'*64)
        base.assets['evaluation_assets_sha256'] = 'e'*64
        stack = base.stack
        stack.enter_context(patch.object(entry, 'ROOT', base.root))
        stack.enter_context(patch.object(ctypes, 'WinDLL', side_effect=AssertionError('Explicit WinDLL forbidden'), create=True))
        stack.enter_context(patch.object(journal, '_kernel', side_effect=AssertionError('Native journal API forbidden')))
        stack.enter_context(patch.object(entry.subprocess, 'check_output', side_effect=AssertionError('Real process/git forbidden')))
        def open_file(path, *, create=False):
            return Path(path).open('xb' if create else 'rb', buffering=0)
        def snapshot(stream):
            value = os.fstat(stream.fileno())
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                raise journal.StatusJournalError('Fixture file must be ordinary and single-link')
            return (value.st_dev, value.st_ino), value.st_size
        stack.enter_context(patch.object(journal, '_open_file', side_effect=open_file))
        stack.enter_context(patch.object(journal, '_snapshot', side_effect=snapshot))
        output = base.root/entry.OUTPUT
        base.eval_preclaim = stack.enter_context(patch.object(entry, 'preclaim', return_value=(
            output, Path(sys.executable), base.root/'gpc', base.checkpoint, base.script, base.config, base.assets)))
        base.state['status_change'] = lambda value: value.update(training_complete=False, model_loads=2,
            optimizer_updates=0, evaluation_only=True, training_applicable=False, source_checkpoint_update=500)
        exclusive = guard._exclusive
        def publish(path, value):
            path = Path(path)
            if path.name != journal.STATUS_NAME:
                return exclusive(path, value)
            # Real Writer framing/fsync and Reader parsing; only OS open/stat is replaced.
            with patch.object(journal.os, 'getpid', return_value=303):
                writer = journal.Writer(path, pid=303, nonce=value['nonce'])
                try:
                    writer.publish(value)
                finally:
                    writer.close()
            if partial_tail:
                with path.open('ab',buffering=0) as stream:stream.write(b'GPCEVS0')
            if first_error:
                exclusive(path.parent/'heartbeat-error.json', dict(kind='gpc_heartbeat_failure.v1',
                    step='status.append.fsync', exception_type='OSError', message='first injected error',
                    actual_worker_pid=303, nonce=value['nonce']))
        stack.enter_context(patch.object(guard, '_exclusive', side_effect=publish))
        return base

    def run_eval_guard(self, base):
        return guard.supervise(base.prepared, base.binding, python_executable=sys.executable,
            gpc_root=base.root/'gpc', checkpoint=base.checkpoint, script_path=base.script,
            config_path=base.config, provider=base.provider, owner_factory=base.owner,
            launcher_factory=base.launcher, eval_only=True)

    def test_actual_fixed_identity_and_authority_reject_crossed_inputs(self):
        base = self.fixture()
        valid = deepcopy(base.prepared['config'])
        entry.identity(base.config, valid, base.binding)
        for field, value in (('candidate_id', 'other'), ('output_dir', 'artifacts/other')):
            wrong = dict(valid, **{field: value})
            with self.assertRaises(PermissionError):entry.identity(base.config, wrong, base.binding)
        with self.assertRaises(PermissionError):entry.identity(base.root/'configs/other.json', valid, base.binding)
        for authority in ('', 'old-pilot-authority'):
            with self.assertRaises(PermissionError):
                entry.verify_execution(base.prepared, authorization_id=authority,
                    reviewed_commit='a'*40, authorized_config_sha256=base.prepared['config_sha256'])
        with self.assertRaises(PermissionError):
            entry.verify_execution(base.prepared, authorization_id=entry.AUTHORIZATION,
                reviewed_commit='a'*40, authorized_config_sha256='0'*64)
        self.assertFalse((base.root/entry.OUTPUT).exists())
        with patch.object(entry.subprocess, 'check_output', side_effect=['a'*40+'\n', '']):
            binding = entry.verify_execution(base.prepared, authorization_id=entry.AUTHORIZATION,
                reviewed_commit='a'*40, authorized_config_sha256=base.prepared['config_sha256'])
        self.assertEqual((binding['mode'], binding['source_checkpoint_update'], binding['max_updates']), ('eval_only', 500, 0))
        with patch.object(entry.subprocess, 'check_output', side_effect=['a'*40+'\n', ' M fake.py']):
            with self.assertRaises(PermissionError):
                entry.verify_execution(base.prepared, authorization_id=entry.AUTHORIZATION,
                    reviewed_commit='a'*40, authorized_config_sha256=base.prepared['config_sha256'])

    def test_actual_plan_is_read_only_with_zero_model_and_native_work(self):
        base = self.fixture()
        cfg = deepcopy(base.prepared['config']); cfg['sources'] = engine.SOURCES
        base.config.write_text(json.dumps(cfg))
        baseline = dict(gaps=[], data=dict(sha256='d'*64), benchmark=dict(sha256='c'*64), training_schedule=dict(cursor=500))
        with patch.object(engine, 'validate_config', side_effect=deepcopy), \
                patch.object(entry.original, 'plan', return_value=baseline) as oldplan, \
                patch.object(asset_api, 'verify_evaluation_assets', return_value=dict(sources=engine.SOURCES)) as verify, \
                patch.object(asset_api, 'check_prepared_assets') as cheap, \
                patch.object(guard, 'supervise', side_effect=AssertionError('Plan must not launch')):
            result = entry.plan(base.config)
        self.assertTrue(result['ready']);self.assertEqual(result['mode'], 'read_only')
        self.assertEqual((result['real_operations'], result['native_resource_queries']), (0, 0))
        self.assertFalse(result['execution_authorized']);self.assertFalse((base.root/entry.OUTPUT).exists())
        oldplan.assert_called_once_with(base.root/'configs/gpc_supervised_adaptation_pilot03.json', claimed=True)
        verify.assert_called_once_with(repo_root=base.root);cheap.assert_called_once()
        base.native.assert_not_called();base.native_owner.assert_not_called();base.native_popen.assert_not_called()

    def test_eval_clocks_and_separate_five_second_heartbeat_cuda_freshness(self):
        binding = guard._clocks({}, 1000., 100., eval_only=True)
        self.assertIsNone(binding['training_deadline_unix']);self.assertIsNone(binding['training_deadline_monotonic'])
        self.assertFalse(binding['training_deadline_applicable'])
        value = guard.status_template(eval_only=True)
        value.update(phase='evaluation', actual_worker_pid=303, nonce='n', at_unix=2000., at_monotonic=1100.,
            cuda_sample_state='fresh', cuda_sample_at_unix=2000., cuda_sample_at_monotonic=1100.,
            cuda_allocated_bytes=0, cuda_reserved_bytes=0, cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0)
        def reason(v):return guard.monitor_reason(v, binding, pid=303, nonce='n', now_unix=2000., now_monotonic=1100., eval_only=True)
        self.assertIsNone(reason(value))  # More than900s elapsed; no fake training completion.
        self.assertEqual(reason(dict(value, at_unix=1994.999)), 'worker_status_stale_or_invalid')
        self.assertEqual(reason(dict(value, at_monotonic=1094.999)), 'worker_status_stale_or_invalid')
        self.assertEqual(reason(dict(value, cuda_sample_at_unix=1994.999)), 'cuda_sample_stale_or_invalid')
        self.assertEqual(reason(dict(value, cuda_sample_at_monotonic=1094.999)), 'cuda_sample_stale_or_invalid')
        self.assertIsNone(reason(dict(value, at_unix=1995., cuda_sample_at_unix=1995., cuda_sample_at_monotonic=1095.)))
        for changes in (dict(training_complete=True), dict(training_applicable=True), dict(source_checkpoint_update=499), dict(phase='training')):
            self.assertEqual(reason(dict(value, **changes)), 'evaluation_only_contract_invalid')
        self.assertEqual(reason(dict(value, optimizer_updates=1)), 'worker_counter_invalid:optimizer_updates')
        self.assertEqual(reason(dict(value, model_loads=3)), 'worker_counter_invalid:model_loads')

    def test_parent_completes_with_real_status_frames_two_loads_zero_updates(self):
        base = self.fixture();report = self.run_eval_guard(base)
        self.assertEqual(report['status'], 'complete', report)
        self.assertIsNone(report['guard_reason']);self.assertFalse(report['native_integration_verified'])
        self.assertEqual(report['job_active_processes_after_cleanup'], 0)
        self.assertFalse(report['single_model_lock_retained']);self.assertFalse(guard.DEFAULT_LOCK.exists())
        self.assertIsNone(report['binding']['training_deadline_unix'])
        self.assertIsNone(report['binding']['training_deadline_monotonic'])
        self.assertFalse(report['binding']['training_deadline_applicable'])
        status = report['last_status']
        self.assertEqual((status['model_loads'], status['optimizer_updates'], status['source_checkpoint_update']), (2, 0, 500))
        self.assertFalse(status['training_complete']);self.assertFalse(status['training_applicable'])
        self.assertEqual((status['forecasts'], status['euler_steps'], status['eval_batches']), (3996, 11988, 32))
        status_path = Path(report['guardian_dir'])/journal.STATUS_NAME
        self.assertEqual(status_path.read_bytes()[:8], journal._MAGIC)
        self.assertEqual(guard._read(Path(report['guardian_dir'])/'stop-record.json')['status'], 'complete')
        self.assertEqual(report['telemetry']['status_journal_reserve_bytes'], 16*guard.MIB)
        self.assertEqual(report['telemetry']['ledger_reserve_bytes'], 16*guard.MIB)
        base.eval_preclaim.assert_called_once();base.audit.assert_not_called()
        base.native.assert_not_called();base.native_owner.assert_not_called();base.native_popen.assert_not_called()

    def test_incomplete_tail_cannot_hide_behind_a_previous_complete_frame(self):
        base=self.fixture(partial_tail=True)
        report=self.run_eval_guard(base)
        self.assertEqual(report['status'],'stopped')
        self.assertEqual(report['guard_reason'],'worker_status_incomplete_tail')
        self.assertEqual(report['job_active_processes_after_cleanup'],0)
        self.assertFalse(report['single_model_lock_retained'])

    def test_first_error_slot_beats_valid_complete_frame_and_cleanup_failure(self):
        base = self.fixture(first_error=True);base.state['quiesce'] = 1
        report = self.run_eval_guard(base)
        wanted = 'worker_heartbeat_failure:status.append.fsync:OSError'
        self.assertEqual(report['status'], 'stopped');self.assertEqual(report['guard_reason'], wanted)
        self.assertTrue(report['cleanup_errors']);self.assertTrue(report['single_model_lock_retained'])
        self.assertEqual(report['job_active_processes_after_cleanup'], 1)
        directory = Path(report['guardian_dir'])
        self.assertEqual((directory/journal.STATUS_NAME).read_bytes()[:8], journal._MAGIC)
        first = guard._read(directory/'heartbeat-error.json')
        self.assertEqual(first['message'], 'first injected error')
        durable = guard._read(directory/'stop-record.json')
        self.assertEqual(durable['guard_reason'], wanted)
        self.assertTrue(durable['cleanup_errors']);self.assertTrue(durable['single_model_lock_retained'])
        self.assertTrue(guard.DEFAULT_LOCK.exists())
        base.native.assert_not_called();base.native_owner.assert_not_called();base.native_popen.assert_not_called()

    def test_telemetry_admission_reserves_journal_ledger_fixed_stdout_and_monitor(self):
        base = self.fixture();directory = base.root/'telemetry-only';directory.mkdir()
        telemetry = guard._Telemetry(directory, journal_reserve_bytes=16*guard.MIB)
        class LengthOnly:
            def __init__(self, size):self.size = size
            def __len__(self):return self.size
        try:
            self.assertEqual(16+16+1+2+29, 64)
            telemetry.admit('monitor', LengthOnly(29*guard.MIB))
            telemetry.admit('stdout', LengthOnly(2*guard.MIB))
            spent = telemetry.monitor_bytes+telemetry.stdout_bytes+telemetry.journal_reserve_bytes+guard.LEDGER_RESERVE+guard.FIXED_RESERVE
            self.assertEqual(spent, 64*guard.MIB)
            before = telemetry.monitor_bytes, telemetry.stdout_bytes
            for kind in ('monitor', 'stdout'):
                with self.assertRaises(OSError):telemetry.admit(kind, b'x')
                self.assertEqual((telemetry.monitor_bytes, telemetry.stdout_bytes), before)
        finally:
            telemetry.close();telemetry.emergency.close()


if __name__ == '__main__':
    unittest.main()
