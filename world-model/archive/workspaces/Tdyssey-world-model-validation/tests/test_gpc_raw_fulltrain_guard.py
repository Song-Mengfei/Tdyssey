"""Stdlib raw-mode guard/journal fixtures; no process, native, array or SDK call.

Root owns execution. File APIs with explicit Win32 handles are replaced by
ordinary fixture streams. Existing mode limits remain independently checked.
"""
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation_guard as guard
from tdyssey import gpc_evaluation_status as journal

EVIDENCE = Path(__file__).resolve().parents[1]/'artifacts/gpc-raw-tl-fulltrain-preparation-20261026/guard-fixtures'


def raw_status(**changes):
    value = guard.status_template(raw_fulltrain=True)
    value.update(actual_worker_pid=os.getpid(),nonce='raw-fixture',at_unix=1000.,at_monotonic=10.,
        phase='training',model_instances=1,model_loads=1,cuda_sample_state='fresh',
        cuda_allocated_bytes=0,cuda_reserved_bytes=0,cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0,
        cuda_sample_at_unix=1000.,cuda_sample_at_monotonic=10.)
    value.update(changes);return value


class RawFulltrainGuardTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='raw-',dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root = Path(self.temp.name)
        self.stack = ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(journal,'_kernel',side_effect=AssertionError('No native API')))
        self.stack.enter_context(patch.object(journal,'_open_file',
            side_effect=lambda path,create=False:Path(path).open('xb' if create else 'rb',buffering=0)))
        def snapshot(stream):
            value=os.fstat(stream.fileno())
            if not stat.S_ISREG(value.st_mode) or value.st_nlink!=1:raise ValueError('Fixture must be ordinary')
            return (value.st_dev,value.st_ino),value.st_size
        self.stack.enter_context(patch.object(journal,'_snapshot',side_effect=snapshot))

    def test_fixed_clocks_include_initial_load_and_old_modes_do_not_change(self):
        raw = guard._clocks({},1000.,10.,raw_fulltrain=True)
        self.assertEqual((raw['deadline_unix'],raw['training_deadline_unix']),(2800.,2200.))
        self.assertEqual((raw['deadline_monotonic'],raw['training_deadline_monotonic']),(1810.,1210.))
        self.assertTrue(raw['training_deadline_applicable'])
        self.assertEqual(guard._clocks({},1000.,10.)['deadline_unix'],4600.)
        self.assertEqual(guard._clocks({},1000.,10.,conditioning_only=True)['deadline_unix'],1300.)
        for mode in (dict(eval_only=True),dict(conditioning_only=True)):
            with self.assertRaises(ValueError):guard._clocks({},1000.,10.,raw_fulltrain=True,**mode)
        altered=dict(raw,deadline_monotonic=1811.)
        with self.assertRaises(ValueError):guard._clocks(altered,1001.,11.,raw_fulltrain=True)

    def test_raw_caps_training_completion_and_independent_freshness(self):
        bound=guard._clocks({},1000.,10.,raw_fulltrain=True)
        value=raw_status()
        def reason(v,wall=1000.,mono=10.):
            return guard.monitor_reason(v,bound,pid=os.getpid(),nonce='raw-fixture',
                now_unix=wall,now_monotonic=mono,raw_fulltrain=True)
        self.assertIsNone(reason(value));journal.validate(value,raw_fulltrain=True)
        for key,number in (('optimizer_updates',2329),('model_loads',5),('forecasts',49),
                           ('euler_steps',145),('eval_batches',49),('eval_examples',49)):
            changed=dict(value,**{key:number})
            self.assertEqual(reason(changed),'worker_counter_invalid:'+key)
            with self.assertRaises(journal.StatusJournalError):journal.validate(changed,raw_fulltrain=True)
        self.assertEqual(reason(dict(value,training_complete=True,optimizer_updates=500)),'premature_training_completion')
        self.assertEqual(reason(dict(value,source_checkpoint_update=500)),'raw_fulltrain_contract_invalid')
        self.assertEqual(reason(dict(value,at_monotonic=4.99)),'worker_status_stale_or_invalid')
        self.assertEqual(reason(dict(value,cuda_sample_at_unix=994.99)),'cuda_sample_stale_or_invalid')
        with self.assertRaises(journal.StatusJournalError):
            journal.validate(dict(value,cuda_sample_at_monotonic=4.99),raw_fulltrain=True)
        now=dict(value,at_unix=2200.,at_monotonic=1210.,cuda_sample_at_unix=2200.,cuda_sample_at_monotonic=1210.)
        self.assertEqual(reason(dict(now,phase='model_loading'),2200.,1210.),'training_phase_deadline_exceeded')
        done=dict(now,phase='evaluation',training_complete=True,optimizer_updates=2328)
        self.assertIsNone(reason(done,2200.,1210.))
        self.assertEqual(reason(done,2800.,1810.),'total_deadline_exceeded')

    def test_cleanup_waits_consume_original_deadline_and_keep_io_reserve(self):
        bound=guard._clocks({},1000.,10.,raw_fulltrain=True)
        with patch.object(guard.time,'time',return_value=2780.),patch.object(guard.time,'monotonic',return_value=1790.):
            self.assertEqual(guard._cleanup_wait_timeout(bound,raw_fulltrain=True),10.)
        with patch.object(guard.time,'time',return_value=2790.),patch.object(guard.time,'monotonic',return_value=1800.):
            self.assertEqual(guard._cleanup_wait_timeout(bound,raw_fulltrain=True),2.)
        with patch.object(guard.time,'time',return_value=2795.),patch.object(guard.time,'monotonic',return_value=1805.):
            self.assertEqual(guard._cleanup_wait_timeout(bound,raw_fulltrain=True),0.)
            self.assertEqual(guard._cleanup_wait_timeout(bound),10.)

    def test_raw_writer_reader_roundtrip_and_default_eval_schema_stays_strict(self):
        path=self.root/journal.STATUS_NAME;pid=os.getpid()
        with journal.Writer(path,pid=pid,nonce='raw-fixture',raw_fulltrain=True,
                clock=lambda:1000.,monotonic=lambda:10.) as writer:
            expected=writer.publish(raw_status(optimizer_updates=2328,training_complete=True))
        with journal.Reader(path,pid=pid,nonce='raw-fixture',raw_fulltrain=True) as reader:
            self.assertEqual(reader.poll(),expected);self.assertEqual(reader.sequence,1)
            self.assertIsNone(reader.poll())
        with journal.Reader(path,pid=pid,nonce='raw-fixture') as old:
            with self.assertRaises(journal.StatusJournalError):old.poll()
        other=self.root/'old'/journal.STATUS_NAME;other.parent.mkdir()
        with journal.Writer(other,pid=pid,nonce='raw-fixture') as old:
            with self.assertRaises(journal.StatusJournalError):old.publish(raw_status())

    def test_bad_intermediate_complete_frame_cannot_hide_behind_a_later_good_frame(self):
        path=self.root/journal.STATUS_NAME
        values=[raw_status(),raw_status(optimizer_updates=2329),raw_status(optimizer_updates=2)]
        path.write_bytes(b''.join(journal._frame(index,json.dumps(value,separators=(',',':')).encode())
                                 for index,value in enumerate(values,1)))
        with journal.Reader(path,pid=os.getpid(),nonce='raw-fixture',raw_fulltrain=True) as reader:
            with self.assertRaises(journal.StatusJournalError):reader.poll()
            self.assertIsNone(reader.latest)

    def test_record_capacity_covers_three_progress_records_per_update_and_heartbeat(self):
        self.assertLess(3*2328+2*48+2*48+2*4+1800+128,journal.MAX_RECORDS)
        self.assertLess(journal.MAX_RECORDS*journal.MAX_ENCODED_RECORD_BYTES,journal.MAX_BYTES)

    def test_injected_guard_routes_raw_handoff_journal_terminal_and_job_cleanup(self):
        from tdyssey import gpc_raw_fulltrain_entry as entry
        spec=importlib.util.spec_from_file_location('_raw_guard_bridge_fixture',
            Path(__file__).with_name('test_gpc_adaptation_guard.py'))
        bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
        with patch.object(bridge,'EVIDENCE',EVIDENCE/'bridge'):
            base=bridge.BridgeTests(methodName='test_owned_ack_success_and_durable_report');base.setUp()
        self.addCleanup(base.doCleanups)
        base.config=base.root/'configs/gpc_raw_tl_fulltrain.json'
        base.script=base.root/'scripts/gpc_raw_tl_fulltrain.py';base.script.write_text('# fake; never executed\n')
        base.prepared['config'].update(candidate_id='gpc-raw-tl-fulltrain-01',
            output_dir='artifacts/gpc-raw-tl-fulltrain-20261026')
        base.prepared['config']['limits']['wall_seconds']=1800
        base.prepared['config']['train']['phase_seconds']=1200
        base.binding.update(authorization_id='20261026-user-raw-tl-fulltrain-01',
                            candidate_id='gpc-raw-tl-fulltrain-01')
        base.sync_identity();output=base.root/base.prepared['config']['output_dir']
        base.stack.enter_context(patch.object(guard,'_read',side_effect=lambda p:json.loads(Path(p).read_bytes())))
        preclaim=base.stack.enter_context(patch.object(entry,'preclaim',return_value=(output,Path(sys.executable),
            base.root/'gpc',base.checkpoint,base.script,base.config,base.assets)))
        def change(value):
            value.update(raw_fulltrain=True,evaluation_only=False,training_applicable=True,training_complete=True,
                model_instances=0,model_loads=4,optimizer_updates=2328,forecasts=48,euler_steps=144,
                eval_batches=48,eval_examples=48)
            value.pop('source_checkpoint_update',None)
        base.state['status_change']=change
        exclusive=guard._exclusive
        def publish(path,value):
            path=Path(path)
            if path.name==journal.STATUS_NAME:
                with patch.object(journal.os,'getpid',return_value=303):
                    with journal.Writer(path,pid=303,nonce=value['nonce'],raw_fulltrain=True) as writer:
                        writer.publish(value)
            elif path.name=='completion.json':
                exclusive(path,dict(status='complete',forecast_records=48,optimizer_updates=2328,
                                    model_loads=4,eval_loss_batches=48))
            else:exclusive(path,value)
        base.stack.enter_context(patch.object(guard,'_exclusive',side_effect=publish))
        report=guard.supervise(base.prepared,base.binding,python_executable=sys.executable,
            gpc_root=base.root/'gpc',checkpoint=base.checkpoint,script_path=base.script,config_path=base.config,
            provider=base.provider,owner_factory=base.owner,launcher_factory=base.launcher,raw_fulltrain=True)
        self.assertEqual(report['status'],'complete',report)
        self.assertEqual(report['last_status']['optimizer_updates'],2328)
        self.assertEqual(report['binding']['deadline_unix']-report['binding']['started_unix'],1800)
        self.assertEqual(report['binding']['training_deadline_unix']-report['binding']['started_unix'],1200)
        self.assertEqual(base.state['handoff']['kind'],'gpc_raw_fulltrain_guard_handoff.v1')
        self.assertEqual(report['telemetry']['status_journal_reserve_bytes'],16*guard.MIB)
        self.assertEqual(report['cleanup_reserve_seconds'],20)
        self.assertEqual(report['job_active_processes_after_cleanup'],0)
        self.assertFalse(report['single_model_lock_retained']);self.assertFalse(report['native_integration_verified'])
        preclaim.assert_called_once();base.plan.assert_not_called()
        base.native.assert_not_called();base.native_owner.assert_not_called();base.native_popen.assert_not_called()


if __name__=='__main__':unittest.main()
