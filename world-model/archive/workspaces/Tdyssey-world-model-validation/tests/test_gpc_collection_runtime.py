"""Small byte/fake-owned-process tests; no simulator/native/GPU/model work."""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from tdyssey import gpc_collection_runtime as runtime

EVIDENCE=Path(__file__).resolve().parents[1]/'artifacts/wm-sampling-b-only-20261026/runtime'


class BudgetTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(prefix='fixture-',dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.patch=patch.object(runtime,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.binding=dict(authorization_id='synthetic-only',source_commit='a'*40);self.wall,self.mono=1000.,100.
    def budget(self,stage):
        budget=runtime.CollectionBudget.create(stage,self.binding,clock=lambda:self.wall,monotonic=lambda:self.mono)
        self.addCleanup(budget.close);return budget

    def test_a_cross_run_episode_caps_and_journal_hash_chain(self):
        budget=self.budget('A')
        for name in ('T-A','T-B','L-A','L-B'):
            budget.begin_run(name)
            for episode in range(8):
                budget.begin_episode(episode)
                operation=budget.reserve(controls=100,control_substeps=1000,resets=20,reset_substeps=20,restores=1,renders=121)
                budget.complete(operation,dict(synthetic=True));budget.finish_episode()
            budget.write_bytes(name+'/manifest.json',b'{"synthetic":true}\n');budget.finish_run()
        budget.write_bytes('completion.json',b'{"status":"complete"}\n');state=budget.finish()
        self.assertEqual(state['counts']['controls'],3200);self.assertEqual(state['counts']['resets'],640)
        self.assertEqual(state['physics_substeps'],32640);self.assertEqual(state['counts'],state['completed'])
        lines=budget._journal.read_bytes().splitlines(keepends=True);tip='0'*64
        for index,line in enumerate(lines,1):
            row=json.loads(line);self.assertEqual(row['sequence'],index);self.assertEqual(row['previous_sha256'],tip);tip=hashlib.sha256(line).hexdigest()
        self.assertEqual(tip,state['journal_tip']);self.assertEqual(len(lines),state['sequence'])
        with self.assertRaises(FileExistsError):runtime.CollectionBudget.create('A',dict(self.binding,source_commit='b'*40))

    def test_failed_operation_stays_reserved_no_second_worker_or_refund(self):
        budget=self.budget('B');budget.begin_run('B');operation=budget.reserve(controls=1,control_substeps=10,renders=1)
        row=json.loads(budget._journal.read_bytes().splitlines()[-1]);self.assertEqual(row['kind'],'reserve')
        self.assertEqual(row['data']['operation_id'],operation);self.assertEqual(row['data']['costs']['controls'],1)
        budget.fail('synthetic environment failure');state=budget.snapshot()
        self.assertEqual(state['counts']['controls'],1);self.assertEqual(state['completed']['controls'],0);self.assertIsNotNone(state['pending'])
        with self.assertRaises(runtime.CollectionError):budget.complete(operation)
        with self.assertRaises(FileExistsError):runtime.CollectionBudget.create('B',self.binding,claim=budget.claim)
        self.assertEqual(json.loads(budget._failure.read_bytes())['reason'],'synthetic environment failure')

    def test_b_exact_limits_render_overflow_stops_before_operation(self):
        budget=self.budget('B');budget.begin_run('B')
        operation=budget.reserve(controls=256,control_substeps=2560,resets=256,reset_substeps=256,restores=64,renders=512)
        budget.complete(operation);self.assertEqual(budget.snapshot()['physics_substeps'],2816);before=budget.counts.copy()
        with self.assertRaisesRegex(runtime.CollectionError,'cap'):budget.reserve(renders=1)
        self.assertEqual(budget.counts,before)

    def test_b2_fixed_distinct_account_same_caps_preserves_spent_a_b(self):
        originals=[]
        for stage in ('A','B'):
            old=self.budget(stage);old.begin_run('T-A' if stage=='A' else 'B')
            if stage=='A':old.begin_episode(0)
            old.reserve(controls=1,control_substeps=10,renders=1);old.fail('Old spent '+stage);old.close()
            originals.extend([Path(old.claim['path']),*old.output.rglob('*')])
        before={str(path):path.read_bytes() for path in originals if path.is_file()}
        budget=self.budget('B2');output,registry=runtime.stage_paths('B2')
        self.assertEqual(output.name,'gpc-data-stage-b2-20261026');self.assertEqual(registry.name,'B2.json')
        self.assertEqual({k:v for k,v in runtime.STAGES['B2'].items() if k!='output'},
            {k:v for k,v in runtime.STAGES['B'].items() if k!='output'})
        budget.begin_run('B');operation=budget.reserve(controls=256,control_substeps=2560,resets=256,reset_substeps=256,restores=64,renders=512)
        budget.complete(operation);self.assertEqual(budget.snapshot()['physics_substeps'],2816)
        self.assertEqual(budget.claim['deadline_unix']-budget.claim['started_unix'],600)
        self.assertEqual(budget.limits['bytes'],64*runtime.MIB)
        with self.assertRaises(ValueError):runtime.stage_paths('B3')
        with self.assertRaises(FileExistsError):runtime.CollectionBudget.create('B2',dict(self.binding,source_commit='b'*40))
        with self.assertRaises(FileExistsError):runtime.CollectionBudget.create('B2',self.binding,claim=budget.claim)
        self.assertEqual({str(path):path.read_bytes() for path in originals if path.is_file()},before)

    def test_episode_cap_and_expiry_during_atomic_publication(self):
        budget=self.budget('A');budget.begin_run('T-A');budget.begin_episode(0)
        operation=budget.reserve(controls=100,control_substeps=1000);budget.complete(operation)
        with self.assertRaisesRegex(runtime.CollectionError,'100 controls'):budget.reserve(controls=1,control_substeps=10)
        self.assertEqual(budget.counts['controls'],100)
        other=self.budget('B');other.begin_run('B');original=runtime._write_all
        def delay(stream,payload):
            result=original(stream,payload)
            if str(stream.name).endswith('.tmp'):self.mono+=601;self.wall+=601
            return result
        with patch.object(runtime,'_write_all',side_effect=delay):
            with self.assertRaisesRegex(runtime.CollectionError,'deadline'):other.write_bytes('late.json',b'{}')
        self.assertFalse((other.output/'late.json').exists());self.assertTrue(list(other.output.glob('late.json.*.tmp')))

    def test_per_run_and_aggregate_deadline_cannot_restart(self):
        budget=self.budget('A');budget.begin_run('T-A');self.mono+=120;self.wall+=120
        with self.assertRaisesRegex(runtime.CollectionError,'Per-run'):budget.check()
        with self.assertRaises(FileExistsError):runtime.CollectionBudget.create('A',self.binding)
        other=self.budget('B');self.mono+=600;self.wall+=600
        with self.assertRaisesRegex(runtime.CollectionError,'Aggregate'):other.check()

    def test_atomic_overlap_charged_and_failed_temp_retained(self):
        budget=self.budget('B');budget.begin_run('B');first=budget.write_bytes('manifest.json',b'{"v":1}')
        before=budget.charged_bytes;budget.write_bytes('manifest.json',b'{"v":200}',replace=True)
        self.assertGreaterEqual(budget.charged_bytes-before,len(b'{"v":200}'))
        self.assertEqual((budget.output/'manifest.json').read_bytes(),b'{"v":200}')
        self.assertEqual(first['sha256'],hashlib.sha256(b'{"v":1}').hexdigest());before=budget.charged_bytes
        with patch.object(runtime,'_publish_json',side_effect=OSError('synthetic publication failure')):
            with self.assertRaisesRegex(OSError,'publication'):budget.write_bytes('payload.npz',b'not-real-numpy')
        self.assertGreater(budget.charged_bytes,before);self.assertTrue(list(budget.output.glob('payload.npz.*.tmp')))
        self.assertFalse((budget.output/'payload.npz').exists());self.assertIn('publication',budget.first_reason)

    def test_small_byte_cap_refuses_before_temp_file(self):
        stages=deepcopy(runtime.STAGES);stages['B']['bytes']=runtime.GUARD_BYTES+3*runtime.SLOT+1000
        with patch.object(runtime,'STAGES',stages):
            budget=self.budget('B');budget.begin_run('B')
            with self.assertRaisesRegex(runtime.CollectionError,'byte cap'):budget.write_bytes('too-large.npz',b'x'*1000)
            self.assertFalse(list(budget.output.glob('too-large*')))

    def test_held_manifest_stat_never_opens_or_rehashes_payload(self):
        budget=self.budget('B');budget.begin_run('B');record=budget.write_bytes('held.npz',b'synthetic-held-only')
        manifest=dict(episodes=[dict(files=[record])])
        with patch.object(Path,'open',side_effect=AssertionError('Held payload read forbidden')):rows=runtime.manifest_stats(budget.output,manifest)
        self.assertEqual(rows[0]['sha256'],record['sha256']);self.assertFalse(rows[0]['payload_read'])
        manifest['episodes'][0]['files'][0]=dict(record,bytes=record['bytes']+1)
        with self.assertRaisesRegex(runtime.CollectionError,'metadata'):runtime.manifest_stats(budget.output,manifest)


class GuardianTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(prefix='guardian-',dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);(self.root/'scripts').mkdir()
        self.script=self.root/'scripts/worker.py';self.script.write_text('# never executed\n')
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(runtime,'ROOT',self.root));self.stack.enter_context(patch.object(runtime,'_verify_authority'))
        self.native=self.stack.enter_context(patch.object(runtime,'NoModelResourceProvider',side_effect=AssertionError('Native reads forbidden')))
        self.native_process=self.stack.enter_context(patch.object(runtime,'WindowsWorker',side_effect=AssertionError('Native process forbidden')))
        self.native_launch=self.stack.enter_context(patch.object(runtime.subprocess,'Popen',side_effect=AssertionError('Process execution forbidden')))
        self.binding=dict(authorization_id='synthetic-only',source_commit='a'*40);self.state=dict(exited=False,runtime=0,launches=0,quiesce=0)
        test=self
        class Provider:
            def startup(self,output):return test.resources()
            def runtime(self,pid,output):
                test.state['runtime']+=1
                if test.state['runtime']==2:
                    handoff=runtime._read(output/'guardian/handoff.json');test.state['handoff']=handoff
                    test.assertTrue((output/'guardian/worker-ack.json').exists());test.assertTrue(test.state['owned'])
                    budget=runtime.CollectionBudget.create('B',test.binding,claim=handoff['claim'])
                    budget.begin_run('B');operation=budget.reserve(controls=1,control_substeps=10,renders=1);budget.complete(operation)
                    budget.write_bytes('completion.json',b'{"status":"complete"}');budget.finish_run();budget.finish()
                    value=runtime.status(budget,'complete');value.update(actual_worker_pid=303,nonce=handoff['nonce'],at_unix=time.time())
                    if 'status_change' in test.state:test.state['status_change'](value)
                    runtime._exclusive(Path(handoff['status_path']),runtime._json(value));test.state['exited']=True
                return test.resources()
            def close(self):test.state['provider_closed']=True
        self.provider=Provider()

    def resources(self):
        return dict(available_ram_bytes=8*runtime.GIB,commit_headroom_bytes=12*runtime.GIB,disk_free_bytes=20*runtime.GIB,
            working_set_bytes=64*runtime.MIB,private_bytes=128*runtime.MIB,process_peak_working_set_bytes=64*runtime.MIB)

    def launcher(self,command,**kwargs):
        self.state['launches']+=1;self.state['env']=kwargs['env'];self.state['command']=command
        path=Path(command[command.index('--handshake')+1]);nonce=command[command.index('--nonce')+1]
        runtime._exclusive(path,runtime._json(dict(pid=303,parent_pid=202,nonce=nonce)))
        test=self
        class Child:
            pid=202;stdout=io.BytesIO(b'fake-no-model-worker\n')
            def poll(self):return 0 if test.state['exited'] else None
            def wait(self,timeout):return 0
            def terminate(self):test.state['exited']=True
        return Child()

    def owner(self,pid,launcher_pid):
        self.assertEqual((pid,launcher_pid),(303,202));self.state['owned']=True;test=self
        self.assertFalse((self.root/'artifacts'/runtime.STAGES['B']['output']/'guardian/worker-ack.json').exists())
        class Owner:
            def poll_exit(self):return test.state['exited']
            def exit_code(self):return 0
            def terminate_and_wait(self,timeout):test.state['exited']=True
            def quiesce(self,timeout):return test.state['quiesce']
            def close(self):test.state['owner_closed']=True
        return Owner()

    def run_guard(self):
        return runtime.supervise('B',self.binding,python_executable=sys.executable,script_path=self.script,
            worker_options=dict(synthetic=True),provider=self.provider,owner_factory=self.owner,launcher_factory=self.launcher)

    def test_fake_launcher_owns_job_before_ack_then_publishes_complete(self):
        report=self.run_guard();self.assertEqual(report['status'],'complete',report)
        self.assertEqual(report['actual_worker_pid'],303);self.assertEqual(report['job_active_processes_after_cleanup'],0)
        self.assertFalse(report['lock_retained']);self.assertFalse(report['native_integration_verified']);self.assertEqual(report['gpu_queries'],0)
        self.assertEqual(runtime._read(Path(report['output_dir'])/'guardian/stop-record.json')['status'],'complete')
        self.native.assert_not_called();self.native_process.assert_not_called();self.native_launch.assert_not_called()

    def test_startup_missing_read_spends_stage_and_does_not_launch(self):
        self.provider.startup=lambda output:dict(self.resources(),available_ram_bytes=None)
        report=self.run_guard();self.assertEqual(report['guard_reason'],'missing_or_invalid_resource:available_ram_bytes')
        self.assertEqual(self.state['launches'],0)
        with self.assertRaises(FileExistsError):self.run_guard()
        self.assertEqual(self.state['launches'],0)

    def test_b2_cannot_launch_while_old_stage_owns_native_slot(self):
        claim=runtime.claim_stage('B',self.binding);lock=Path(claim['path']).parent/'active-native.lock'
        runtime._exclusive(lock,runtime._json(dict(stage='B',nonce=claim['nonce'],pid=202)))
        original_claim=Path(claim['path']).read_bytes();original_lock=lock.read_bytes()
        report=runtime.supervise('B2',self.binding,python_executable=sys.executable,script_path=self.script,
            provider=self.provider,owner_factory=self.owner,launcher_factory=self.launcher)
        self.assertEqual(report['status'],'stopped');self.assertEqual(report['guard_reason'],'collection_guard_failure')
        self.assertEqual(self.state['launches'],0);self.assertEqual(self.state['runtime'],0)
        self.assertEqual(Path(claim['path']).read_bytes(),original_claim);self.assertEqual(lock.read_bytes(),original_lock)
        self.assertTrue(runtime.stage_paths('B2')[1].exists())
        with self.assertRaises(FileExistsError):runtime.claim_stage('B2',dict(self.binding,authorization_id='replacement forbidden'))

    def test_first_resource_error_kept_nonempty_job_retains_owner(self):
        self.provider.runtime=lambda pid,output:dict(self.resources(),private_bytes=4*runtime.GIB);self.state['quiesce']=1
        report=self.run_guard();self.assertEqual(report['guard_reason'],'process_memory_limit:private_bytes')
        self.assertTrue(report['lock_retained']);self.assertTrue(report['cleanup_errors'])

    def test_live_run_deadline_and_model_operation_fail_closed(self):
        self.state['status_change']=lambda value:value.update(active_run='B',run_deadline_unix=time.time()-1,run_deadline_monotonic=time.monotonic()-1)
        report=self.run_guard();self.assertEqual(report['guard_reason'],'per_run_deadline_exceeded')
        value=dict(report['last_status'],model_loads=1)
        self.assertEqual(runtime._status_reason(value,report['claim'],303),'forbidden_model_policy_operation:model_loads')

    def test_final_stop_close_crossing_deadline_cannot_return_complete(self):
        wall,mono=time.time(),time.monotonic();offset=[0];original=Path.open
        class SlowClose:
            def __init__(self,stream):self.stream=stream
            def __getattr__(self,key):return getattr(self.stream,key)
            def close(self):self.stream.close();offset[0]=601
        def opening(path,mode='r',*args,**kwargs):
            stream=original(path,mode,*args,**kwargs)
            return SlowClose(stream) if path.name=='stop-record.json' and mode=='xb+' else stream
        with patch.object(Path,'open',opening),patch.object(runtime.time,'time',side_effect=lambda:wall+offset[0]),patch.object(runtime.time,'monotonic',side_effect=lambda:mono+offset[0]):
            report=self.run_guard()
        self.assertEqual(report['guard_reason'],'aggregate_deadline_exceeded');self.assertGreaterEqual(report['wall_seconds'],600)
        saved=runtime._read(Path(report['output_dir'])/'guardian/stop-record.json')
        self.assertEqual(saved['status'],'stopped');self.assertEqual(saved['guard_reason'],'aggregate_deadline_exceeded')

    def test_worker_handoff_fixture_verifies_without_native_checks(self):
        report=self.run_guard();path=Path(report['output_dir'])/'guardian/handoff.json';handoff=self.state['handoff']
        env=dict(GPC_COLLECTION_HANDOFF=str(path),GPC_COLLECTION_NONCE=handoff['nonce'],GPC_COLLECTION_GUARD_PID=str(handoff['guard_pid']))
        with patch.dict(os.environ,env),patch.object(runtime.os,'getpid',return_value=303),patch.object(runtime.os,'getppid',return_value=202):
            self.assertEqual(runtime.verify_worker(path,native_check=False)['claim'],report['claim'])
            with patch.dict(os.environ,dict(GPC_COLLECTION_NONCE='wrong')):
                with self.assertRaises(PermissionError):runtime.verify_worker(path,native_check=False)


class ResourceTests(unittest.TestCase):
    def test_provider_uses_existing_cpu_disk_readers_without_any_gpu_query(self):
        from tdyssey import policy_memory
        memory=dict(ok=True,physical=dict(available_bytes=8*runtime.GIB),commit=dict(headroom_bytes=12*runtime.GIB))
        process=dict(ok=True,working_set_bytes=runtime.MIB,private_bytes=2*runtime.MIB,peak_working_set_bytes=3*runtime.MIB)
        with patch.object(policy_memory,'system_memory',return_value=memory),patch.object(policy_memory,'disk_snapshot',return_value=dict(ok=True,free_bytes=8*runtime.GIB)),patch.object(policy_memory,'process_memory',return_value=process),patch.object(policy_memory,'gpu_memory',side_effect=AssertionError('GPU read forbidden')) as gpu:
            provider=runtime.NoModelResourceProvider();startup=provider.startup(Path('.'));live=provider.runtime(123,Path('.'))
            self.assertIsNone(runtime.resource_reason(startup,startup=True));self.assertIsNone(runtime.resource_reason(live));gpu.assert_not_called()
        self.assertEqual(runtime.resource_reason(dict(live,process_peak_working_set_bytes=3*runtime.GIB)),'process_memory_limit:process_peak_working_set_bytes')


if __name__=='__main__':unittest.main()
