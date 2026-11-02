"""Injected, stdlib-only adaptation bridge tests: no native query or process."""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation_guard as guard

EVIDENCE=Path(__file__).resolve().parents[1]/'artifacts/gpc-supervised-pilot03-review-20261026/guard-fixtures'
GIB=1024**3


class BridgeTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='fixture-',dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'scripts').mkdir();(self.root/'artifacts').mkdir();(self.root/'gpc').mkdir()
        self.script=self.root/'scripts/worker.py';self.script.write_text('# fake; never executed\n')
        self.checkpoint=self.root/'initial.pt';self.checkpoint.write_bytes(b'not-a-model')
        (self.root/'configs').mkdir();self.config=self.root/'configs/gpc_supervised_adaptation.json'
        cfg=dict(candidate_id='gpc-tl-supervised-one-step-01',output_dir='artifacts/'+guard.OUTPUT,
            limits=dict(wall_seconds=3600,telemetry_bytes=64*1024**2,active_gpu_models=1),train=dict(phase_seconds=900))
        self.config.write_text(json.dumps(cfg));config_sha=hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.prepared=dict(ready=True,config=cfg,config_sha256=config_sha,config_path=str(self.config))
        self.binding=dict(authorization_id='20261026-user-autonomous-gpc-supervised-pilot-01',
            candidate_id=cfg['candidate_id'],output_dir=cfg['output_dir'],config_path=str(self.config),source_commit='a'*40,config_sha256=config_sha)
        self.assets=dict(source_audit_sha256='b'*64,checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),checkpoint_bytes=self.checkpoint.stat().st_size)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(guard,'ROOT',self.root))
        self.stack.enter_context(patch.object(guard.protocol,'ROOT',self.root))
        self.stack.enter_context(patch.object(guard,'DEFAULT_LOCK',self.root/'artifacts/model.lock'))
        self.plan=self.stack.enter_context(patch.object(guard.protocol,'plan',return_value=deepcopy(self.prepared)))
        self.verify=self.stack.enter_context(patch.object(guard.protocol,'verify_execution',return_value=deepcopy(self.binding)))
        self.audit=self.stack.enter_context(patch.object(guard,'_assets',return_value=self.assets))
        self.native=self.stack.enter_context(patch.object(guard,'ResourceProvider',side_effect=AssertionError('Native query forbidden')))
        self.native_owner=self.stack.enter_context(patch.object(guard,'WindowsWorker',side_effect=AssertionError('Native process forbidden')))
        self.native_popen=self.stack.enter_context(patch.object(guard.subprocess,'Popen',side_effect=AssertionError('Process launch forbidden')))
        self.state=dict(exited=False,closed=False,owned=False,launches=0,runtime=0,quiesce=0)
        self.provider=self.make_provider()

    def resources(self):
        return dict(available_ram_bytes=8*GIB,commit_headroom_bytes=12*GIB,gpu_free_bytes=6*GIB,disk_free_bytes=20*GIB,
            gpu_age_seconds=0,working_set_bytes=64*1024**2,private_bytes=128*1024**2,process_peak_working_set_bytes=64*1024**2)

    def make_provider(self):
        test=self
        class Provider:
            def startup(self,output):return test.resources()
            def start(self):test.state['started']=True
            def runtime(self,pid,output):
                test.state['runtime']+=1
                if test.state['runtime']==2:
                    handoff=guard._read(output/'guardian/handoff.json');test.state['handoff']=handoff
                    test.assertTrue(test.state['owned']);test.assertTrue(Path(handoff['ack_path']).is_file())
                    value=guard.status_template();value.update(phase='complete',training_complete=True,model_loads=3,optimizer_updates=500,
                        forecasts=3996,euler_steps=11988,eval_batches=32,eval_examples=32,actual_worker_pid=303,nonce=handoff['nonce'],at_unix=time.time(),
                        cuda_allocated_bytes=0,cuda_reserved_bytes=0,cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0,
                        cuda_sample_state='fresh',cuda_sample_at_unix=time.time(),cuda_sample_at_monotonic=time.monotonic())
                    if 'status_change' in test.state:test.state['status_change'](value)
                    guard._exclusive(Path(handoff['status_path']),value)
                    guard._exclusive(output/'completion.json',dict(status='complete',forecast_records=3996))
                    test.state['exited']=True
                return test.resources()
            def close(self):test.state['provider_closed']=True
        return Provider()

    def launcher(self,command,**kwargs):
        self.state['launches']+=1;self.state['command']=command;self.state['env']=kwargs['env']
        nonce=command[command.index('--nonce')+1];path=Path(command[command.index('--handshake')+1])
        guard._exclusive(path,dict(pid=303,parent_pid=202,nonce=self.state.get('handshake_nonce',nonce)))
        test=self
        class Child:
            pid=202
            stdout=io.BytesIO(b'fake-worker-only\n')
            def poll(self):return 0 if test.state['exited'] else None
            def wait(self,timeout):return 0
            def terminate(self):test.state['exited']=True
        return Child()

    def owner(self,pid,launcher_pid):
        self.assertEqual((pid,launcher_pid),(303,202));self.state['owned']=True;test=self
        self.assertFalse((self.root/self.prepared['config']['output_dir']/'guardian/worker-ack.json').exists())
        class Owner:
            def poll_exit(self):return test.state['exited']
            def exit_code(self):return 0
            def terminate_and_wait(self,timeout):test.state['exited']=True
            def quiesce(self,timeout):return test.state['quiesce']
            def close(self):test.state['closed']=True
        return Owner()

    def run_guard(self,**kwargs):
        return guard.supervise(self.prepared,self.binding,python_executable=sys.executable,gpc_root=self.root/'gpc',
            checkpoint=self.checkpoint,script_path=self.script,config_path=self.config,
            provider=kwargs.get('provider',self.provider),owner_factory=self.owner,launcher_factory=self.launcher)

    def use_pilot02(self):
        self.config=self.root/'configs/gpc_supervised_adaptation_pilot02.json'
        self.prepared['config'].update(candidate_id='gpc-tl-supervised-one-step-02',
            output_dir='artifacts/gpc-supervised-adaptation-pilot02-20261026')
        self.binding.update(authorization_id='20261026-user-autonomous-gpc-supervised-pilot-02',
            candidate_id=self.prepared['config']['candidate_id'])
        self.sync_identity()

    def use_pilot03(self):
        self.config=self.root/'configs/gpc_supervised_adaptation_pilot03.json'
        self.prepared['config'].update(candidate_id='gpc-tl-supervised-one-step-03',
            output_dir='artifacts/gpc-supervised-adaptation-pilot03-20261026')
        self.binding.update(authorization_id='20261026-user-autonomous-gpc-supervised-pilot-03',
            candidate_id=self.prepared['config']['candidate_id'])
        self.sync_identity()

    def sync_identity(self):
        self.config.write_text(json.dumps(self.prepared['config']))
        self.prepared['config_sha256']=hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.prepared['config_path']=str(self.config)
        self.binding['config_sha256']=self.prepared['config_sha256']
        self.binding['config_path']=str(self.config);self.binding['output_dir']=self.prepared['config']['output_dir']
        self.plan.return_value=deepcopy(self.prepared);self.verify.return_value=deepcopy(self.binding)

    def test_owned_ack_success_and_durable_report(self):
        before={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')}
        report=self.run_guard()
        self.assertEqual(report['status'],'complete',report);self.assertIsNone(report['guard_reason'])
        self.assertEqual(report['actual_worker_pid'],303);self.assertEqual(report['job_active_processes_after_cleanup'],0)
        self.assertFalse(report['single_model_lock_retained']);self.assertFalse(guard.DEFAULT_LOCK.exists())
        self.assertFalse(report['native_integration_verified']);self.assertTrue(all(report['injected'].values()))
        self.assertEqual(report['binding']['deadline_unix']-report['binding']['started_unix'],3600)
        self.assertEqual(report['binding']['training_deadline_monotonic']-report['binding']['started_monotonic'],900)
        self.assertEqual(guard._read(Path(report['guardian_dir'])/'stop-record.json')['status'],'complete')
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):self.assertEqual(self.state['env'][key],'2')
        self.assertEqual(before,{k:os.environ.get(k) for k in before})
        self.native.assert_not_called();self.native_owner.assert_not_called();self.native_popen.assert_not_called()

    def test_readiness_and_source_failure_before_claim_and_query(self):
        self.prepared['ready']=False
        with self.assertRaisesRegex(ValueError,'not ready'):self.run_guard()
        self.assertFalse((self.root/'artifacts'/guard.OUTPUT).exists());self.assertEqual(self.state['launches'],0)
        self.prepared['ready']=True;self.audit.side_effect=ValueError('source audit fails')
        with self.assertRaisesRegex(ValueError,'source audit'):self.run_guard()
        self.assertFalse((self.root/'artifacts'/guard.OUTPUT).exists());self.native.assert_not_called()

    def test_startup_missing_read_spends_claim_without_launch(self):
        self.provider.startup=lambda output:dict(self.resources(),gpu_free_bytes=None)
        report=self.run_guard();self.assertEqual(report['guard_reason'],'missing_or_invalid_resource:gpu_free_bytes')
        self.assertEqual(self.state['launches'],0);self.assertFalse(report['single_model_lock_retained'])
        with self.assertRaises(FileExistsError):self.run_guard()
        self.assertEqual(self.state['launches'],0)

    def test_wrong_nonce_never_ack_and_retains_uncertain_lock(self):
        self.state['handshake_nonce']='wrong';report=self.run_guard()
        self.assertEqual(report['status'],'stopped');self.assertIn('nonce',report['error'])
        self.assertFalse((Path(report['guardian_dir'])/'worker-ack.json').exists())
        self.assertTrue(report['single_model_lock_retained']);self.assertTrue(guard.DEFAULT_LOCK.exists())

    def test_first_resource_failure_survives_cleanup_failure(self):
        original=self.provider.runtime
        self.provider.runtime=lambda pid,output:dict(original(pid,output),disk_free_bytes=GIB)
        self.state['quiesce']=1;report=self.run_guard()
        self.assertEqual(report['guard_reason'],'resource_below_floor:disk_free_bytes')
        self.assertTrue(report['single_model_lock_retained']);self.assertTrue(report['cleanup_errors'])

    def test_forbidden_and_allocator_peak_fail_closed(self):
        self.state['status_change']=lambda value:value.update(real_renders=1,cuda_peak_allocated_bytes=513*1024**2)
        report=self.run_guard();self.assertEqual(report['guard_reason'],'forbidden_worker_operation:real_renders')

    def test_worker_handoff_verifies_exact_paths_hash_nonce_and_original_deadline(self):
        report=self.run_guard();handoff=self.state['handoff'];path=Path(report['guardian_dir'])/'handoff.json'
        env={guard.ENV_HANDOFF:str(path),guard.ENV_NONCE:handoff['nonce'],guard.ENV_PARENT:str(handoff['guard_pid'])}
        with patch.dict(os.environ,env),patch.object(guard.os,'getpid',return_value=303),patch.object(guard.os,'getppid',return_value=202):
            verified=guard.verify_worker(path,native_check=False);self.assertEqual(verified['binding'],report['binding'])
            with patch.dict(os.environ,{guard.ENV_NONCE:'other'}):
                with self.assertRaisesRegex(RuntimeError,'handoff'):guard.verify_worker(path,native_check=False)
            self.checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'changed'):guard.verify_worker(path,native_check=False)

    def test_initial_checkpoint_is_rechecked_after_model_cleanup(self):
        self.state['status_change']=lambda value:self.checkpoint.write_bytes(b'changed initial fixture')
        report=self.run_guard()
        self.assertEqual(report['guard_reason'],'initial_checkpoint_recheck_failed')
        self.assertIn('initialization checkpoint changed',report['error'])
        self.assertTrue(self.state['closed']);self.assertEqual(report['job_active_processes_after_cleanup'],0)
        self.assertFalse(report['single_model_lock_retained'])

    def test_final_close_time_is_in_original_deadline_and_durable_stop(self):
        wall,mono=time.time(),time.monotonic();offset=[0]
        original=guard._Telemetry.__init__
        class SlowClose:
            def __init__(self,stream):self.stream=stream
            def __getattr__(self,key):return getattr(self.stream,key)
            def close(self):self.stream.close();offset[0]=3601
        def initialization(telemetry,directory):
            original(telemetry,directory);telemetry.emergency=SlowClose(telemetry.emergency)
        with patch.object(guard.time,'time',side_effect=lambda:wall+offset[0]),patch.object(guard.time,'monotonic',side_effect=lambda:mono+offset[0]),patch.object(guard._Telemetry,'__init__',initialization):
            report=self.run_guard()
        self.assertEqual(report['guard_reason'],'total_deadline_exceeded')
        self.assertEqual(report['wall_seconds'],3601);self.assertFalse(report['native_integration_verified'])
        saved=guard._read(Path(report['guardian_dir'])/'stop-record.json')
        self.assertEqual(saved['guard_reason'],'total_deadline_exceeded');self.assertEqual(saved['status'],'stopped')

    def test_uninjected_failure_cannot_claim_native_integration_verified(self):
        self.provider.startup=lambda output:dict(self.resources(),gpu_free_bytes=None)
        self.native.side_effect=lambda:self.provider
        report=guard.supervise(self.prepared,self.binding,python_executable=sys.executable,gpc_root=self.root/'gpc',
            checkpoint=self.checkpoint,script_path=self.script,config_path=self.config)
        self.assertFalse(any(report['injected'].values()));self.assertFalse(report['native_integration_verified'])
        self.assertEqual(report['status'],'stopped');self.native_owner.assert_not_called();self.native_popen.assert_not_called()


class Pilot02IdentityTests(unittest.TestCase):
    def fixture(self):
        fixture=BridgeTests(methodName='test_owned_ack_success_and_durable_report');fixture.setUp()
        self.addCleanup(fixture.doCleanups);fixture.use_pilot02();return fixture

    def test_new_fixed_output_does_not_touch_or_reopen_old_spent_attempt(self):
        fixture=self.fixture();old=fixture.root/'artifacts'/guard.OUTPUT;old.mkdir()
        old_claim=old/'attempt.json';old_claim.write_bytes(b'{"spent":true,"pilot":1}\n');before=old_claim.read_bytes()
        old_config=fixture.root/'configs/gpc_supervised_adaptation.json';old_cfg_before=old_config.read_bytes()
        report=fixture.run_guard()
        self.assertEqual(report['status'],'complete',report)
        self.assertEqual(report['output_dir'],str(fixture.root/'artifacts/gpc-supervised-adaptation-pilot02-20261026'))
        self.assertEqual(report['binding']['candidate_id'],'gpc-tl-supervised-one-step-02')
        self.assertEqual(report['binding']['authorization_id'],'20261026-user-autonomous-gpc-supervised-pilot-02')
        self.assertEqual(report['binding']['deadline_unix']-report['binding']['started_unix'],3600)
        self.assertEqual(report['binding']['training_deadline_unix']-report['binding']['started_unix'],900)
        self.assertEqual(old_claim.read_bytes(),before);self.assertEqual(old_config.read_bytes(),old_cfg_before)
        with self.assertRaises(FileExistsError):fixture.run_guard()
        fixture.native.assert_not_called();fixture.native_owner.assert_not_called();fixture.native_popen.assert_not_called()

    def test_crossed_candidate_authority_config_and_output_refused_before_claim(self):
        fixture=self.fixture();baseline_cfg=deepcopy(fixture.prepared['config']);baseline_binding=deepcopy(fixture.binding)
        canonical=fixture.config
        for case in ('authority','candidate','output','config_path','unknown_candidate'):
            with self.subTest(field=case):
                fixture.prepared['config']=deepcopy(baseline_cfg);fixture.binding=deepcopy(baseline_binding);fixture.config=canonical
                if case=='authority':fixture.binding['authorization_id']='20261026-user-autonomous-gpc-supervised-pilot-01'
                elif case=='candidate':fixture.binding['candidate_id']='gpc-tl-supervised-one-step-01'
                elif case=='output':fixture.prepared['config']['output_dir']='artifacts/'+guard.OUTPUT
                elif case=='config_path':fixture.config=fixture.root/'configs/gpc_supervised_adaptation.json'
                else:fixture.prepared['config']['candidate_id']='gpc-tl-supervised-one-step-04'
                fixture.sync_identity()
                with self.assertRaises((ValueError,PermissionError)):fixture.run_guard()
                self.assertFalse((fixture.root/'artifacts/gpc-supervised-adaptation-pilot02-20261026').exists())
                self.assertFalse((fixture.root/'artifacts'/guard.OUTPUT).exists());self.assertEqual(fixture.state['launches'],0)

    def test_pilot02_worker_rechecks_auth_config_hash_output_and_nonce(self):
        fixture=self.fixture();report=fixture.run_guard();handoff=fixture.state['handoff'];path=Path(report['guardian_dir'])/'handoff.json'
        original=deepcopy(handoff);env={guard.ENV_HANDOFF:str(path),guard.ENV_NONCE:handoff['nonce'],guard.ENV_PARENT:str(handoff['guard_pid'])}
        with patch.dict(os.environ,env),patch.object(guard.os,'getpid',return_value=303),patch.object(guard.os,'getppid',return_value=202):
            self.assertEqual(guard.verify_worker(path,native_check=False)['binding']['candidate_id'],'gpc-tl-supervised-one-step-02')
            for case in ('authority','config_path','config_hash','output','nonce'):
                with self.subTest(field=case):
                    changed=deepcopy(original)
                    if case=='authority':changed['binding']['authorization_id']='20261026-user-autonomous-gpc-supervised-pilot-01'
                    elif case=='config_path':
                        old_config=fixture.root/'configs/gpc_supervised_adaptation.json';old_config.write_bytes(fixture.config.read_bytes());changed['config_path']=str(old_config)
                    elif case=='config_hash':changed['binding']['config_sha256']='f'*64
                    elif case=='output':changed['output_dir']=str(fixture.root/'artifacts'/guard.OUTPUT)
                    else:changed['nonce']='different'
                    path.write_text(json.dumps(changed))
                    with self.assertRaises((ValueError,PermissionError,RuntimeError)):guard.verify_worker(path,native_check=False)
            path.write_text(json.dumps(original))


class Pilot03IdentityTests(unittest.TestCase):
    def fixture(self):
        fixture=BridgeTests(methodName='test_owned_ack_success_and_durable_report');fixture.setUp()
        self.addCleanup(fixture.doCleanups);fixture.use_pilot03();return fixture

    def test_new_fixed_output_preserves_both_spent_pilots_and_all_attempts_are_once(self):
        fixture=self.fixture();old=[]
        for number,output,config in (
                ('01',guard.OUTPUT,'gpc_supervised_adaptation.json'),
                ('02','gpc-supervised-adaptation-pilot02-20261026','gpc_supervised_adaptation_pilot02.json')):
            directory=fixture.root/'artifacts'/output;directory.mkdir()
            claim=directory/'attempt.json';claim.write_text(json.dumps(dict(spent=True,pilot=number)))
            path=fixture.root/'configs'/config
            cfg=deepcopy(fixture.prepared['config']);cfg.update(candidate_id='gpc-tl-supervised-one-step-'+number,output_dir='artifacts/'+output)
            path.write_text(json.dumps(cfg));old.append((number,claim,claim.read_bytes(),path,path.read_bytes()))
        report=fixture.run_guard()
        self.assertEqual(report['status'],'complete',report)
        self.assertEqual(report['output_dir'],str(fixture.root/'artifacts/gpc-supervised-adaptation-pilot03-20261026'))
        self.assertEqual(report['binding']['candidate_id'],'gpc-tl-supervised-one-step-03')
        self.assertEqual(report['binding']['authorization_id'],'20261026-user-autonomous-gpc-supervised-pilot-03')
        self.assertEqual(report['binding']['config_path'],str(fixture.config))
        self.assertEqual(report['binding']['config_sha256'],hashlib.sha256(fixture.config.read_bytes()).hexdigest())
        self.assertEqual(report['binding']['deadline_unix']-report['binding']['started_unix'],3600)
        self.assertEqual(report['binding']['training_deadline_unix']-report['binding']['started_unix'],900)
        with self.assertRaises(FileExistsError):fixture.run_guard()
        for number,claim,claim_bytes,path,config_bytes in old:
            fixture.config=path;fixture.prepared['config']=json.loads(config_bytes)
            fixture.binding.update(authorization_id='20261026-user-autonomous-gpc-supervised-pilot-'+number,
                candidate_id=fixture.prepared['config']['candidate_id'])
            fixture.sync_identity()
            with self.subTest(spent_pilot=number),self.assertRaises(FileExistsError):fixture.run_guard()
            self.assertEqual(claim.read_bytes(),claim_bytes);self.assertEqual(path.read_bytes(),config_bytes)
        self.assertEqual(fixture.state['launches'],1)
        fixture.native.assert_not_called();fixture.native_owner.assert_not_called();fixture.native_popen.assert_not_called()

    def test_old_identity_components_cannot_cross_into_pilot03_before_claim(self):
        fixture=self.fixture();baseline_cfg=deepcopy(fixture.prepared['config']);baseline_binding=deepcopy(fixture.binding)
        canonical=fixture.config
        for number,output,config in (
                ('01',guard.OUTPUT,'gpc_supervised_adaptation.json'),
                ('02','gpc-supervised-adaptation-pilot02-20261026','gpc_supervised_adaptation_pilot02.json')):
            for case in ('authority','candidate','output','config_path'):
                with self.subTest(pilot=number,field=case):
                    fixture.prepared['config']=deepcopy(baseline_cfg);fixture.binding=deepcopy(baseline_binding);fixture.config=canonical
                    if case=='authority':fixture.binding['authorization_id']='20261026-user-autonomous-gpc-supervised-pilot-'+number
                    elif case=='candidate':fixture.binding['candidate_id']='gpc-tl-supervised-one-step-'+number
                    elif case=='output':fixture.prepared['config']['output_dir']='artifacts/'+output
                    else:fixture.config=fixture.root/'configs'/config
                    fixture.sync_identity()
                    with self.assertRaises((ValueError,PermissionError)):fixture.run_guard()
                    for name in (guard.OUTPUT,'gpc-supervised-adaptation-pilot02-20261026','gpc-supervised-adaptation-pilot03-20261026'):
                        self.assertFalse((fixture.root/'artifacts'/name).exists())
                    self.assertEqual(fixture.state['launches'],0)
        fixture.native.assert_not_called();fixture.native_owner.assert_not_called();fixture.native_popen.assert_not_called()

    def test_pilot03_worker_rechecks_old_identity_components_sha_and_nonce(self):
        fixture=self.fixture();report=fixture.run_guard();handoff=fixture.state['handoff'];path=Path(report['guardian_dir'])/'handoff.json'
        original=deepcopy(handoff);env={guard.ENV_HANDOFF:str(path),guard.ENV_NONCE:handoff['nonce'],guard.ENV_PARENT:str(handoff['guard_pid'])}
        cases=('authority01','authority02','candidate01','candidate02','config_path01','config_path02',
            'config_hash','checkpoint_hash','script_hash','output01','output02','nonce')
        with patch.dict(os.environ,env),patch.object(guard.os,'getpid',return_value=303),patch.object(guard.os,'getppid',return_value=202):
            self.assertEqual(guard.verify_worker(path,native_check=False)['binding']['candidate_id'],'gpc-tl-supervised-one-step-03')
            for case in cases:
                with self.subTest(field=case):
                    changed=deepcopy(original);number=case[-2:]
                    if case.startswith('authority'):changed['binding']['authorization_id']='20261026-user-autonomous-gpc-supervised-pilot-'+number
                    elif case.startswith('candidate'):changed['binding']['candidate_id']='gpc-tl-supervised-one-step-'+number
                    elif case.startswith('config_path'):
                        name='gpc_supervised_adaptation.json' if number=='01' else 'gpc_supervised_adaptation_pilot02.json'
                        old_config=fixture.root/'configs'/name;old_config.write_bytes(fixture.config.read_bytes());changed['config_path']=str(old_config)
                    elif case=='config_hash':changed['binding']['config_sha256']='f'*64
                    elif case=='checkpoint_hash':changed['checkpoint_sha256']='f'*64
                    elif case=='script_hash':changed['script_sha256']='f'*64
                    elif case.startswith('output'):
                        output=guard.OUTPUT if number=='01' else 'gpc-supervised-adaptation-pilot02-20261026'
                        changed['output_dir']=str(fixture.root/'artifacts'/output)
                    else:changed['nonce']='different'
                    path.write_text(json.dumps(changed))
                    with self.assertRaises((ValueError,PermissionError,RuntimeError)):guard.verify_worker(path,native_check=False)
            path.write_text(json.dumps(original))


class PrimitiveTests(unittest.TestCase):
    def test_slow_monitor_io_does_not_block_stdout_admission_or_write(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='parallel-io-',dir=EVIDENCE) as name:
            root=Path(name);telemetry=guard._Telemetry(root);blocked=threading.Event();release=threading.Event();errors=[]
            original=guard._write_all
            def slow_monitor(stream,payload):
                if stream is telemetry.stream:
                    blocked.set()
                    if not release.wait(2):raise TimeoutError('Test did not release monitor IO')
                return original(stream,payload)
            def event():
                try:telemetry.event('blocked',{'fixture':True})
                except BaseException as error:errors.append(error)
            with patch.object(guard,'_write_all',side_effect=slow_monitor):
                thread=threading.Thread(target=event);thread.start()
                try:
                    self.assertTrue(blocked.wait(1))
                    pump=guard._Pump(io.BytesIO(b'parallel stdout'),root/'stdout',telemetry)
                    pump.thread.join(timeout=.5)
                    self.assertFalse(pump.thread.is_alive(),'Slow monitor held the admission lock during IO')
                    self.assertEqual((root/'stdout').read_bytes(),b'parallel stdout')
                    pump.close()
                finally:release.set();thread.join(timeout=2)
            self.assertFalse(errors);self.assertEqual(telemetry.stdout_bytes,len(b'parallel stdout'))
            telemetry.close();telemetry.emergency.close()

    def test_shared_deadlines_and_missing_counter_and_peak(self):
        binding=guard._clocks({},100,10);status=guard.status_template()
        status.update(actual_worker_pid=3,nonce='n',at_unix=1000)
        self.assertEqual(guard.monitor_reason(status,binding,pid=3,nonce='n',now_unix=1000,now_monotonic=910),'training_phase_deadline_exceeded')
        status.update(training_complete=True,optimizer_updates=500,phase='evaluation',at_unix=3700)
        self.assertEqual(guard.monitor_reason(status,binding,pid=3,nonce='n',now_unix=3700,now_monotonic=3610),'total_deadline_exceeded')
        status.update(at_unix=101,cuda_allocated_bytes=0,cuda_reserved_bytes=0,cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=1024**3+1,
            cuda_sample_state='fresh',cuda_sample_at_unix=100,cuda_sample_at_monotonic=10)
        self.assertIn('cuda_peak_reserved',guard.monitor_reason(status,binding,pid=3,nonce='n',now_unix=101,now_monotonic=11))
        status['cuda_peak_reserved_bytes']=0;del status['controls']
        self.assertEqual(guard.monitor_reason(status,binding,pid=3,nonce='n',now_unix=101,now_monotonic=11),'forbidden_worker_operation:controls')
        with self.assertRaisesRegex(ValueError,'extended'):guard._clocks(dict(binding,deadline_unix=9999),101,11)

    def test_stdout_quota_precedes_write_and_partial_failure_close_cannot_replay(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='io-',dir=EVIDENCE) as name:
            root=Path(name);telemetry=guard._Telemetry(root)
            with patch.object(guard,'STDOUT_CAP',4):
                pump=guard._Pump(io.BytesIO(b'12345'),root/'stdout',telemetry)
                with self.assertRaisesRegex(RuntimeError,'cap exceeded'):pump.close()
            self.assertEqual((root/'stdout').stat().st_size,0);self.assertEqual(telemetry.stdout_bytes,0)
            original=guard._write_all
            def partial(stream,payload):
                if stream.name==str(root/'partial'):
                    stream.write(payload[:2]);raise OSError('injected after two bytes')
                return original(stream,payload)
            with patch.object(guard,'_write_all',side_effect=partial):
                pump=guard._Pump(io.BytesIO(b'12345'),root/'partial',telemetry)
                with self.assertRaisesRegex(RuntimeError,'injected'):pump.close()
            self.assertEqual((root/'partial').read_bytes(),b'12');self.assertEqual(telemetry.stdout_bytes,5)
            telemetry.close();telemetry.emergency.close()

    def test_exact_status_transient_only_and_no_age_refresh(self):
        path=Path('worker-status.json').absolute();previous=dict(at_unix=1)
        with patch.object(guard,'_read',side_effect=FileNotFoundError(2,'missing',str(path))):self.assertIs(guard._latest(path,previous),previous)
        with patch.object(guard,'_read',side_effect=PermissionError(13,'ACL',str(path))):
            with self.assertRaises(PermissionError):guard._latest(path,previous)
        with patch.object(guard,'_read',side_effect=FileNotFoundError(2,'other',str(path.parent/'claim'))):
            with self.assertRaises(FileNotFoundError):guard._latest(path,previous)


if __name__=='__main__':unittest.main()
