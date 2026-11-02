"""SDK diagnostic: stdlib fake objects only, never Torch/native/process probes."""
from contextlib import ExitStack,redirect_stderr,redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('sdk_diagnostic_under_test',ROOT/'scripts/diagnose_gpc_sdk_startup.py')
SDK=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SDK)
REAL_AUTHORITY=SDK._authority
EVIDENCE=ROOT/'artifacts/gpc-sdk-startup-engineering-20261026'


class FakeCuda:
    def __init__(self):self.calls=[]
    def is_available(self):self.calls.append('available');return True
    def init(self):self.calls.append('init')
    def reset_peak_memory_stats(self):self.calls.append('reset_peak')
    def memory_allocated(self,device):return 0
    def memory_reserved(self,device):return 0
    def max_memory_allocated(self,device):return 0
    def max_memory_reserved(self,device):return 0


class FakeTorch:
    def __init__(self):self.cuda=FakeCuda();self.threads=None
    def set_num_threads(self,value):self.threads=value
    def __getattr__(self,name):raise AssertionError('Forbidden Torch operation: '+name)


class SDKTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(prefix='fixture-',dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);(self.root/'artifacts').mkdir()
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(SDK,'ROOT',self.root));self.stack.enter_context(patch.object(SDK,'DEFAULT_LOCK',self.root/'artifacts/model.lock'))
        self.stack.enter_context(patch.object(SDK,'_authority'))
        self.native=self.stack.enter_context(patch.object(SDK,'ResourceProvider',side_effect=AssertionError('Native resources forbidden')))
        self.native_owner=self.stack.enter_context(patch.object(SDK,'WindowsWorker',side_effect=AssertionError('Native process forbidden')))
        self.native_launch=self.stack.enter_context(patch.object(SDK.subprocess,'Popen',side_effect=AssertionError('Process execution forbidden')))
        self.stack.enter_context(patch.object(SDK.shared.protocol,'plan',side_effect=AssertionError('GPC planning/weights forbidden')))
        self.stack.enter_context(patch.object(SDK,'_protected_inputs',return_value={'fake_protected_metadata':'unchanged'}))
        path=SDK.plan_path();path.parent.mkdir(parents=True);path.write_bytes(SDK._json(SDK.diagnostic_plan('a'*40)))
        self.state=dict(exited=False,runtime=0,launches=0,quiesce=0);self.torch=FakeTorch();self.stderr=io.StringIO();test=self
        class Provider:
            def startup(self,output):return test.resources()
            def start(self):test.state['provider_started']=True
            def runtime(self,pid,output):
                test.state['runtime']+=1
                if test.state['runtime']==2:
                    handoff=SDK._read(output/'guardian/handoff.json');test.state['handoff']=handoff
                    test.assertTrue(test.state['owned']);test.assertTrue(Path(handoff['ack_path']).is_file())
                    if test.state.get('blocked'):
                        test.state['clock'][0]=SDK.WORK_SECONDS
                        value=SDK.shared.status_template();value.update(policy_calls=0,nonce=handoff['nonce'],actual_worker_pid=303,at_unix=time.time())
                        SDK._exclusive(Path(handoff['status_path']),value);return test.resources()
                    with patch.dict(os.environ,test.state['env']),patch.object(SDK.os,'getpid',return_value=303),patch.object(SDK.os,'getppid',return_value=202),redirect_stderr(test.stderr):
                        try:
                            if test.state.get('heartbeat_error'):
                                value=SDK.shared.status_template();value.update(policy_calls=0,nonce=handoff['nonce'],actual_worker_pid=303,at_unix=time.time()-6)
                                SDK._exclusive(Path(handoff['status_path']),value)
                                SDK._exclusive(output/'guardian/heartbeat-error.json',dict(kind='gpc_heartbeat_failure.v1',step='status.atomic_publish',exception_type='PermissionError',winerror=32))
                            else:
                                test.state['result']=SDK.worker(output/'guardian/handoff.json',startup_helper=test.startup,native_check=False)
                                test.state['exited']=True
                        except BaseException as error:test.state['worker_error']=error;test.state['exited']=True
                return test.resources()
            def close(self):test.state['provider_closed']=True
        self.provider=Provider()

    def resources(self):
        return dict(available_ram_bytes=8*1024**3,commit_headroom_bytes=12*1024**3,gpu_free_bytes=6*1024**3,disk_free_bytes=20*1024**3,
            gpu_age_seconds=0,working_set_bytes=64*1024**2,private_bytes=128*1024**2,process_peak_working_set_bytes=64*1024**2)
    def startup(self,heartbeat,trace,**kwargs):
        self.state['sdk_kwargs']=kwargs
        return SDK.shared.initialize_sdk(heartbeat,trace,importer=lambda name:self.torch,**kwargs)
    def launcher(self,command,**kwargs):
        self.state['launches']+=1;self.state['env']=kwargs['env'];self.state['command']=command
        path=Path(command[command.index('--handshake')+1]);nonce=command[command.index('--nonce')+1]
        SDK._exclusive(path,dict(pid=303,parent_pid=202,nonce=nonce));test=self
        class Child:
            pid=202;stdout=io.BytesIO(b'fake-sdk-worker-only\n')
            def poll(self):return 1 if test.state['exited']and test.state.get('worker_error')else 0 if test.state['exited']else None
            def wait(self,timeout):test.state.setdefault('wait_timeouts',[]).append(timeout);return self.poll()or 0
            def terminate(self):test.state['exited']=True
        return Child()
    def owner(self,pid,parent):
        self.assertEqual((pid,parent),(303,202));self.state['owned']=True;test=self
        self.assertFalse((SDK.paths()[0]/'guardian/worker-ack.json').exists())
        class Owner:
            def poll_exit(self):return test.state['exited']
            def exit_code(self):return 1 if test.state.get('worker_error')else 0
            def terminate_and_wait(self,timeout):
                test.state['terminate_timeout']=timeout;test.state['exited']=True
                if test.state.get('blocked'):test.state['clock'][0]+=timeout if test.state.get('full_timeout')else min(2,timeout)
            def quiesce(self,timeout):
                test.state['quiesce_timeout']=timeout
                if test.state.get('blocked'):test.state['clock'][0]+=min(1,timeout)
                return test.state['quiesce']
            def close(self):test.state['owner_closed']=True
        return Owner()
    def supervise(self):
        return SDK.supervise(authorization_id=SDK.AUTHORIZATION,reviewed_commit='a'*40,python_executable=sys.executable,
            provider=self.provider,owner_factory=self.owner,launcher_factory=self.launcher)

    def test_default_plan_has_no_claim_query_or_process(self):
        with redirect_stdout(io.StringIO())as text:self.assertEqual(SDK.main([]),0)
        value=json.loads(text.getvalue());self.assertEqual(value['status'],'plan_only');self.assertEqual(value['total_seconds'],60)
        self.assertEqual(value['total_bytes'],8*1024**2);self.assertFalse(SDK.paths()[0].exists());self.assertFalse(SDK.paths()[1].exists())
        self.native.assert_not_called();self.native_owner.assert_not_called();self.native_launch.assert_not_called()

    def test_fake_sdk_only_success_owned_before_ack_never_training(self):
        report=self.supervise();self.assertEqual(report['status'],'sdk_ready',report)
        self.assertEqual(report['actual_worker_pid'],303);self.assertEqual(report['job_active_processes_after_cleanup'],0)
        self.assertFalse(report['single_model_lock_retained']);self.assertFalse(report['native_sdk_path_verified'])
        self.assertEqual(report['claim']['deadline_monotonic']-report['claim']['started_monotonic'],60)
        self.assertEqual(report['claim']['execution_deadline_monotonic']-report['claim']['started_monotonic'],48)
        self.assertEqual(report['claim']['cleanup_reserve_seconds'],12);self.assertEqual(report['total_cap_bytes'],8*1024**2)
        self.assertEqual(self.torch.threads,2);self.assertEqual(self.torch.cuda.calls,['available','init','reset_peak'])
        self.assertEqual(self.state['sdk_kwargs'],dict(cpu_threads=2,require_cuda=True))
        self.assertEqual(report['last_status']['phase'],'complete');self.assertTrue(report['last_status']['sdk_complete']);self.assertFalse(report['last_status']['training_complete'])
        for key in SDK.ZERO_FIELDS:self.assertEqual(report['last_status'][key],0)
        output=SDK.paths()[0];self.assertTrue((output/'guardian/sdk-startup.jsonl').is_file())
        self.assertIsNone(SDK.shared.heartbeat_failure(output/'guardian/handoff.json'));self.assertFalse((output/'ledger').exists());self.assertFalse(list(output.rglob('*.pt')))
        self.assertEqual(report['claim']['plan_sha256'],SDK._sha(SDK.plan_path()))
        self.native.assert_not_called();self.native_owner.assert_not_called();self.native_launch.assert_not_called()

    def test_startup_failure_permanently_spends_external_claim(self):
        self.provider.startup=lambda output:dict(self.resources(),gpu_free_bytes=None)
        report=self.supervise();self.assertEqual(report['guard_reason'],'missing_or_invalid_resource:gpu_free_bytes');self.assertEqual(self.state['launches'],0)
        output,claim=SDK.paths();before=claim.read_bytes()
        self.assertTrue(output.is_relative_to(self.root));self.assertEqual(output.name,SDK.OUTPUT);shutil.rmtree(output)
        with self.assertRaises(FileExistsError):self.supervise()
        self.assertEqual(claim.read_bytes(),before);self.assertEqual(self.state['launches'],0)

    def test_first_heartbeat_error_precedes_stale_status(self):
        self.state['heartbeat_error']=True;report=self.supervise()
        self.assertEqual(report['guard_reason'],'heartbeat_failure');self.assertEqual(report['heartbeat_failure']['winerror'],32)
        self.assertEqual(report['heartbeat_failure']['step'],'status.atomic_publish')

    def test_shared_sdk_failure_durable_without_result(self):
        self.torch.cuda.is_available=lambda:False;report=self.supervise()
        self.assertEqual(report['guard_reason'],'heartbeat_failure',report)
        self.assertEqual(report['heartbeat_failure']['step'],'sdk.cuda_available');self.assertIn('CUDA unavailable',report['heartbeat_failure']['message'])
        self.assertFalse((SDK.paths()[0]/'sdk-result.json').exists());self.assertIn('gpc_heartbeat_failure',self.stderr.getvalue())

    def test_cleanup_uncertainty_retains_lock_and_primary_reason(self):
        self.provider.runtime=lambda pid,output:dict(self.resources(),private_bytes=4*1024**3);self.state['quiesce']=1
        report=self.supervise();self.assertEqual(report['guard_reason'],'process_memory_limit:private_bytes')
        self.assertTrue(report['single_model_lock_retained']);self.assertTrue(SDK.DEFAULT_LOCK.exists())

    def test_unknown_cuda_and_five_second_status_age_fail_closed(self):
        handoff=dict(actual_worker_pid=303,nonce='n',claim=dict(deadline_unix=time.time()+60,deadline_monotonic=time.monotonic()+60))
        value=SDK.shared.status_template();value.update(policy_calls=0,actual_worker_pid=303,nonce='n',at_unix=time.time())
        self.assertIsNone(SDK._status_reason(value,handoff));self.assertIsNone(value['cuda_allocated_bytes'])
        value['at_unix']-=5.01;self.assertEqual(SDK._status_reason(value,handoff),'worker_status_stale_or_invalid')
        value['at_unix']=time.time();value['model_loads']=1;self.assertEqual(SDK._status_reason(value,handoff),'forbidden_sdk_operation:model_loads')
        value['model_loads']=0;value.update(phase='complete',sdk_complete=True)
        self.assertEqual(SDK._status_reason(value,handoff),'cuda_sample_unavailable_or_inconsistent')
        value.update(SDK.shared.cuda_allocator_sample(self.torch));self.assertIsNone(SDK._status_reason(value,handoff))
        value['cuda_peak_allocated_bytes']=512*1024**2+1;self.assertIn('cuda_peak_allocated',SDK._status_reason(value,handoff))

    def test_stdout_bound_is_admitted_before_write(self):
        directory=self.root/'io';directory.mkdir();telemetry=SDK._Telemetry(directory)
        try:
            with self.assertRaisesRegex(OSError,'8MiB'):telemetry.admit('stdout',b'x'*(SDK.MIB+1))
            self.assertEqual(telemetry.stdout_bytes,0)
        finally:telemetry.close();telemetry.emergency.close()

    def test_fixed_plan_caps_or_source_mutation_refused_before_claim(self):
        path=SDK.plan_path();original=path.read_bytes()
        for key,value in(('total_seconds',61),('script_sha256','f'*64),('shared_guard_sha256','e'*64),('reviewed_commit','b'*40)):
            with self.subTest(field=key):
                plan=json.loads(original);plan[key]=value;path.write_bytes(SDK._json(plan))
                with self.assertRaises(PermissionError):self.supervise()
                self.assertFalse(SDK.paths()[0].exists());self.assertFalse(SDK.paths()[1].exists())
        path.write_bytes(original);self.assertEqual(self.state['launches'],0)

    def test_worker_rechecks_frozen_plan_before_sdk(self):
        self.supervise();path=SDK.paths()[0]/'guardian/handoff.json'
        plan=json.loads(SDK.plan_path().read_bytes());plan['total_bytes']+=1;SDK.plan_path().write_bytes(SDK._json(plan))
        with patch.dict(os.environ,self.state['env']),patch.object(SDK.os,'getpid',return_value=303),patch.object(SDK.os,'getppid',return_value=202):
            with self.assertRaises(PermissionError):SDK.verify_worker(path,native_check=False)

    def test_worker_cannot_start_sdk_in_the_cleanup_reserve(self):
        report=self.supervise();path=SDK.paths()[0]/'guardian/handoff.json';claim=report['claim']
        with patch.dict(os.environ,self.state['env']),patch.object(SDK.os,'getpid',return_value=303),patch.object(SDK.os,'getppid',return_value=202),patch.object(SDK.time,'time',return_value=claim['started_unix']+49),patch.object(SDK.time,'monotonic',return_value=claim['started_monotonic']+49):
            with self.assertRaisesRegex(PermissionError,'deadline'):SDK.verify_worker(path,native_check=False)

    def test_exact_authority_and_cli_weight_flags_refused(self):
        with self.assertRaises(PermissionError):REAL_AUTHORITY('other-diagnostic','a'*40)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):SDK.main(['--execute','--checkpoint','forbidden'])
        self.assertFalse(SDK.paths()[0].exists());self.assertFalse(SDK.paths()[1].exists())

    def test_git_checks_share_the_original_work_deadline(self):
        wall,mono=time.time(),time.monotonic();offset=[0];timeouts=[]
        def git(command,*,text,timeout):
            timeouts.append(timeout);offset[0]+=17 if command[-1]=='HEAD'else 5
            return 'a'*40+'\n'if command[-1]=='HEAD'else ''
        with patch.object(SDK.time,'time',side_effect=lambda:wall+offset[0]),patch.object(SDK.time,'monotonic',side_effect=lambda:mono+offset[0]),patch.object(SDK.subprocess,'check_output',side_effect=git):
            REAL_AUTHORITY(SDK.AUTHORIZATION,'a'*40,work_deadline_unix=wall+48,work_deadline_monotonic=mono+48)
        self.assertEqual(timeouts,[48,31]);self.assertEqual(offset[0],22)
        self.assertFalse(SDK.paths()[0].exists());self.assertFalse(SDK.paths()[1].exists())

    def test_git_timeout_refuses_before_claim_or_resource_query(self):
        with patch.object(SDK,'_authority',side_effect=REAL_AUTHORITY),patch.object(SDK.subprocess,'check_output',side_effect=SDK.subprocess.TimeoutExpired('git',48))as git:
            with self.assertRaises(SDK.subprocess.TimeoutExpired):self.supervise()
        self.assertEqual(git.call_count,1);self.assertFalse(SDK.paths()[0].exists());self.assertFalse(SDK.paths()[1].exists())
        self.assertEqual(self.state['launches'],0);self.assertEqual(self.state['runtime'],0)
        self.native.assert_not_called();self.native_owner.assert_not_called();self.native_launch.assert_not_called()

    def test_expiry_after_second_git_check_cannot_claim_or_restart_clock(self):
        wall,mono=time.time(),time.monotonic();offset=[0];timeouts=[]
        def git(command,*,text,timeout):
            timeouts.append(timeout);offset[0]+=20 if command[-1]=='HEAD'else 28
            return 'a'*40+'\n'if command[-1]=='HEAD'else ''
        with patch.object(SDK.time,'time',side_effect=lambda:wall+offset[0]),patch.object(SDK.time,'monotonic',side_effect=lambda:mono+offset[0]),patch.object(SDK,'_authority',side_effect=REAL_AUTHORITY),patch.object(SDK.subprocess,'check_output',side_effect=git):
            with self.assertRaisesRegex(TimeoutError,'Original SDK work deadline'):self.supervise()
        self.assertEqual(timeouts,[48,28]);self.assertFalse(SDK.paths()[0].exists());self.assertFalse(SDK.paths()[1].exists())
        self.assertEqual(self.state['launches'],0);self.assertEqual(self.state['runtime'],0)

    def test_cleanup_waits_share_original_sixty_second_deadline(self):
        wall,mono=time.time(),time.monotonic();offset=[0];self.state.update(blocked=True,clock=offset)
        with patch.object(SDK.time,'time',side_effect=lambda:wall+offset[0]),patch.object(SDK.time,'monotonic',side_effect=lambda:mono+offset[0]):report=self.supervise()
        self.assertEqual(report['guard_reason'],'sdk_execution_deadline_cleanup_reserve')
        self.assertEqual(self.state['terminate_timeout'],4);self.assertEqual(self.state['quiesce_timeout'],2)
        self.assertTrue(all(timeout<=1 for timeout in self.state['wait_timeouts']));self.assertEqual(report['wall_seconds'],51)
        self.assertEqual(report['job_active_processes_after_cleanup'],0)

    def test_exhausted_early_cleanup_preserves_seven_seconds_for_thread_joins(self):
        wall,mono=time.time(),time.monotonic();offset=[0];self.state.update(blocked=True,clock=offset,full_timeout=True);test=self
        class Join:
            def __init__(self,name):self.name=name
            def join(self,timeout):test.state[self.name+'_timeout']=timeout;offset[0]+=timeout
            def is_alive(self):return False
        class Pump:
            def __init__(self,source,path,telemetry):self.source=source;self.stream=path.open('xb',buffering=0);self.thread=Join('pipe');self.error=None
            def check(self):pass
        class Stop:
            def set(self):test.state['resource_stop_requested']=True
        self.provider._stop=Stop();self.provider._thread=Join('resource');self.native.side_effect=lambda:self.provider
        with patch.object(SDK.time,'time',side_effect=lambda:wall+offset[0]),patch.object(SDK.time,'monotonic',side_effect=lambda:mono+offset[0]),patch.object(SDK.shared,'_Pump',Pump):
            report=SDK.supervise(authorization_id=SDK.AUTHORIZATION,reviewed_commit='a'*40,python_executable=sys.executable,
                owner_factory=self.owner,launcher_factory=self.launcher)
        self.assertEqual(self.state['terminate_timeout'],4);self.assertNotIn('quiesce_timeout',self.state)
        self.assertEqual(self.state['pipe_timeout'],3);self.assertEqual(self.state['resource_timeout'],4)
        self.assertEqual(self.state['wait_timeouts'],[0]);self.assertEqual(report['wall_seconds'],59)
        self.assertTrue(report['single_model_lock_retained']);self.assertIsNone(report['job_active_processes_after_cleanup'])

    def test_final_io_crossing_total_deadline_returns_and_persists_stop(self):
        wall,mono=time.time(),time.monotonic();offset=[0];original=SDK._Telemetry.close
        def closing(telemetry):original(telemetry);offset[0]=61
        with patch.object(SDK.time,'time',side_effect=lambda:wall+offset[0]),patch.object(SDK.time,'monotonic',side_effect=lambda:mono+offset[0]),patch.object(SDK._Telemetry,'close',closing):report=self.supervise()
        self.assertEqual(report['guard_reason'],'sdk_total_deadline_exceeded');self.assertEqual(report['wall_seconds'],61)
        saved=SDK._read(SDK.paths()[0]/'guardian/stop-record.json');self.assertEqual(saved['guard_reason'],'sdk_total_deadline_exceeded')


if __name__=='__main__':unittest.main()
