"""Ordinary-object guard fixtures; no DLL, GPU query, Torch or real child."""
import io
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest.mock import patch

from tdyssey import raw_training_guard as guard


def token(**changes):
    value=dict(pid=303,nonce='fixture',parent=202,max_updates=2,deadline=100.,deadline_wall=1100.)
    value.update(changes);return value


def status(**changes):
    value=dict(pid=303,nonce='fixture',at=1000.,mono=10.,phase='training',model_instances=1,
        model_loads=1,reserved_updates=1,completed_updates=0,cuda_at=10.,cuda_wall=1000.,
        **dict.fromkeys(guard.CUDA_CAPS,0))
    value.update(changes);return value


def resources():
    return dict(ram=8*guard.GIB,commit=12*guard.GIB,gpu=6*guard.GIB,disk=20*guard.GIB,
        gpu_age=0.,working_set=64*1024**2,peak_working_set=64*1024**2,private=128*1024**2)


class RawGuardTests(unittest.TestCase):
    def setUp(self):
        evidence=Path(__file__).resolve().parents[1]/'artifacts/raw-training-guard-engineering'
        evidence.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix='fake-',dir=evidence)
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.config=dict(output_dir=str(self.root/'output'),max_updates=2,wall_seconds=30.)

    def test_plan_and_linux_refusal_do_not_query_or_create_output(self):
        with (patch.object(guard,'_Native',side_effect=AssertionError('No native query')) as native,
              patch.object(guard.subprocess,'Popen',side_effect=AssertionError('No process')) as popen):
            self.assertEqual(guard.run_guarded(self.config,['worker.py'])['status'],'plan')
            with patch.object(guard.sys,'platform','linux'):
                with self.assertRaisesRegex(RuntimeError,'Linux execution is unavailable'):
                    guard.run_guarded(self.config,['worker.py'],execute=True)
            native.assert_not_called();popen.assert_not_called()
        self.assertFalse((self.root/'output').exists())
        with self.assertRaises(ValueError):guard.run_guarded(dict(self.config,wall_seconds=float('inf')),['worker.py'])

    def test_exact_resource_floors_and_native_process_peaks(self):
        for startup in (False,True):
            self.assertIsNone(guard.resource_reason(resources(),startup=startup))
            for key in ('ram','commit','gpu','disk'):
                self.assertIn(key,guard.resource_reason(dict(resources(),**{key:None}),startup=startup))
        self.assertEqual(guard.resource_reason(dict(resources(),gpu_age=5.001)),'gpu_sample_stale')
        self.assertIn('peak_working_set',guard.resource_reason(dict(resources(),peak_working_set=2*guard.GIB+1)))
        self.assertIn('private',guard.resource_reason(dict(resources(),private=3*guard.GIB+1)))
        self.assertIsNone(guard.resource_reason(dict(resources(),ram=3*guard.GIB//2,commit=3*guard.GIB,
            gpu=3*guard.GIB//2,disk=2*guard.GIB,working_set=2*guard.GIB,peak_working_set=2*guard.GIB,private=3*guard.GIB)))

    def test_worker_counts_cuda_and_independent_clock_boundaries(self):
        def reason(value,**kw):return guard.status_reason(value,token(),now=kw.get('now',1000.),mono=kw.get('mono',10.))
        self.assertIsNone(reason(status()))
        for key,value in (('model_instances',2),('model_loads',2),('reserved_updates',3),('completed_updates',3)):
            self.assertIn(key,reason(status(**{key:value})))
        self.assertEqual(reason(status(completed_updates=2)),'worker_updates_not_reserved')
        self.assertEqual(reason(status(at=994.999)),'worker_status_stale')
        self.assertEqual(reason(status(mono=4.999)),'worker_status_stale')
        self.assertEqual(reason(status(cuda_at=4.999)),'cuda_sample_stale')
        self.assertIn('cuda_peak_allocated',reason(status(cuda_peak_allocated_bytes=guard.GIB//2+1)))

    def test_actual_status_callback_samples_after_import_and_never_refunds_failure(self):
        class CUDA:
            def __init__(self):self.initialized=False;self.resets=0;self.calls=[]
            def is_available(self):self.calls.append('available');return True
            def init(self):self.calls.append('init');self.initialized=True
            def reset_peak_memory_stats(self,device):
                self.calls.append(('reset',device));self.resets+=1
            def memory_allocated(self,_device):
                if not self.initialized:raise AssertionError('Uninitialized CUDA must not be reported fresh')
                return 0
            memory_reserved=max_memory_allocated=max_memory_reserved=memory_allocated
        class Torch:
            cuda=CUDA()
            def set_num_threads(self,count):self.threads=count
        sdk=Torch();stream=io.BytesIO();control=guard.WorkerStatus(token(),stream)
        with patch.object(guard.time,'time',return_value=1000.),patch.object(guard.time,'monotonic',return_value=10.):
            control.attach_torch(sdk)
            control.check('training',dict(model_instances=1,model_loads=1,reserved_updates=1,completed_updates=0))
            control.check('training',dict(completed_updates=1))
            with self.assertRaisesRegex(ValueError,'backward'):control.check('training',dict(reserved_updates=0))
            first=control.error
            with self.assertRaisesRegex(RuntimeError,'heartbeat failed'):control.check('training',dict(reserved_updates=2))
        self.assertIs(control.error,first);self.assertEqual(sdk.threads,2)
        rows=[json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(rows[-1]['completed_updates'],1)
        self.assertEqual(rows[-1]['cuda_at'],10.)
        self.assertEqual(sdk.cuda.calls,['available','init',('reset',0)])
        self.assertEqual(sdk.cuda.resets,1)

    def test_unavailable_cuda_never_initializes_or_publishes_fresh_zero(self):
        class CUDA:
            def __init__(self):self.calls=[]
            def is_available(self):self.calls.append('available');return False
            def init(self):raise AssertionError('Unavailable CUDA must not initialize')
            def reset_peak_memory_stats(self,_device):raise AssertionError('No initialized CUDA peaks')
            def memory_allocated(self,_device):raise AssertionError('No uninitialized zero observation')
            memory_reserved=max_memory_allocated=max_memory_reserved=memory_allocated
        class Torch:
            cuda=CUDA()
            def set_num_threads(self,count):self.threads=count
        sdk=Torch();stream=io.BytesIO();control=guard.WorkerStatus(token(),stream)
        with self.assertRaisesRegex(RuntimeError,'CUDA is unavailable'):control.attach_torch(sdk)
        first=control.error
        with self.assertRaisesRegex(RuntimeError,'heartbeat failed'):control.attach_torch(sdk)
        self.assertIs(control.error,first);self.assertIsNone(control.refresh)
        self.assertIsNone(control.value['cuda_at']);self.assertEqual(stream.getvalue(),b'')
        self.assertEqual(sdk.cuda.calls,['available'])

    def test_pipe_rejects_bad_intermediate_record_and_counter_rollback(self):
        for bad in (status(reserved_updates=3),status(reserved_updates=0)):
            rows=[status(),bad,status(completed_updates=1)]
            pipe=guard._Pipe(io.BytesIO(b''.join(guard._encode(row) for row in rows)),token())
            pipe.thread.join(1)
            self.assertIsNotNone(pipe.error);self.assertEqual(pipe.latest,rows[0])
            self.assertFalse(pipe.thread.is_alive())

    def _run_fake_parent(self, *, own_error=False, cleanup_error=False):
        state={};test=self
        class Stream:
            def __init__(self):self.lines=queue.Queue();self.closed=False
            def readline(self,_bound):return self.lines.get(timeout=1)
            def close(self):self.closed=True
        class Input(io.BytesIO):
            def flush(self):
                if state.get('acked'):return
                proof=json.loads(self.getvalue());state['acked']=True
                test.assertTrue(state.get('owned'))
                sample_mono,sample_wall=time.monotonic(),time.time()
                row=status(pid=proof['pid'],nonce=proof['nonce'],at=time.time(),mono=time.monotonic(),
                    cuda_at=sample_mono,cuda_wall=sample_wall,phase='complete',model_instances=0,
                    reserved_updates=2,completed_updates=2)
                child=state['child'];child.stdout.lines.put(guard._encode(row));child.stdout.lines.put(b'');child.returncode=0
        class Child:
            pid=303
            def __init__(self,command,**kw):
                state.update(child=self,command=command,env=kw['env']);self.returncode=None
                self.stdin=Input();self.stdout=Stream()
                self.stdout.lines.put(guard._encode(dict(ready=True,pid=303,nonce=kw['env']['TDYSSEY_RAW_NONCE'])))
            def poll(self):return self.returncode
            def wait(self,timeout):
                if self.returncode is None:raise TimeoutError('Fake still alive')
                return self.returncode
            def terminate(self):state['terminated']=True;self.returncode=1;self.stdout.lines.put(b'')
        class Native:
            def __init__(self):self.owned=False
            def snapshot(self,_child,_output):return resources()
            def own(self,child):
                test.assertFalse(state.get('acked',False))
                if own_error:raise OSError('Injected Job assignment failure')
                self.owned=True;state['owned']=True
            def cleanup(self,deadline):
                state['cleanup_deadline']=deadline
                if cleanup_error:raise TimeoutError('Injected Job not quiet')
                return 0
            def close(self):state['job_closed']=True
        class GPU:
            def __init__(self,_deadline):pass
            def sample(self):pass
            def start(self):pass
            def add(self,value):return value
            def close(self,_timeout):state['gpu_closed']=True
        with (patch.object(guard.sys,'platform','win32'),patch.object(guard,'_Native',Native),
              patch.object(guard,'_GPU',GPU),patch.object(guard.subprocess,'Popen',Child),
              patch.object(guard,'LOCK',self.root/'model.lock')):
            report=guard.run_guarded(self.config,['worker.py','--worker'],execute=True)
        return report,state

    def test_parent_owns_before_ack_and_success_releases_job_and_model_lock(self):
        report,state=self._run_fake_parent()
        self.assertEqual(report['status'],'complete',report);self.assertTrue(report['job_empty'])
        self.assertFalse(report['lock_retained']);self.assertFalse(report['native_verified'])
        self.assertFalse((self.root/'model.lock').exists())
        self.assertEqual(state['command'][0],guard.sys._base_executable)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            self.assertEqual(state['env'][key],'2')
        saved=(self.root/'output/guard-report.json').read_bytes()
        again,_=self._run_fake_parent();self.assertEqual(again['status'],'stopped')
        self.assertEqual((self.root/'output/guard-report.json').read_bytes(),saved)

    def test_assignment_failure_terminates_only_retained_gated_child(self):
        report,state=self._run_fake_parent(own_error=True)
        self.assertIn('Injected Job assignment failure',report['reason'])
        self.assertTrue(state['terminated']);self.assertFalse(state.get('acked',False))
        self.assertFalse(report['lock_retained']);self.assertTrue(state['job_closed'])

    def test_uncertain_job_cleanup_keeps_lock_and_preserves_first_failure(self):
        report,_=self._run_fake_parent(cleanup_error=True)
        self.assertEqual(report['status'],'stopped');self.assertTrue(report['lock_retained'])
        self.assertTrue((self.root/'model.lock').exists());self.assertFalse(report['job_empty'])


if __name__=='__main__':unittest.main()
