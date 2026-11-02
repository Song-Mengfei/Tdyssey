"""Harmless heartbeat observability fakes; no SDK, process, model or physics."""
import errno
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation_guard as guard

EVIDENCE=guard.ROOT/"artifacts/gpc-sdk-startup-review-20261026"


class FakeTorch:
    """Only scalar counters; none of these methods invokes an SDK or device."""
    def __init__(self,available=True):
        self.calls=[];self.available=available;self.cuda=type("FakeCuda",(),{})()
        def record(name,result=None):
            def invoke(*args,**kwargs):self.calls.append(name);return result
            return invoke
        self.set_num_threads=record("set_num_threads")
        for name,value in (("is_available",available),("init",None),("reset_peak_memory_stats",None),
                ("memory_allocated",7),("memory_reserved",11),("max_memory_allocated",13),("max_memory_reserved",17)):
            setattr(self.cuda,name,record(name,value))
    def importer(self,*args,**kwargs):self.calls.append("torch_import");return self


class StartupObservabilityChecks(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix="fake-observer-",dir=EVIDENCE)
        self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name)
        self.handoff=self.root/"guardian/handoff.json";self.handoff.parent.mkdir()
        self.status_path=self.handoff.parent/"worker-status.json"
        self.handoff.write_text(json.dumps(dict(status_path=str(self.status_path),actual_worker_pid=os.getpid(),nonce="fixture")),encoding="utf-8")
        self.reader=patch.object(guard,"_read",side_effect=lambda path:json.loads(Path(path).read_bytes()))
        self.reader.start();self.addCleanup(self.reader.stop)
        # Avoid the real Win32 atomic-publication helper in these harmless tests.
        self.publisher=patch.object(guard,"_publish_json",side_effect=lambda source,target,**kwargs:os.replace(source,target))
        self.publisher.start();self.addCleanup(self.publisher.stop)
        self.stderr=io.StringIO();self.stderr_patch=patch("sys.stderr",self.stderr)
        self.stderr_patch.start();self.addCleanup(self.stderr_patch.stop)

    def test_heartbeat_continues_while_main_thread_waits_without_stage_progress(self):
        published=[];third=threading.Event();main_started=threading.Event()
        def publication(path,payload,*args,**kwargs):
            published.append(dict(payload,observed_unix=time.time(),main_waiting=main_started.is_set()))
            if len(published)>=3:third.set()
            return published[-1]
        with patch.object(guard,"publish_status",side_effect=publication):
            with guard.Heartbeat(self.handoff,guard.status_template()) as heartbeat:
                main_started.set()
                self.assertTrue(third.wait(3.5),"Ordinary waiting operation prevented two periodic heartbeats")
                heartbeat.check()
            self.assertFalse(heartbeat.thread.is_alive())
            self.assertTrue(heartbeat.stop.is_set())
        self.assertGreaterEqual(sum(row["main_waiting"] for row in published),2)
        self.assertTrue(all(row["phase"]=="setup" and not row["training_complete"] for row in published))
        self.assertTrue(all(row["model_loads"]==row["optimizer_updates"]==row["forecasts"]==0 for row in published))

    def test_heartbeat_freshness_limit_remains_exactly_five_seconds(self):
        binding=guard._clocks({},100.,10.);status=guard.status_template()
        status.update(actual_worker_pid=3,nonce="fixture",at_unix=100.)
        self.assertIsNone(guard.monitor_reason(status,binding,pid=3,nonce="fixture",now_unix=105.,now_monotonic=15.))
        self.assertEqual(guard.monitor_reason(status,binding,pid=3,nonce="fixture",now_unix=105.000001,now_monotonic=15.),
                         "worker_status_stale_or_invalid")

    def test_startup_metadata_cannot_claim_a_model_or_completed_training(self):
        value=guard.status_template()
        self.assertFalse(value["training_complete"])
        for key in ("model_instances","model_loads","optimizer_updates","forecasts","euler_steps","eval_batches","eval_examples"):
            self.assertEqual(value[key],0)
        value.update(actual_worker_pid=3,nonce="fixture",at_unix=100.,training_complete=True)
        self.assertEqual(guard.monitor_reason(value,guard._clocks({},100.,10.),pid=3,nonce="fixture",now_unix=100.,now_monotonic=10.),
                         "premature_training_completion")

    def first_error(self):
        record=guard.heartbeat_failure(self.handoff)
        self.assertIsInstance(record,dict)
        return record.get("first_reason",record)

    def test_refresh_failure_is_durable_and_thread_cleanup_keeps_first_cause(self):
        failing=threading.Event();calls=[0]
        def refresh():
            calls[0]+=1
            if calls[0]>1:
                failing.set();raise OSError(errno.ENOMEM,"injected refresh failure")
            return {}
        with patch.object(guard,"publish_status",side_effect=lambda path,value,**kwargs:value):
            with self.assertRaisesRegex(RuntimeError,"heartbeat"):
                with guard.Heartbeat(self.handoff,guard.status_template(),refresh=refresh) as heartbeat:
                    self.assertTrue(failing.wait(2.5))
                    # Exit joins the failed publisher and surfaces its first error.
            self.assertFalse(heartbeat.thread.is_alive())
        error=self.first_error()
        self.assertEqual(error["errno"],errno.ENOMEM);self.assertIn("refresh",error["step"])
        self.assertTrue(error["exception_type"].endswith("OSError"))
        self.assertIn("injected refresh failure",error.get("trace",error.get("traceback","")))
        self.assertEqual((self.handoff.parent/"heartbeat-error.json").stat().st_size,guard.SLOT)

    def test_partial_status_write_and_cleanup_error_preserve_original_error(self):
        original_write=guard._write_all;original_unlink=Path.unlink;active=[False]
        def partial(stream,payload):
            if active[0] and Path(stream.name).name.startswith("worker-status.json."):
                stream.write(payload[:7]);raise OSError(errno.EIO,"injected partial status write")
            return original_write(stream,payload)
        def cleanup(path,*args,**kwargs):
            if active[0] and path.name.startswith("worker-status.json."):
                raise PermissionError(errno.EACCES,"injected cleanup failure",str(path))
            return original_unlink(path,*args,**kwargs)
        with patch.object(guard,"_write_all",side_effect=partial),patch.object(Path,"unlink",cleanup):
            with self.assertRaises(OSError) as caught:
                with guard.Heartbeat(self.handoff,guard.status_template()) as heartbeat:
                    active[0]=True;heartbeat.update(fixture="write-failure")
            self.assertFalse(heartbeat.thread.is_alive())
        self.assertEqual(caught.exception.errno,errno.EIO)
        error=self.first_error();self.assertEqual(error["errno"],errno.EIO);self.assertIn("write",error["step"])
        self.assertIn("partial status write",error.get("trace",error.get("traceback","")))
        partials=list(self.status_path.parent.glob("worker-status.json.*.tmp"))
        self.assertEqual(len(partials),1);self.assertEqual(len(partials[0].read_bytes()),7)

    def test_publication_sharing_error_is_recorded_once_without_retry(self):
        calls=[];primary=PermissionError(errno.EACCES,"injected sharing failure");primary.winerror=32
        def publication(source,target,**kwargs):
            calls.append(str(target))
            if len(calls)>1:raise primary
            return os.replace(source,target)
        with patch.object(guard,"_publish_json",side_effect=publication):
            with self.assertRaises(PermissionError):
                with guard.Heartbeat(self.handoff,guard.status_template()) as heartbeat:
                    heartbeat.update(fixture="sharing-failure")
            self.assertFalse(heartbeat.thread.is_alive())
        self.assertEqual(len(calls),2)
        first=self.first_error();self.assertEqual(first["errno"],errno.EACCES);self.assertEqual(first["winerror"],32)
        self.assertIn("publish",first["step"])
        heartbeat.record_error(OSError(errno.ENOSPC,"secondary after sharing"),step="fixture.secondary")
        self.assertEqual(self.first_error(),first)

    def test_fake_sdk_stages_are_durable_and_do_not_count_models_or_training(self):
        sdk=FakeTorch();trace=guard.StartupTrace(self.handoff);self.addCleanup(trace.close)
        with guard.Heartbeat(self.handoff,guard.status_template(),refresh=trace.snapshot) as heartbeat:
            loaded,diagnostics=guard.initialize_sdk(heartbeat,trace,importer=sdk.importer)
            self.assertIs(loaded,sdk);self.assertTrue(diagnostics["sdk_complete"])
            for key in ("model_instances","model_loads","optimizer_updates","forecasts","euler_steps","eval_batches","eval_examples"):
                self.assertEqual(heartbeat.payload[key],0)
            self.assertFalse(heartbeat.payload["training_complete"])
            self.assertEqual(heartbeat.payload["cuda_sample_state"],"fresh")
        self.assertEqual(sdk.calls,["torch_import","set_num_threads","is_available","init","reset_peak_memory_stats",
            "memory_allocated","memory_reserved","max_memory_allocated","max_memory_reserved"])
        stages=("torch_import","set_num_threads","cuda_available","cuda_init","cuda_reset_peak","allocator_sample")
        journal=[json.loads(line) for line in trace.path.read_bytes().splitlines()]
        observed=[(row["stage"],row["state"]) for row in journal if row["stage"] in stages]
        self.assertEqual(observed,[(name,state) for name in stages for state in ("before","after")])
        self.assertEqual(trace.snapshot()["startup_stage"],"sdk_initialized")
        self.assertEqual(trace.snapshot()["startup_stage_state"],"after")

    def test_fake_sdk_unavailable_stops_before_cuda_initialization_and_retains_stage(self):
        sdk=FakeTorch(available=False);trace=guard.StartupTrace(self.handoff);self.addCleanup(trace.close)
        with self.assertRaisesRegex(RuntimeError,"CUDA unavailable"):
            with guard.Heartbeat(self.handoff,guard.status_template(),refresh=trace.snapshot) as heartbeat:
                guard.initialize_sdk(heartbeat,trace,importer=sdk.importer)
        self.assertEqual(sdk.calls,["torch_import","set_num_threads","is_available"])
        self.assertEqual(self.first_error()["step"],"sdk.cuda_available")
        self.assertEqual(trace.snapshot()["startup_stage_state"],"error")
        self.assertFalse(heartbeat.payload["training_complete"]);self.assertEqual(heartbeat.payload["model_loads"],0)

    def test_fake_long_import_keeps_heartbeats_but_does_not_advance_main_stage(self):
        sdk=FakeTorch();trace=guard.StartupTrace(self.handoff);self.addCleanup(trace.close)
        importing=threading.Event();two=threading.Event();during=[]
        def publication(path,payload,**kwargs):
            if importing.is_set():
                during.append(dict(payload))
                if len(during)>=2:two.set()
            return payload
        def importer(*args,**kwargs):
            importing.set()
            self.assertTrue(two.wait(3.5),"Heartbeat did not run while harmless importer waited")
            importing.clear();return sdk.importer(*args,**kwargs)
        with patch.object(guard,"publish_status",side_effect=publication):
            with guard.Heartbeat(self.handoff,guard.status_template(),refresh=trace.snapshot) as heartbeat:
                guard.initialize_sdk(heartbeat,trace,importer=importer)
        self.assertGreaterEqual(len(during),2)
        self.assertTrue(all(row["startup_stage"]=="torch_import" and row["startup_stage_state"]=="before" for row in during))
        self.assertEqual(len({row["startup_progress_unix"] for row in during}),1)
        self.assertTrue(all(row["cuda_sample_state"]=="unavailable" and row["cuda_allocated_bytes"] is None for row in during))

    def test_stale_main_stage_is_distinct_from_fresh_heartbeat_and_unknown_cuda(self):
        with patch.object(guard.time,"time",return_value=100.),patch.object(guard.time,"monotonic",return_value=10.):
            trace=guard.StartupTrace(self.handoff);self.addCleanup(trace.close)
            trace.stage("torch_import","before")
        value=dict(guard.status_template(),**trace.snapshot(),actual_worker_pid=3,nonce="fixture",at_unix=200.)
        self.assertEqual(value["startup_progress_unix"],100.)
        self.assertIsNone(guard.monitor_reason(value,guard._clocks({},100.,10.),pid=3,nonce="fixture",now_unix=200.,now_monotonic=110.))
        self.assertTrue(all(value[key] is None for key in ("cuda_allocated_bytes","cuda_reserved_bytes",
            "cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","cuda_sample_at_unix","cuda_sample_at_monotonic")))
        value.update(phase="evaluation",model_instances=1,training_complete=True,optimizer_updates=500)
        self.assertIn("cuda",guard.monitor_reason(value,guard._clocks({},100.,10.),pid=3,nonce="fixture",now_unix=200.,now_monotonic=110.))

    def test_cuda_sample_age_and_state_cannot_be_replaced_by_fresh_heartbeat(self):
        sdk=FakeTorch()
        with patch.object(guard.time,"time",return_value=100.),patch.object(guard.time,"monotonic",return_value=10.):
            sample=guard.cuda_allocator_sample(sdk)
        value=dict(guard.status_template(),**sample)
        value.update(actual_worker_pid=3,nonce="fixture",at_unix=105.,phase="evaluation",model_instances=1,
                     training_complete=True,optimizer_updates=500)
        binding=guard._clocks({},100.,10.)
        self.assertIsNone(guard.monitor_reason(value,binding,pid=3,nonce="fixture",now_unix=105.,now_monotonic=15.))
        self.assertEqual(guard.monitor_reason(value,binding,pid=3,nonce="fixture",now_unix=105.000001,now_monotonic=15.),"cuda_sample_stale_or_invalid")
        for state in (None,"invented"):
            with self.subTest(state=state):
                wrong=dict(value,cuda_sample_state=state,cuda_sample_at_unix=None,cuda_sample_at_monotonic=None)
                self.assertIsNotNone(guard.monitor_reason(wrong,binding,pid=3,nonce="fixture",now_unix=105.,now_monotonic=15.),
                    "Unrecognized CUDA sample state bypassed its freshness contract")


if __name__=="__main__":unittest.main()
