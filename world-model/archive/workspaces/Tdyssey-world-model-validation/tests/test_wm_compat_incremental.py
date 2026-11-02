"""Synthetic complete flow and cross-process quota tests; no real assets."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("NumPy is required for constructed fake arrays")
import numpy as np
from tdyssey.wm_compat_budget import CompatBudget, EXACT_CHECKPOINT
from tdyssey.wm_compat_engine import ChunkSink, SavedBranch
from tdyssey.wm_compat_evaluation import run_evaluation, save_summary
from tdyssey.wm_compat_storage import Coordinator, StorageError, create_lock_file
from tests.test_wm_compat_reporting import ANCHORS, CANDIDATES, real_rows, fake_measurement, fake_truth

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("torch", "pymunk", "pygame", "diffusion_policy", "tdyssey.gpc_world_model", "tdyssey.wm_image_measurement_v2")

def _capacity_denial(error,phase):
    details=dict(error.details)
    assert details["phase"]==phase+".capacity",details
    assert details["exception_type"]=="ValueError",details
    assert details["message"]=="output_bytes limit exceeded including reservation",details
    return details

def constructed_inputs():
    image = np.ones((3,96,96),dtype=np.float32)
    image[:,0,0] = np.asarray([119,136,153],dtype=np.float32)*np.float32(1/255)
    items=[]
    for row in real_rows():
        if row["arm"] != "A": continue
        key=row["source_key"]; parts=key.split("-"); calibration=row["source"]=="calibration"
        items.append(dict(source_key=key,source=row["source"],shape=row["shape"],
            anchor=key if calibration else parts[0],candidate=None if calibration else parts[1],
            horizon=None if calibration else int(parts[-1].lstrip("H")),
            rgb=None if row["status"]=="missing" else image,goal=None,truth=deepcopy(row["truth"])))
    frames=np.stack([image]*4); actions=np.full((4,2),200.,dtype=np.float32)
    branches=[SavedBranch(anchor,candidate,anchor[0],frames,actions[:3],actions,frames,
        goal=None,truth=[fake_truth() for _ in range(4)],latest=fake_truth(),
        contact_by_h4=candidate in ("toward","side"))
        for anchor in ANCHORS for candidate in CANDIDATES if (anchor,candidate)!=("L01","hold")]
    assert len(items)==536 and sum(r["rgb"] is not None for r in items)==528 and len(branches)==63
    return dict(real_items=items,branches=branches)

class FakeBackend:
    def __init__(self): self.value=1; self.sample_calls=self.euler_calls=self.close_calls=0
    def capture(self): return struct.pack("<I",self.value)
    def restore(self,value): self.value=struct.unpack("<I",value)[0]
    def seed(self,value): self.value=value
    def fake_denoiser(self):
        self.euler_calls+=1; self.value=(1664525*self.value+1013904223)%2**32
    def sample(self,images,actions):
        assert images.shape==(4,3,96,96) and actions.shape==(4,2)
        assert images.dtype==actions.dtype==np.float32
        assert not images.flags.writeable and not actions.flags.writeable
        assert np.isfinite(images).all() and np.isfinite(actions).all()
        self.sample_calls+=1
        for _ in range(3): self.fake_denoiser()
        return np.full((3,96,96),-.1,dtype=np.float32)
    def stats(self):
        return dict(model_instances=1,model_loads=1,optimizer_updates=0,cuda_allocated_bytes=0,
            cuda_reserved_bytes=0,cuda_peak_allocated_bytes=0,cuda_peak_reserved_bytes=0)
    def close(self):
        self.close_calls+=1; assert self.value==1,"Caller RNG was not restored"

def run_constructed_flow(base,storage,*,io_hook=None,heartbeat=None,seconds=1800):
    base=Path(base); inputs=constructed_inputs()
    binding=dict(protocol_sha256="a"*64,code_sha256={"synthetic.py":"b"*64},
        asset_sha256={"synthetic.npz":"c"*64},checkpoint_sha256=EXACT_CHECKPOINT)
    budget=CompatBudget.create(base,base/"artifacts/wm-input-compat-l",
        base/"artifacts/wm-input-compat-o/results",binding,storage=storage,io_hook=io_hook,limits={"seconds":seconds})
    sink,backend=ChunkSink(budget),FakeBackend()
    counters=dict(M_measurements=0,forecast_measurements=0,factory_calls=0)
    def measure(image,shape,goal):
        assert image.shape==(3,96,96) and shape in ("L","T") and goal is None
        counters["forecast_measurements" if counters["factory_calls"] else "M_measurements"]+=1
        return fake_measurement()
    def factory():
        assert counters["M_measurements"]==1056
        counters["factory_calls"]+=1
        return backend
    sink.record("binding.json",binding,maximum=128*1024)
    result=run_evaluation(inputs,budget=budget,backend_factory=factory,measure=measure,sink=sink,heartbeat=heartbeat)
    if heartbeat is not None: heartbeat.update(phase="export")
    save_summary(result,sink,budget,branches=inputs["branches"])
    assert result["M"]["passed"] and result["T"]["passed"]
    assert len(result["real_rows"])==1072 and sum(r["status"]=="completed" for r in result["real_rows"])==1056
    assert len(result["rows"])==2304 and sum(r["status"]=="completed" for r in result["rows"])==2268
    assert sum(r["status"]=="missing" for r in result["rows"])==36
    assert not any(r["status"]=="not_attempted" for r in result["rows"])
    assert counters==dict(M_measurements=1056,forecast_measurements=2268,factory_calls=1)
    assert (backend.sample_calls,backend.euler_calls,backend.close_calls)==(2268,6804,1)
    state=budget.snapshot()
    assert state["status"]=="complete"
    assert state["counts"]==dict(images=2268,euler_steps=6804,model_loads=1,T_images=576,L_images=1692)
    raw_outputs=[row for row in state["outputs"].values() if row["raw_bytes"]]
    assert len(raw_outputs)==2268 and all(row["status"]=="complete" for row in raw_outputs)
    raw_payload_bytes=sum(row["raw_bytes"] for row in raw_outputs)
    assert raw_payload_bytes==state["allocated_bytes"]["raw"]==250822656
    serialized_raw_bytes=0
    for row in raw_outputs:
        actual=Path(row["path"]).stat().st_size
        assert actual==row["bytes"] and row["raw_bytes"]<actual<=row["max_bytes"]
        serialized_raw_bytes+=actual
    usage=storage.usage()
    assert usage["output_bytes"]<=budget.limits["result_bytes"]
    assert usage["telemetry_bytes"]<=budget.limits["telemetry_bytes"]
    assert all(state["allocated_bytes"][kind]<=budget.limits[limit] for kind,limit in
               (("raw","raw_bytes"),("headers","header_bytes"),("metadata","metadata_bytes")))
    completed={r["key"]:r for r in result["rows"] if r["status"]=="completed"}
    paired=0
    for row in completed.values():
        if row["arm"]=="A" and row["mode"]=="AR": continue
        ref=completed[f"{row['anchor']}/{row['candidate']}/{row['seed']}/AR/A/H{row['horizon']}"]
        assert (row["rng_entry"],row["rng_exit"])==(ref["rng_entry"],ref["rng_exit"])
        paired+=1
    assert paired==1512
    for name in ("binding.json","M.json","ranking.json","forecast-diagnostics.json","case-plates.json",
                 "summary.json","planned-status.json","completion.json"):
        assert (budget.output_dir/name).is_file(),name
    assert not any(name in sys.modules for name in FORBIDDEN)
    return dict(kind="synthetic_complete_M_T_L_flow",passed=True,counters=counters,
        fake_sampler_calls=backend.sample_calls,fake_denoiser_calls=backend.euler_calls,
        fake_backend_close_calls=backend.close_calls,paired_rng_checks=paired,counts=state["counts"],
        stage_status=state["stage_status"],limits=budget.limits,
        completed_AR=sum(r["mode"]=="AR" for r in completed.values()),completed_TF=sum(r["mode"]=="TF" for r in completed.values()),
        raw_payload_bytes=raw_payload_bytes,serialized_raw_bytes=serialized_raw_bytes,
        verified_raw_outputs=len(raw_outputs),allocated_bytes=state["allocated_bytes"],
        real_model_loads=0,real_decoder_calls=0,
        saved_real_image_reads=0,physical_operations=0,production_claims=0,
        output_dir=str(budget.output_dir),ledger_dir=str(budget.ledger_dir),registry_dir=str(budget.registry_dir))

def _worker(config_path):
    import tdyssey.wm_compat_guard as guard
    import tdyssey.wm_compat_storage as storage_module
    import tdyssey.wm_compat_budget as budget_module
    config=json.loads(Path(config_path).read_text()); base=Path(config["base"])
    assert base.parent==ROOT/"artifacts" and base.name.startswith("wm-input-compat-iflow-")
    guard.REGISTRY=config["registry_relative"]
    handoff=guard.verify_guardian()  # Actual harmless child belongs to the native Job.
    storage=guard.from_handoff(handoff)
    emitted=threading.Event(); failures=[]
    def emitter():
        sequence=0
        while not emitted.wait(.03):
            sequence+=1; print(json.dumps(dict(fake_stdout_tick=sequence)),flush=True)
    thread=threading.Thread(target=emitter,daemon=True); thread.start()
    observations=dict(stage_audits=0,hot_tree_scans=0,new_file_hashes=0,barriers=[])
    seen=set(); depth=[0]
    original_scan,original_fingerprint=storage_module.scan_files,storage_module.fingerprint
    original_audit=CompatBudget.audit_storage
    original_fsync,original_copy=os.fsync,budget_module.shutil.copyfileobj
    def audit(owner):
        observations["stage_audits"]+=1; depth[0]+=1
        try: return original_audit(owner)
        finally: depth[0]-=1
    def scan(*args,**kwargs):
        assert depth[0] and kwargs.get("phase")=="budget.audit","Runtime global tree scan"
        return original_scan(*args,**kwargs)
    def fingerprint(path,*args,**kwargs):
        assert not storage.accounting_lock_held,"Hashing while quota lock held"
        if not depth[0]:
            assert kwargs.get("phase")=="publication.file",kwargs
            key=str(path); assert key not in seen,"Repeated hot-path immutable hash"
            seen.add(key); observations["new_file_hashes"]+=1
        return original_fingerprint(path,*args,**kwargs)
    done=set()
    def barrier(tag):
        if tag in done: return
        done.add(tag); assert not storage.accounting_lock_held
        started=time.monotonic()
        print(json.dumps(dict(event="stall",tag=tag,directory=handoff["directory"],policy=handoff["storage_policy"])),file=sys.stderr,flush=True)
        assert sys.stdin.readline().strip()=="release "+tag
        elapsed=time.monotonic()-started; assert elapsed>=1.2
        observations["barriers"].append(dict(tag=tag,seconds=elapsed,quota_lock_held=False))
    def hook(operation,path):
        assert not storage.accounting_lock_held,"Payload hook while global quota held"
        if operation=="chunk_fsync" and "real" in Path(path).parts: barrier("M_payload_fsync")
    def copy(source,destination,*args,**kwargs):
        assert not storage.accounting_lock_held,"Payload write while global quota held"
        if "predictions" in Path(destination.name).parts: barrier("forecast_payload_write")
        return original_copy(source,destination,*args,**kwargs)
    def fsync(fd):
        assert not storage.accounting_lock_held,"fsync while global quota held"
        return original_fsync(fd)
    initial=dict(phase="measurement",model_instances=0,model_loads=0,cuda_allocated_bytes=0,cuda_reserved_bytes=0,
                 **{name:0 for name in guard.FORBIDDEN_COUNTERS})
    try:
        with patch.object(CompatBudget,"audit_storage",audit),patch.object(storage_module,"scan_files",scan), \
                patch.object(budget_module,"scan_files",scan),patch.object(storage_module,"fingerprint",fingerprint), \
                patch.object(storage_module.os,"fsync",fsync),patch.object(budget_module.shutil,"copyfileobj",copy):
            with guard.WorkerHeartbeat(initial,interval_seconds=1.,handoff=handoff) as heartbeat:
                seconds=int(handoff["deadline_monotonic"]-time.monotonic())
                result=run_constructed_flow(base,storage,io_hook=hook,heartbeat=heartbeat,seconds=seconds)
                heartbeat.update(phase="complete")
        result.update(observations,worker_pid=os.getpid(),native_handoff_verified=True,quota=storage.usage(),forbidden_imports=[n for n in FORBIDDEN if n in sys.modules])
        assert observations["stage_audits"]==3 and len(done)==2
        print(json.dumps(dict(event="done",result=result)),file=sys.stderr,flush=True)
    finally:
        emitted.set(); thread.join(3); storage.close()

def _save_evidence(value):
    import uuid
    folder=ROOT/"artifacts/wm-input-compat-incremental-engineering-20261025"
    folder.mkdir(exist_ok=True)
    path=folder/("fake-flow-"+uuid.uuid4().hex+".json")
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False); stream.write("\n")
    return path

def _quiescent_journal_summary(ledger):
    """Read synthetic numeric metadata after the owned process tree has exited."""
    path=Path(ledger)/"journal.jsonl"
    if not path.exists(): return dict(available=False)
    payload=path.read_bytes(); kinds={}; prefixes={}; reserved=set(); completed=set(); gates={}; last=None
    lines=payload.splitlines(keepends=True); trailing=0
    for line in lines:
        if not line.endswith(b"\n"):
            trailing+=len(line); continue
        event=json.loads(line); kind=event["kind"]; data=event["data"]; last=event
        kinds[kind]=kinds.get(kind,0)+1
        if kind=="call_reserved": reserved.add(data["key"])
        if kind=="call_complete": completed.add(data["key"])
        if kind in ("M_gate","T_gate"): gates[kind]=data
        if kind=="output_complete":
            prefix=Path(data["name"]).parts[0]; prefixes[prefix]=prefixes.get(prefix,0)+1
    return dict(available=True,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
        event_counts=kinds,completed_output_prefixes=prefixes,durable_reserved_calls=len(reserved),
        durable_completed_calls=len(completed),unfinished_reserved_call_keys=sorted(reserved-completed),
        gates=gates,last_event=last,trailing_incomplete_bytes=trailing,
        actual_runtime_counts_are_not_inferred=True)

class IncrementalCompleteFlowChecks(unittest.TestCase):
    def test_capacity_oracle_does_not_hide_permission_failure(self):
        with self.assertRaises(AssertionError):
            _capacity_denial(StorageError("fake.lock","test.child_quota.lock_open",PermissionError(13,"denied")),
                             "test.child_quota")

    def test_postexit_journal_counts_do_not_reconstruct_partial_events(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=[dict(kind="call_reserved",data={"key":"first"}),
                  dict(kind="output_complete",data={"name":"real/fake.json"}),
                  dict(kind="call_complete",data={"key":"first"}),
                  dict(kind="call_reserved",data={"key":"pending"})]
            payload=b"".join((json.dumps(row)+"\n").encode() for row in rows)+b'{"kind":"call_complete"'
            (Path(directory)/"journal.jsonl").write_bytes(payload)
            result=_quiescent_journal_summary(directory)
            self.assertEqual((result["durable_reserved_calls"],result["durable_completed_calls"]),(2,1))
            self.assertEqual(result["unfinished_reserved_call_keys"],["pending"])
            self.assertEqual(result["completed_output_prefixes"],{"real":1})
            self.assertEqual(result["trailing_incomplete_bytes"],len(b'{"kind":"call_complete"'))
            self.assertTrue(result["actual_runtime_counts_are_not_inferred"])

    def test_cross_process_quota_admission_and_abandoned_charge(self):
        evidence=dict(kind="synthetic_cross_process_quota",passed=False,formal_runs=0,cases=[])
        try:
            for abandoned in (False,True):
                with tempfile.TemporaryDirectory() as directory:
                    base=Path(directory); result=base/"results"; telemetry=base/"telemetry"
                    result.mkdir(); telemetry.mkdir(); lock=telemetry/"storage.lock"; create_lock_file(lock)
                    policy=dict(lock_path=str(lock),result_roots=[str(result)],telemetry_roots=[str(telemetry)],
                        limits=dict(output_bytes=10,telemetry_bytes=65536))
                    storage=Coordinator(**policy)
                    config=base/"quota-config.json"
                    config.write_text(json.dumps(dict(policy=policy,path=str(result/"child.bin"),abandoned=abandoned)))
                    child=subprocess.Popen([sys.executable,"-B","-m","tests.test_wm_compat_incremental","--quota-child",str(config)],
                        cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    try:
                        self.assertEqual(child.stdout.readline().strip(),b"ready")
                        child.stdin.write(b"go\n"); child.stdin.flush()
                        parent_admitted=False; parent_denial=None
                        if not abandoned:
                            try:
                                ticket=storage.reserve_bytes(result/"parent.bin",8,phase="test.quota_race")
                                (result/"parent.bin").write_bytes(b"x"*8); storage.commit(ticket,8); parent_admitted=True
                            except StorageError as error: parent_denial=_capacity_denial(error,"test.quota_race")
                        answer=json.loads(child.stdout.readline())
                        self.assertEqual(child.wait(timeout=10),0,child.stderr.read().decode(errors="replace"))
                        usage=storage.usage()
                        self.assertNotEqual(answer["pid"],os.getpid())
                        if abandoned:
                            self.assertTrue(answer["admitted"])
                            self.assertEqual((usage["output_bytes"],usage["output_observed_bytes"]),(8,0))
                            with self.assertRaises(StorageError) as denied:
                                storage.reserve_bytes(result/"later.bin",3,phase="no-refund")
                            _capacity_denial(denied.exception,"no-refund")
                        else:
                            self.assertEqual(int(answer["admitted"])+int(parent_admitted),1)
                            self.assertEqual((usage["output_bytes"],usage["output_observed_bytes"]),(8,8))
                            self.assertEqual(len(list(result.iterdir())),1)
                        evidence["cases"].append(dict(abandoned=abandoned,parent_pid=os.getpid(),child=answer,
                            parent_denial=parent_denial,quota=usage))
                    finally:
                        if child.poll() is None: child.kill(); child.wait(timeout=10)
                        for stream in (child.stdin,child.stdout,child.stderr): stream.close()
                        storage.close()
            evidence["passed"]=True
        finally: _save_evidence(evidence)

    @unittest.skipUnless(sys.platform=="win32","Native harmless guardian requires Windows")
    def test_complete_flow_actual_guard_and_slow_io_without_global_lock(self):
        import tdyssey.wm_compat_guard as guard
        import tdyssey.wm_compat_storage as storage_module
        from tests.test_wm_compat_guard import resources
        evidence=dict(kind="incremental_native_synthetic_complete_flow",passed=False,formal_runs=0,
            real_model_loads=0,real_decoder_calls=0,saved_real_image_reads=0,physical_operations=0)
        started=time.monotonic(); folder=ROOT/"artifacts/wm-input-compat-incremental-engineering-20261025"
        folder.mkdir(exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/"artifacts",prefix="wm-input-compat-iflow-") as directory:
                base=Path(directory)
                self.assertEqual(base.resolve().parent,(ROOT/"artifacts").resolve())
                self.assertTrue(base.name.startswith("wm-input-compat-iflow-"))
                out=base/"artifacts/wm-input-compat-o"
                ledger=base/"artifacts/wm-input-compat-l"
                from tdyssey.wm_compat_identity import REGISTRY
                registry=base/"artifacts"/REGISTRY
                relative=str(registry.relative_to(ROOT/"artifacts"))
                config=base/"synthetic-worker.json"
                config.write_text(json.dumps(dict(base=str(base),registry_relative=relative)))
                children=[]; progress=[]; messages=[]; controls=[]; errors=[]; results=[]
                class Provider:
                    calls=0
                    def startup(self,path): return resources()
                    def start(self): pass
                    def runtime(self,pid,path): self.calls+=1; return resources()
                    def close(self): pass
                provider=Provider()
                def control(child):
                    try:
                        for raw in iter(child.stderr.readline,b""):
                            text=raw.decode("utf-8",errors="replace").strip(); messages.append(text)
                            if not text.startswith("{"): continue
                            message=json.loads(text)
                            if message.get("event")=="done": results.append(message["result"])
                            if message.get("event")!="stall": continue
                            tag=message["tag"]; entered=time.monotonic(); location=Path(message["directory"])
                            quota=None; release=False
                            try:
                                quota=Coordinator(**message["policy"])
                                status=location/"worker-status.json"; monitor=location/"monitor.jsonl"; stdout=location/"worker-output.log"
                                before=guard._read(status)["at_unix"]
                                before_monitor,before_stdout=monitor.stat().st_size,stdout.stat().st_size
                                then=time.monotonic(); path=location/(tag+"-probe.bin")
                                ticket=quota.reserve_bytes(path,1,phase="test.concurrent_probe")
                                assert not quota.accounting_lock_held
                                path.write_bytes(b"p"); quota.commit(ticket,1)
                                latency=time.monotonic()-then; self.assertLess(latency,.5)
                                time.sleep(max(0,1.25-(time.monotonic()-entered)))
                                self.assertGreater(guard._read(status)["at_unix"],before)
                                self.assertGreater(monitor.stat().st_size,before_monitor)
                                self.assertGreater(stdout.stat().st_size,before_stdout)
                                progress.append(dict(tag=tag,quota_admission_seconds=latency,heartbeat_advanced=True,
                                    monitor_advanced=True,stdout_advanced=True,parent_pid=os.getpid()))
                                release=True
                            finally:
                                child.stdin.write((("release " if release else "abort ")+tag+"\n").encode()); child.stdin.flush()
                                if quota is not None: quota.close()
                    except BaseException as error: errors.append(error)
                def launch(argv,**kwargs):
                    kwargs.update(stdin=subprocess.PIPE,stderr=subprocess.PIPE)
                    child=subprocess.Popen(argv,**kwargs); children.append(child)
                    thread=threading.Thread(target=control,args=(child,),daemon=True); thread.start(); controls.append(thread)
                    return child
                original_scan=storage_module.scan_files
                def scan(*args,**kwargs):
                    assert not children or children[0].poll() is not None,"Guardian scanned globally during worker IO"
                    return original_scan(*args,**kwargs)
                with patch.object(guard,"REGISTRY",relative),patch.object(storage_module,"scan_files",scan):
                    report=guard.supervise([str(Path(__file__).resolve()),"--fake-worker",str(config)],output_dir=out,
                        ledger_dir=ledger,lock_path=base/"one-model.lock",provider=provider,
                        telemetry_dirs=(ledger,registry),launcher_factory=launch,poll_seconds=.2,deadline_seconds=1800)
                for thread in controls: thread.join(10); self.assertFalse(thread.is_alive())
                for child in children:
                    for stream in (child.stdin,child.stderr): stream.close()
                evidence.update(guard_report=report,worker_messages=messages[-30:],cross_process_progress=progress)
                try: evidence["quiescent_synthetic_journal"]=_quiescent_journal_summary(ledger)
                except BaseException as error: evidence["quiescent_synthetic_journal_error"]=type(error).__name__+": "+str(error)
                if errors: raise errors[0]
                self.assertEqual(report["status"],"exited",str(report)+"\n"+"\n".join(messages[-20:]))
                self.assertTrue(report["job_empty_verified_before_close"])
                self.assertEqual(len(results),1,"\n".join(messages[-20:]))
                self.assertEqual(len(progress),2)
                result=results[0]
                self.assertEqual(result["worker_pid"],report["actual_worker_pid"])
                self.assertTrue(result["native_handoff_verified"])
                self.assertNotEqual(result["worker_pid"],os.getpid())
                self.assertEqual((result["completed_AR"],result["completed_TF"]),(1512,756))
                self.assertEqual(result["raw_payload_bytes"],250822656)
                evidence.update(passed=True,result=result,guard_report=report,cross_process_progress=progress,
                    provider_calls=provider.calls,native_job_exercised=True,resources_injected=True,
                    forbidden_runtime_modules_imported=[n for n in FORBIDDEN if n in sys.modules])
        except BaseException as error:
            evidence["passed"]=False
            evidence["error"]=type(error).__name__+": "+str(error); raise
        finally:
            evidence["wall_seconds"]=time.monotonic()-started
            print("synthetic full-flow evidence: "+str(_save_evidence(evidence)),flush=True)


def _quota_child(config_path):
    config=json.loads(Path(config_path).read_text()); storage=Coordinator(**config["policy"])
    print("ready",flush=True); assert sys.stdin.readline().strip()=="go"
    admitted=False; denial=None
    try:
        ticket=storage.reserve_bytes(Path(config["path"]),8,phase="test.child_quota")
        admitted=True
        if not config["abandoned"]:
            Path(config["path"]).write_bytes(b"x"*8); storage.commit(ticket,8)
    except StorageError as error: denial=_capacity_denial(error,"test.child_quota")
    print(json.dumps(dict(pid=os.getpid(),admitted=admitted,denial=denial)),flush=True)
    if config["abandoned"]: os._exit(0)
    storage.close()

if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--fake-worker": _worker(sys.argv[2])
    elif len(sys.argv)==3 and sys.argv[1]=="--quota-child": _quota_child(sys.argv[2])
    else: unittest.main()
