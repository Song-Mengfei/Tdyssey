"""Bounded Linux native-GPC GPU probes and independent training stage chains.

The parent process never imports Torch. GPU probes use disposable child processes,
reserve two Adam moment buffers, and perform zero optimizer updates. No distributed
training or shared model/optimizer state is used.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATCHES = (8, 16, 32, 64)
EFFECTIVE_BATCH = 128


def write_json(path, value):
    from tdyssey.native_rgb_training import atomic_json
    atomic_json(path, value)


class ProcessHandle:
    """Own a process group and drain output into a size-bounded log."""
    def __init__(self, command, cwd, env, log_path, max_log_bytes=4*1024**2):
        self.command = list(command)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.started_monotonic, self.started_unix = time.monotonic(), time.time()
        self.log_stats = dict(total_bytes=0, retained_bytes=0, truncated=False)
        self.stop_drain = threading.Event()
        self.finished = None
        process_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        process_env.update(env or {})
        # Open synchronously so a bad log destination cannot leave an unowned
        # child behind. Existing evidence is never silently truncated.
        self.log_stream = self.log_path.open("xb")
        try:
            self.process = subprocess.Popen(self.command, cwd=cwd, env=process_env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, start_new_session=os.name != "nt")
        except BaseException:
            self.log_stream.close()
            raise
        try:
            os.set_blocking(self.process.stdout.fileno(), False)
        except BaseException:
            self.terminate_owned()
            self.process.stdout.close()
            self.log_stream.close()
            raise

        def drain():
            with self.log_stream as target:
                while True:
                    try:
                        block = os.read(self.process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        if self.stop_drain.wait(.025):
                            break
                        continue
                    except (OSError, ValueError):
                        break
                    if not block:
                        break
                    self.log_stats["total_bytes"] += len(block)
                    remaining = max(0, max_log_bytes - self.log_stats["retained_bytes"])
                    if remaining:
                        target.write(block[:remaining])
                        self.log_stats["retained_bytes"] += min(remaining, len(block))
                    if len(block) > remaining:
                        self.log_stats["truncated"] = True
        self.thread = threading.Thread(target=drain, name="bounded-child-log", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.terminate_owned()
            self.process.stdout.close()
            self.log_stream.close()
            raise

    def terminate_owned(self):
        """Only the process/group created by this handle may be terminated."""
        if self.process.poll() is None:
            try:
                if os.name == "nt":
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    self.process.kill()
                else:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.process.wait(timeout=1)

    def finish(self):
        if self.finished is not None:
            return dict(self.finished)
        if self.process.poll() is None:
            raise RuntimeError("Cannot finish a live owned child; terminate it first")
        self.thread.join(timeout=.5)
        if self.thread.is_alive():
            # A descendant may have inherited stdout. Nonblocking reads plus
            # this event bound log cleanup without waiting on an inherited pipe.
            self.stop_drain.set()
            self.thread.join(timeout=.5)
        if self.thread.is_alive():
            raise RuntimeError("Bounded child log drainer did not stop")
        if self.process.stdout is not None:
            self.process.stdout.close()
        self.finished = dict(pid=self.process.pid, returncode=self.process.returncode,
            started_unix=self.started_unix, finished_unix=time.time(),
            wall_seconds=time.monotonic()-self.started_monotonic,
            log=str(self.log_path), **self.log_stats)
        return dict(self.finished)


def _positive_finite(value, name):
    if isinstance(value, bool):
        raise ValueError(name+" must be a positive finite duration")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(name+" must be a positive finite duration")
    return number


def _validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Provide a nonempty list of independent jobs")
    seen = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError("Each job must be a metadata object")
        identifier = str(job.get("id", f"job-{index}"))
        if identifier in seen or not identifier or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in identifier):
            raise ValueError("Job IDs must be distinct simple names")
        seen.add(identifier)
        stages = job.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("Each job needs a nonempty list of argv stages")
        for command in stages:
            if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
                raise ValueError("Commands must be argv string lists; shell strings are unsupported")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in job.get("env", {}).items()):
            raise ValueError("Environment overrides must be string pairs")
        if "timeout_seconds" in job:
            _positive_finite(job["timeout_seconds"], "job timeout_seconds")


def run_jobs(jobs, concurrency=1, output=None, job_timeout_seconds=172800,
             max_log_bytes=4*1024**2, fail_fast=True, deadline=None):
    """Run independent per-seed stage chains, with bounded logs and deadlines.

    Each job is {id, stages:[[argv...],...], cwd?, env?, timeout_seconds?}.
    A failing stage cancels pending jobs and terminates only owned running groups.
    The caller must provide separate output/ledger/model paths in each command;
    this launcher never merges training state or writes the shared dataset.
    """
    job_timeout_seconds = _positive_finite(job_timeout_seconds, "job_timeout_seconds")
    if concurrency not in (1, 2) or type(max_log_bytes) is not int or max_log_bytes < 1:
        raise ValueError("Choose concurrency1/2 and positive finite bounds")
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError("An overall deadline must be a finite monotonic timestamp")
    _validate_jobs(jobs)
    folder = Path(output or "artifacts/native-rgb-jobs").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    states = [dict(id=str(job.get("id", f"job-{index}")), status="pending", next_stage=0,
                   stages=[], started_monotonic=None) for index, job in enumerate(jobs)]
    pending = deque(range(len(jobs)))
    active = {}
    failure = None
    max_active = 0
    try:
        while pending or active:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                failure = "overall_deadline"
            while pending and len(active) < concurrency and failure is None:
                index = pending.popleft()
                job, state = jobs[index], states[index]
                if state["started_monotonic"] is None:
                    state["started_monotonic"] = now
                stage = state["next_stage"]
                try:
                    handle = ProcessHandle(job["stages"][stage], job.get("cwd", str(ROOT)), job.get("env"),
                        folder/state["id"]/f"stage-{stage:02d}.log", max_log_bytes)
                except Exception as error:
                    state["status"] = "failed"
                    state["error"] = type(error).__name__ + ": " + str(error)
                    failure = "launch_failure:" + state["id"]
                    break
                state["status"] = "running"
                active[index] = handle
                max_active = max(max_active, len(active))
            for index, handle in list(active.items()):
                state, job = states[index], jobs[index]
                timeout = float(job.get("timeout_seconds", job_timeout_seconds))
                timed_out = time.monotonic() - state["started_monotonic"] >= timeout
                if timed_out and handle.process.poll() is None:
                    handle.terminate_owned()
                if handle.process.poll() is None:
                    continue
                detail = handle.finish()
                detail.update(stage=state["next_stage"], timed_out=timed_out)
                state["stages"].append(detail)
                del active[index]
                if timed_out or detail["returncode"] != 0:
                    state["status"] = "timed_out" if timed_out else "failed"
                    if fail_fast:
                        failure = state["status"] + ":" + state["id"]
                else:
                    state["next_stage"] += 1
                    if state["next_stage"] < len(job["stages"]):
                        state["status"] = "pending_next_stage"
                        pending.appendleft(index)
                    else:
                        state["status"] = "complete"
            if failure is not None:
                for index, handle in list(active.items()):
                    handle.terminate_owned()
                    detail = handle.finish()
                    detail.update(stage=states[index]["next_stage"], cancelled=True)
                    states[index]["stages"].append(detail)
                    states[index]["status"] = "cancelled_running"
                active.clear()
                while pending:
                    index = pending.popleft()
                    states[index]["status"] = "cancelled_pending"
                break
            if pending or active:
                time.sleep(.05)
    finally:
        for handle in active.values():
            handle.terminate_owned()
            handle.finish()
    for state in states:
        state.pop("started_monotonic", None)
    report = dict(kind="native-gpc-independent-jobs.v1", status="complete" if all(state["status"] == "complete" for state in states) else "failed",
        concurrency=concurrency, max_active=max_active, jobs=states, failure=failure,
        wall_seconds=time.monotonic()-started, shared_optimizer=False, distributed_training=False)
    write_json(folder/"jobs-result.json", report)
    return report


def safe_candidate(result, memory_fraction=.85):
    return (result.get("status") == "complete" and result.get("optimizer_updates") == 0
            and result.get("samples_per_second", 0) > 0
            and result.get("peak_reserved_bytes", float("inf")) <= result.get("gpu_total_bytes", 0)*memory_fraction
            and result.get("minimum_global_free_bytes", 0) >= result.get("gpu_total_bytes", float("inf"))*(1-memory_fraction))


def choose_recommendation(candidates, concurrent=None, memory_fraction=.85):
    safe = [row for row in candidates if safe_candidate(row, memory_fraction)]
    if not safe:
        return dict(status="no_safe_batch", microbatch=None, effective_batch=EFFECTIVE_BATCH,
                    concurrency=1, reason="No completed candidate retained the global memory margin")
    winner = max(safe, key=lambda row: row["samples_per_second"])
    concurrency, reason, improvement = 1, "Single process is the safe default", None
    identity = winner.get("dataset_identity") or {}
    operations = winner.get("measured_operations") or {}
    real_pipeline = (winner.get("measurement_scope") == "audited_train_end_to_end"
        and winner.get("effective_batch") == EFFECTIVE_BATCH and identity.get("split") == "train"
        and all(isinstance(identity.get(key), str) and len(identity[key]) == 64
                for key in ("manifest_sha256", "physics_audit_sha256"))
        and bool(winner.get("render_configuration"))
        and all(operations.get(key) is True for key in ("data_read","rendering","host_to_device","forward","backward")))
    if concurrent and concurrent.get("status") == "complete" and real_pipeline:
        children = concurrent["children"]
        combined_peak = sum(row["peak_reserved_bytes"] for row in children)
        safe_pair = (len(children) == 2 and all(safe_candidate(row, memory_fraction) for row in children)
                     and combined_peak <= winner["gpu_initial_free_bytes"]*memory_fraction
                     and all(row.get("measurement_scope") == "audited_train_end_to_end"
                             and row.get("dataset_identity") == winner.get("dataset_identity")
                             and row.get("render_configuration") == winner.get("render_configuration")
                             and row.get("effective_batch") == EFFECTIVE_BATCH for row in children)
                     and concurrent.get("measurement_overlap_seconds", 0) > 0)
        improvement = concurrent["samples_per_second"] / winner["samples_per_second"] - 1
        if safe_pair and improvement > .05:
            concurrency, reason = 2, "Two independent processes improved audited train read/render/H2D/F-B aggregate throughput by more than5% with memory margin"
        else:
            reason = "Concurrent probe lacked a safe memory margin or more than5% throughput improvement"
    if not real_pipeline:
        reason = "GPU-compute-only evidence: real data/render total throughput unmeasured; concurrency is fixed at1"
    return dict(status="complete", microbatch=winner["batch"], effective_batch=EFFECTIVE_BATCH,
        concurrency=concurrency, reason=reason, concurrent_throughput_improvement=improvement,
        chosen_single_samples_per_second=winner["samples_per_second"], optimizer_updates=0,
        model="original full GPC96 phase-two residual objective", image_size=96,
        memory_fraction=memory_fraction, measurement_scope=winner.get("measurement_scope", "gpu_compute_only"),
        dataset_identity=winner.get("dataset_identity"), render_configuration=winner.get("render_configuration"),
        end_to_end_throughput_measured=real_pipeline)


def effective_pipeline_batch(order, renderer, microbatch, device, backward):
    """One bounded, fresh effective128 batch, including lazy data rendering.

    The enclosing timer starts before this function. It advances the window order
    exactly once, prefetches only this batch and never requests a final split.
    This orchestration is CPU-mockable without importing Torch.
    """
    indices = order.take(EFFECTIVE_BATCH)
    samples = microbatches = 0
    for batch in renderer.batches(indices, microbatch, device):
        count = int(batch["image"].shape[0])
        if count < 1 or count > microbatch:
            raise ValueError("Invalid current-batch renderer output")
        backward(batch, count/EFFECTIVE_BATCH)
        samples += count
        microbatches += 1
    if samples != EFFECTIVE_BATCH:
        raise ValueError("Renderer did not supply the full effective128 batch")
    return dict(samples=samples, microbatches=microbatches,
                selected_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest())


def gpu_child(args):
    """Disposable Linux process. This is the only function importing Torch."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("GPU probes run only on the user-managed Linux server")
    import torch
    from tdyssey.native_rgb_training import construct_model, BatchRenderer, WindowOrder, sha256
    from tdyssey.raw_world_model_training import make_generator, supervised_denoising_loss
    real_pipeline = args.data is not None
    result = dict(status="failed", batch=args.batch, optimizer_updates=0, warmup_steps=1, measured_steps=2,
        effective_batch=EFFECTIVE_BATCH if real_pipeline else args.batch,
        measurement_scope="audited_train_end_to_end" if real_pipeline else "gpu_compute_only",
        measured_operations=dict(data_read=real_pipeline, rendering=real_pipeline,
            host_to_device=real_pipeline, forward=True, backward=True, optimizer_step=False),
        final_split_windows_read=False)
    started = time.monotonic()
    renderer = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda:0")
        initial_free, total = torch.cuda.mem_get_info(device)
        model, _ = construct_model(args.gpc_root, args.checkpoint, device="cuda", trainable=True)
        # These are allocation/touch probes, not an optimizer: .step() is absent.
        moments = [torch.zeros_like(parameter) for parameter in model.parameters() for _ in range(2)]
        moment_bytes = sum(tensor.numel()*tensor.element_size() for tensor in moments)
        if real_pipeline:
            from tdyssey.native_rgb_data import NativeRGBDataset, TRAIN_LETTERS
            dataset = NativeRGBDataset(args.data, "train", source_root=args.physics_root,
                                       cache_bytes=args.cache_bytes, require_gate=True)
            if (not dataset.entries or any(entry["split"] != "train" or entry["shape"] not in TRAIN_LETTERS
                                           for entry in dataset.entries)):
                raise ValueError("Probe windows must come only from the 18 audited training letters")
            if not dataset.audit.get("physics_collection_gate_passed"):
                raise ValueError("An intact passed physical collection gate is required")
            order = WindowOrder(dataset, args.seed)
            renderer = BatchRenderer(dataset, workers=args.render_workers,
                                     cache_bytes=args.cache_bytes, max_pending=EFFECTIVE_BATCH)
            result.update(dataset_identity=dict(
                manifest_sha256=sha256(Path(args.data)/"manifest.json"),
                physics_audit_sha256=sha256(Path(args.data)/"physics-audit.json"),
                split="train", letters=sorted({entry["shape"] for entry in dataset.entries})),
                render_configuration=dict(workers=args.render_workers, total_cache_bytes=args.cache_bytes,
                    max_pending=EFFECTIVE_BATCH, source="actual_NativeRGBDataset_and_BatchRenderer"))
        else:
            image = torch.rand(args.batch, 5, 3, 96, 96, device=device)
            action = torch.rand(args.batch, 4, 2, device=device)*2-1
        generator = make_generator(args.seed, device)
        minimum_free = [torch.cuda.mem_get_info(device)[0]]

        def step():
            model.zero_grad(set_to_none=True)
            losses = []
            def backward(batch, weight):
                with torch.autocast("cuda", dtype=torch.float16, enabled=not args.no_amp):
                    loss, _ = supervised_denoising_loss(model, batch, generator=generator)
                (loss*(128.*weight)).backward()
                if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite native residual loss")
                losses.append(float(loss)*weight)
            if real_pipeline:
                current = effective_pipeline_batch(order, renderer, args.batch, device, backward)
                result.setdefault("effective_batch_windows", []).append(current)
            else:
                backward(dict(image=image, action=action), 1.)
            torch.cuda.synchronize(device)
            minimum_free.append(torch.cuda.mem_get_info(device)[0])
            if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                   for parameter in model.parameters()):
                raise FloatingPointError("Missing/nonfinite native full-model gradient")
            return sum(losses)

        step()
        if args.ready:
            write_json(args.ready, dict(pid=os.getpid(), ready_unix=time.time()))
            wait_deadline = time.monotonic()+args.child_seconds
            while not Path(args.start_gate).exists():
                if time.monotonic() >= wait_deadline:
                    raise TimeoutError("Concurrent start barrier expired")
                time.sleep(.025)
            gate = json.loads(Path(args.start_gate).read_text())
            while time.time() < gate["start_unix"]:
                time.sleep(.005)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        measured_start_unix, measured_start = time.time(), time.monotonic()
        losses = [step() for _ in range(2)]
        elapsed = time.monotonic()-measured_start
        result.update(status="complete", pid=os.getpid(), parameter_count=sum(p.numel() for p in model.parameters()),
            adam_moment_reserved_bytes=moment_bytes, gpu_name=torch.cuda.get_device_name(device),
            gpu_initial_free_bytes=initial_free, gpu_total_bytes=total,
            minimum_global_free_bytes=min(minimum_free), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            samples_per_second=2*result["effective_batch"]/elapsed,
            measured_seconds=elapsed, measured_started_unix=measured_start_unix,
            measured_finished_unix=time.time(), loss=losses, amp=not args.no_amp)
    except BaseException as error:
        result.update(status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
                      error=type(error).__name__+": "+str(error), traceback=traceback.format_exc()[-6000:])
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception as error:
                result.update(status="failed", cleanup_error=type(error).__name__+": "+str(error))
        result["wall_seconds"] = time.monotonic()-started
        write_json(args.output, result)
    return 0 if result["status"] == "complete" else 2


def _child_command(args, batch, result, ready=None, gate=None):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "_gpu-child", "--batch", str(batch),
        "--gpc-root", str(Path(args.gpc_root).resolve()), "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--output", str(result), "--child-seconds", str(args.per_child_seconds)]
    command.extend(["--render-workers", str(args.render_workers), "--cache-bytes", str(args.cache_bytes)])
    if args.data is not None:
        command.extend(["--data", str(Path(args.data).resolve()),
                        "--physics-root", str(Path(args.physics_root).resolve())])
    if args.no_amp:
        command.append("--no-amp")
    if ready is not None:
        command.extend(["--ready", str(ready), "--start-gate", str(gate)])
    return command


def _run_probe_group(args, batch, folder, count, overall_deadline):
    folder.mkdir(parents=True, exist_ok=False)
    gate = folder/"start-gate.json"
    handles = []
    files = [folder/f"child-{index}.json" for index in range(count)]
    ready = [folder/f"ready-{index}.json" for index in range(count)]
    # Reserve finite process/log cleanup inside both the group and total bound.
    deadline = min(overall_deadline, time.monotonic()+args.per_child_seconds)-5
    failure = None
    exception = None
    try:
        for index in range(count):
            result = files[index]
            marker = ready[index]
            command = _child_command(args, batch, result, marker if count == 2 else None, gate if count == 2 else None)
            handles.append(ProcessHandle(command, str(ROOT), {}, folder/f"child-{index}.log"))
        while any(handle.process.poll() is None for handle in handles):
            if time.monotonic() >= deadline:
                failure = "deadline"
                break
            if count == 2 and not gate.exists() and all(path.exists() for path in ready):
                write_json(gate, dict(start_unix=time.time()+.15))
            if any(handle.process.poll() not in (None, 0) for handle in handles):
                failure = "child_failure"
                break
            time.sleep(.025)
    except BaseException as error:
        exception = error
        failure = "parent_exception:"+type(error).__name__+": "+str(error)
    finally:
        for handle in handles:
            if handle.process.poll() is None:
                handle.terminate_owned()
        process_results = [handle.finish() for handle in handles]
    children = []
    for index, path in enumerate(files):
        if path.exists():
            children.append(json.loads(path.read_text()))
        else:
            children.append(dict(status="timed_out" if failure == "deadline" else "failed",
                                 batch=batch, optimizer_updates=0, error=failure or "child_result_missing"))
    result = dict(status="complete" if failure is None and all(row["status"] == "complete" for row in children) else "failed",
                  batch=batch, children=children, processes=process_results, failure=failure)
    if result["status"] == "complete":
        start = min(row["measured_started_unix"] for row in children)
        finish = max(row["measured_finished_unix"] for row in children)
        overlap = min(row["measured_finished_unix"] for row in children)-max(row["measured_started_unix"] for row in children)
        result.update(samples_per_second=sum(row.get("effective_batch",row["batch"])*row["measured_steps"] for row in children)/(finish-start),
                      measurement_overlap_seconds=max(0, overlap))
    write_json(folder/"group-result.json", result)
    if exception is not None and not isinstance(exception, Exception):
        raise exception
    return result


def probe(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run the actual GPU probe manually on the Linux server; use selftest locally")
    if not 0 < args.per_child_seconds <= 180 or not 0 < args.total_seconds <= 900 or not .5 <= args.memory_fraction <= .95:
        raise ValueError("Probe bounds: child<=180s, total<=900s, memory fraction0.5..0.95")
    if (args.data is None) != (args.physics_root is None):
        raise ValueError("Provide --data and --physics-root together for real pipeline measurements")
    if type(args.render_workers) is not int or not 1 <= args.render_workers <= 16 or args.cache_bytes < 0:
        raise ValueError("Renderer workers1..16 and nonnegative bounded cache are required")
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    started, candidates, concurrent = time.monotonic(), [], None
    deadline = started+args.total_seconds
    for batch in BATCHES:
        if time.monotonic() >= deadline:
            break
        group = _run_probe_group(args, batch, folder/f"batch-{batch}", 1, deadline)
        candidate = group["children"][0]
        candidates.append(candidate)
    preliminary = choose_recommendation(candidates, memory_fraction=args.memory_fraction)
    if preliminary["status"] == "complete" and args.data is not None and time.monotonic() < deadline:
        concurrent = _run_probe_group(args, preliminary["microbatch"], folder/"two-process-probe", 2, deadline)
    recommendation = choose_recommendation(candidates, concurrent, args.memory_fraction)
    report = dict(kind="native-gpc-server-probe.v1", recommendation=recommendation, candidates=candidates,
                  concurrent_probe=concurrent, wall_seconds=time.monotonic()-started,
                  per_child_seconds=args.per_child_seconds, total_seconds=args.total_seconds,
                  optimizer_updates=0, actual_training_executed=False)
    write_json(folder/"probe-result.json", report)
    write_json(folder/"recommendation.json", recommendation)
    return report


def selftest(output):
    """Real CPU subprocesses only; never imports or initializes Torch/CUDA."""
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)

    def command(marker, duration=.05, code=0, emit_bytes=0):
        return [sys.executable, "-u", str(Path(__file__).resolve()), "_cpu-task", "--marker", str(marker),
                "--seconds", str(duration), "--exit-code", str(code), "--emit-bytes", str(emit_bytes)]
    jobs = [dict(id=f"seed-{index}", stages=[command(folder/f"seed-{index}-{stage}.json") for stage in range(2)])
            for index in range(2)]
    normal = run_jobs(jobs, 2, folder/"parallel", job_timeout_seconds=10)
    markers = [json.loads((folder/f"seed-{index}-{stage}.json").read_text()) for index in range(2) for stage in range(2)]
    assert normal["status"] == "complete" and normal["max_active"] == 2
    assert len({marker["pid"] for marker in markers}) == 4
    for index in range(2):
        assert markers[index*2]["finished_unix"] <= markers[index*2+1]["started_unix"]
    failure = run_jobs([dict(id="fail", stages=[command(folder/"failed.json", code=7)]),
                        dict(id="pending", stages=[command(folder/"must-not-run.json")])],
                       1, folder/"failure", job_timeout_seconds=10)
    assert failure["jobs"][0]["stages"][0]["returncode"] == 7
    assert failure["jobs"][1]["status"] == "cancelled_pending" and not (folder/"must-not-run.json").exists()
    timeout = run_jobs([dict(id="timeout", stages=[command(folder/"timeout.json", duration=3)], timeout_seconds=.15)],
                       1, folder/"timeout", job_timeout_seconds=10)
    assert timeout["jobs"][0]["status"] == "timed_out"
    bounded = run_jobs([dict(id="log", stages=[command(folder/"log.json", emit_bytes=8192)])],
                       1, folder/"bounded", job_timeout_seconds=10, max_log_bytes=256)
    assert bounded["jobs"][0]["stages"][0]["retained_bytes"] == 256
    assert bounded["jobs"][0]["stages"][0]["truncated"]
    startup = run_jobs([dict(id="owned", stages=[command(folder/"owned-must-stop.json", duration=3)]),
                        dict(id="bad-executable", stages=[[str(folder/"does-not-exist-python")]])],
                       2, folder/"launch-failure", job_timeout_seconds=10)
    assert startup["jobs"][0]["status"] == "cancelled_running"
    assert startup["jobs"][0]["stages"][0]["returncode"] is not None
    assert not (folder/"owned-must-stop.json").exists()
    for invalid in (float("inf"), float("nan"), 0, -1):
        try:
            run_jobs(jobs, output=folder/"invalid-bound", job_timeout_seconds=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("A nonfinite/nonpositive job bound was accepted")
    from unittest.mock import patch
    created = []
    original_handle = ProcessHandle

    def partial_launch(*args, **kwargs):
        if created:
            raise OSError("injected second-child launch failure")
        handle = original_handle(*args, **kwargs)
        created.append(handle)
        return handle

    def cpu_command(args, batch, result, ready=None, gate=None):
        return command(result, duration=3)

    with patch.object(sys.modules[__name__], "ProcessHandle", side_effect=partial_launch), \
         patch.object(sys.modules[__name__], "_child_command", side_effect=cpu_command):
        group = _run_probe_group(argparse.Namespace(per_child_seconds=10), 8, folder/"partial-group", 2,
                                 time.monotonic()+10)
    assert group["status"] == "failed" and len(group["children"]) == 2
    assert len(created) == 1 and created[0].process.poll() is not None
    # A finite inherited stdout lifetime must not block finish indefinitely.
    inherited = ProcessHandle([sys.executable, "-u", "-c",
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(.8)']); print('parent exited')"],
        str(ROOT), {}, folder/"inherited-stdout.log")
    inherited.process.wait(timeout=10)
    finished_started = time.monotonic()
    inherited.finish()
    assert time.monotonic()-finished_started < 2
    time.sleep(.9)  # Allow this selftest's finite CPU descendant to finish.
    base = dict(status="complete", optimizer_updates=0, gpu_total_bytes=1000, gpu_initial_free_bytes=950,
                minimum_global_free_bytes=500, peak_reserved_bytes=100,
                measurement_scope="audited_train_end_to_end", effective_batch=128,
                dataset_identity=dict(manifest_sha256="a"*64, physics_audit_sha256="b"*64, split="train"),
                render_configuration=dict(workers=4, total_cache_bytes=1024, max_pending=128),
                measured_operations=dict(data_read=True, rendering=True, host_to_device=True, forward=True, backward=True))
    safe = [dict(base, batch=8, samples_per_second=10), dict(base, batch=16, samples_per_second=20)]
    pair = dict(status="complete", children=[safe[1], safe[1]], samples_per_second=21.2, measurement_overlap_seconds=1)
    assert choose_recommendation(safe)["microbatch"] == 16
    assert choose_recommendation(safe, pair)["concurrency"] == 2
    compute = [dict(row, measurement_scope="gpu_compute_only") for row in safe]
    assert choose_recommendation(compute, pair)["concurrency"] == 1
    bad_pair = dict(pair, children=[dict(safe[1], dataset_identity=dict(base["dataset_identity"], manifest_sha256="c"*64)), safe[1]])
    assert choose_recommendation(safe, bad_pair)["concurrency"] == 1
    pair["samples_per_second"] = 20.8
    assert choose_recommendation(safe, pair)["concurrency"] == 1
    assert choose_recommendation([dict(base, batch=8, samples_per_second=10, minimum_global_free_bytes=1)])["status"] == "no_safe_batch"
    class FakeOrder:
        cursor = 0
        def take(self, count):
            result = list(range(self.cursor, self.cursor+count))
            self.cursor += count
            return result
    class FakeImage:
        def __init__(self, batch):
            self.shape = (batch,5,3,96,96)
    class FakeRenderer:
        selected = []
        def batches(self, indices, microbatch, device):
            self.selected.append(list(indices))
            time.sleep(.01)  # This preprocessing is inside the enclosing timer.
            for start in range(0,len(indices),microbatch):
                yield dict(image=FakeImage(min(microbatch,len(indices)-start)))
    weights, renderer, order = [], FakeRenderer(), FakeOrder()
    timer = time.monotonic()
    first = effective_pipeline_batch(order,renderer,16,"mock-device",lambda batch,weight:weights.append(weight))
    assert time.monotonic()-timer >= .009
    second = effective_pipeline_batch(order,renderer,16,"mock-device",lambda batch,weight:None)
    assert first["samples"] == 128 and first["microbatches"] == 8 and sum(weights) == 1
    assert not set(renderer.selected[0]).intersection(renderer.selected[1])
    assert first["selected_indices_sha256"] != second["selected_indices_sha256"]
    namespace = argparse.Namespace(gpc_root=ROOT, checkpoint=ROOT/"unused-original.pth",per_child_seconds=180,
        render_workers=4,cache_bytes=1024,data=ROOT/"unused-audited-data",physics_root=ROOT/"unused-physics",no_amp=False)
    forwarded = _child_command(namespace,16,folder/"unused-child.json")
    assert "--data" in forwarded and "--physics-root" in forwarded and "--render-workers" in forwarded and "--cache-bytes" in forwarded
    assert "torch" not in sys.modules
    report = dict(status="complete", independent_processes=len({marker["pid"] for marker in markers}),
                  parallel_stagechains=True, failure_code7=True, pending_cancelled=True, deadline_enforced=True,
                  logs_bounded=True, finite_bounds=True, partial_launch_cleanup=True,
                  inherited_stdout_cleanup_bounded=True, throughput_memory_selection=True,
                  real_pipeline_argument_forwarding=True, effective128_render_accumulation_mock=True,
                  compute_only_never_selects_concurrency2=True, mismatched_data_never_selects_concurrency2=True,
                  cuda_initialized=False, optimizer_updates=0)
    write_json(folder/"selftest-result.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    actual = modes.add_parser("probe", help="Run bounded original-GPC GPU probes on Linux")
    actual.add_argument("--gpc-root", type=Path, required=True)
    actual.add_argument("--checkpoint", type=Path, required=True)
    actual.add_argument("--output", type=Path, required=True)
    actual.add_argument("--per-child-seconds", type=float, default=180)
    actual.add_argument("--total-seconds", type=float, default=900)
    actual.add_argument("--memory-fraction", type=float, default=.85)
    actual.add_argument("--no-amp", action="store_true")
    actual.add_argument("--data", type=Path, help="Audited dataset; omission restricts evidence to GPU compute")
    actual.add_argument("--physics-root", type=Path)
    actual.add_argument("--render-workers", type=int, default=4)
    actual.add_argument("--cache-bytes", type=int, default=256*1024**2)
    jobs = modes.add_parser("run", help="Launch independent seed stage chains")
    jobs.add_argument("--jobs", type=Path, required=True)
    jobs.add_argument("--output", type=Path, required=True)
    jobs.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    jobs.add_argument("--job-timeout-seconds", type=float, default=172800)
    cpu = modes.add_parser("selftest", help="CPU-only real subprocess isolation/deadline tests")
    cpu.add_argument("--output", type=Path, required=True)
    child = modes.add_parser("_gpu-child", help=argparse.SUPPRESS)
    child.add_argument("--gpc-root", type=Path, required=True)
    child.add_argument("--checkpoint", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--batch", type=int, choices=BATCHES, required=True)
    child.add_argument("--seed", type=int, default=5478)
    child.add_argument("--child-seconds", type=float, default=180)
    child.add_argument("--ready", type=Path)
    child.add_argument("--start-gate", type=Path)
    child.add_argument("--no-amp", action="store_true")
    child.add_argument("--data", type=Path)
    child.add_argument("--physics-root", type=Path)
    child.add_argument("--render-workers", type=int, default=4)
    child.add_argument("--cache-bytes", type=int, default=256*1024**2)
    task = modes.add_parser("_cpu-task", help=argparse.SUPPRESS)
    task.add_argument("--marker", type=Path, required=True)
    task.add_argument("--seconds", type=float, default=.05)
    task.add_argument("--exit-code", type=int, default=0)
    task.add_argument("--emit-bytes", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode in ("probe", "_gpu-child"):
        if (args.data is None) != (args.physics_root is None):
            parser.error("--data and --physics-root must be supplied together")
        if not 1 <= args.render_workers <= 16 or not 0 <= args.cache_bytes <= 1024**3:
            parser.error("render-workers must be1..16 and cache-bytes0..1GiB")
    if args.mode == "_gpu-child":
        return gpu_child(args)
    if args.mode == "_cpu-task":
        started = time.time()
        if args.emit_bytes:
            sys.stdout.write("x"*args.emit_bytes)
            sys.stdout.flush()
        time.sleep(args.seconds)
        write_json(args.marker, dict(pid=os.getpid(), started_unix=started, finished_unix=time.time()))
        return args.exit_code
    if args.mode == "probe":
        result = probe(args)
        print(json.dumps(result["recommendation"]))
        return 0 if result["recommendation"]["status"] == "complete" else 2
    if args.mode == "selftest":
        result = selftest(args.output)
    else:
        payload = json.loads(args.jobs.read_text())
        result = run_jobs(payload["jobs"] if isinstance(payload, dict) else payload,
                          args.concurrency, args.output, args.job_timeout_seconds)
    print(json.dumps({key: result[key] for key in ("status", "optimizer_updates", "concurrency", "max_active") if key in result}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
