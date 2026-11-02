"""Plan one GPC adaptation; explicit execute requires complete data and authority."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.gpc_adaptation import CONFIG, NativeSampler, OutputWriter, plan, repository_path, verify_execution


def worker(handoff_path):
    from tdyssey.gpc_adaptation_guard import (Heartbeat,StartupTrace,initialize_sdk,
        cuda_allocator_sample,status_template,verify_worker)
    from tdyssey.gpc_adaptation import run_candidate
    handoff = verify_worker(handoff_path)
    prepared = plan(handoff["config_path"], claimed=True)
    binding = handoff["binding"]
    if not prepared["ready"] or any(binding[name] != prepared[source]["sha256"] for name, source in
            (("data_sha256", "data"), ("benchmark_sha256", "benchmark"))):
        raise ValueError("Worker assets changed after launch binding")
    if binding["config_sha256"] != prepared["config_sha256"]:
        raise ValueError("Worker configuration changed")
    cfg = prepared["config"]
    if cfg.get("training_schedule") is not None:
        schedule=prepared.get("training_schedule") or {}
        for field in ("sha256","order_sha256","rule_sha256"):
            if not schedule.get(field) or binding.get("training_schedule_"+field)!=schedule[field]:
                raise ValueError("Worker training schedule identity/order/rule changed")
    output = repository_path(cfg["output_dir"])
    trace=StartupTrace(handoff_path)
    runtime = dict(torch=None, model_instances=0)
    def refresh():
        value = dict(model_instances=runtime["model_instances"],**trace.snapshot())
        torch = runtime["torch"]
        if torch is not None:
            value.update(cuda_allocator_sample(torch))
        return value
    # Maintain fresh status even during a long import, model load or forward.
    # Only the main thread synchronizes CUDA; heartbeat memory reads do not wait.
    with trace, Heartbeat(handoff_path, status_template(), refresh=refresh) as heartbeat:
        try:
            torch,sdk_diagnostics=initialize_sdk(heartbeat,trace,cpu_threads=cfg["limits"]["cpu_threads"])
            from tdyssey.gpc_supervised_adaptation import make_real_denoiser
            runtime["torch"] = torch
            def memory_check():
                heartbeat.check(); torch.cuda.synchronize(0)
                if (torch.cuda.max_memory_allocated(0) > cfg["limits"]["cuda_allocated_bytes"]
                        or torch.cuda.max_memory_reserved(0) > cfg["limits"]["cuda_reserved_bytes"]):
                    raise MemoryError("Fixed training/evaluation CUDA memory ceiling exceeded")
            def progress(phase, step, state):
                heartbeat.update(phase=phase, training_complete=state["selection_count"]==1 and state["completed_updates"]==cfg["train"]["selected_update"],
                    optimizer_updates=state["spent_reserved_updates"], model_loads=state["spent_reserved_model_loads"],
                    forecasts=state["spent_reserved_forecasts"], euler_steps=state["spent_reserved_euler_steps"],
                    eval_batches=state["spent_reserved_eval_batches"], eval_examples=state["spent_eval_examples"])
            def close_model():
                gc.collect(); torch.cuda.empty_cache()
                runtime["model_instances"] = 0
                heartbeat.update()
            def factory(role):
                runtime["model_instances"] = 1
                heartbeat.update(phase="model_loading")
                return make_real_denoiser(handoff["gpc_root"], handoff["checkpoint"], "cuda:0", dict(cfg, execute=True))
            report = run_candidate(prepared, binding, factory=factory, close_model=close_model,
                make_sampler=lambda model: NativeSampler(model, memory_check), writer=OutputWriter(output, cfg["limits"]),
                progress=progress, memory_check=memory_check)
        except BaseException as error:
            try: heartbeat.update(phase="error", error=type(error).__name__+": "+str(error))
            except BaseException as secondary: error.add_note("Status publication also failed: "+str(secondary))
            raise
    print(json.dumps(dict(status=report["status"], forecast_records=report["forecast_records"])))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--authorization-id", default="")
    parser.add_argument("--reviewed-commit", default="")
    parser.add_argument("--authorized-config-sha256", default="")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpc-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            return worker(args.worker)
        prepared = plan(args.config)
        if not args.execute and not args.preflight:
            print(json.dumps({key: prepared[key] for key in
                ("kind", "mode", "ready", "gaps", "config_sha256", "real_operations", "native_resource_queries")}, indent=2))
            return 0
        if args.preflight:
            from tdyssey.gpc_adaptation_runtime import native_resource_preflight
            gib = 1024**3
            result = native_resource_preflight(repository_path(prepared["config"]["output_dir"]),
                dict(available_ram_bytes=4*gib, commit_headroom_bytes=6*gib, gpu_free_bytes=4*gib, disk_free_bytes=2*gib))
            print(json.dumps(result, indent=2)); return 0 if result["passed"] else 2
        binding = verify_execution(prepared, authorization_id=args.authorization_id,
            reviewed_commit=args.reviewed_commit, authorized_config_sha256=args.authorized_config_sha256)
        if args.gpc_root is None or args.checkpoint is None:
            raise ValueError("Execute needs explicit pinned source and initialization paths")
        from tdyssey.gpc_adaptation_guard import supervise
        result = supervise(prepared, binding, python_executable=args.python, gpc_root=args.gpc_root,
            checkpoint=args.checkpoint, script_path=Path(__file__), config_path=args.config)
        print(json.dumps(result, indent=2)); return 0 if result["status"] == "complete" else 2
    except Exception as error:
        print(json.dumps(dict(status="refused_or_stopped", reason=type(error).__name__+": "+str(error))))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
