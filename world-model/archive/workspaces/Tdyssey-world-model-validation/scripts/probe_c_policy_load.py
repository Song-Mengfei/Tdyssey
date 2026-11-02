"""Single-instance C model load diagnostic, with no rollout or optimizer step.

This child is intended for an externally monitored, explicitly authorized run.
It uses the formal C runtime and actor factory once, then creates only the value
head and empty optimizer shells. Importing this script performs no runtime work.
The measurements cover loading and migration, not PPO or inference memory.
"""

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MIN_AVAILABLE_BYTES = 1024**3
MIN_COMMIT_HEADROOM_BYTES = 2 * 1024**3
MAX_SECONDS = 180
MAX_STAGE_BYTES = 256 * 1024


class LoadProbeStopped(RuntimeError):
    """The diagnostic stopped before beginning its next expensive operation."""


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/policy_c.example.json")
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--base-checkpoint", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    payload = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _loader_stage_lines(path):
    """Find post-operation source lines before importing torch or opening weights.

    A line event precedes its statement, so the next top-level statement is the
    observation point after each preceding operation. Refuse a changed loader
    layout instead of silently attaching a stage to the wrong operation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "OfficialDiffusionPolicy"]
    if len(classes) != 1:
        raise ValueError("Expected exactly one official policy class")
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
               and node.name == "__init__"]
    if len(methods) != 1:
        raise ValueError("Expected exactly one official policy constructor")
    body = methods[0].body
    operations = {
        "mmap_done": 'payload = torch.load(checkpoint, map_location=\'cpu\', weights_only=True, mmap=True)',
        "model_constructed": "self.network = DiffusionUnetHybridImagePolicy(noise_scheduler=DDPMScheduler(**scheduler_config), **config)",
        "strict_load_done": "self.network.load_state_dict(payload['state_dict'], strict=True)",
        "cuda_transfer_done": "self.network.eval().requires_grad_(False).to(self.device)",
    }
    stages = {}
    indices = []
    for stage, expression in operations.items():
        expected = ast.dump(ast.parse(expression).body[0], include_attributes=False)
        matches = [index for index, node in enumerate(body)
                   if ast.dump(node, include_attributes=False) == expected]
        if len(matches) != 1 or matches[0] + 1 >= len(body):
            raise ValueError(f"Official loader layout changed at {stage}; review instrumentation")
        index = matches[0]
        line = body[index + 1].lineno
        if line in stages:
            raise ValueError("Loader observation stages must have distinct next statements")
        stages[line] = stage
        indices.append(index)
    if indices != sorted(indices):
        raise ValueError("Official loader operation order changed")
    return methods[0].lineno, stages


class _Recorder:
    def __init__(self, directory, memory, *, config_hash, loader_sha256):
        self.directory, self.memory = directory, memory
        self.config_hash, self.loader_sha256 = config_hash, loader_sha256
        self.started = time.perf_counter()
        self.last_stage = None
        self.stages = []
        self.torch = None
        self.device = None
        # Exclusive creation makes accidental diagnostic reuse fail before load.
        self.journal = (directory / "stages.jsonl").open("xb")

    def close(self):
        self.journal.close()

    def _cuda(self):
        if self.torch is None or self.device != "cuda":
            return {"initialized": False}
        cuda = self.torch.cuda
        if not cuda.is_initialized():
            return {"initialized": False}
        return {"initialized": True,
                "allocated_bytes": cuda.memory_allocated(),
                "reserved_bytes": cuda.memory_reserved(),
                "peak_allocated_bytes": cuda.max_memory_allocated(),
                "peak_reserved_bytes": cuda.max_memory_reserved()}

    def emit(self, stage, *, check=True, **details):
        system = self.memory.system_memory()
        reason = self.memory.guard_reason(system, minimum_available_bytes=MIN_AVAILABLE_BYTES,
                                          minimum_commit_bytes=MIN_COMMIT_HEADROOM_BYTES)
        elapsed = time.perf_counter() - self.started
        if elapsed >= MAX_SECONDS:
            reason = f"Load-only child exceeded {MAX_SECONDS} seconds"
        record = {"stage": stage, "at_utc": _utc_now(), "elapsed_seconds": elapsed,
                  "pid": os.getpid(), "config_hash": self.config_hash,
                  "loader_sha256": self.loader_sha256,
                  "system": system, "process": self.memory.process_memory(os.getpid()),
                  "gpu": self.memory.gpu_memory(timeout=3.0), "cuda": self._cuda(),
                  "guard_reason": reason, **details}
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        if self.journal.tell() + len(payload) > MAX_STAGE_BYTES:
            raise LoadProbeStopped("Load diagnostic stage journal exceeded its byte cap")
        self.journal.write(payload)
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.last_stage = stage
        self.stages.append(stage)
        if check and reason:
            raise LoadProbeStopped(str(reason))
        return record


@contextmanager
def _observe_loader(code, stages, recorder):
    """Observe the original function once; do not hold frames or change its calls."""
    if sys.gettrace() is not None:
        raise LoadProbeStopped("An existing Python tracer prevents isolated loader instrumentation")
    seen = set()
    state = {"constructor_calls": 0}

    def local_trace(frame, event, arg):
        if event == "line" and frame.f_lineno in stages:
            stage = stages[frame.f_lineno]
            if stage in seen:
                raise LoadProbeStopped("Official loader repeated an observed operation")
            seen.add(stage)
            recorder.emit(stage)
        return local_trace

    def global_trace(frame, event, arg):
        if event != "call":
            return None
        if frame.f_code is code:
            state["constructor_calls"] += 1
            if state["constructor_calls"] != 1:
                raise LoadProbeStopped("Only one official model construction is permitted")
            return local_trace
        # These are fail-closed checks, not replacement implementations. The
        # unchanged constructor uses output-shape arithmetic and no forward.
        name = frame.f_code.co_name
        filename = frame.f_code.co_filename.replace("\\", "/")
        if name == "forward":
            raise LoadProbeStopped("A forward call is forbidden in the load-only child")
        if (name == "step" and "/torch/optim/" in filename
                and not filename.endswith("/lr_scheduler.py")):
            raise LoadProbeStopped("An optimizer step is forbidden in the load-only child")
        if name in ("reset", "step", "_set_state") and "/diffusion_policy/env/" in filename:
            raise LoadProbeStopped("Simulator operations are forbidden in the load-only child")
        if (name in ("sample", "conditional_sample", "predict_action", "predict_actions")
                and ("/tdyssey/" in filename or "/diffusion_policy/policy/" in filename)):
            raise LoadProbeStopped("Policy sampling is forbidden in the load-only child")
        if frame.f_code.co_qualname == "ExperimentBudget.__init__" and filename.endswith("/tdyssey/policy_protocol.py"):
            raise LoadProbeStopped("A C experiment ledger is forbidden in the load-only child")
        return None

    sys.settrace(global_trace)
    try:
        yield state
        if state["constructor_calls"] != 1 or seen != set(stages.values()):
            raise LoadProbeStopped("The original official loader did not complete every required stage exactly once")
    finally:
        sys.settrace(None)


def run(args):
    # These imports are stdlib-only. In particular, reading CConfig starts no job.
    from tdyssey import policy_memory as memory
    from tdyssey.policy_protocol import CConfig

    config = CConfig.from_json(args.config)
    loader_path = ROOT / "tdyssey/official.py"
    constructor_line, stage_lines = _loader_stage_lines(loader_path)
    loader_hash = _digest(loader_path)
    directory = args.output_dir.resolve()
    if (directory / "report.json").exists() or (directory / "stages.jsonl").exists():
        raise FileExistsError("Diagnostic output already contains a stage journal or report")
    directory.mkdir(parents=True, exist_ok=True)
    recorder = _Recorder(directory, memory, config_hash=config.config_hash, loader_sha256=loader_hash)
    report = {"kind": "tdyssey.C.load_only_probe", "status": "blocked",
              "config_hash": config.config_hash, "config": config.to_dict(),
              "source_root": str(args.source_root.resolve()),
              "base_checkpoint": str(args.base_checkpoint.resolve()),
              "loader_sha256": loader_hash,
              "limits": {"minimum_available_ram_bytes": MIN_AVAILABLE_BYTES,
                         "minimum_commit_headroom_bytes": MIN_COMMIT_HEADROOM_BYTES,
                         "wall_seconds": MAX_SECONDS, "stage_bytes": MAX_STAGE_BYTES},
              "counts": {"official_model_constructions": 0, "forward_calls": 0,
                         "policy_samples": 0, "resets": 0, "control_steps": 0,
                         "optimizer_steps": 0, "evaluation_episodes": 0},
              "formal_C_experiment_initialized": False, "formal_C_clock_started": False,
              "scope": "load/migration and empty optimizer shells; no inference or PPO activation measurement"}
    try:
        recorder.emit("before_import")
        from scripts.run_c_policy import _new_actor, _optimizers, _runtime

        device = _runtime(args, config)
        import torch

        recorder.torch, recorder.device = torch, device
        if device == "cuda" and torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        recorder.emit("after_runtime")
        from tdyssey.official import OfficialDiffusionPolicy

        code = OfficialDiffusionPolicy.__init__.__code__
        if (Path(code.co_filename).resolve() != loader_path.resolve()
                or code.co_firstlineno != constructor_line or _digest(loader_path) != loader_hash):
            raise LoadProbeStopped("Official loader source differs from the reviewed observation points")
        observer_state = None
        try:
            with _observe_loader(code, stage_lines, recorder) as observer_state:
                official, actor = _new_actor(args, config, device)
                if official.network is not actor.network:
                    raise LoadProbeStopped("The actor must share the single loaded official network")
                recorder.emit("actor_ready", actor_parameters=official.parameter_count,
                              selected_parameters=actor.selection.total)
                from tdyssey.policy_training import VisualCritic

                critic = VisualCritic(actor.network.obs_feature_dim * actor.network.n_obs_steps).to(device)
                actor_optimizer, critic_optimizer = _optimizers(actor, critic, config.to_dict())
                if actor_optimizer.state or critic_optimizer.state:
                    raise LoadProbeStopped("Fresh optimizer shells must not contain Adam state")
                if any(parameter.grad is not None for parameter in actor.network.parameters()):
                    raise LoadProbeStopped("A load-only actor must have no gradients")
                if device == "cuda":
                    torch.cuda.synchronize()
                ready = recorder.emit("critic_optimizer_ready",
                    actor_parameters=official.parameter_count, selected_parameters=actor.selection.total,
                    actor_parameter_bytes=sum(p.numel() * p.element_size() for p in actor.network.parameters()),
                    actor_buffer_bytes=sum(p.numel() * p.element_size() for p in actor.network.buffers()),
                    critic_parameters=sum(p.numel() for p in critic.parameters()),
                    optimizer_state_entries={"actor": len(actor_optimizer.state), "critic": len(critic_optimizer.state)})
        finally:
            if observer_state is not None:
                report["counts"]["official_model_constructions"] = observer_state["constructor_calls"]
        report.update(status="complete", ready=ready,
                      checkpoint_sha256=official.checkpoint_sha256,
                      actor_shares_official_network=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            recorder.emit("blocked", check=False, error=report["error"])
        except Exception as diagnostic_error:
            report["diagnostic_error"] = {"type": type(diagnostic_error).__name__, "message": str(diagnostic_error)}
    finally:
        report["stages"] = list(recorder.stages)
        report["elapsed_seconds"] = time.perf_counter() - recorder.started
        recorder.close()
    _write_json(directory / "report.json", report)
    return report


def main():
    args = parser().parse_args()
    try:
        report = run(args)
        print(json.dumps({"status": report["status"], "report": str(args.output_dir.resolve() / "report.json"),
                          "stages": report["stages"], "error": report.get("error")},
                         sort_keys=True, allow_nan=False))
        return 0 if report["status"] == "complete" else 1
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
