"""Flushed local scalar events, optionally mirrored to Weights & Biases.

No torch/W&B import for local logging. Credentials and model/data files are never
passed to the service. Every recovery invocation owns a separate tracking run.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import warnings
import uuid


@dataclass(frozen=True)
class TrackingConfig:
    mode: str = "local"
    project: str = "tdyssey"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    log_every: int = 10
    max_event_bytes: int = 16*1024**2

    def __post_init__(self):
        if self.mode not in ("local", "offline", "online"):
            raise ValueError("tracking mode must be local, offline or online")
        if type(self.log_every) is not int or self.log_every < 1:
            raise ValueError("log_every must be a positive integer")
        if type(self.max_event_bytes) is not int or self.max_event_bytes < 1024:
            raise ValueError("tracking byte cap must be at least 1024")


def add_tracking_arguments(parser):
    parser.add_argument("--tracking", choices=("local", "offline", "online"), default="local",
                        help="Always write metrics.jsonl; optionally mirror scalars to W&B")
    parser.add_argument("--wandb-project", default="tdyssey")
    parser.add_argument("--wandb-entity", help="W&B account or team; credentials stay outside the repo")
    parser.add_argument("--run-name")
    parser.add_argument("--run-group", help="Group repeated seeds/methods in the same comparison")
    parser.add_argument("--log-every", type=int, default=10, help="Optimizer minibatches per live loss event")


def tracking_from_args(args):
    return TrackingConfig(args.tracking, args.wandb_project, args.wandb_entity,
                          args.run_name, args.run_group, args.log_every)


class RunLogger:
    def __init__(self, output, config=None, metadata=None):
        self.output = Path(output)
        self.config = config or TrackingConfig()
        self.started = time.monotonic()
        self.events = (self.output/"metrics.jsonl").open("x", buffering=1)
        self.bytes = 0
        self.run = None
        self.closed = False
        self.state = {"mode": self.config.mode, "status": "running", "local_events": "metrics.jsonl",
                      "external_status": "disabled", "event_byte_cap": self.config.max_event_bytes}
        # Failures in optional remote monitoring must not discard GPU training.
        if self.config.mode != "local":
            try:
                import wandb
                directory = self.output/"tracking"
                directory.mkdir()
                self.run = wandb.init(project=self.config.project, entity=self.config.entity,
                    name=self.config.name or self.output.name, group=self.config.group,
                    mode=self.config.mode, dir=str(directory), config=metadata or {},
                    id=uuid.uuid4().hex[:8], resume="never",
                    settings=wandb.Settings(disable_git=True, disable_code=True, console="off",
                                           _disable_stats=True, _disable_meta=True))
                self.run.define_metric("progress/optimizer_step")
                self.run.define_metric("*", step_metric="progress/optimizer_step")
                self.state.update(external_status=self.config.mode, run_id=self.run.id,
                                  url=self.run.url if self.config.mode == "online" else None)
            except Exception as exc:
                self._external_failure(exc)
        self._save()

    def _external_failure(self, exc):
        self.state["external_status"] = "failed"
        self.state["external_error_type"] = type(exc).__name__
        warnings.warn("W&B monitoring unavailable; training continues with local metrics.jsonl", stacklevel=2)
        self._remote_failed = True

    def _save(self):
        temporary = self.output/"tracking.tmp"
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False)+"\n")
        temporary.replace(self.output/"tracking.json")

    def log(self, values, step):
        if self.closed:
            raise ValueError("Tracking run is closed")
        scalars = {}
        for key, value in values.items():
            if value is None:
                continue
            if not isinstance(value, (bool, int, float)) or not math.isfinite(value):
                raise ValueError(f"Metric must be a finite scalar: {key}")
            scalars[key] = value
        scalars["progress/optimizer_step"] = step
        event = {"elapsed_seconds": time.monotonic()-self.started, "metrics": scalars}
        data = json.dumps(event, allow_nan=False, separators=(",", ":"))+"\n"
        if self.bytes+len(data.encode()) <= self.config.max_event_bytes:
            self.events.write(data)
            self.events.flush()
            self.bytes += len(data.encode())
        elif not self.state.get("local_events_capped"):
            self.state["local_events_capped"] = True
            warnings.warn("Local scalar log reached its byte cap; reports/checkpoints continue", stacklevel=2)
            self._save()
        if self.run is not None and not getattr(self, "_remote_failed", False):
            try:
                self.run.log(scalars)
            except Exception as exc:
                self._external_failure(exc)
                self._save()

    def finish(self, status):
        if self.closed:
            return
        self.events.close()
        self.closed = True
        self.state.update(status=status, event_bytes=self.bytes)
        if self.run is not None:
            try:
                self.run.finish(exit_code=0 if status == "complete" else 1)
            except Exception as exc:
                self._external_failure(exc)
        self._save()


def evaluation_scalars(episodes, prefix="eval"):
    """Per-shape development outcomes, never pool away a failing shape."""
    result = {}
    for shape in sorted({row["shape"] for row in episodes}):
        rows = [row for row in episodes if row["shape"] == shape]
        for field in ("success", "peak_coverage", "final_coverage", "controls"):
            result[f"{prefix}/{shape}/{field}"] = sum(r[field] for r in rows)/len(rows)
    return result


def evaluation_score(episodes):
    """Macro shape success, then final coverage: declared checkpoint selection."""
    values = evaluation_scalars(episodes)
    def mean(field):
        rows = [v for k, v in values.items() if k.endswith("/"+field)]
        return sum(rows)/len(rows)
    return mean("success"), mean("final_coverage")
