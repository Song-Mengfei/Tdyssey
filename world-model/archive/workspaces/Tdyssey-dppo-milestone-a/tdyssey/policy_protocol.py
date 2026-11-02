"""Milestone C proposals and durable experiment budgets, using only stdlib.

Reading/validating a config never authorizes or starts an experiment. New ledger
creation requires the caller's explicit create=True initialization. The stable
registry is authoritative across run directories; spent reservations never reset.
No torch, simulator, sampler, optimizer or milestone B ledger is imported here.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
import uuid


PHASES = ("train", "development", "diagnostic")
PHASE_KINDS = ("control", "plan", "reset")
GLOBAL_KINDS = ("actor_step", "critic_step", "batch")
SCHEMA_VERSION = 1
_HARD_CAPS = {
    "train": {"control": 2048, "plan": 384, "reset": 384},
    "development": {"control": 24000, "plan": 3040, "reset": 160},
    "diagnostic": {"control": 0, "plan": 0, "reset": 0},
    "global": {"actor_step": 72, "critic_step": 72, "batch": 24},
    "wall_time_seconds": 7200, "storage_bytes": 512 * 1024**2, "max_checkpoints": 2,
}


class BudgetExceeded(RuntimeError):
    """A C limit was exhausted before the requested operation began."""


class BudgetBusy(RuntimeError):
    """A live, stale or corrupt registry owner lock prevents operation."""


def _integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value, name, minimum=0, maximum=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    return value


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Unexpected or missing {name} fields")


def _seed_list(value, name, minimum=1, maximum=32):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} must be a fixed list of {minimum}..{maximum} scene/seed IDs")
    for seed in value:
        _integer(seed, name, 0, 2**32 - 1)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate IDs")


class CConfig:
    """Versioned future protocol; the example is a proposal, not permission.

    data/to_dict/canonical_dict return independent dictionaries. config_hash is
    SHA-256 of sorted compact JSON; array order is preserved because scene order
    and policy seed order are protocol choices rather than unordered metadata.
    """

    def __init__(self, data):
        self._data = deepcopy(data)
        self.validate()

    @classmethod
    def from_dict(cls, value):
        return cls(value)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def data(self):
        return self.to_dict()

    def to_dict(self):
        return deepcopy(self._data)

    def canonical_dict(self):
        return self.to_dict()

    @property
    def config_hash(self):
        encoded = json.dumps(self._data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def hash(self):
        return self.config_hash

    def validate(self):
        data = self._data
        _keys(data, ("schema_version", "status", "base", "shapes", "train_seed", "scenes",
                     "evaluation", "runtime", "training", "optimization", "budgets"), "C config")
        if type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported C config schema")
        if data["status"] != "proposed_not_executed" or data["shapes"] != ["T", "L"]:
            raise ValueError("C v1 describes a proposed T/L development protocol")
        _keys(data["base"], ("sha256", "bytes"), "base identity")
        if not isinstance(data["base"]["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", data["base"]["sha256"]):
            raise ValueError("Base identity requires a recorded lowercase SHA-256")
        _integer(data["base"]["bytes"], "base bytes", 1)
        _integer(data["train_seed"], "train_seed", 0, 2**32 - 1)
        scenes, evaluation = data["scenes"], data["evaluation"]
        _keys(scenes, ("training", "excluded_evaluation"), "scenes")
        _keys(scenes["training"], ("T", "L"), "training scenes")
        _keys(evaluation, ("scenes", "policy_seeds", "max_groups", "episode_controls",
                           "new_development_scenes", "untouched_test"), "evaluation")
        _seed_list(scenes["excluded_evaluation"], "excluded evaluation", 5, 32)
        _seed_list(evaluation["scenes"], "development scenes", 5, 5)
        _seed_list(evaluation["policy_seeds"], "development policy seeds", 2, 2)
        _seed_list(evaluation["new_development_scenes"], "new development scenes", 0, 5)
        excluded = set(scenes["excluded_evaluation"])
        if (not {100000, 100001, 100002}.issubset(excluded)
                or not set(evaluation["scenes"]).issubset(excluded)
                or not set(evaluation["new_development_scenes"]).issubset(evaluation["scenes"])
                or evaluation["untouched_test"] is not False):
            raise ValueError("All known/development scenes must be excluded, and are not an untouched test")
        for shape in data["shapes"]:
            _seed_list(scenes["training"][shape], f"{shape} training scenes")
            if excluded.intersection(scenes["training"][shape]):
                raise ValueError("Training and the full evaluation exclusion set must be disjoint for each shape")
        _integer(evaluation["max_groups"], "development groups", 1, 4)
        _integer(evaluation["episode_controls"], "development episode controls", 1, 300)
        runtime = data["runtime"]
        _keys(runtime, ("device", "precision", "threads"), "runtime")
        if runtime["device"] not in ("cpu", "cuda") or runtime["precision"] != "fp32":
            raise ValueError("C v1 uses explicit CPU/CUDA FP32 without automatic device fallback")
        _integer(runtime["threads"], "CPU threads", 1, 4)
        training = data["training"]
        _keys(training, ("batch_plans", "batch_controls", "episode_controls"), "training")
        _integer(training["batch_plans"], "batch plans", 1, 16)
        _integer(training["batch_controls"], "batch controls", 1, 128)
        _integer(training["episode_controls"], "training episode controls", 1, 300)
        if training["batch_controls"] > 8 * training["batch_plans"]:
            raise ValueError("Batch physical controls exceed its sampled-plan capacity")
        optimization = data["optimization"]
        _keys(optimization, ("subset", "std_floor", "gamma", "beta", "gae_lambda", "clip", "denoising_discount",
                             "actor_lr", "critic_lr", "max_grad_norm", "target_kl", "max_passes",
                             "first_batch_passes", "plan_microbatch", "step_microbatch"), "optimization")
        fixed = {"subset": "output", "std_floor": .01, "gamma": .99, "beta": 1.0, "gae_lambda": .95,
                 "clip": .2, "denoising_discount": 1.0, "actor_lr": 1e-5, "critic_lr": 3e-4,
                 "max_grad_norm": 1.0, "target_kl": .02, "first_batch_passes": 1}
        if any(optimization[key] != expected or type(optimization[key]) is bool for key, expected in fixed.items()):
            raise ValueError("C v1 fixed optimization settings changed; version the protocol explicitly")
        _integer(optimization["max_passes"], "maximum passes", 1, 3)
        _integer(optimization["plan_microbatch"], "plan microbatch", 1, 4)
        _integer(optimization["step_microbatch"], "step microbatch", 1, 4)
        budgets = data["budgets"]
        _keys(budgets, (*PHASES, "global", "wall_time_seconds", "storage_bytes", "max_checkpoints"), "budgets")
        for phase in (*PHASES, "global"):
            kinds = GLOBAL_KINDS if phase == "global" else PHASE_KINDS
            _keys(budgets[phase], kinds, f"{phase} limits")
            for kind in kinds:
                _integer(budgets[phase][kind], f"{phase} {kind} cap", 0, _HARD_CAPS[phase][kind])
        _finite(budgets["wall_time_seconds"], "wall time", 0.001, _HARD_CAPS["wall_time_seconds"])
        _integer(budgets["storage_bytes"], "storage bytes", 1, _HARD_CAPS["storage_bytes"])
        _integer(budgets["max_checkpoints"], "retained checkpoints", 1, _HARD_CAPS["max_checkpoints"])
        batches = budgets["global"]["batch"]
        if (budgets["train"]["control"] > batches * training["batch_controls"]
                or budgets["train"]["control"] > budgets["train"]["plan"] * 8
                or budgets["train"]["plan"] > batches * training["batch_plans"]):
            raise ValueError("Training controls/plans cannot fit the declared batch capacities")
        steps = (1 + (batches - 1) * optimization["max_passes"]) if batches else 0
        if any(budgets["global"][kind] < steps for kind in ("actor_step", "critic_step")):
            raise ValueError("Optimizer step caps cannot cover the declared first/subsequent batch passes")
        # Report capacities only: choosing evaluation groups remains caller-driven.
        return self


def _atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _identifier(value, name):
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if (not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value)
            or value in reserved):
        raise ValueError(f"{name} must be an explicit safe lowercase identifier")
    return value


def _lock(path, token):
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise BudgetBusy(f"Existing registry lock; no automatic stale takeover: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"token": token, "pid": os.getpid(), "host": socket.gethostname()}, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _unlock(path, token):
    if _read_json(path).get("token") != token:
        raise BudgetBusy("Registry lock ownership changed")
    path.unlink()


def _counters():
    return {**{phase: {kind: 0 for kind in PHASE_KINDS} for phase in PHASES},
            "global": {kind: 0 for kind in GLOBAL_KINDS}}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


class ExperimentBudget:
    """Registry-owned C ledger; opening a new run directory never resets it.

    create=True is explicit experiment initialization, never an automatic retry.
    The registry index retains the ID/config even if a run directory is removed.
    A removed/corrupt authoritative ledger fails closed rather than reinitializing.
    Keep registry_root stable; deleting/replacing that authority is not recovery.
    Wall time starts at initialization and includes downtime across resumes.

    Completed operation history is bounded to 64 recent entries; exact aggregate
    counters, monotonically increasing IDs and every incomplete reservation are
    durable. Per-control fsync therefore does not rewrite an unbounded event log.
    """

    def __init__(self, registry_root, experiment_id, config, run_dir, create=False):
        self.config = config if isinstance(config, CConfig) else CConfig.from_dict(config)
        self.config.validate()  # Before mkdir, index, lock or ledger creation.
        self.experiment_id = _identifier(experiment_id, "experiment_id")
        if type(create) is not bool:
            raise ValueError("create must be an explicit boolean initialization choice")
        self.registry_root = Path(registry_root).resolve()
        self.run_dir = Path(run_dir).resolve()
        if self.run_dir == self.registry_root or self.registry_root.is_relative_to(self.run_dir):
            raise ValueError("A run directory cannot contain the authoritative registry")
        self.directory = self.registry_root / self.experiment_id
        self.path = self.directory / "ledger.json"
        self.lock_path = self.directory / "owner.lock"
        self.index_path = self.registry_root / "index.json"
        self._token = uuid.uuid4().hex
        self._mutex = threading.RLock()
        self._closed = True
        self._poisoned = False
        if not create and (not self.index_path.is_file() or not self.path.is_file()):
            raise FileNotFoundError("No existing C experiment; explicit initialization is required")
        if create:
            self.registry_root.mkdir(parents=True, exist_ok=True)
        index_lock = self.registry_root / "index.lock"
        index_token = uuid.uuid4().hex
        _lock(index_lock, index_token)
        try:
            index = _read_json(self.index_path) if self.index_path.exists() else {
                "schema_version": SCHEMA_VERSION, "experiments": {}, "run_bindings": {}}
            self._validate_index(index)
            entry = index["experiments"].get(self.experiment_id)
            if create and (entry is not None or self.directory.exists()):
                raise FileExistsError("Experiment ID already registered; resume its authority rather than reinitialize")
            if not create and (entry is None or entry["config_hash"] != self.config.config_hash):
                raise ValueError("Registered experiment/config identity differs")
            self._check_output_binding(self.run_dir, index)
            if create:
                self.directory.mkdir()
            _lock(self.lock_path, self._token)
            self._closed = False
            now = time.time()
            if create:
                self._data = {
                    "schema_version": SCHEMA_VERSION, "experiment_id": self.experiment_id,
                    "registry_root": str(self.registry_root), "config_hash": self.config.config_hash,
                    "config": self.config.to_dict(), "started_at": now, "max_observed_time": now,
                    "elapsed_wall_seconds": 0.0, "output_dirs": [], "reserved": _counters(),
                    "completed": _counters(), "last_event_id": 0, "last_train_operation_id": 0,
                    "pending": {}, "recent_events": [], "by_shape": {}, "written_files": {},
                    "checkpoint_files": [], "recovery_boundary": None, "evaluation_groups": {},
                }
                index["experiments"][self.experiment_id] = {"config_hash": self.config.config_hash,
                    "ledger": f"{self.experiment_id}/ledger.json", "created_at": now}
            else:
                self._data = _read_json(self.path)
                self._validate_data()
            self._opened_time = max(now, self._data["max_observed_time"])
            self._opened_monotonic = time.monotonic()
            self._bind_output(self.run_dir, index)
            self._save(self._data)
            _atomic_json(self.index_path, index)
        except BaseException:
            if not self._closed:
                _unlock(self.lock_path, self._token)
                self._closed = True
            raise
        finally:
            _unlock(index_lock, index_token)

    def _validate_index(self, index):
        _keys(index, ("schema_version", "experiments", "run_bindings"), "registry index")
        if index["schema_version"] != SCHEMA_VERSION or not isinstance(index["experiments"], dict) or not isinstance(index["run_bindings"], dict):
            raise ValueError("Invalid C registry index")
        for experiment_id, entry in index["experiments"].items():
            _identifier(experiment_id, "registered experiment ID")
            if entry.get("ledger") != f"{experiment_id}/ledger.json" or not re.fullmatch(r"[0-9a-f]{64}", entry.get("config_hash", "")):
                raise ValueError("Corrupt registry experiment identity")

    def _check_output_binding(self, path, index):
        if path == self.registry_root or self.registry_root.is_relative_to(path):
            raise ValueError("Output directory cannot contain the authoritative registry")
        binding = index["run_bindings"].get(str(path))
        reference = path / "c-experiment.json"
        expected = {"schema_version": SCHEMA_VERSION, "experiment_id": self.experiment_id,
                    "registry_root": str(self.registry_root), "config_hash": self.config.config_hash}
        if binding is not None and binding != self.experiment_id:
            raise ValueError("Output directory is already associated with a different experiment ID")
        if reference.exists():
            if _read_json(reference) != expected or binding != self.experiment_id:
                raise ValueError("Output-directory reference differs from the registry authority")
        elif path.exists() and any(path.iterdir()):
            raise ValueError("Refusing an existing non-C output directory, including milestone B artifacts")
        return expected

    def _bind_output(self, path, index):
        reference = self._check_output_binding(path, index)
        path.mkdir(parents=True, exist_ok=True)
        if not (path / "c-experiment.json").exists():
            _atomic_json(path / "c-experiment.json", reference)
        index["run_bindings"][str(path)] = self.experiment_id
        if str(path) not in self._data["output_dirs"]:
            self._data["output_dirs"].append(str(path))

    def _validate_data(self):
        data = self._data
        if (data.get("schema_version") != SCHEMA_VERSION or data.get("experiment_id") != self.experiment_id
                or data.get("registry_root") != str(self.registry_root)
                or data.get("config_hash") != self.config.config_hash or data.get("config") != self.config.to_dict()):
            raise ValueError("Authoritative ledger/config identity differs; refusing counter reset")
        _finite(data["started_at"], "ledger start time")
        _finite(data["max_observed_time"], "ledger observed time", data["started_at"])
        _finite(data["elapsed_wall_seconds"], "ledger elapsed time")
        for mode in ("reserved", "completed"):
            _keys(data[mode], (*PHASES, "global"), f"{mode} counters")
            for phase, values in data[mode].items():
                kinds = GLOBAL_KINDS if phase == "global" else PHASE_KINDS
                _keys(values, kinds, f"{mode}/{phase}")
                for kind in kinds:
                    _integer(values[kind], f"{mode} {phase}/{kind}", 0, self.config.data["budgets"][phase][kind])
                    if data["completed"][phase][kind] > data["reserved"][phase][kind]:
                        raise ValueError("Completed C counters exceed reserved attempts")
        _integer(data["last_event_id"], "last reservation ID")
        _integer(data["last_train_operation_id"], "last training reservation ID", 0, data["last_event_id"])
        pending_counts = _counters()
        for identifier, event in data["pending"].items():
            if str(event["id"]) != identifier or not 1 <= event["id"] <= data["last_event_id"]:
                raise ValueError("Corrupt pending reservation ID")
            _integer(event["amount"], "pending reservation amount", 1)
            if event["kind"] == "storage":
                continue
            bucket = self._bucket(event["kind"], event["phase"])
            pending_counts[bucket][event["kind"]] += event["amount"]
        for phase, values in pending_counts.items():
            for kind, count in values.items():
                if count != data["reserved"][phase][kind] - data["completed"][phase][kind]:
                    raise ValueError("Pending operations disagree with durable reservation counters")
        if not isinstance(data["recent_events"], list) or len(data["recent_events"]) > 64:
            raise ValueError("Invalid bounded recent-event history")
        if len(data["evaluation_groups"]) > self.config.data["evaluation"]["max_groups"]:
            raise ValueError("Evaluation claims exceed the protocol group cap")

    def _ensure_open(self):
        if self._closed:
            raise BudgetBusy("C budget is closed")
        if self._poisoned:
            raise BudgetBusy("C persistence failed; this owner is poisoned and must not retry from stale memory")
        try:
            if _read_json(self.lock_path).get("token") != self._token:
                raise BudgetBusy("C budget lock ownership changed")
        except (OSError, ValueError) as exc:
            raise BudgetBusy("C owner lock missing/corrupt; refusing further operations") from exc

    def _touch_time(self, data):
        now = max(time.time(), self._opened_time + max(0.0, time.monotonic() - self._opened_monotonic), data["max_observed_time"])
        data["max_observed_time"] = now
        data["elapsed_wall_seconds"] = max(data["elapsed_wall_seconds"], now - data["started_at"])

    def _save(self, data):
        self._touch_time(data)
        try:
            _atomic_json(self.path, data)
        except BaseException:
            # os.replace may already have committed before directory fsync or
            # another post-publication operation raises. Never overwrite that
            # authoritative state with this owner's potentially stale memory.
            self._poisoned = True
            raise
        self._data = data

    @staticmethod
    def _bucket(kind, phase):
        if phase not in PHASES:
            raise ValueError("Unknown C phase")
        if kind in GLOBAL_KINDS:
            if phase != "train":
                raise ValueError("Optimizer steps and training batches are train-only global budgets")
            return "global"
        if kind not in PHASE_KINDS:
            raise ValueError("Unknown C operation kind")
        return phase

    def add_output_dir(self, path):
        with self._mutex:
            self._ensure_open()
            candidate = Path(path).resolve()
            lock_path, token = self.registry_root / "index.lock", uuid.uuid4().hex
            _lock(lock_path, token)
            try:
                index = _read_json(self.index_path)
                self._validate_index(index)
                self._bind_output(candidate, index)
                self._save(self._data)
                _atomic_json(self.index_path, index)
            finally:
                _unlock(lock_path, token)
            return str(candidate)

    def check_time(self):
        with self._mutex:
            self._ensure_open()
            self._save(deepcopy(self._data))
            left = max(0.0, self.config.data["budgets"]["wall_time_seconds"] - self._data["elapsed_wall_seconds"])
            if left <= 0:
                raise BudgetExceeded("C wall-time limit exhausted, including previous sessions/downtime")
            return left

    def reserved_count(self, kind, phase="train"):
        with self._mutex:
            self._ensure_open()
            return self._data["reserved"][self._bucket(kind, phase)][kind]

    def remaining(self, kind, phase="train"):
        with self._mutex:
            self._ensure_open()
            bucket = self._bucket(kind, phase)
            return max(0, int(self.config.data["budgets"][bucket][kind] - self._data["reserved"][bucket][kind]))

    def _remember(self, data, event):
        data["recent_events"] = [item for item in data["recent_events"] if item["id"] != event["id"]]
        data["recent_events"].append(deepcopy(event))
        data["recent_events"] = data["recent_events"][-64:]

    def reserve(self, kind, phase="train", shape=None, amount=1):
        with self._mutex:
            self._ensure_open()
            bucket = self._bucket(kind, phase)
            _integer(amount, "reservation amount", 1)
            if shape not in (None, "T", "L") or (bucket == "global" and shape is not None):
                raise ValueError("Only T/L physical metadata is allowed; mixed optimizer/batch operations have no shape")
            self.check_time()
            self.check_storage()
            if amount > self.remaining(kind, phase):
                raise BudgetExceeded(f"C {phase}/{kind} cap exhausted; no operation was attempted")
            data = deepcopy(self._data)
            data["last_event_id"] += 1
            identifier = data["last_event_id"]
            event = {"id": identifier, "kind": kind, "phase": phase, "shape": shape,
                     "amount": amount, "completed": False, "reserved_at": time.time()}
            data["reserved"][bucket][kind] += amount
            if phase == "train":
                data["last_train_operation_id"] = identifier
            if bucket != "global":
                counters = data["by_shape"].setdefault(phase, {}).setdefault(shape or "unspecified",
                            {"reserved": {k: 0 for k in PHASE_KINDS}, "completed": {k: 0 for k in PHASE_KINDS}})
                counters["reserved"][kind] += amount
            data["pending"][str(identifier)] = event
            self._remember(data, event)
            self._save(data)
            return identifier

    def mark_completed(self, identifier):
        with self._mutex:
            self._ensure_open()
            _integer(identifier, "reservation ID", 1)
            event = self._data["pending"].get(str(identifier))
            if event is None or event["kind"] == "storage":
                raise ValueError("Unknown/completed operation reservation; storage uses mark_storage_written")
            data = deepcopy(self._data)
            event = data["pending"].pop(str(identifier))
            bucket = self._bucket(event["kind"], event["phase"])
            data["completed"][bucket][event["kind"]] += event["amount"]
            if bucket != "global":
                data["by_shape"][event["phase"]][event["shape"] or "unspecified"]["completed"][event["kind"]] += event["amount"]
            event["completed"] = True
            self._remember(data, event)
            self._save(data)  # Completion telemetry remains writable after time expiry.

    def _owned_path(self, path):
        path = Path(path).resolve()
        if not any(path.is_relative_to(Path(directory)) and path != Path(directory) for directory in self._data["output_dirs"]):
            raise ValueError("Output file is outside this experiment's registered output directories")
        return path

    def _storage_usage(self):
        files = {}
        roots = [self.directory, *(Path(path) for path in self._data["output_dirs"])]
        for root in roots:
            if not root.exists():
                continue
            for directory, subdirectories, names in os.walk(root, followlinks=False):
                for name in subdirectories:
                    if (Path(directory) / name).is_symlink():
                        raise ValueError("Symlinked output directories are not allowed in budgeted storage")
                for name in names:
                    path = Path(directory) / name
                    if path.is_symlink():
                        raise ValueError("Symlinked output files are not allowed in budgeted storage")
                    files[str(path.resolve())] = path.stat().st_size
        if self.index_path.exists():
            files[str(self.index_path)] = self.index_path.stat().st_size
        pending = 0
        for event in self._data["pending"].values():
            if event["kind"] == "storage":
                grown = max(0, files.get(event.get("path"), 0) - event.get("existing_bytes", 0))
                pending += max(0, event["amount"] - grown)
        return {"current_bytes": sum(files.values()), "pending_bytes": pending,
                "limit_bytes": self.config.data["budgets"]["storage_bytes"]}

    def _retention_candidates(self):
        live = [item["path"] for item in self._data["checkpoint_files"] if Path(item["path"]).is_file()]
        maximum = self.config.data["budgets"]["max_checkpoints"]
        return live[:-maximum] if len(live) > maximum else []

    def check_storage(self):
        with self._mutex:
            self._ensure_open()
            usage = self._storage_usage()
            if usage["current_bytes"] + usage["pending_bytes"] > usage["limit_bytes"]:
                raise BudgetExceeded("C storage cap exhausted across all registered outputs and pending writes")
            if self._retention_candidates():
                raise BudgetExceeded("Retire the registered old checkpoints before another operation")
            return usage

    def reserve_storage(self, amount, path=None):
        """Reserve a worst-case write before IO; also accepts (path, amount)."""
        if isinstance(amount, (str, Path)) and type(path) is int:
            amount, path = path, amount
        with self._mutex:
            self._ensure_open()
            _integer(amount, "write byte reservation", 1)
            if amount > self.config.data["budgets"]["storage_bytes"]:
                raise BudgetExceeded("Requested write exceeds the entire C storage cap")
            self.check_time()
            usage = self.check_storage()
            if usage["current_bytes"] + usage["pending_bytes"] + amount > usage["limit_bytes"]:
                raise BudgetExceeded("Insufficient C storage capacity for the reserved write")
            target = self._owned_path(path) if path is not None else None
            if target is not None and any(event["kind"] == "storage" and event["path"] == str(target)
                                          for event in self._data["pending"].values()):
                raise ValueError("An output path already has an incomplete write reservation")
            data = deepcopy(self._data)
            data["last_event_id"] += 1
            identifier = data["last_event_id"]
            event = {"id": identifier, "kind": "storage", "phase": None, "shape": None, "amount": amount,
                     "path": str(target) if target is not None else None,
                     "existing_bytes": target.stat().st_size if target is not None and target.exists() else 0,
                     "completed": False, "reserved_at": time.time()}
            data["pending"][str(identifier)] = event
            self._remember(data, event)
            self._save(data)
            return identifier

    def mark_storage_written(self, identifier, actual_bytes):
        with self._mutex:
            self._ensure_open()
            event = self._data["pending"].get(str(identifier))
            if event is None or event["kind"] != "storage":
                raise ValueError("Unknown/completed storage reservation")
            _integer(actual_bytes, "actual write bytes", 0, event["amount"])
            target = Path(event["path"]) if event["path"] is not None else None
            if target is not None and (not target.is_file()
                    or target.stat().st_size != event["existing_bytes"] + actual_bytes):
                raise ValueError("Completed write growth differs from the reserved path's previous byte count")
            data = deepcopy(self._data)
            event = data["pending"].pop(str(identifier))
            event.update(completed=True, actual_bytes=actual_bytes)
            if target is not None:
                current_bytes = target.stat().st_size
                data["written_files"][str(target)] = {"bytes": current_bytes, "reservation_id": identifier}
                if target.name.startswith("checkpoint-") and target.suffix == ".pt":
                    if any(item["path"] == str(target) for item in data["checkpoint_files"]):
                        raise ValueError("Checkpoint outputs are immutable; refuse a repeated registered path")
                    data["checkpoint_files"].append({"path": str(target), "bytes": current_bytes, "reservation_id": identifier})
            self._remember(data, event)
            self._save(data)
            usage = self._storage_usage()
            if usage["current_bytes"] + usage["pending_bytes"] > usage["limit_bytes"]:
                raise BudgetExceeded("Completed writes exceed the C storage cap")
            return {"retire_paths": self._retention_candidates(), **usage}

    def retire_paths(self, paths):
        """Delete only explicitly requested registered checkpoint files, never trees."""
        with self._mutex:
            self._ensure_open()
            allowed = {item["path"] for item in self._data["checkpoint_files"]}
            targets = [self._owned_path(path) for path in paths]
            if any(str(path) not in allowed or not path.name.startswith("checkpoint-") or path.suffix != ".pt" for path in targets):
                raise ValueError("Retirement is restricted to this experiment's registered checkpoint-*.pt files")
            for path in targets:
                if path.exists():
                    path.unlink()
            return self._storage_usage()

    def record_boundary(self, checkpoint_path, metadata=None):
        with self._mutex:
            self._ensure_open()
            path = self._owned_path(checkpoint_path)
            if str(path) not in self._data["written_files"] or not path.is_file():
                raise ValueError("A boundary requires an existing successfully registered checkpoint")
            if not path.name.startswith("checkpoint-") or path.suffix != ".pt":
                raise ValueError("A recovery boundary must identify checkpoint-*.pt")
            # Do not run retention checks here: caller must first publish a new
            # safe checkpoint, then retire older files returned below.
            marker = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path),
                      "reserved": {key: deepcopy(self._data["reserved"][key]) for key in ("train", "global")},
                      "completed": {key: deepcopy(self._data["completed"][key]) for key in ("train", "global")},
                      "last_train_operation_id": self._data["last_train_operation_id"],
                      "config_hash": self.config.config_hash, "metadata": deepcopy(metadata or {})}
            json.dumps(marker, allow_nan=False)  # Primitive-only metadata before a persistent mutation.
            data = deepcopy(self._data)
            data["recovery_boundary"] = marker
            self._save(data)
            return {**deepcopy(marker), "retire_paths": self._retention_candidates()}

    def validate_boundary(self, checkpoint_path=None):
        with self._mutex:
            self._ensure_open()
            marker = self._data["recovery_boundary"]
            if marker is None:
                raise ValueError("No resumable checkpoint boundary was recorded")
            path = self._owned_path(checkpoint_path or marker["path"])
            if (str(path) != marker["path"] or marker["config_hash"] != self.config.config_hash
                    or marker["last_train_operation_id"] != self._data["last_train_operation_id"]
                    or any(marker[mode][key] != self._data[mode][key] for mode in ("reserved", "completed") for key in ("train", "global"))
                    or not path.is_file() or path.stat().st_size != marker["bytes"] or _sha256(path) != marker["sha256"]):
                raise ValueError("Checkpoint is not the current exact training boundary; cannot replay consumed operations")
            return deepcopy(marker)

    def claim_evaluation_group(self, group_id, *, method, protocol, case_digest):
        with self._mutex:
            self._ensure_open()
            _identifier(group_id, "evaluation group ID")
            if not isinstance(method, str) or not method.strip() or not isinstance(protocol, str) or not protocol.strip():
                raise ValueError("Evaluation method/protocol must be explicit nonempty names")
            if not isinstance(case_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", case_digest):
                raise ValueError("Evaluation case list requires a SHA-256 digest")
            self.check_time()
            self.check_storage()
            if group_id in self._data["evaluation_groups"] or len(self._data["evaluation_groups"]) >= self.config.data["evaluation"]["max_groups"]:
                raise BudgetExceeded("Evaluation group already claimed or group cap exhausted; failed claims are not refunded")
            token = uuid.uuid4().hex
            data = deepcopy(self._data)
            data["evaluation_groups"][group_id] = {"token": token, "method": method, "protocol": protocol,
                "case_digest": case_digest, "claimed_at": time.time(), "completed": False}
            self._save(data)
            return token

    def complete_evaluation_group(self, group_id, report_sha256):
        with self._mutex:
            self._ensure_open()
            if not isinstance(report_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", report_sha256):
                raise ValueError("Completed evaluation report requires its SHA-256")
            group = self._data["evaluation_groups"].get(group_id)
            if group is None or group["completed"]:
                raise ValueError("Unknown/already completed evaluation group")
            data = deepcopy(self._data)
            data["evaluation_groups"][group_id].update(completed=True, report_sha256=report_sha256)
            self._save(data)

    def snapshot(self):
        with self._mutex:
            self._ensure_open()
            data = deepcopy(self._data)
            self._touch_time(data)
            data["storage"] = self._storage_usage()
            return data

    def close(self):
        with self._mutex:
            if not self._closed:
                if self._poisoned:
                    # The failed operation never returned a reservation token.
                    # Release ownership without writing; a new owner must read
                    # the actual on-disk ledger, including any committed spend.
                    _unlock(self.lock_path, self._token)
                    self._closed = True
                    return
                self._ensure_open()
                self._save(deepcopy(self._data))
                _unlock(self.lock_path, self._token)
                self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()
