"""Durable, evaluation-only paired account; no tensor or simulator imports.

This is independent of C's training ledger. The registry permanently binds one
evaluation identity to the original experiment, artifacts and ordered cases.
New output directories and process restarts never replenish its fixed limits.
An interrupted operation/group can be audited, but cannot be silently retried.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

from .policy_protocol import (BudgetBusy, BudgetExceeded, _atomic_json, _identifier,
                              _lock, _read_json, _unlock)


LIMITS = {"groups": 2, "control": 12000, "plan": 1520, "reset": 80,
          "wall_time_seconds": 7200, "storage_bytes": 512 * 1024**2}
METHODS = ("base_training", "delta_training")
PROTOCOL = "official-t-l-com-v1"
_KINDS = ("control", "plan", "reset")
_INDEX_KIND = "tdyssey.paired_evaluation_registry"
_LEDGER_KIND = "tdyssey.paired_evaluation_ledger"


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + " must be a bounded integer")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(name + " must be a lowercase SHA-256 digest")
    return value


def validate_binding(value):
    expected = {"training_experiment_id", "config_hash", "base_identity", "delta_identity",
                "case_digest", "methods", "max_controls"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Evaluation binding must declare the exact training/artifact/case identities")
    _identifier(value["training_experiment_id"], "training experiment ID")
    _digest(value["config_hash"], "training config hash")
    _digest(value["case_digest"], "ordered case digest")
    for name in ("base_identity", "delta_identity"):
        identity = value[name]
        if not isinstance(identity, dict) or set(identity) != {"bytes", "sha256"}:
            raise ValueError(name + " requires exact byte count and SHA-256")
        _integer(identity["bytes"], name + " bytes", 1)
        _digest(identity["sha256"], name + " SHA-256")
    if value["methods"] != list(METHODS) or type(value["max_controls"]) is not int or value["max_controls"] != 300:
        raise ValueError("Only ordered base_training/delta_training, each capped at 300 controls/case, is authorized")
    return deepcopy(value)


def _binding_hash(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _counts():
    return dict.fromkeys(_KINDS, 0)


class EvaluationBudget:
    """One permanent paired-evaluation account, with an exclusive owner lock.

    create=True initializes only this evaluation account and starts its wall
    clock. Existing incomplete operations or group claims make a reopened owner
    audit-only. Storage telemetry remains permitted within its byte cap, even
    after the physical deadline, so failure evidence need not be discarded.
    """

    def __init__(self, registry_root, evaluation_id, binding, run_dir, create=False):
        self._binding = validate_binding(binding)
        self.binding_hash = _binding_hash(self.binding)
        self.evaluation_id = _identifier(evaluation_id, "evaluation ID")
        if type(create) is not bool:
            raise ValueError("create must be an explicit bool")
        self.registry_root, self.run_dir = Path(registry_root).resolve(), Path(run_dir).resolve()
        if (self.registry_root == self.run_dir or self.registry_root.is_relative_to(self.run_dir)
                or self.run_dir.is_relative_to(self.registry_root)):
            raise ValueError("Evaluation output and registry directories must be separate")
        self.directory = self.registry_root / self.evaluation_id
        self.path, self.index_path = self.directory / "ledger.json", self.registry_root / "index.json"
        self.lock_path = self.directory / "owner.lock"
        self._token, self._mutex = uuid.uuid4().hex, threading.RLock()
        self._closed, self._poisoned, self._active_group = True, False, None
        self._blocked_on_open = False
        self._pending_at_open = set()
        # Refuse a training registry before writing even an evaluation lock.
        if self.index_path.exists():
            self._validate_index(_read_json(self.index_path))
        if not create and (not self.index_path.is_file() or not self.path.is_file()):
            raise FileNotFoundError("No existing evaluation authority; explicit create=True is required")
        if create:
            self.registry_root.mkdir(parents=True, exist_ok=True)
        index_lock, index_token = self.registry_root / "index.lock", uuid.uuid4().hex
        _lock(index_lock, index_token)
        try:
            index = _read_json(self.index_path) if self.index_path.exists() else {
                "schema_version": 1, "kind": _INDEX_KIND, "evaluations": {}, "run_bindings": {}}
            self._validate_index(index)
            entry = index["evaluations"].get(self.evaluation_id)
            if create:
                if entry is not None or self.directory.exists():
                    raise FileExistsError("Evaluation ID already exists; it cannot be reinitialized")
                if any(item["binding_hash"] == self.binding_hash for item in index["evaluations"].values()):
                    raise FileExistsError("This paired evaluation binding already has an account; a new ID cannot refill it")
            elif entry is None or entry["binding_hash"] != self.binding_hash:
                raise ValueError("Evaluation registry identity/binding differs")
            self._check_output(self.run_dir, index)
            if create:
                self.directory.mkdir()
            _lock(self.lock_path, self._token)
            self._closed = False
            now = time.time()
            self._opened_time, self._opened_monotonic = now, time.monotonic()
            if create:
                self._data = {"schema_version": 1, "kind": _LEDGER_KIND,
                    "evaluation_id": self.evaluation_id, "registry_root": str(self.registry_root),
                    "binding": deepcopy(self.binding), "binding_hash": self.binding_hash,
                    "limits": dict(LIMITS), "started_at": now, "max_observed_time": now,
                    "elapsed_wall_seconds": 0.0, "reserved": {"development": _counts()},
                    "completed": {"development": _counts()}, "by_shape": {}, "last_event_id": 0,
                    "pending": {}, "recent_events": [], "evaluation_groups": {},
                    "output_dirs": [], "written_files": {}}
                index["evaluations"][self.evaluation_id] = {"binding_hash": self.binding_hash,
                    "ledger": f"{self.evaluation_id}/ledger.json", "created_at": now}
            else:
                self._data = _read_json(self.path)
                self._validate_data()
                self._opened_time = max(now, self._data["max_observed_time"])
                self._blocked_on_open = bool(self._data["pending"] or any(
                    not group["completed"] for group in self._data["evaluation_groups"].values()))
                self._pending_at_open = set(self._data["pending"])
            self._bind_output(self.run_dir, index)
            self._save(deepcopy(self._data))
            _atomic_json(self.index_path, index)
        except BaseException:
            if not self._closed:
                _unlock(self.lock_path, self._token)
                self._closed = True
            raise
        finally:
            _unlock(index_lock, index_token)

    def _validate_index(self, index):
        if (not isinstance(index, dict) or set(index) != {"schema_version", "kind", "evaluations", "run_bindings"}
                or index["schema_version"] != 1 or index["kind"] != _INDEX_KIND
                or not isinstance(index["evaluations"], dict) or not isinstance(index["run_bindings"], dict)):
            raise ValueError("Not a valid evaluation-only registry; training registries are unsupported")
        hashes = []
        for name, entry in index["evaluations"].items():
            _identifier(name, "registered evaluation ID")
            _digest(entry["binding_hash"], "registered binding hash")
            if entry["ledger"] != f"{name}/ledger.json":
                raise ValueError("Registry ledger identity differs")
            hashes.append(entry["binding_hash"])
        if len(hashes) != len(set(hashes)):
            raise ValueError("Duplicate paired evaluation binding in registry")

    def _check_output(self, path, index):
        expected = {"schema_version": 1, "kind": _LEDGER_KIND, "evaluation_id": self.evaluation_id,
                    "registry_root": str(self.registry_root), "binding_hash": self.binding_hash}
        reference = path / "evaluation-account.json"
        owner = index["run_bindings"].get(str(path))
        if owner is not None and owner != self.evaluation_id:
            raise ValueError("Output directory belongs to another evaluation")
        if reference.exists():
            if _read_json(reference) != expected or owner != self.evaluation_id:
                raise ValueError("Output reference differs from permanent evaluation authority")
        elif path.exists() and any(path.iterdir()):
            raise ValueError("Refusing an unbound nonempty output directory, including training/B artifacts")
        return expected

    def _bind_output(self, path, index):
        reference = self._check_output(path, index)
        path.mkdir(parents=True, exist_ok=True)
        if not (path / "evaluation-account.json").exists():
            _atomic_json(path / "evaluation-account.json", reference)
        index["run_bindings"][str(path)] = self.evaluation_id
        if str(path) not in self._data["output_dirs"]:
            self._data["output_dirs"].append(str(path))

    def _validate_data(self):
        data = self._data
        if (data.get("schema_version") != 1 or data.get("kind") != _LEDGER_KIND
                or data.get("evaluation_id") != self.evaluation_id or data.get("registry_root") != str(self.registry_root)
                or data.get("binding") != self.binding or data.get("binding_hash") != self.binding_hash
                or data.get("limits") != LIMITS):
            raise ValueError("Evaluation ledger identity/configuration differs; refusing reset")
        for key in ("started_at", "max_observed_time", "elapsed_wall_seconds"):
            if type(data[key]) not in (int, float) or not math.isfinite(data[key]) or data[key] < 0:
                raise ValueError("Invalid persistent evaluation clock")
        if data["max_observed_time"] < data["started_at"]:
            raise ValueError("Persistent evaluation clock moved backward")
        _integer(data["last_event_id"], "last event ID")
        if not set(data["by_shape"]).issubset({"T", "L"}):
            raise ValueError("Evaluation shape counters contain an unsupported shape")
        for stats in data["by_shape"].values():
            if set(stats) != {"reserved", "completed"}:
                raise ValueError("Invalid per-shape evaluation counters")
            for mode in ("reserved", "completed"):
                if set(stats[mode]) != set(_KINDS):
                    raise ValueError("Invalid per-shape operation kinds")
                for kind in _KINDS:
                    _integer(stats[mode][kind], "shape " + mode + " " + kind, 0, LIMITS[kind])
                    if stats["completed"][kind] > stats["reserved"][kind]:
                        raise ValueError("Shape completions exceed reservations")
        pending = _counts()
        for identifier, event in data["pending"].items():
            if str(event["id"]) != identifier or event["completed"] is not False:
                raise ValueError("Invalid pending evaluation operation")
            _integer(event["id"], "event ID", 1, data["last_event_id"])
            _integer(event["amount"], "event amount", 1)
            if event["kind"] != "storage":
                self._kind(event["kind"], event["phase"])
                if (event["shape"] not in ("T", "L") or event["group_id"] not in data["evaluation_groups"]
                        or data["evaluation_groups"][event["group_id"]]["completed"]):
                    raise ValueError("Pending evaluation operation has no active group/shape")
                pending[event["kind"]] += event["amount"]
            else:
                _integer(event["existing_bytes"], "existing output bytes")
        for mode in ("reserved", "completed"):
            if set(data[mode]) != {"development"} or set(data[mode]["development"]) != set(_KINDS):
                raise ValueError("Evaluation-only ledger contains unsupported training/global counters")
            for kind in _KINDS:
                _integer(data[mode]["development"][kind], mode + " " + kind, 0, LIMITS[kind])
        for kind in _KINDS:
            if data["reserved"]["development"][kind] - data["completed"]["development"][kind] != pending[kind]:
                raise ValueError("Pending evaluation operations disagree with durable counters")
            for mode in ("reserved", "completed"):
                if sum(stats[mode][kind] for stats in data["by_shape"].values()) != data[mode]["development"][kind]:
                    raise ValueError("Per-shape evaluation counters disagree")
        groups = sorted(data["evaluation_groups"].values(), key=lambda group: group["claim_id"])
        if (len(groups) > 2 or [group["method"] for group in groups] != list(METHODS[:len(groups)])
                or len({group["claim_id"] for group in groups}) != len(groups)):
            raise ValueError("Invalid ordered paired evaluation group claims")
        if len(data["recent_events"]) > 64:
            raise ValueError("Evaluation recent history exceeds its bound")
        for group in groups:
            if group["protocol"] != PROTOCOL or group["case_digest"] != self.binding["case_digest"]:
                raise ValueError("Evaluation group case/protocol binding differs")
            _integer(group["claim_id"], "group claim ID", 1, data["last_event_id"])
            if type(group["completed"]) is not bool:
                raise ValueError("Group completion must be explicit")
            if group["completed"]:
                _digest(group["report_sha256"], "completed group report digest")
            for mode, key in (("reserved", "reserved"), ("completed", "completed_counts")):
                if set(group[key]) != set(_KINDS):
                    raise ValueError("Group operation counters differ")
                for kind in _KINDS:
                    _integer(group[key][kind], "group " + key + " " + kind, 0, LIMITS[kind] // 2)
                    if group["completed_counts"][kind] > group["reserved"][kind]:
                        raise ValueError("Group completed operations exceed reservations")
        if any(not group["completed"] for group in groups[:-1]):
            raise ValueError("A later group cannot follow an incomplete group")
        for mode, key in (("reserved", "reserved"), ("completed", "completed_counts")):
            for kind in _KINDS:
                if sum(group[key][kind] for group in groups) != data[mode]["development"][kind]:
                    raise ValueError("Group and total operation counters disagree")

    @staticmethod
    def _kind(kind, phase):
        if phase != "development" or kind not in _KINDS:
            raise ValueError("Evaluation account accepts only development control/plan/reset; training is forbidden")
        return kind

    @property
    def binding(self):
        """Expose a copy so report construction cannot change the account identity."""
        return deepcopy(self._binding)

    def _ensure_open(self):
        if self._closed or self._poisoned:
            raise BudgetBusy("Evaluation owner is closed or poisoned after persistence failure")
        try:
            if _read_json(self.lock_path).get("token") != self._token:
                raise BudgetBusy("Evaluation lock ownership changed")
        except (OSError, ValueError) as exc:
            raise BudgetBusy("Evaluation lock is missing/corrupt") from exc

    def _touch_time(self, data):
        now = max(time.time(), self._opened_time + max(0.0, time.monotonic() - self._opened_monotonic),
                  data["max_observed_time"])
        data["max_observed_time"] = now
        data["elapsed_wall_seconds"] = max(data["elapsed_wall_seconds"], now - data["started_at"])

    def _save(self, data):
        self._touch_time(data)
        try:
            _atomic_json(self.path, data)
        except BaseException:
            self._poisoned = True  # os.replace may already have committed; never overwrite stale memory.
            raise
        self._data = data

    def _remember(self, data, event):
        data["recent_events"] = [row for row in data["recent_events"] if row["id"] != event["id"]]
        data["recent_events"] = (data["recent_events"] + [deepcopy(event)])[-64:]

    def check_time(self):
        with self._mutex:
            self._ensure_open()
            self._save(deepcopy(self._data))
            remaining = max(0.0, LIMITS["wall_time_seconds"] - self._data["elapsed_wall_seconds"])
            if not remaining:
                raise BudgetExceeded("Evaluation wall deadline exhausted, including downtime")
            return remaining

    def remaining(self, kind, phase="development"):
        with self._mutex:
            self._ensure_open(); self._kind(kind, phase)
            return LIMITS[kind] - self._data["reserved"]["development"][kind]

    def reserved_count(self, kind, phase="development"):
        with self._mutex:
            self._ensure_open(); self._kind(kind, phase)
            return self._data["reserved"]["development"][kind]

    def _physical_ready(self):
        if self._blocked_on_open or self._data["pending"]:
            raise BudgetBusy("Interrupted evaluation remains consumed; automatic retry/reclaim is forbidden")

    def reserve(self, kind, phase="development", shape=None, amount=1):
        with self._mutex:
            self._ensure_open(); self._kind(kind, phase)
            _integer(amount, "physical reservation amount", 1)
            if shape not in ("T", "L"):
                raise ValueError("Evaluation physical operations require T/L shape metadata")
            self._physical_ready()
            if self._active_group is None:
                raise BudgetBusy("Claim the fixed evaluation group before any physical operation")
            self.check_time(); self.check_storage()
            group = self._data["evaluation_groups"][self._active_group]
            if amount > self.remaining(kind) or group["reserved"][kind] + amount > LIMITS[kind] // 2:
                raise BudgetExceeded("Fixed paired evaluation operation cap exhausted; no operation attempted")
            data = deepcopy(self._data); data["last_event_id"] += 1
            event = {"id": data["last_event_id"], "kind": kind, "phase": phase, "shape": shape,
                     "group_id": self._active_group, "amount": amount, "completed": False,
                     "reserved_at": time.time()}
            data["reserved"]["development"][kind] += amount
            stats = data["by_shape"].setdefault(shape, {"reserved": _counts(), "completed": _counts()})
            stats["reserved"][kind] += amount
            data["evaluation_groups"][self._active_group]["reserved"][kind] += amount
            data["pending"][str(event["id"])] = event
            self._remember(data, event); self._save(data)
            return event["id"]

    def mark_completed(self, identifier):
        with self._mutex:
            self._ensure_open(); _integer(identifier, "reservation ID", 1)
            event = self._data["pending"].get(str(identifier))
            if self._blocked_on_open or event is None or event["kind"] == "storage" or event["group_id"] != self._active_group:
                raise ValueError("Unknown, interrupted-owner, or nonphysical evaluation reservation")
            data = deepcopy(self._data); event = data["pending"].pop(str(identifier))
            data["completed"]["development"][event["kind"]] += event["amount"]
            data["by_shape"][event["shape"]]["completed"][event["kind"]] += event["amount"]
            data["evaluation_groups"][event["group_id"]]["completed_counts"][event["kind"]] += event["amount"]
            event["completed"] = True
            self._remember(data, event); self._save(data)  # Completion recording survives the deadline.

    def claim_evaluation_group(self, group_id, *, method, protocol, case_digest):
        with self._mutex:
            self._ensure_open(); self._physical_ready()
            _identifier(group_id, "evaluation group ID")
            groups = self._data["evaluation_groups"]
            if group_id in groups or len(groups) >= 2:
                raise BudgetExceeded("Evaluation group is consumed or paired group cap exhausted")
            if self._active_group is not None or any(not group["completed"] for group in groups.values()):
                raise BudgetBusy("Complete the existing group; interrupted claims cannot be retried")
            if method != METHODS[len(groups)] or protocol != PROTOCOL or case_digest != self.binding["case_digest"]:
                raise ValueError("Group method/order/protocol/ordered case digest differs from fixed binding")
            self.check_time(); self.check_storage()
            data = deepcopy(self._data); data["last_event_id"] += 1
            token = uuid.uuid4().hex
            data["evaluation_groups"][group_id] = {"token": token, "claim_id": data["last_event_id"],
                "method": method, "protocol": protocol, "case_digest": case_digest,
                "claimed_at": time.time(), "completed": False, "reserved": _counts(), "completed_counts": _counts()}
            # Keep operation counters distinct from the group's completion flag.
            data["evaluation_groups"][group_id]["completed"] = False
            self._save(data); self._active_group = group_id
            return token

    def complete_evaluation_group(self, group_id, report_sha256):
        with self._mutex:
            self._ensure_open(); _digest(report_sha256, "evaluation report SHA-256")
            if self._blocked_on_open or group_id != self._active_group or self._data["pending"]:
                raise BudgetBusy("Interrupted/unowned group or pending operation cannot be declared complete")
            data = deepcopy(self._data)
            data["evaluation_groups"][group_id].update(completed=True, report_sha256=report_sha256,
                                                       completed_at=time.time())
            self._save(data); self._active_group = None

    def _owned_path(self, path):
        target = Path(path).resolve()
        if not any(target != Path(directory) and target.is_relative_to(Path(directory)) for directory in self._data["output_dirs"]):
            raise ValueError("Evaluation output path is outside its registered directories")
        return target

    def _storage_usage(self):
        files = {}
        for root in [self.directory, *(Path(path) for path in self._data["output_dirs"])]:
            for directory, subdirectories, names in os.walk(root, followlinks=False):
                if any((Path(directory) / name).is_symlink() for name in [*subdirectories, *names]):
                    raise ValueError("Symlinked evaluation output is unsupported")
                for name in names:
                    path = Path(directory) / name
                    files[str(path.resolve())] = path.stat().st_size
        if self.index_path.exists():
            files[str(self.index_path)] = self.index_path.stat().st_size
        pending = sum(max(0, event["amount"] - max(0, files.get(event.get("path"), 0) - event["existing_bytes"]))
                      for event in self._data["pending"].values() if event["kind"] == "storage")
        return {"current_bytes": sum(files.values()), "pending_bytes": pending,
                "limit_bytes": LIMITS["storage_bytes"]}

    def check_storage(self):
        with self._mutex:
            self._ensure_open(); usage = self._storage_usage()
            if usage["current_bytes"] + usage["pending_bytes"] > usage["limit_bytes"]:
                raise BudgetExceeded("Evaluation aggregate storage cap exhausted")
            return usage

    def reserve_storage(self, amount, path=None):
        with self._mutex:
            self._ensure_open(); _integer(amount, "storage reservation bytes", 1)
            if amount > LIMITS["storage_bytes"]:
                raise BudgetExceeded("Evaluation write reservation exceeds its entire fixed storage cap")
            usage = self.check_storage()
            if usage["current_bytes"] + usage["pending_bytes"] + amount > LIMITS["storage_bytes"]:
                raise BudgetExceeded("Evaluation storage reservation would exceed its cap")
            target = self._owned_path(path) if path is not None else None
            if target is not None and any(event["kind"] == "storage" and event["path"] == str(target)
                                          for event in self._data["pending"].values()):
                raise BudgetBusy("Evaluation output already has an incomplete write reservation")
            data = deepcopy(self._data); data["last_event_id"] += 1
            event = {"id": data["last_event_id"], "kind": "storage", "amount": amount,
                     "path": str(target) if target else None,
                     "existing_bytes": target.stat().st_size if target is not None and target.exists() else 0,
                     "completed": False, "reserved_at": time.time()}
            data["pending"][str(event["id"])] = event
            self._remember(data, event); self._save(data)
            return event["id"]

    def mark_storage_written(self, identifier, actual_bytes):
        with self._mutex:
            self._ensure_open(); _integer(identifier, "storage reservation ID", 1)
            event = self._data["pending"].get(str(identifier))
            if str(identifier) in self._pending_at_open or event is None or event["kind"] != "storage":
                raise ValueError("Unknown/already completed evaluation storage reservation")
            _integer(actual_bytes, "actual written bytes", 0, event["amount"])
            target = Path(event["path"]) if event["path"] is not None else None
            if target is not None and (not target.is_file() or target.stat().st_size != event["existing_bytes"] + actual_bytes):
                raise ValueError("Evaluation file growth differs from its durable write reservation")
            data = deepcopy(self._data); event = data["pending"].pop(str(identifier))
            event.update(completed=True, actual_bytes=actual_bytes)
            if target is not None:
                data["written_files"][str(target)] = {"bytes": target.stat().st_size, "reservation_id": identifier}
            self._remember(data, event); self._save(data)
            return self.check_storage()

    def snapshot(self):
        with self._mutex:
            self._ensure_open(); data = deepcopy(self._data); self._touch_time(data)
            data["storage"] = self._storage_usage()
            data["audit_only_after_interruption"] = self._blocked_on_open
            return data

    def close(self):
        with self._mutex:
            if self._closed:
                return
            try:
                if not self._poisoned:
                    self._ensure_open(); self._save(deepcopy(self._data))
            finally:
                _unlock(self.lock_path, self._token); self._closed = True

    def __enter__(self):
        self._ensure_open(); return self

    def __exit__(self, *args):
        self.close()


EvalBudget = EvaluationBudget
