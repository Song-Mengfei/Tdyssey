"""One durable, nonrenewable frozen-model diagnostic attempt; stdlib only.

Reservations precede operations and are never refunded, including aborted work.
This authority has no interface to the B/C training or evaluation accounts.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = ROOT / "artifacts/frozen-wm-diagnostic-registry-v1"
ATTEMPT_ID = "frozen-wm-v1-20261024"
LIMITS = {"branches": 64, "calibration": 24, "reset": 88, "restore": 88,
          "settle": 88, "controls": 448, "substeps": 4480,
          "forecast_images": 768, "euler_steps": 2304, "custom_images": 192,
          "optimizer_updates": 0, "output_bytes": 256 * 1024**2,
          "telemetry_bytes": 64 * 1024**2}
WALL_SECONDS = 1800
KIND = "tdyssey.frozen_world_model_diagnostic.v1"


class BudgetExceeded(RuntimeError):
    """A fixed diagnostic resource cap has been reached."""


class BudgetBusy(RuntimeError):
    """The authority is unavailable or an operation is already pending."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(name + " must be a lowercase SHA-256 digest")


def validate_binding(binding):
    if not isinstance(binding, dict):
        raise ValueError("Diagnostic binding must be an object")
    expected = {"protocol_sha256", "config_sha256", "source_sha256", "checkpoint_sha256"}
    if not expected.issubset(binding):
        raise ValueError("Binding requires protocol/config/source/checkpoint SHA-256 identities")
    result = deepcopy(binding)
    for name in expected - {"source_sha256"}:
        _digest(result[name], name)
    sources = result["source_sha256"]
    if isinstance(sources, dict):
        if not sources or any(not isinstance(name, str) or not name for name in sources):
            raise ValueError("Source digest mapping must have nonempty file names")
        for name, digest in sources.items():
            _digest(digest, name)
    else:
        _digest(sources, "source_sha256")
    canonical_hash(result)  # Reject non-JSON/nonfinite metadata before any writes.
    return result


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _exclusive_json(path, value):
    with Path(path).open("xb") as out:
        out.write(_json_bytes(value))
        out.flush()
        os.fsync(out.fileno())


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        _exclusive_json(temporary, value)
        replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def replace_with_retry(source, destination):
    """Bounded retry for transient Windows sharing/access rejection only."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            transient = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)
            if not transient or attempt == 4:
                raise
            time.sleep(.02 * (attempt + 1))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _same_non_source_binding(original, current):
    return {k: v for k, v in original.items() if k != "source_sha256"} == {
        k: v for k, v in current.items() if k != "source_sha256"}


def replay_journal(claim, rows, *, snapshot_event=None):
    """Rebuild authority from fsynced events, including a journal-ahead reserve."""
    original = validate_binding(claim["binding"])
    original_hash = canonical_hash(original)
    started = claim["started_at"]
    if type(started) not in (int, float) or not math.isfinite(started) or started < 0:
        raise ValueError("Invalid original diagnostic start time")
    data = {"schema_version": 1, "kind": KIND, "binding": original, "binding_hash": original_hash,
        "output_dir": claim["output_dir"], "ledger_dir": claim["ledger_dir"],
        "registry_claim": claim["registry_claim"], "started_at": started, "max_observed_time": started,
        "elapsed_seconds": 0.0, "wall_limit_seconds": WALL_SECONDS, "limits": dict(LIMITS),
        "reserved": dict.fromkeys(LIMITS, 0), "completed": dict.fromkeys(LIMITS, 0),
        "aborted": dict.fromkeys(LIMITS, 0), "pending": {}, "operations": {}, "event_id": 0,
        "policy_optimizer_updates": 0, "world_model_optimizer_updates": 0,
        "source_lineage": [], "active_binding": deepcopy(original)}
    states = {}
    for event in rows:
        if type(event.get("event_id")) is not int or event["event_id"] != data["event_id"] + 1:
            raise ValueError("Diagnostic journal event IDs are not contiguous and monotone")
        elapsed = event.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < data["elapsed_seconds"]:
            raise ValueError("Diagnostic journal elapsed clock moved backward or is invalid")
        kind, detail = event.get("kind"), event.get("detail")
        if not isinstance(detail, dict):
            raise ValueError("Diagnostic journal event detail must be an object")
        if kind == "initialize":
            if event["event_id"] != 1 or detail.get("binding_hash") != original_hash:
                raise ValueError("Diagnostic journal initialization binding differs")
        elif event["event_id"] == 1:
            raise ValueError("Diagnostic journal must begin with initialization")
        elif kind == "reserve":
            identifier, costs = detail.get("operation_id"), detail.get("costs")
            if identifier != str(len(data["operations"]) + 1) or data["pending"]:
                raise ValueError("Diagnostic journal operation order/pending reservation differs")
            if (not isinstance(costs, dict) or not costs or any(name not in LIMITS for name in costs)
                    or any(type(v) is not int or v < 0 for v in costs.values()) or not any(costs.values())):
                raise ValueError("Diagnostic journal reservation costs are invalid")
            for name, amount in costs.items():
                if data["reserved"][name] + amount > LIMITS[name]:
                    raise ValueError("Diagnostic journal exceeded fixed cap: " + name)
                data["reserved"][name] += amount
            operation = {"costs": deepcopy(costs), "status": "pending", "detail": None}
            data["pending"][identifier] = deepcopy(operation)
            data["operations"][identifier] = operation
        elif kind in ("completed", "aborted"):
            identifier = detail.get("operation_id")
            if identifier not in data["pending"]:
                raise ValueError("Diagnostic journal completes a nonpending reservation")
            operation = data["pending"].pop(identifier)
            for name, amount in operation["costs"].items():
                data[kind][name] += amount
            data["operations"][identifier].update(status=kind, detail=deepcopy(detail.get("detail")))
        elif kind == "resume":
            binding = validate_binding(detail.get("binding"))
            if not _same_non_source_binding(original, binding) or detail.get("original_binding_hash") != original_hash:
                raise ValueError("Diagnostic resume changed protocol/config/checkpoint or account identity")
            data["source_lineage"].append({"event_id": event["event_id"], "binding": deepcopy(binding),
                                            "binding_hash": canonical_hash(binding)})
            data["active_binding"] = deepcopy(binding)
        elif kind == "close":
            if detail.get("pending_operations") != list(data["pending"]):
                raise ValueError("Diagnostic journal close pending operations differ")
        else:
            raise ValueError("Unknown diagnostic journal event kind")
        data["event_id"] = event["event_id"]
        data["elapsed_seconds"] = float(elapsed)
        data["max_observed_time"] = max(data["max_observed_time"], started+elapsed)
        if data["event_id"] == snapshot_event:
            states[data["event_id"]] = {key: deepcopy(data[key]) for key in (
                "reserved", "completed", "aborted", "pending", "operations", "binding", "binding_hash", "started_at")}
    if not rows:
        raise ValueError("Cannot resume an empty diagnostic journal")
    return data, states


def storage_usage(output_dir, ledger_dir):
    """Actual allocated payload lengths; symlinks and inaccessible files fail closed."""
    output, ledger = Path(output_dir).resolve(), Path(ledger_dir).resolve()
    result = {"output_bytes": 0, "telemetry_bytes": 0}
    for directory, category in ((output, None), (ledger, "telemetry_bytes")):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            try:
                if path.is_symlink():
                    raise ValueError("Diagnostic storage cannot contain symbolic links")
                if path.is_file():
                    size = path.stat().st_size
                    name = category or ("telemetry_bytes" if path.relative_to(output).parts[0] == "telemetry"
                                        else "output_bytes")
                    result[name] += size
            except FileNotFoundError:
                # The worker atomically replaces JSON through short-lived tmp
                # files. A tmp disappearing after enumeration is not missing
                # evidence; later samples count the replacement destination.
                continue
    return result


class Budget:
    """Create exactly one attempt; existing ledgers are always read-only evidence.

    API: reserve(**costs) -> operation ID, complete(id, detail), abort(id, detail),
    check(), snapshot(), close(). An aborted reservation continues to consume its
    whole declared cost. A crash leaves the reservation pending and unretryable.
    """

    def __init__(self, output, ledger, binding, *, resume=False):
        if type(resume) is not bool:
            raise ValueError("resume must be explicit bool")
        if resume:
            self._resume(output, ledger, binding)
            return
        self.output, self.ledger = Path(output).resolve(), Path(ledger).resolve()
        if self.output == self.ledger or self.output.is_relative_to(self.ledger) or self.ledger.is_relative_to(self.output):
            raise ValueError("Diagnostic output and ledger directories must be separate")
        self.binding = validate_binding(binding)
        self.binding_hash = canonical_hash(self.binding)
        self.path, self.journal = self.ledger / "ledger.json", self.ledger / "journal.jsonl"
        self._mutex, self._closed, self._poisoned = threading.RLock(), True, False
        if self.ledger.exists() and any(self.ledger.iterdir()):
            raise FileExistsError("Existing diagnostic ledger cannot be restarted or replenished")
        if self.output.exists() and any(p.name != "telemetry" or not p.is_dir() for p in self.output.iterdir()):
            raise FileExistsError("Refusing nonempty diagnostic output; previous evidence is immutable")
        registry = Path(REGISTRY_ROOT).resolve()
        registry.mkdir(parents=True, exist_ok=True)
        # The authorization is for one attempt, regardless of later source,
        # checkpoint or configuration changes. The binding is evidence, not a
        # renewable account key. Never remove/replace this singleton claim.
        self.registry_path = registry / (ATTEMPT_ID + ".json")
        now, self._started_monotonic = time.time(), time.monotonic()
        self._started_time = now
        # Never delete this claim, even if initialization or the worker fails.
        _exclusive_json(self.registry_path, {"kind": KIND, "attempt_id": ATTEMPT_ID, "binding_hash": self.binding_hash,
            "binding": self.binding, "output_dir": str(self.output), "ledger_dir": str(self.ledger),
            "started_at": now})
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger.mkdir(parents=True, exist_ok=True)
        self._data = {"schema_version": 1, "kind": KIND, "binding": deepcopy(self.binding),
            "binding_hash": self.binding_hash, "output_dir": str(self.output), "ledger_dir": str(self.ledger),
            "registry_claim": str(self.registry_path), "started_at": now, "max_observed_time": now,
            "elapsed_seconds": 0.0, "wall_limit_seconds": WALL_SECONDS, "limits": dict(LIMITS),
            "reserved": dict.fromkeys(LIMITS, 0), "completed": dict.fromkeys(LIMITS, 0),
            "aborted": dict.fromkeys(LIMITS, 0), "pending": {}, "operations": {}, "event_id": 0,
            "policy_optimizer_updates": 0, "world_model_optimizer_updates": 0}
        self._closed = False
        self._record("initialize", {"binding_hash": self.binding_hash})

    @classmethod
    def resume(cls, output, ledger, binding):
        """Continue the same interrupted attempt, with its original deadline."""
        return cls(output, ledger, binding, resume=True)

    def _resume(self, output, ledger, binding):
        self.output, self.ledger = Path(output).resolve(), Path(ledger).resolve()
        current = validate_binding(binding)
        self.path, self.journal = self.ledger / "ledger.json", self.ledger / "journal.jsonl"
        self.registry_path = Path(REGISTRY_ROOT).resolve() / (ATTEMPT_ID + ".json")
        self._mutex, self._closed, self._poisoned = threading.RLock(), True, False
        claim = _read_json(self.registry_path)
        if (claim.get("kind") != KIND or claim.get("attempt_id") != ATTEMPT_ID
                or claim.get("output_dir") != str(self.output) or claim.get("ledger_dir") != str(self.ledger)
                or canonical_hash(claim.get("binding")) != claim.get("binding_hash")):
            raise ValueError("Diagnostic recovery must use the original singleton claim and paths")
        if not _same_non_source_binding(claim["binding"], current):
            raise ValueError("Diagnostic recovery can change source only, not protocol/config/checkpoint")
        original_bytes = self.path.read_bytes()
        saved = json.loads(original_bytes)
        claim["registry_claim"] = str(self.registry_path)
        lines = self.journal.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line for line in lines):
            raise ValueError("Diagnostic journal contains an empty/truncated event")
        rows = [json.loads(line) for line in lines]
        saved_event = saved.get("event_id")
        rebuilt, states = replay_journal(claim, rows, snapshot_event=saved_event)
        if (saved.get("kind") != KIND or saved.get("limits") != LIMITS or saved_event not in states
                or saved.get("output_dir") != str(self.output) or saved.get("ledger_dir") != str(self.ledger)
                or any(saved.get(key) != value for key, value in states[saved_event].items())):
            raise ValueError("Saved diagnostic ledger is not a matching prefix of its durable journal")
        now = time.time()
        elapsed = max(rebuilt["elapsed_seconds"], now-rebuilt["started_at"])
        if elapsed >= WALL_SECONDS:
            raise BudgetExceeded("Original diagnostic 1800-second deadline expired during downtime")
        # Serialize recovery owners without modifying or replacing the singleton.
        self._resume_lock = self.ledger / "recovery-owner.lock"
        self._resume_token = uuid.uuid4().hex
        try:
            _exclusive_json(self._resume_lock, {"pid": os.getpid(), "token": self._resume_token})
        except FileExistsError as exc:
            raise BudgetBusy("Another recovery owner is active or an earlier recovery needs audit") from exc
        try:
            backup = self.ledger / f"ledger-before-recovery-event-{saved_event}-{uuid.uuid4().hex}.json"
            with backup.open("xb") as out:
                out.write(original_bytes)
                out.flush()
                os.fsync(out.fileno())
            self._data = rebuilt
            self.binding, self.binding_hash = deepcopy(claim["binding"]), claim["binding_hash"]
            self._started_time = rebuilt["started_at"]
            self._started_monotonic = time.monotonic() - max(0.0, elapsed)
            self._closed = False
            self._data["source_lineage"].append({"event_id": self._data["event_id"]+1,
                "binding": deepcopy(current), "binding_hash": canonical_hash(current)})
            self._data["active_binding"] = deepcopy(current)
            self._record("resume", {"binding": current, "original_binding_hash": self.binding_hash,
                "saved_ledger_event": saved_event, "replayed_journal_event": rows[-1]["event_id"], "backup": backup.name})
            for identifier in list(self._data["pending"]):
                self.abort(identifier, {"reason": "interrupted_operation_conservatively_charged_no_retry",
                                        "recovered_from_journal": True})
        except BaseException:
            self._closed = True
            self._release_resume_lock()
            raise

    def _release_resume_lock(self):
        path = getattr(self, "_resume_lock", None)
        if path is not None and path.exists():
            identity = _read_json(path)
            if identity.get("token") != self._resume_token:
                raise BudgetBusy("Diagnostic recovery lock changed owner")
            path.unlink()

    def _ensure_open(self):
        if self._closed or self._poisoned:
            raise BudgetBusy("Diagnostic authority is closed or poisoned after persistence failure")

    def _touch(self):
        now = max(time.time(), self._started_time + max(0.0, time.monotonic() - self._started_monotonic),
                  self._data["max_observed_time"])
        self._data["max_observed_time"] = now
        self._data["elapsed_seconds"] = max(self._data["elapsed_seconds"], now - self._started_time)

    def _record(self, kind, detail):
        self._touch()
        self._data["event_id"] += 1
        row = {"event_id": self._data["event_id"], "kind": kind, "detail": deepcopy(detail),
               "at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": self._data["elapsed_seconds"]}
        try:
            with self.journal.open("ab") as out:
                out.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
                out.flush()
                os.fsync(out.fileno())
            _atomic_json(self.path, self._data)
        except BaseException:
            self._poisoned = True
            raise

    def check(self):
        with self._mutex:
            self._ensure_open()
            self._touch()
            if self._data["elapsed_seconds"] >= WALL_SECONDS:
                raise BudgetExceeded("Frozen diagnostic 1800-second wall deadline reached")
            used = storage_usage(self.output, self.ledger)
            for key, amount in used.items():
                if max(amount, self._data["reserved"][key]) > LIMITS[key]:
                    raise BudgetExceeded("Frozen diagnostic storage cap exceeded: " + key)
            return {"elapsed_seconds": self._data["elapsed_seconds"], **used}

    def reserve(self, **costs):
        with self._mutex:
            observed = self.check()
            if not costs or any(name not in LIMITS for name in costs):
                raise ValueError("Only fixed frozen-diagnostic cost names are accepted")
            if any(type(value) is not int or value < 0 for value in costs.values()) or not any(costs.values()):
                raise ValueError("Reservations require nonnegative integer costs and a positive total")
            if self._data["pending"]:
                raise BudgetBusy("Complete or abort the current operation before reserving another")
            for name, amount in costs.items():
                baseline = self._data["reserved"][name]
                if name in ("output_bytes", "telemetry_bytes"):
                    baseline = max(baseline, observed[name])
                if baseline + amount > LIMITS[name]:
                    raise BudgetExceeded("Frozen diagnostic reservation exceeds " + name)
            identifier = str(len(self._data["operations"]) + 1)
            operation = {"costs": dict(costs), "status": "pending", "detail": None}
            for name, amount in costs.items():
                self._data["reserved"][name] += amount
            self._data["pending"][identifier] = deepcopy(operation)
            self._data["operations"][identifier] = operation
            self._record("reserve", {"operation_id": identifier, "costs": costs})
            return identifier

    def _finish(self, operation, detail, status):
        with self._mutex:
            self._ensure_open()
            canonical_hash(detail)  # Validate before changing durable counts.
            if operation not in self._data["pending"]:
                raise ValueError("Operation is not a pending diagnostic reservation")
            row = self._data["pending"].pop(operation)
            for name, amount in row["costs"].items():
                self._data[status][name] += amount
            self._data["operations"][operation].update(status=status, detail=deepcopy(detail))
            self._record(status, {"operation_id": operation, "detail": detail})

    def complete(self, operation, detail=None):
        self._finish(operation, detail, "completed")

    def abort(self, operation, detail=None):
        self._finish(operation, detail, "aborted")

    def snapshot(self):
        with self._mutex:
            self._touch()
            return deepcopy(self._data)

    def close(self):
        with self._mutex:
            if not self._closed:
                try:
                    if not self._poisoned:
                        self._record("close", {"pending_operations": list(self._data["pending"])})
                finally:
                    self._closed = True
                    self._release_resume_lock()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
