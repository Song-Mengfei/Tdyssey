"""Independent, non-resumable accounting for the input-compatibility proposal.

Only standard-library bookkeeping lives here. Creating this object is not
execution authority. The caller must validate assets/resources and must enter a
sampler only after ``reserve_call`` returns successfully. The append-only journal
is authoritative; an existing claim is never reopened, including after a crash.
"""

from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

from .wm_compat_storage import Coordinator, StorageError, scan_files, unlinked_path
from .wm_compat_identity import ATTEMPT_ID, REGISTRY


MIB = 1024 * 1024
IMAGE_BYTES = 3 * 96 * 96 * 4
EXACT_CHECKPOINT = "4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892"
DEFAULT_LIMITS = {
    "seconds": 1800,
    "images": 2268,
    "euler_steps": 6804,
    "model_loads": 1,
    "result_bytes": 256 * MIB,
    "telemetry_bytes": 16 * MIB,
    "raw_bytes": 250822656,
    "metadata_bytes": 12 * MIB,
    "header_bytes": MIB,
    "temporary_bytes": MIB,
    "chunk_bytes": MIB,
}


class BudgetError(RuntimeError):
    """The attempt cannot continue; reservations are never refunded."""
    def __init__(self,message,*,details=None):
        self.details = details
        super().__init__(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_json(value))


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _hash_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def call_key(anchor, candidate, seed, arm, mode, horizon):
    """Return the sole canonical key format used by the fixed plan."""
    if (anchor not in [f"T{i}" for i in range(12, 16)] + [f"L{i:02d}" for i in range(12)]
            or candidate not in ("hold", "toward", "side", "away")
            or type(seed) is not int or seed not in (5478, 5479, 5480)
            or arm not in ("A", "B") or mode not in ("AR", "TF")
            or type(horizon) is not int or horizon not in ((1, 2, 3, 4) if mode == "AR" else (2, 4))):
        raise BudgetError("Call is outside the fixed compatibility plan")
    return f"{anchor}/{candidate}/{seed}/{mode}/{arm}/H{horizon}"


def canonical_plan():
    """Full 64-branch table: 2268 available calls and 36 unavailable calls."""
    branches, calls = [], []
    for anchor in [f"T{i}" for i in range(12, 16)] + [f"L{i:02d}" for i in range(12)]:
        phase = anchor[0]
        for candidate in ("hold", "toward", "side", "away"):
            missing = anchor == "L01" and candidate == "hold"
            branches.append(dict(anchor=anchor, shape=phase, candidate=candidate,
                                 status="missing" if missing else "available"))
            for seed in (5478, 5479, 5480):
                for mode, horizons in (("AR", (1, 2, 3, 4)), ("TF", (2, 4))):
                    for arm in ("A", "B"):
                        for horizon in horizons:
                            calls.append(dict(key=call_key(anchor, candidate, seed, arm, mode, horizon),
                                              phase=phase, anchor=anchor, candidate=candidate,
                                              seed=seed, arm=arm, mode=mode, horizon=horizon,
                                              status="missing" if missing else "planned"))
    return dict(schema_version=1, measurement_rows=528, seeds=[5478, 5479, 5480],
                branches=branches, calls=calls)


def _safe_root(repo, value):
    path = Path(value)
    try:
        path = unlinked_path(path if path.is_absolute() else repo / path)
    except StorageError as error:
        raise BudgetError(str(error),details=_copy(error.details)) from error
    artifacts = repo / "artifacts"
    try:
        relative = path.relative_to(artifacts)
    except ValueError as error:
        raise BudgetError("New accounting/output paths must be under this repository's artifacts") from error
    if not relative.parts or not relative.parts[0].startswith("wm-input-compat-"):
        raise BudgetError("Only the new wm-input-compat-* namespace is writable")
    if any(part.lower().startswith(("frozen", "milestone-b", "milestone-c")) for part in relative.parts):
        raise BudgetError("Old frozen/B/C paths are forbidden")
    return path


def _files(root):
    return {path: record.size for path, record in scan_files(root, phase="budget.files").items()}


def _transaction(method):
    """Order this owner's IO without retaining the cross-process quota lock."""
    @wraps(method)
    def locked(self, *args, **kwargs):
        try:
            with self._io_lock:
                return method(self, *args, **kwargs)
        except StorageError as error:
            if method.__name__ == "_append":
                # _emit owns the journal's first failure record; do not wrap
                # it in a generic storage abort before that owner sees it.
                raise
            self._abort("storage_failed: "+str(error), error=error)
    return locked


class CompatBudget:
    """Create once, reserve before entry, and never resume a claimed attempt."""

    @classmethod
    def create(cls, repo, ledger_dir, output_dir, binding, plan=None, *, clock=time.monotonic,
               io_hook=None, limits=None, telemetry_dirs=(), storage=None):
        started = float(clock())
        self = cls.__new__(cls)
        try:
            self.repo = unlinked_path(repo)
        except StorageError as error:
            raise BudgetError(str(error),details=_copy(error.details)) from error
        self.ledger_dir = _safe_root(self.repo, ledger_dir)
        self.output_dir = _safe_root(self.repo, output_dir)
        self.registry_dir = self.repo / "artifacts" / REGISTRY
        self.telemetry_dirs = [_safe_root(self.repo, path) for path in telemetry_dirs]
        telemetry_roots = [self.ledger_dir, self.registry_dir, *self.telemetry_dirs]
        for index, path in enumerate(telemetry_roots):
            if path.is_relative_to(self.output_dir) or self.output_dir.is_relative_to(path):
                raise BudgetError("Telemetry and result roots must be disjoint")
            for other in telemetry_roots[:index]:
                if path.is_relative_to(other) or other.is_relative_to(path):
                    raise BudgetError("Telemetry roots must not overlap")
        if (self.ledger_dir == self.output_dir or self.ledger_dir.is_relative_to(self.output_dir)
                or self.output_dir.is_relative_to(self.ledger_dir)
                or self.registry_dir.is_relative_to(self.ledger_dir)
                or self.registry_dir.is_relative_to(self.output_dir)
                or self.ledger_dir.is_relative_to(self.registry_dir)
                or self.output_dir.is_relative_to(self.registry_dir)):
            raise BudgetError("Ledger, results and fixed registry must be disjoint")
        for path in (self.ledger_dir, self.output_dir):
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise BudgetError("New ledger and results directories must be empty")
        if not math.isfinite(started):
            raise BudgetError("Clock must be finite")
        self.clock, self.io_hook = clock, io_hook
        self._last_clock = started
        self.limits = dict(DEFAULT_LIMITS)
        if limits:
            if set(limits) - set(self.limits):
                raise BudgetError("Unknown budget limit")
            self.limits.update(limits)
        for key, value in self.limits.items():
            adjustable = key in ("header_bytes", "temporary_bytes", "chunk_bytes")
            maximum = DEFAULT_LIMITS["result_bytes"] if adjustable else DEFAULT_LIMITS[key]
            if type(value) is not int or value <= 0 or value > maximum:
                raise BudgetError(f"Limits can only tighten the frozen maximum: {key}")
        if any(self.limits[key] != DEFAULT_LIMITS[key] for key in ("raw_bytes", "metadata_bytes")):
            raise BudgetError("Full raw payload and 12MiB metadata reserves cannot be reduced")
        self.serialization_bound = sum(self.limits[k] for k in
                                       ("raw_bytes", "metadata_bytes", "header_bytes", "temporary_bytes"))
        if self.serialization_bound > self.limits["result_bytes"]:
            raise BudgetError("Worst-case raw/metadata/header/temporary reservation exceeds results cap")
        if self.limits["chunk_bytes"] > self.limits["temporary_bytes"]:
            raise BudgetError("Each immutable chunk must fit its temporary reservation")
        if self.limits["telemetry_bytes"] <= 65536:
            raise BudgetError("Telemetry must retain emergency-stop headroom")
        self.binding = _copy(binding)
        if not isinstance(binding, dict) or not _hash_string(binding.get("protocol_sha256")):
            raise BudgetError("Binding must include protocol_sha256")
        if self.binding.get("attempt_id",ATTEMPT_ID) != ATTEMPT_ID:
            raise BudgetError("Binding differs from the fixed v3 attempt identity")
        for section in ("code_sha256", "asset_sha256"):
            values = binding.get(section)
            if not isinstance(values, dict) or not values or not all(
                    isinstance(key, str) and _hash_string(value) for key, value in values.items()):
                raise BudgetError(f"Binding must include nonempty {section} path/hash map")
        if binding.get("checkpoint_sha256") != EXACT_CHECKPOINT:
            raise BudgetError("Only the exact pinned phase-two checkpoint is allowed")
        self.plan = canonical_plan() if plan is None else _copy(plan)
        if self.plan != canonical_plan():
            raise BudgetError("The complete plan, including missing L01/hold, is immutable")
        self._call_plan = {row["key"]: row for row in self.plan["calls"]}
        self._calls, self._outputs = {}, {}
        self._output_tickets, self._owned_files = {}, {}
        self._io_lock = threading.RLock()
        self._journal_hash = hashlib.sha256()
        self._journal_size = 0
        self._journal_identity = self._journal_mtime_ns = None
        self._allocated = dict(raw=0, metadata=0, headers=0)
        self._seq, self._previous, self._journal_broken = 0, "0" * 64, False
        self._secondary_count = 0
        self._last_telemetry_bytes = None
        self._emergency_bytes = 0
        self._abort_state_lock = self._io_lock
        self._state = dict(status="active", phase="M", stop_reason=None, secondary_failure=None, started_monotonic=started,
                           deadline_monotonic=started + self.limits["seconds"], model_load_status="not_attempted",
                           counts=dict(images=0, euler_steps=0, model_loads=0, T_images=0, L_images=0),
                           gates=dict(M=None, T=None), stage_status=dict(M="active", T="not_attempted", L="not_attempted"))
        if storage is not None and getattr(storage,"emergency_headroom",None) != 65536:
            raise BudgetError("Shared storage must protect exactly 64KiB of emergency telemetry headroom")
        self.storage = storage if storage is not None else Coordinator(
            result_roots=(self.output_dir,), telemetry_roots=tuple(telemetry_roots),
            append_only=(self.ledger_dir / "journal.jsonl",),
            pending_roots=(self.output_dir,self.ledger_dir / "exports"),pending_reserve=2*MIB,
            emergency_headroom=65536,
            limits=dict(output_bytes=self.limits["result_bytes"],telemetry_bytes=self.limits["telemetry_bytes"]))
        self.nonce = uuid.uuid4().hex
        claim = dict(attempt_id=ATTEMPT_ID, nonce=self.nonce, pid=os.getpid(),
                     created_utc=datetime.now(timezone.utc).isoformat(),
                     started_monotonic=started, deadline_monotonic=self._state["deadline_monotonic"],
                     binding_sha256=_sha(_json(self.binding).encode()), plan_sha256=_sha(_json(self.plan).encode()),
                     ledger_dir=str(self.ledger_dir), output_dir=str(self.output_dir),
                     resume_allowed=False)
        return self._create_claim(claim)

    @_transaction
    def _create_claim(self, claim):
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        claim_path = self.registry_dir / "attempt.json"
        try:
            self._exclusive(claim_path, (_json(claim) + "\n").encode(), "claim")
        except StorageError as error:
            if error.details["exception_type"] == "FileExistsError":
                raise BudgetError("The fixed compatibility attempt is already claimed; no resume or replacement",
                                  details=_copy(error.details)) from error
            raise
        self.claim = claim
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            declaration = dict(claim=claim, binding=self.binding, plan=self.plan, limits=self.limits,
                               serialization_bound=self.serialization_bound,
                               telemetry_dirs=[str(path) for path in self.telemetry_dirs],
                               phase_caps=dict(T=dict(images=576, euler_steps=1728),
                                               L=dict(images=1692, euler_steps=5076)))
            payload = (_json(declaration) + "\n").encode()
            if self._telemetry_bytes() + len(payload) > self.limits["telemetry_bytes"]:
                raise BudgetError("Declaration does not fit the telemetry budget")
            self._exclusive(self.ledger_dir / "declaration.json", payload, "declaration")
            self._emit("created", dict(serialization_bound=self.serialization_bound))
            self.check()
        except Exception as error:
            self._abort(f"create_failed: {type(error).__name__}: {error}", error=error)
        return self

    def _hook(self, operation, path):
        if self.io_hook is not None:
            self.io_hook(operation, path)

    def _remember_owned(self, path):
        record = self.storage.remember_file(path)
        self._owned_files[str(path)] = dict(bytes=record.size,sha256=record.sha256,
            mtime_ns=record.mtime_ns,identity=list(record.identity))
        return record

    def _exclusive(self, path, payload, operation):
        ticket = None
        if operation not in ("emergency","chunk"):
            if self._telemetry_bytes()+len(payload) > self.limits["telemetry_bytes"]:
                raise StorageError(path,"budget."+operation+".capacity",
                    message="Budget telemetry cap including precharged emergency allowance")
            ticket = self.storage.reserve_bytes(path,len(payload),phase="budget."+operation+".reserve")
        phase = "budget."+operation+".open"
        try:
            self._hook(operation + "_open", path)
            with path.open("xb") as stream:
                phase = "budget."+operation+".write"
                stream.write(payload)
                stream.flush()
                phase = "budget."+operation+".fsync"
                self._hook(operation + "_fsync", path)
                os.fsync(stream.fileno())
        except OSError as error:
            raise StorageError(path,phase,error) from error
        if ticket is not None:
            self.storage.commit(ticket,actual_bytes=len(payload))
        if operation != "chunk":
            self._remember_owned(path)

    def _elapsed(self):
        now = float(self.clock())
        if not math.isfinite(now) or now < self._last_clock:
            raise BudgetError("Clock is invalid or moved backwards")
        self._last_clock = now
        return now - self._state["started_monotonic"]

    def _pending_files(self):
        pending = {}
        for row in self._outputs.values():
            if row["status"] == "pending":
                pending[row["path"]] = row["max_bytes"]
                pending[row["temporary_path"]] = row["temp_bytes"]
        return pending

    def _telemetry_bytes(self):
        value = self.storage.usage()["telemetry_bytes"]
        self._last_telemetry_bytes = value
        return value

    @_transaction
    def _append(self, kind, data):
        if self._journal_broken:
            raise BudgetError("A failed journal cannot be continued or repaired")
        event = dict(sequence=self._seq + 1, previous_sha256=self._previous, kind=kind,
                     elapsed_seconds=self._elapsed(), data=data)
        payload = (_json(event) + "\n").encode()
        if len(payload) > 65536:
            raise BudgetError("Journal event is too large")
        current = self._telemetry_bytes()
        if current + len(payload) > self.limits["telemetry_bytes"]:
            raise BudgetError("Telemetry cap exceeded, including precharged emergency allowance")
        path = self.ledger_dir / "journal.jsonl"
        ticket = self.storage.reserve_bytes(path,len(payload),phase="budget.journal_append.reserve")
        phase = "budget.journal_append.open"
        try:
            self._hook("journal_append", path)
            with path.open("ab") as stream:
                self.storage.validate_append(path,stream)
                phase = "budget.journal_append.write"
                stream.write(payload)
                stream.flush()
                phase = "budget.journal_append.fsync"
                self._hook("journal_fsync", path)
                os.fsync(stream.fileno())
                self.storage.record_append(path,payload,stream)
                current = os.fstat(stream.fileno())
        except OSError as error:
            raise StorageError(path,phase,error) from error
        self.storage.commit(ticket,actual_bytes=len(payload))
        self._journal_hash.update(payload)
        self._journal_size += len(payload)
        self._journal_identity = [current.st_dev,current.st_ino]
        self._journal_mtime_ns = current.st_mtime_ns
        self._seq, self._previous = event["sequence"], _sha(payload)

    def _emit(self, kind, data):
        try:
            self._append(kind, data)
        except Exception as error:
            self._journal_broken = True
            kind = error.details["exception_type"] if isinstance(error,StorageError) else type(error).__name__
            self._abort(f"journal_failed: {kind}: {error}", error=error)

    def _abort(self, reason, *, error=None):
        latest = str(reason)
        # Stop memory state even when quota admission failed. State and prepaid
        # emergency IO use only this owner's mutex, never the quota mutex.
        with self._abort_state_lock:
            if isinstance(error, StorageError):
                details = _copy(error.details)
                if self._state.get("storage_error") is None:
                    self._state["storage_error"] = details
                else:
                    self._state["secondary_storage_error"] = details
            primary = self._state.get("stop_reason")
            if primary is None:
                primary = latest
            elif latest != primary:
                self._state["secondary_failure"] = latest
            self._state.update(status="stopped", stop_reason=primary)
            for stage in ("M", "T", "L"):
                if self._state["stage_status"][stage] == "active":
                    self._state["stage_status"][stage] = "stopped"
        try:
            with self._io_lock:
                with self._abort_state_lock:
                    emergency = dict(attempt_id=ATTEMPT_ID, nonce=self.nonce, status="stopped", reason=primary,
                        secondary=self._state.get("secondary_failure"), latest_failure=latest,
                        state=self._state, pending=[key for key,row in self._calls.items() if row["status"]=="pending"])
                    payload = (_json(emergency)+"\n").encode()
                path = self.ledger_dir / "emergency-stop.json"
                if path.exists():
                    self._secondary_count += 1
                    path = self.ledger_dir / f"emergency-secondary-{self._secondary_count:06d}.json"
                # The fixed allowance is precharged; emergency IO needs no
                # shared quota lock, including after a quota-lock failure.
                if (self.ledger_dir.exists()
                        and self._emergency_bytes+len(payload) <= 65536 and not path.exists()):
                    # Partial/open/fsync failures retain this reservation too.
                    self._emergency_bytes += len(payload)
                    self._exclusive(path, payload, "emergency")
        except Exception:
            pass  # A durable fixed claim still prevents a new attempt after total IO failure.
        raise BudgetError(str(reason),details=_copy(self._state.get("storage_error")))

    @_transaction
    def check(self, binding=None):
        if self._state["status"] != "active":
            raise BudgetError(f"Attempt is {self._state['status']}: {self._state['stop_reason']}")
        try:
            if binding is not None and _json(binding) != _json(self.binding):
                raise BudgetError("Source/config/asset binding changed")
            if self._elapsed() >= self.limits["seconds"]:
                raise BudgetError("The original attempt deadline expired")
            self._check_storage()
        except Exception as error:
            self._abort(str(error), error=error)
        return self._state_snapshot()

    def _check_storage(self):
        usage = self.storage.usage()
        self._last_telemetry_bytes = usage["telemetry_bytes"]
        future = sum(self.limits[limit] - self._allocated[kind] for kind, limit in
                     (("raw", "raw_bytes"), ("metadata", "metadata_bytes"), ("headers", "header_bytes")))
        worst = usage["output_bytes"]+future+max(0,self.limits["temporary_bytes"]-self.storage.pending_reserve)
        if worst > self.limits["result_bytes"]:
            raise BudgetError("Result/current/outstanding/temporary/worst-case storage cap exceeded")
        if usage["telemetry_bytes"] > self.limits["telemetry_bytes"]:
            raise BudgetError("Telemetry/current/outstanding storage cap exceeded")

    @_transaction
    def audit_storage(self):
        """Full owner-only integrity boundary; never scan mutable guardian IO."""
        self._check_storage()
        if any(row["status"]=="pending" for row in self._outputs.values()):
            self._abort("Integrity boundary requires no pending output")
        actual = {}
        for root in (self.output_dir,self.ledger_dir,self.registry_dir):
            actual.update(scan_files(root,phase="budget.audit"))
        expected = dict(self._owned_files)
        journal = str(self.ledger_dir/"journal.jsonl")
        if self._seq:
            expected[journal] = dict(bytes=self._journal_size,sha256=self._journal_hash.hexdigest(),
                mtime_ns=self._journal_mtime_ns,identity=self._journal_identity)
        external_claims = {name:sha for name,sha in self.binding["asset_sha256"].items()
                           if Path(name).is_relative_to(self.registry_dir)}
        for name in set(actual)-set(expected)-set(external_claims):
            raise StorageError(name,"budget.audit.untracked",message="Untracked owner file at integrity boundary")
        for name,record in expected.items():
            current = actual.get(name)
            if current is None:
                raise StorageError(name,"budget.audit.missing",FileNotFoundError("Owned immutable file disappeared"))
            if (current.size,current.sha256,current.mtime_ns,list(current.identity)) != (
                    record["bytes"],record["sha256"],record["mtime_ns"],record["identity"]):
                raise StorageError(name,"budget.audit.immutable",message="Owned immutable content or identity changed")
        for name,sha in external_claims.items():
            if name not in actual or actual[name].sha256 != sha:
                raise StorageError(name,"budget.audit.claim",message="Bound launcher claim changed or disappeared")
        if self._elapsed() >= self.limits["seconds"]:
            self._abort("Original deadline includes integrity audits")
        return dict(files=len(actual),journal_sha256=self._journal_hash.hexdigest())

    def _state_snapshot(self):
        """Bounded status/counters without walking call or output histories."""
        state = _copy(self._state)
        state.update(attempt_id=ATTEMPT_ID, nonce=self.nonce, journal_sequence=self._seq,
                     serialization_bound=self.serialization_bound, allocated_bytes=dict(self._allocated))
        return state

    def snapshot(self):
        state = self._state_snapshot()
        state["calls"] = []
        for row in self.plan["calls"]:
            current = dict(row)
            if row["status"] != "missing":
                current.update(self._calls.get(row["key"], dict(status="not_attempted")))
                if current["status"] == "not_attempted":
                    current["reason"] = self._state["stop_reason"] or "not_reached"
            state["calls"].append(current)
        state["outputs"] = _copy(self._outputs)
        return state

    @_transaction
    def advance_m(self, passed, evidence=None):
        self.check()
        if self._state["phase"] != "M" or type(passed) is not bool:
            self._abort("M gate is not repeatable")
        self.audit_storage()
        self._state["gates"]["M"] = dict(passed=passed, evidence=_copy(evidence))
        self._state["stage_status"]["M"] = "passed" if passed else "failed"
        if passed:
            self._state["phase"] = "T"
            self._state["stage_status"]["T"] = "active"
        else:
            self._state.update(status="stopped", stop_reason="M_gate_failed")
        self._emit("M_gate", self._state["gates"]["M"])
        return self.snapshot()

    @_transaction
    def reserve_model_load(self):
        self.check()
        if self._state["phase"] != "T" or self._state["counts"]["model_loads"]:
            self._abort("One model load is allowed only after M passes")
        self._state["counts"]["model_loads"] += 1
        self._state["model_load_status"] = "pending"
        self._emit("model_load_reserved", dict(charged=1))
        self.check()

    @_transaction
    def complete_model_load(self):
        self.check()
        if self._state["model_load_status"] != "pending":
            self._abort("No pending model load")
        self._state["model_load_status"] = "complete"
        self._emit("model_load_complete", {})

    @_transaction
    def reserve_call(self, key, rng_entry):
        self.check()
        row = self._call_plan.get(key)
        if (row is None or row["status"] == "missing" or key in self._calls
                or row["phase"] != self._state["phase"]
                or self._state["model_load_status"] != "complete"
                or any(call["status"] == "pending" for call in self._calls.values())
                or not _hash_string(rng_entry)):
            self._abort("Unknown, missing, duplicate, pending, wrong-stage or invalid-RNG call")
        reference_key = call_key(row["anchor"], row["candidate"], row["seed"], "A", "AR", row["horizon"])
        if key != reference_key:
            reference = self._calls.get(reference_key)
            if not reference or reference["status"] != "complete" or rng_entry != reference["rng_entry"]:
                self._abort("A/B/TF sampler entry RNG does not match completed A-AR reference")
        if row["mode"] == "AR" and row["horizon"] > 1:
            previous = self._calls.get(call_key(row["anchor"], row["candidate"], row["seed"],
                                               row["arm"], "AR", row["horizon"] - 1))
            if not previous or previous["status"] != "complete":
                self._abort("AR horizons must complete in causal order")
        counts = self._state["counts"]
        if (counts["images"] + 1 > self.limits["images"]
                or counts["euler_steps"] + 3 > self.limits["euler_steps"]
                or counts[row["phase"] + "_images"] + 1 > (576 if row["phase"] == "T" else 1692)):
            self._abort("Forecast budget exhausted")
        counts["images"] += 1
        counts["euler_steps"] += 3
        counts[row["phase"] + "_images"] += 1
        self._calls[key] = dict(status="pending", rng_entry=rng_entry, charged_images=1, charged_euler_steps=3)
        self._emit("call_reserved", dict(key=key, **self._calls[key]))
        self.check()  # fsync time also consumes the original deadline, before sampler entry.
        return _copy(self._calls[key])

    @_transaction
    def complete_call(self, key, rng_exit, output_sha256, output_bytes=IMAGE_BYTES):
        self.check()
        row = self._calls.get(key)
        if (not row or row["status"] != "pending" or not _hash_string(rng_exit)
                or not _hash_string(output_sha256) or output_bytes != IMAGE_BYTES):
            self._abort("Invalid completion; call charges are retained")
        planned = self._call_plan[key]
        reference_key = call_key(planned["anchor"], planned["candidate"], planned["seed"],
                                 "A", "AR", planned["horizon"])
        if key != reference_key and rng_exit != self._calls[reference_key]["rng_exit"]:
            self._abort("A/B/TF sampler exit RNG differs from A-AR reference")
        row.update(status="complete", rng_exit=rng_exit, output_sha256=output_sha256, output_bytes=output_bytes)
        self._emit("call_complete", dict(key=key, **row))

    @_transaction
    def fail_call(self, key, reason):
        row = self._calls.get(key)
        if not row or row["status"] != "pending":
            self._abort("Failure has no pending call; no retry permitted")
        row.update(status="failed", reason=str(reason))
        self._emit("call_failed", dict(key=key, reason=str(reason), charges_retained=True))
        self._abort(f"call_failed: {key}: {reason}")

    complete = complete_call
    fail = fail_call

    @_transaction
    def stop(self, reason):
        """Persist a non-call failure without granting another operation/resume."""
        if self._state["status"] != "active":
            return self.snapshot()
        self._state.update(status="stopped", stop_reason=str(reason))
        for stage, status in self._state["stage_status"].items():
            if status == "active":
                self._state["stage_status"][stage] = "stopped"
        self._emit("stopped", dict(reason=str(reason), counts=self._state["counts"],
                                   pending=[key for key, row in self._calls.items() if row["status"] == "pending"]))
        return self.snapshot()

    @_transaction
    def advance_t(self, passed, evidence=None):
        self.check()
        completed = sum(row["status"] == "complete" for key, row in self._calls.items()
                        if self._call_plan[key]["phase"] == "T")
        if self._state["phase"] != "T" or completed != 576 or type(passed) is not bool:
            self._abort("T gate requires every fixed T call to complete once")
        self.audit_storage()
        self._state["gates"]["T"] = dict(passed=passed, evidence=_copy(evidence))
        self._state["stage_status"]["T"] = "passed" if passed else "failed"
        if passed:
            self._state["phase"] = "L"
            self._state["stage_status"]["L"] = "active"
        else:
            self._state.update(status="stopped", stop_reason="T_gate_failed")
        self._emit("T_gate", dict(**self._state["gates"]["T"], L_status=self._state["stage_status"]["L"]))
        return self.snapshot()

    def _check_output(self, final):
        if self._journal_broken:
            raise BudgetError("Failed journal permits no further output accounting; emergency record is final")
        if not final or self._state["status"] == "active":
            self.check()
        else:
            try:
                if self._elapsed() >= self.limits["seconds"]:
                    raise BudgetError("Original deadline includes final exports")
                self._check_storage()
            except Exception as error:
                self._abort(str(error), error=error)

    @_transaction
    def reserve_output(self, name, max_bytes, *, raw_bytes=0, temp_bytes=None, category="results", final=False):
        self._check_output(final)
        relative = Path(name)
        if (relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts)
                or category not in ("results", "telemetry") or type(max_bytes) is not int
                or not 0 < max_bytes <= self.limits["chunk_bytes"] or type(raw_bytes) is not int
                or not 0 <= raw_bytes <= max_bytes or raw_bytes % IMAGE_BYTES
                or type(final) is not bool or (final and raw_bytes)):
            self._abort("Invalid bounded immutable chunk reservation")
        root = self.output_dir if category == "results" else self.ledger_dir / "exports"
        path = unlinked_path(root / relative)
        if not path.is_relative_to(root) or str(path) in self._outputs or path.exists():
            self._abort("Chunk path is outside its root or already used")
        if any(row["status"] == "pending" for row in self._outputs.values()):
            self._abort("Only one bounded chunk may be outstanding")
        temp_bytes = max_bytes if temp_bytes is None else temp_bytes
        if type(temp_bytes) is not int or not max_bytes <= temp_bytes <= self.limits["temporary_bytes"]:
            self._abort("Temporary chunk does not fit its reservation")
        allocation = dict(raw=raw_bytes, headers=max_bytes - raw_bytes, metadata=0) if raw_bytes else dict(raw=0, headers=0, metadata=max_bytes)
        if category == "results":
            for kind, limit in (("raw", "raw_bytes"), ("metadata", "metadata_bytes"), ("headers", "header_bytes")):
                if self._allocated[kind] + allocation[kind] > self.limits[limit]:
                    self._abort(f"Chunk {kind} allocation exhausted")
            for kind, amount in allocation.items():
                self._allocated[kind] += amount
        temporary = path.with_name(path.name + ".pending-" + self.nonce)
        row = dict(name=str(relative), path=str(path), temporary_path=str(temporary), max_bytes=max_bytes,
                   raw_bytes=raw_bytes, temp_bytes=temp_bytes, category=category, final=final, status="pending")
        ticket = self.storage.reserve_bytes(path,max_bytes,temporary_bytes=temp_bytes,
                                            phase="budget.output.reserve")
        self._output_tickets[str(path)] = ticket
        self._outputs[str(path)] = row
        self._emit("output_reserved", {key:value for key,value in row.items()
                                      if key not in ("path","temporary_path")})
        self._check_output(final)
        return _copy(row)

    @_transaction
    def complete_output(self, name, *, category="results"):
        root = self.output_dir if category == "results" else self.ledger_dir / "exports"
        row = self._outputs.get(str(unlinked_path(root / name)))
        if not row or row["status"] != "pending":
            self._abort("No pending immutable chunk")
        self._check_output(row["final"])
        path = Path(row["path"])
        try:
            record = self._remember_owned(path)
            if record.size > row["max_bytes"] or Path(row["temporary_path"]).exists():
                raise BudgetError("Chunk size/temp/immutability mismatch")
            self.storage.commit(self._output_tickets[str(path)],actual_bytes=record.size)
            row.update(status="complete", bytes=record.size, sha256=record.sha256,
                       mtime_ns=record.mtime_ns, identity=list(record.identity))
            self._emit("output_complete", {key:value for key,value in row.items()
                                          if key not in ("path","temporary_path")})
            self._check_output(row["final"])
        except Exception as error:
            self._abort(f"output_failed: {error}", error=error)
        return _copy(row)

    @_transaction
    def write_chunk(self, name, payload, *, raw_bytes=0, max_bytes=None, category="results", final=False):
        if not isinstance(payload, bytes):
            self._abort("Chunk payload must be bytes")
        maximum = len(payload) if max_bytes is None else max_bytes
        if len(payload) > maximum:
            self._abort("Serialized chunk exceeds its declared bound")
        row = self.reserve_output(name, maximum, raw_bytes=raw_bytes, category=category, final=final)
        path, temporary = Path(row["path"]), Path(row["temporary_path"])
        phase = "budget.chunk.mkdir"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._exclusive(temporary, payload, "chunk")
            self._check_output(final)
            phase = "budget.chunk_publish.open"
            self._hook("chunk_publish", path)
            # Exclusive creation works in Windows sandboxes without hard-link
            # privileges. A partial final file on failure is retained, charged,
            # and never treated as complete or overwritten on a later attempt.
            with temporary.open("rb") as source, path.open("xb") as destination:
                phase = "budget.chunk_publish.write"
                shutil.copyfileobj(source, destination, length=65536)
                destination.flush()
                phase = "budget.chunk_publish.fsync"
                self._hook("chunk_publish_fsync", path)
                os.fsync(destination.fileno())
            phase = "budget.chunk_publish.unlink"
            temporary.unlink()
            return self.complete_output(name, category=category)
        except Exception as error:
            if isinstance(error,OSError):
                failed_path = getattr(error,"filename",None) or (temporary if phase.endswith(".unlink") else path)
                error = StorageError(failed_path,phase,error)
            kind = error.details["exception_type"] if isinstance(error,StorageError) else type(error).__name__
            self._abort(f"chunk_failed: {kind}: {error}", error=error)

    @_transaction
    def finish(self, evidence=None):
        self.check()
        if (self._state["phase"] != "L" or self._state["counts"]["images"] != 2268
                or any(row["status"] != "complete" for row in self._calls.values())
                or any(row["status"] != "complete" for row in self._outputs.values())):
            self._abort("Finish requires the complete fixed plan and no pending operations")
        self.audit_storage()
        # Full shared-root audit belongs to the parent after worker quiescence.
        usage = self.storage.usage()
        for field, roots in (("output_bytes",self.storage.result_roots),
                             ("telemetry_bytes",self.storage.telemetry_roots)):
            if usage[field] > self.storage.limits[field]:
                raise StorageError(roots[0] if roots else self.output_dir,
                    "budget.finish.full_verify.capacity",message=(
                        f"Aggregate {field}={usage[field]} exceeds {self.storage.limits[field]}; "
                        f"all declared roots={[str(path) for path in roots]}"))
        self._state.update(status="complete", stop_reason=None)
        self._state["stage_status"]["L"] = "complete"
        self._emit("finished", dict(counts=self._state["counts"], evidence=_copy(evidence)))
        return self.snapshot()
