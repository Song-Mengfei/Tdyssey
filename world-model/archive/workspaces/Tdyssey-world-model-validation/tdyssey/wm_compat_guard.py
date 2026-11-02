"""Model-free Windows guardian for a separately authorized compatibility entry.

Importing or planning does no resource query, model work or file write. The real
interpreter waits at the existing stdlib bootstrap gate until its exact retained
handle belongs to a kill-on-close, 3-GiB Job Object. Resource sampling is a stop
rule, not a guarantee of instantaneous free-memory floors between observations.
"""

from copy import deepcopy
from contextlib import contextmanager, nullcontext
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import stat
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from . import policy_memory
from .wm_compat_identity import ATTEMPT_ID, REGISTRY, validate_execution_identity

ROOT = Path(__file__).resolve().parents[1]
GIB, MIB = 1024**3, 1024**2
DEFAULT_LOCK = ROOT / "artifacts/wm-compat-single-model.lock"
LIMITS = {"startup_ram_bytes": 4*GIB, "startup_commit_bytes": 6*GIB,
    "startup_gpu_bytes": 4*GIB, "startup_disk_bytes": 2*GIB,
    "runtime_ram_bytes": 3*GIB//2, "runtime_commit_bytes": 3*GIB,
    "runtime_gpu_bytes": 3*GIB//2, "working_set_bytes": 2*GIB,
    "private_bytes": 3*GIB, "job_memory_bytes": 3*GIB,
    "cuda_allocated_bytes": 512*MIB, "cuda_reserved_bytes": GIB,
    "output_bytes": 256*MIB, "telemetry_bytes": 16*MIB,
    "wall_seconds": 1800, "worker_status_max_age_seconds": 5,
    "loading_max_seconds": 30, "gpu_max_age_seconds": 15}
PHASES = frozenset(("setup", "measurement", "M", "model_loading", "running", "export", "complete", "error"))
FORBIDDEN_COUNTERS = ("environment_creations", "resets", "restores", "controls", "physics_substeps",
                      "real_renders", "policy_sampling", "optimizer_updates")
ENV_DIRECTORY, ENV_NONCE, ENV_PARENT = ("WM_COMPAT_GUARD_DIRECTORY", "WM_COMPAT_GUARD_NONCE", "WM_COMPAT_GUARD_PARENT_PID")
ENV_STORAGE = "WM_COMPAT_STORAGE_POLICY"
STATUS_BYTES = 65536


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _json(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode("utf-8")


def _exclusive(path, value):
    _exclusive_bytes(path, _json(value))


def _exclusive_bytes(path, payload):
    with Path(path).open("xb", buffering=0) as stream:
        _write_all(stream, payload); os.fsync(stream.fileno())


def _transaction(storage, phase):
    return nullcontext() if storage is None else storage.local_transaction(phase)


def _reserve(storage, path, maximum, *, temporary=0, phase="guardian.write"):
    if storage is None:
        return None
    return storage.reserve_bytes(path, maximum, temporary_bytes=temporary, phase=phase)


def _commit(storage, ticket, actual):
    if storage is not None:
        storage.commit(ticket, actual_bytes=actual)


def _capacity(storage, category, additional=0):
    if storage is not None:
        usage = storage.usage()
        if usage[category]+additional > LIMITS[category]:
            raise RuntimeError("storage_limit:"+category)


def _growth(storage, path, additional=0, phase="guardian.write"):
    if storage is not None:
        storage.check_growth(path, additional, phase=phase)


def _storage_error_details(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if hasattr(error, "details"):
            return dict(error.details)
        error = error.__cause__ or error.__context__
    return None


def _io_error(path, phase, error):
    from .wm_compat_storage import StorageError
    return StorageError(path, phase, error)


def _write_all(stream, payload):
    """Write only the already admitted payload; reject zero/invalid progress."""
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if type(written) is not int or not 0 < written <= len(view)-offset:
            raise OSError("Append write made no progress or returned an invalid byte count")
        offset += written


def _publish_json(temporary, path, *, shared_status=False):
    """Choose the Windows replacement primitive before attempting publication."""
    from .wm_compat_storage import unlinked_path
    temporary, path = unlinked_path(temporary), unlinked_path(path)
    if shared_status:
        value = temporary.lstat()
        if (temporary.parent != path.parent or not stat.S_ISREG(value.st_mode)
                or getattr(value,"st_nlink",1)>1 or getattr(value,"st_file_attributes",0)&0x400
                or value.st_size>STATUS_BYTES):
            raise _io_error(temporary,"atomic.source_type",ValueError("Invalid status temporary"))
    if not path.exists():
        os.rename(temporary,path)  # First publication must not overwrite a target.
    elif shared_status:
        value = path.lstat()
        if (not stat.S_ISREG(value.st_mode) or getattr(value,"st_nlink",1)>1
                or getattr(value,"st_file_attributes",0)&0x400 or value.st_size>STATUS_BYTES):
            raise _io_error(path,"atomic.target_type",ValueError("Invalid existing status target"))
        if sys.platform == "win32":
            kernel = ctypes.WinDLL("kernel32",use_last_error=True)
            kernel.ReplaceFileW.argtypes = [wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.LPCWSTR,
                wintypes.DWORD,ctypes.c_void_p,ctypes.c_void_p]
            kernel.ReplaceFileW.restype = wintypes.BOOL
            if not kernel.ReplaceFileW(str(path),str(temporary),None,0,None,None):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.replace(temporary,path)
    else:
        os.replace(temporary,path)


def atomic_json(path, value, *, storage=None):
    """Publish one prepaid bounded status slot with owner-local serialization."""
    from .wm_compat_storage import StorageError
    path = Path(path)
    payload = _json(value)
    if len(payload) > STATUS_BYTES:
        raise ValueError("Atomic status payload exceeds its declared 64KiB slot")
    if storage is not None and path not in storage.atomic_targets:
        raise ValueError("Atomic publication requires an exactly declared mutable target")
    if storage is not None and len(payload) > storage.atomic_targets[path]:
        raise ValueError("Atomic status payload exceeds its exact declared slot")
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with _transaction(storage, "guardian.atomic_status"):
        primary = None
        try:
            if storage is not None:
                storage.claim_fixed(path)
            if storage is not None and any(path.parent.glob(path.name+".*.tmp")):
                raise StorageError(path, "atomic.orphan", message="Orphaned temporary retained; another publication is forbidden")
            try:
                _exclusive_bytes(temporary, payload)
            except OSError as exc:
                raise StorageError(temporary, "atomic.create", exc) from exc
            attempts = 1 if storage is not None else 5
            for attempt in range(attempts):
                try:
                    _publish_json(temporary,path,shared_status=storage is not None)
                    break
                except OSError as exc:
                    if attempt == attempts-1 or getattr(exc, "winerror", None) not in (32, 33):
                        raise StorageError(path, "atomic.replace", exc) from exc
                    time.sleep(.02*(attempt+1))
        except BaseException as exc:
            primary = exc
            raise
        finally:
            # A prepaid shared slot keeps the failed bytes and forbids another
            # publication through the existing orphan check. Preserve its cause.
            if primary is None or storage is None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                except BaseException as secondary:
                    if isinstance(secondary, OSError):
                        secondary = StorageError(temporary, "atomic.cleanup", secondary)
                    if primary is None:
                        raise secondary
                    primary.add_note("Secondary atomic temporary cleanup failure: "+type(secondary).__name__+": "+str(secondary))


class _ReadFileInformation(ctypes.Structure):
    _fields_ = [("attributes", wintypes.DWORD), ("created", wintypes.FILETIME),
        ("accessed", wintypes.FILETIME), ("written", wintypes.FILETIME),
        ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD), ("size_low", wintypes.DWORD),
        ("links", wintypes.DWORD), ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]


@contextmanager
def _open_shared_read(path):
    """Keep a complete JSON generation readable without blocking replacement."""
    from .wm_compat_storage import unlinked_path
    path = unlinked_path(path)
    stream = fd = handle = kernel = None
    phase = "guardian.json_read.open"
    mutable = path.name.casefold() == "worker-status.json"
    try:
        if sys.platform == "win32":
            import msvcrt
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateFileW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
            kernel.CreateFileW.restype = wintypes.HANDLE
            kernel.GetFileType.argtypes, kernel.GetFileType.restype = [wintypes.HANDLE], wintypes.DWORD
            kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE,ctypes.POINTER(_ReadFileInformation)]
            kernel.GetFileInformationByHandle.restype = wintypes.BOOL
            kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL
            # GENERIC_READ; SHARE_READ|WRITE|DELETE; OPEN_EXISTING; OPEN_REPARSE_POINT.
            handle = kernel.CreateFileW(str(path),0x80000000,0x7,None,3,0x00200000,None)
            if handle in (None, ctypes.c_void_p(-1).value):
                handle = None
                raise ctypes.WinError(ctypes.get_last_error())
            phase = "guardian.json_read.type"
            info = _ReadFileInformation()
            if not kernel.GetFileInformationByHandle(handle,ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if kernel.GetFileType(handle) != 1 or info.attributes & (0x400|0x10) or info.links > 1:
                raise _io_error(path,phase,ValueError("JSON reader requires an ordinary non-reparse file"))
            fd = msvcrt.open_osfhandle(int(handle),os.O_RDONLY|os.O_BINARY|getattr(os,"O_NOINHERIT",0))
            handle = None  # The descriptor now owns the exact share-delete HANDLE.
        else:
            fd = os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0))
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink > 1:
            raise _io_error(path,phase,ValueError("JSON reader requires a single-link ordinary file"))
        if mutable and opened.st_size > STATUS_BYTES:
            raise _io_error(path,phase,ValueError("Worker status exceeds its 64KiB slot"))
        stream = os.fdopen(fd,"rb",buffering=0); fd = None
        phase = "guardian.json_read.read"
        yield stream
        if not mutable:
            current, after = unlinked_path(path).stat(), os.fstat(stream.fileno())
            signature = lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns)
            if signature(current) != signature(opened) or signature(after) != signature(opened):
                raise _io_error(path,"guardian.json_read.identity",ValueError("Immutable JSON identity/content changed"))
    except OSError as exc:
        raise _io_error(path,phase,exc) from exc
    finally:
        primary = sys.exc_info()[1]
        try:
            if stream is not None:
                stream.close()
            elif fd is not None:
                os.close(fd)
            elif handle is not None and not kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except OSError as exc:
            detail = _io_error(path,"guardian.json_read.close",exc)
            if primary is None:
                raise detail from exc
            primary.add_note("Secondary JSON reader close failure: "+str(detail))


def _read(path):
    with _open_shared_read(path) as stream:
        return json.loads(stream.read().decode("utf-8"))


def _locked_read(path, storage):
    with _transaction(storage, "guardian.json_read"):
        return _read(path)


def _latest_worker_status(path, storage, previous):
    """An absent/sharing-busy generation may reuse only the still-aged snapshot."""
    from .wm_compat_storage import StorageError
    path = Path(path)
    if path.name != "worker-status.json" or path not in storage.atomic_targets:
        raise ValueError("Worker status continuity requires its exactly declared mutable slot")
    def unavailable(error):
        return isinstance(error, FileNotFoundError) or (isinstance(error, OSError)
            and getattr(error, "winerror", None) in (32, 33))
    try:
        snapshot = _locked_read(path, storage)
        if not isinstance(snapshot, dict):
            raise ValueError("Worker status JSON must be an object")
        return snapshot
    except OSError as exc:
        if not unavailable(exc) or exc.filename is None or Path(exc.filename) != path:
            raise
        return previous
    except StorageError as exc:
        if (not unavailable(exc.__cause__)
                or exc.details.get("path") != str(path)
                or exc.details.get("phase") != "guardian.json_read.open"):
            raise
        return previous


def _immutable_json(path, value, storage):
    with _transaction(storage, "guardian.immutable_json"):
        payload = _json(value)
        ticket = _reserve(storage, path, len(payload), temporary=len(payload), phase="guardian.immutable_json")
        temporary = Path(path).with_name(Path(path).name+".pending-"+uuid.uuid4().hex)
        try:
            _exclusive_bytes(temporary, payload)
            os.rename(temporary, path)  # Windows refuses replacing an existing immutable destination.
        except OSError as exc:
            raise _io_error(path, "guardian.immutable_json", exc) from exc
        if storage is not None:
            storage.remember_file(path)
        _commit(storage, ticket, len(payload))


def _completed_output_manifest(ledger, output, *, require_terminal=False):
    """Read the worker's durable chain once, after its process tree is quiet."""
    if ledger is None:
        return {}, None
    path = ledger/"journal.jsonl"
    try:
        if require_terminal:
            with _open_shared_read(ledger/"declaration.json"):
                pass  # A normal ledger exit requires this ordinary immutable file.
        elif not (ledger/"declaration.json").exists():
            return {}, None
        raw = path.read_bytes()
        expected, previous, elapsed = {}, "0"*64, -1
        target = None
        lines = raw.splitlines(keepends=True)
        for sequence, line in enumerate(lines, 1):
            row = json.loads(line)
            if (not line.endswith(b"\n") or row["sequence"] != sequence or row["previous_sha256"] != previous
                    or not _finite(row["elapsed_seconds"]) or row["elapsed_seconds"] < elapsed):
                raise ValueError("Budget journal sequence/hash/elapsed chain differs")
            previous, elapsed = hashlib.sha256(line).hexdigest(), row["elapsed_seconds"]
            if row["kind"] == "output_complete":
                value = row["data"]
                roots = {"results": output/"results", "telemetry": ledger/"exports"}
                category, name = value.get("category"), value.get("name")
                if "category" in value and category not in roots:
                    raise ValueError("Unknown budget output category")
                legacy = value.get("path")
                if "path" in value:
                    if not isinstance(legacy, str) or not Path(legacy).is_absolute():
                        raise ValueError("Legacy completion path must be absolute")
                    legacy = _unlinked_path(legacy)
                if "name" in value:
                    if not isinstance(name, str) or category not in roots:
                        raise ValueError("Compact completion requires a category and relative name")
                    relative = Path(name)
                    if (relative.is_absolute() or relative.drive or not relative.parts
                            or any(part in (".", "..") for part in relative.parts)):
                        raise ValueError("Invalid relative budget completion name")
                    target = _unlinked_path(roots[category]/relative)
                    if legacy is not None and legacy != target:
                        raise ValueError("Legacy and compact completion paths differ")
                elif legacy is not None:
                    target = legacy
                else:
                    raise ValueError("Budget completion has no output path")
                allowed = (roots[category],) if category in roots else tuple(roots.values())
                if (not any(target != root and target.is_relative_to(root) for root in allowed)
                        or str(target) in expected):
                    raise ValueError("Budget completion is outside worker outputs or repeated")
                expected[str(target)] = dict(value, path=str(target))
        if require_terminal:
            terminal = output/"results/completion.json"
            if not lines or row["kind"] != "output_complete" or target != terminal:
                raise ValueError("Normal ledger exit requires its unique terminal completion record")
            completion = expected[str(terminal)]
            if type(completion.get("bytes")) is not int or not 0 < completion["bytes"] <= 8192:
                raise ValueError("Terminal completion exceeds its bounded export")
            with _open_shared_read(terminal) as stream:
                payload = stream.read(8193)
            if len(payload) != completion["bytes"] or hashlib.sha256(payload).hexdigest() != completion["sha256"]:
                raise ValueError("Terminal completion differs from its journal record")
            value = json.loads(payload)
            if not isinstance(value, dict) or value.get("status") not in ("complete", "stopped"):
                raise ValueError("Terminal completion is not a completed or controlled-stop report")
        return expected, {"records":len(lines), "completed_outputs":len(expected),
            "tip_sha256":previous, "sha256":hashlib.sha256(raw).hexdigest()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _io_error(path, "guardian.journal_audit", exc) from exc


def _storage_policy(output, ledger, directory, telemetry_dirs):
    roots = tuple(dict.fromkeys((output/"guardian", *telemetry_dirs)))
    return dict(lock_path=str(directory/"storage.lock"), result_roots=[str(output)],
        telemetry_roots=[str(path) for path in roots], atomic_targets=[str(directory/"worker-status.json")],
        append_only=[str(directory/"monitor.jsonl"), str(directory/"worker-output.log"),
                     *([] if ledger is None else [str(ledger/"journal.jsonl")])],
        fixed_slots={str(directory/"stop-record.json"):65536}, timeout=.5,
        pending_roots=[str(output/"results"), *([] if ledger is None else [str(ledger/"exports")])],
        pending_reserve=2*MIB, telemetry_reserve=65536,
        emergency_headroom=65536 if ledger is not None else 0,
        limits={key:LIMITS[key] for key in ("output_bytes", "telemetry_bytes")})


def from_handoff(handoff):
    """Reconstruct the same cross-process storage coordinator without a model."""
    from .wm_compat_storage import Coordinator
    policy = deepcopy(handoff.get("storage_policy"))
    if not isinstance(policy, dict):
        raise RuntimeError("Guardian storage policy is missing")
    output = _new_namespace(handoff["output_dir"])
    ledger = None if handoff.get("ledger_dir") is None else _new_namespace(handoff["ledger_dir"])
    directory = _new_namespace(output/"guardian"/handoff["nonce"])
    if (_unlinked_path(policy.get("lock_path", "")) != directory/"storage.lock"
            or handoff.get("storage_lock") != str(directory/"storage.lock")):
        raise RuntimeError("Guardian storage lock differs from its nonce directory")
    telemetry = [_new_namespace(path) for path in handoff.get("storage_telemetry_roots", ())]
    expected = _storage_policy(output, ledger, directory, telemetry)
    if policy != expected or handoff.get("storage_telemetry_roots") != expected["telemetry_roots"]:
        raise RuntimeError("Guardian storage policy differs from the complete canonical policy")
    return Coordinator(**policy)


def bootstrap_child(argv=None):
    """Stdlib worker PID gate whose handshake also participates in storage IO."""
    import argparse
    import runpy
    from .wm_compat_storage import Coordinator
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--handshake", type=Path, required=True)
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    target = args.target[1:] if args.target[:1] == ["--"] else args.target
    if not target:
        parser.error("one target script is required")
    storage = Coordinator(**json.loads(os.environ[ENV_STORAGE]))
    _immutable_json(args.handshake, {"pid":os.getpid(), "parent_pid":os.getppid(), "nonce":args.nonce}, storage)
    began = time.monotonic()
    while time.monotonic()-began < 30:
        with storage.local_transaction("bootstrap.wait_ack"):
            accepted = _read(args.ack) if args.ack.exists() else None
        if accepted is not None:
            if accepted != {"pid":os.getpid(), "nonce":args.nonce}:
                raise ValueError("Parent ACK identity differs")
            sys.argv = target
            sys.path.insert(0, str(Path(target[0]).resolve().parent.parent))
            runpy.run_path(target[0], run_name="__main__")
            return 0
        time.sleep(.05)
    raise TimeoutError("No parent ownership ACK; no target code started")


def _unlinked_path(value):
    """Canonicalize without following a symlink or Windows directory junction."""
    value = Path(value)
    path = Path(os.path.abspath(value if value.is_absolute() else ROOT/value))
    for parent in (path, *path.parents):
        if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
            raise ValueError("Symbolic links and junctions are forbidden in guardian paths")
    return path


def _new_namespace(value):
    path = _unlinked_path(value)
    try:
        relative = path.relative_to(ROOT/"artifacts")
    except ValueError as exc:
        raise ValueError("Guardian writes require this repository's artifacts/wm-input-compat-* namespace") from exc
    if (not relative.parts or not relative.parts[0].lower().startswith("wm-input-compat-")
            or any(part.lower().startswith(("frozen", "milestone-b", "milestone-c")) for part in relative.parts)):
        raise ValueError("Only the new wm-input-compat-* namespace is writable")
    return path


def validate_paths(output_dir, *, ledger_dir=None, telemetry_dirs=(), lock_path=DEFAULT_LOCK):
    """Read-only preflight before any output, lock, or monitor mutation.

    The output root and optional new ledger must be empty and disjoint. The
    fixed registry may already exist; none of these paths may follow links.
    """
    output = _new_namespace(output_dir)
    ledger = None if ledger_dir is None else _new_namespace(ledger_dir)
    registry = _new_namespace(ROOT/"artifacts"/REGISTRY)
    if ledger is not None and (ledger.is_relative_to(registry) or registry.is_relative_to(ledger)):
        raise ValueError("New ledger and fixed registry must be disjoint")
    telemetry = tuple(dict.fromkeys(_new_namespace(path) for path in telemetry_dirs))
    roots = tuple(dict.fromkeys(path for path in (ledger, registry, *telemetry) if path is not None))
    for index, path in enumerate(roots):
        if path.is_relative_to(output) or output.is_relative_to(path):
            raise ValueError("Guardian output, ledger and telemetry roots must be disjoint")
        for other in roots[:index]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Guardian telemetry roots must be disjoint")
        if path.exists():
            if not path.is_dir():
                raise ValueError("Guardian storage roots must be directories")
            for folder, directories, files in os.walk(path, followlinks=False):
                for name in (*directories, *files):
                    _unlinked_path(Path(folder)/name)
    for path in (output, ledger):
        if path is not None and path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError("New guardian output and ledger directories must be empty")
    lock = _unlinked_path(lock_path)
    if lock != DEFAULT_LOCK:
        _new_namespace(lock)
    if any(lock.is_relative_to(path) or path.is_relative_to(lock) for path in (output, *roots)):
        raise ValueError("Single-model lock must be separate from output and telemetry roots")
    return output, ledger, telemetry, lock


def _launch_identity(command):
    """Read only the explicit identity flags; duplicates never select a winner."""
    if command.count("--execute") != 1:
        raise ValueError("Sealed execution requires exactly one explicit --execute")
    result = {}
    for flag, name in (("--authorization-id", "authorization_id"), ("--reviewed-commit", "reviewed_commit")):
        values = []
        for index, token in enumerate(command):
            if token == flag:
                values.append(command[index+1] if index+1 < len(command) else "")
            elif token.startswith(flag+"="):
                values.append(token[len(flag)+1:])
        if len(values) != 1 or not values[0].strip() or values[0].startswith("--"):
            raise ValueError("Sealed execution requires exactly one nonempty "+flag)
        result[name] = values[0]
    return result


def validate_launch_seal(guardian, *, command=None):
    """Verify the immutable launch claim; no claim is created by this reader.

    Generic engineering children may have no seal. The production entry must
    require one; the launcher always uses its fixed permanent registry.
    """
    reference = guardian.get("launch_seal")
    if reference is None:
        return None
    try:
        if not isinstance(reference, dict):
            raise ValueError("Invalid launch seal reference")
        path = _new_namespace(reference["path"])
        if path.name != "launcher-attempt.json" or str(path) != reference["path"]:
            raise ValueError("Invalid permanent launch seal path")
        for name in ("output_dir", "ledger_dir"):
            root = _new_namespace(guardian[name])
            if root.is_relative_to(path.parent) or path.parent.is_relative_to(root):
                raise ValueError("Launch registry overlaps output or ledger")
        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        if sha256 != reference["sha256"]:
            raise ValueError("Launch seal bytes changed")
        seal = json.loads(raw)
        if seal.get("kind") != "tdyssey.wm_compat_launch_seal.v1" or seal.get("attempt_id") != ATTEMPT_ID:
            raise ValueError("Launch seal kind/attempt differs")
        for name in ("nonce", "guard_pid", "output_dir", "ledger_dir", "started_monotonic",
                     "deadline_monotonic", "deadline_seconds"):
            if seal.get(name) != guardian.get(name):
                raise ValueError("Launch seal differs from guardian: "+name)
        if reference.get("nonce") != seal["nonce"] or seal.get("resume_allowed") is not False:
            raise ValueError("Launch seal nonce/restart policy differs")
        if (not all(_finite(seal[name]) for name in ("started_monotonic", "deadline_monotonic", "deadline_seconds"))
                or not 0 < seal["deadline_seconds"] <= LIMITS["wall_seconds"]
                or abs(seal["deadline_monotonic"]-seal["started_monotonic"]-seal["deadline_seconds"]) > 1e-6
                or seal["started_monotonic"] > time.monotonic() or seal["deadline_monotonic"] <= time.monotonic()):
            raise ValueError("Launch seal original deadline is invalid or expired")
        declared = seal.get("command")
        identity = _launch_identity(declared)
        if (not isinstance(declared, list) or any(not isinstance(value, str) for value in declared)
                or hashlib.sha256(_json(declared)).hexdigest() != seal.get("command_sha256")
                or any(seal.get(name) != value for name, value in identity.items())):
            raise ValueError("Launch seal command/identity differs")
        if command is not None and list(command) != declared:
            raise ValueError("Actual worker command differs from immutable launch command")
        return dict(seal, path=str(path), sha256=sha256)
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise RuntimeError("Compatibility launch seal validation failed: "+str(exc)) from exc


class _JobAccounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_longlong) for name in ("TotalUserTime", "TotalKernelTime",
        "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [(name, wintypes.DWORD) for name in (
        "TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]


class WindowsWorker(policy_memory.OwnedProcess):
    """Extend retained exact-PID ownership with a verified aggregate job cap."""
    def __init__(self, pid, launcher_pid):
        super().__init__(pid, launcher_pid)
        try:
            limits = policy_memory._JobExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
            limits.JobMemoryLimit = LIMITS["job_memory_bytes"]
            if not self._kernel.SetInformationJobObject(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError("SetInformationJobObject(3GiB aggregate job memory) failed")
            query = self._kernel.QueryInformationJobObject
            query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
            query.restype = wintypes.BOOL
            observed = policy_memory._JobExtendedLimits()
            if not query(self._job, 9, ctypes.byref(observed), ctypes.sizeof(observed), None):
                raise OSError("QueryInformationJobObject failed")
            if observed.JobMemoryLimit != LIMITS["job_memory_bytes"] or observed.BasicLimitInformation.LimitFlags & 0x2200 != 0x2200:
                raise OSError("Windows Job Object memory/kill-on-close limits differ")
        except BaseException:
            self.close()
            raise

    def quiesce(self, timeout=10):
        """Terminate the retained Job and prove all its processes have exited."""
        if not _finite(timeout) or not 0 < timeout <= 10 or not self._job:
            raise ValueError("A retained Job and positive cleanup timeout <=10 seconds are required")
        query = self._kernel.QueryInformationJobObject
        query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        query.restype = wintypes.BOOL
        terminate = self._kernel.TerminateJobObject
        terminate.argtypes, terminate.restype = [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL
        def active_processes():
            accounting = _JobAccounting()
            if not query(self._job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                raise OSError("QueryInformationJobObject(BasicAccounting) failed during cleanup")
            if accounting.ActiveProcesses > accounting.TotalProcesses:
                raise OSError("Windows returned inconsistent Job process accounting")
            return int(accounting.ActiveProcesses)
        if active_processes() == 0:
            return 0
        if not terminate(self._job, 1):
            raise OSError("TerminateJobObject failed during cleanup")
        until = time.monotonic()+timeout
        while True:
            active = active_processes()
            if active == 0:
                return 0
            if time.monotonic() >= until:
                raise TimeoutError("Owned Job still contains active processes after termination")
            time.sleep(.02)


def verify_guardian(*, native_check=True):
    """Entry-side handoff check, before imports/backend construction or M work."""
    directory = Path(os.environ.get(ENV_DIRECTORY, "")).resolve()
    nonce = os.environ.get(ENV_NONCE)
    parent = os.environ.get(ENV_PARENT)
    if not nonce or not parent or not parent.isdigit() or not os.environ.get(ENV_DIRECTORY):
        raise RuntimeError("No verified compatibility guardian handoff")
    worker, ack, guardian = (_read(directory/name) for name in ("worker.json", "worker-ack.json", "guardian.json"))
    pid = os.getpid()
    if (worker != {"pid": pid, "parent_pid": os.getppid(), "nonce": nonce}
            or ack != {"pid": pid, "nonce": nonce}
            or guardian.get("actual_worker_pid") != pid or guardian.get("nonce") != nonce
            or guardian.get("guard_pid") != int(parent) or guardian.get("kill_on_job_close") is not True
            or guardian.get("job_memory_bytes") != LIMITS["job_memory_bytes"]
            or guardian.get("status") != "owned_before_ack"):
        raise RuntimeError("Compatibility guardian PID/nonce/job marker differs")
    started, deadline, seconds = (guardian.get(key) for key in (
        "started_monotonic", "deadline_monotonic", "deadline_seconds"))
    now = time.monotonic()
    if (not all(_finite(value) for value in (started, deadline, seconds))
            or not 0 < seconds <= LIMITS["wall_seconds"] or started > now
            or abs(deadline-(started+seconds)) > 1e-6 or deadline <= now):
        raise RuntimeError("Compatibility guardian original monotonic deadline is missing, invalid or expired")
    try:
        output = _new_namespace(guardian["output_dir"])
        ledger = None if guardian.get("ledger_dir") is None else _new_namespace(guardian["ledger_dir"])
        registry = _new_namespace(ROOT/"artifacts"/REGISTRY)
        paths = (output, registry) if ledger is None else (output, ledger, registry)
        if any(path.is_relative_to(other) or other.is_relative_to(path)
                for index, path in enumerate(paths) for other in paths[:index]):
            raise ValueError("Guardian output, ledger and registry overlap")
        if guardian["output_dir"] != str(output) or (ledger is not None and guardian["ledger_dir"] != str(ledger)):
            raise ValueError("Guardian paths must be canonical absolute paths")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Compatibility guardian output/ledger path binding is invalid") from exc
    if native_check:
        kernel, _ = policy_memory._libraries()
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        kernel.IsProcessInJob.restype = wintypes.BOOL
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        inside = wintypes.BOOL()
        if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)) or not inside.value:
            raise RuntimeError("Worker is not assigned to a Windows Job Object")
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        limits = policy_memory._JobExtendedLimits()
        if (not kernel.QueryInformationJobObject(None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None)
                or limits.JobMemoryLimit != LIMITS["job_memory_bytes"]
                or limits.BasicLimitInformation.LimitFlags & 0x2200 != 0x2200):
            raise RuntimeError("Worker's actual Windows Job Object is missing the guardian limits")
    launch_seal = validate_launch_seal(guardian, command=sys.argv)
    return dict(guardian, ledger_dir=None if ledger is None else str(ledger), launch_seal=launch_seal,
                directory=str(directory), limits=dict(LIMITS))


def publish_worker_status(payload, *, handoff=None, storage=None):
    """Publish caller-supplied counters; this helper never imports a backend."""
    handoff = handoff or verify_guardian()
    value = dict(payload, actual_worker_pid=os.getpid(), nonce=handoff["nonce"],
                 at_unix=time.time(), at_utc=datetime.now(timezone.utc).isoformat())
    _json(value)
    storage = storage or from_handoff(handoff)
    atomic_json(Path(handoff["directory"])/"worker-status.json", value, storage=storage)
    return value


class WorkerHeartbeat:
    """Refresh setup/M/loading status once per second without loading a model.

    ``refresh`` can sample an already-owned backend's allocator counters. Its
    failures become an explicit error status and are also raised by ``check``.
    """
    def __init__(self, initial_status, *, refresh=None, interval_seconds=1., handoff=None):
        if not _finite(interval_seconds) or not 0 < interval_seconds <= 2:
            raise ValueError("Heartbeat interval must be in (0,2] seconds")
        self.handoff = handoff or verify_guardian()
        self.storage = from_handoff(self.handoff)
        self.payload, self.refresh, self.interval = dict(initial_status), refresh, interval_seconds
        self._lock, self._stop, self._error = threading.RLock(), threading.Event(), None
        self._thread = threading.Thread(target=self._loop, daemon=True)
    def _publish(self):
        with self._lock:
            value = dict(self.payload)
            if self.refresh is not None:
                value.update(self.refresh())
            publish_worker_status(value, handoff=self.handoff, storage=self.storage)
    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self._publish()
            except BaseException as exc:
                self._error = exc
                try:
                    publish_worker_status(dict(self.payload, phase="error", error=f"{type(exc).__name__}: {exc}"), handoff=self.handoff, storage=self.storage)
                except BaseException:
                    pass
                return
    def update(self, **fields):
        self.check()
        with self._lock:
            self.payload.update(fields)
        self._publish()
    def check(self):
        if self._error is not None:
            raise RuntimeError("Worker heartbeat failed") from self._error
    def __enter__(self):
        self._publish(); self._thread.start(); return self
    def __exit__(self, exc_type, exc, traceback):
        self._stop.set(); self._thread.join(timeout=3)
        if exc_type is None:
            self.check()


class ResourceProvider:
    """Native system/process reads plus an independently timed GPU reader."""
    def __init__(self):
        self._stop = threading.Event()
        self._gpu = {"value": None, "at": None}
        self._thread = None
    def startup(self, output_dir):
        gpu = policy_memory.gpu_memory(timeout=3)
        self._gpu = {"value": gpu, "at": time.monotonic()}
        return self._snapshot(None, output_dir)
    def start(self):
        def sample():
            while not self._stop.wait(5):
                try:
                    self._gpu = {"value": policy_memory.gpu_memory(timeout=3), "at": time.monotonic()}
                except BaseException as exc:
                    self._gpu = {"value": {"ok": False, "error": str(exc)}, "at": time.monotonic()}
        self._thread = threading.Thread(target=sample, daemon=True); self._thread.start()
    def _snapshot(self, pid, output_dir):
        system, disk = policy_memory.system_memory(), policy_memory.disk_snapshot(output_dir)
        gpu = self._gpu["value"] or {}
        process = policy_memory.process_memory(pid) if pid is not None else {}
        return {"available_ram_bytes": system.get("physical", {}).get("available_bytes") if system.get("ok") else None,
            "commit_headroom_bytes": system.get("commit", {}).get("headroom_bytes") if system.get("ok") else None,
            "disk_free_bytes": disk.get("free_bytes") if disk.get("ok") else None,
            "gpu_free_bytes": min((d["free_bytes"] for d in gpu.get("devices", [])), default=None) if gpu.get("ok") else None,
            "gpu_age_seconds": time.monotonic()-self._gpu["at"] if self._gpu["at"] is not None else None,
            "working_set_bytes": process.get("working_set_bytes") if process.get("ok") else None,
            "private_bytes": process.get("private_bytes") if process.get("ok") else None,
            "process_peak_working_set_bytes": process.get("peak_working_set_bytes") if process.get("ok") else None,
            "errors": {key: value.get("error") for key, value in (("system", system), ("gpu", gpu), ("disk", disk), ("process", process)) if value.get("ok") is False}}
    def runtime(self, pid, output_dir):
        return self._snapshot(pid, output_dir)
    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)


def resource_reason(snapshot, *, startup=False):
    required = {"available_ram_bytes": LIMITS["startup_ram_bytes" if startup else "runtime_ram_bytes"],
        "commit_headroom_bytes": LIMITS["startup_commit_bytes" if startup else "runtime_commit_bytes"],
        "gpu_free_bytes": LIMITS["startup_gpu_bytes" if startup else "runtime_gpu_bytes"],
        "disk_free_bytes": LIMITS["startup_disk_bytes"] if startup else LIMITS["output_bytes"]+LIMITS["telemetry_bytes"]}
    if not isinstance(snapshot, dict):
        return "invalid_resource_snapshot"
    for name, floor in required.items():
        value = snapshot.get(name)
        if type(value) is not int or value < 0:
            return "missing_or_invalid_resource:"+name
        if value < floor:
            return "resource_below_floor:"+name
    age = snapshot.get("gpu_age_seconds")
    if not _finite(age) or not 0 <= age <= LIMITS["gpu_max_age_seconds"]:
        return "gpu_measurement_stale_or_invalid"
    if not startup:
        for name in ("working_set_bytes", "private_bytes"):
            value = snapshot.get(name)
            if type(value) is not int or value < 0:
                return "missing_or_invalid_resource:"+name
            if value > LIMITS[name]:
                return "process_memory_limit:"+name
        peak = snapshot.get("process_peak_working_set_bytes")
        if peak is not None:
            if not _finite(peak) or peak < 0:
                return "missing_or_invalid_resource:process_peak_working_set_bytes"
            if peak > LIMITS["working_set_bytes"]:
                return "process_memory_limit:process_peak_working_set_bytes"
    return None


def worker_status_reason(status, *, pid, nonce, now=None, loading_elapsed=None):
    now = time.time() if now is None else now
    if not isinstance(status, dict) or status.get("actual_worker_pid") != pid or status.get("nonce") != nonce:
        return "worker_status_identity_missing_or_invalid"
    at = status.get("at_unix")
    if not _finite(at) or not -.5 <= now-at <= LIMITS["worker_status_max_age_seconds"]:
        return "worker_status_stale_or_invalid"
    phase = status.get("phase")
    if not isinstance(phase, str) or phase not in PHASES:
        return "worker_phase_missing_or_invalid"
    if phase == "error" or status.get("error"):
        return "worker_error:"+str(status.get("error", "unspecified"))
    for name in ("model_instances", "model_loads", "optimizer_updates"):
        value = status.get(name)
        if type(value) is not int or value < 0:
            return "worker_counter_missing_or_invalid:"+name
        if value > (0 if name == "optimizer_updates" else 1):
            return "worker_counter_limit:"+name
    for name in FORBIDDEN_COUNTERS:
        if type(status.get(name)) is not int or status[name] != 0:
            return "forbidden_worker_operation:"+name
    if "policy_calls" in status and (type(status["policy_calls"]) is not int or status["policy_calls"] != 0):
        return "forbidden_worker_operation:policy_calls"
    if phase in ("setup", "measurement", "M"):
        if any(status[name] != 0 for name in ("model_instances", "model_loads", "optimizer_updates")):
            return "model_loaded_before_measurement_gate"
    if phase in ("model_loading", "running") and any(status[name] != 1 for name in ("model_instances", "model_loads")):
        return "model_missing_in_backend_phase"
    if phase == "model_loading" and loading_elapsed is not None and loading_elapsed > LIMITS["loading_max_seconds"]:
        return "model_loading_deadline"
    for name in ("cuda_allocated_bytes", "cuda_reserved_bytes"):
        value = status.get(name)
        if phase == "model_loading" and value is None:
            continue  # The independent whole-device read remains mandatory.
        if type(value) is not int or value < 0:
            return "cuda_measurement_missing_or_invalid:"+name
        if value > LIMITS[name]:
            return "cuda_allocator_limit:"+name
        if phase in ("setup", "measurement", "M") and value != 0:
            return "cuda_allocation_before_measurement_gate"
    return None


def storage_usage(output_dir, telemetry_dirs=(), *, storage=None, full_verify=False):
    from .wm_compat_storage import Coordinator, unlinked_path
    output = unlinked_path(output_dir)
    extra = [path for path in (unlinked_path(value) for value in telemetry_dirs) if not path.is_relative_to(output)]
    storage = storage or Coordinator(result_roots=(output,), telemetry_roots=(output/"guardian", *dict.fromkeys(extra)))
    return storage.usage(full_verify=full_verify)


class _Recorder:
    """Preallocated retained-handle stop slot survives ordinary replace failures."""
    def __init__(self, directory, storage=None):
        self.directory, self.path = directory, directory/"stop-record.json"
        self.storage = storage
        if storage is not None and storage.fixed_slots.get(self.path) != 65536:
            raise ValueError("Guardian stop record requires its exact predeclared 64KiB fixed slot")
        with _transaction(storage, "guardian.recorder_create"):
            if storage is not None:
                storage.claim_fixed(self.path)
            self.slot = self.path.open("xb+", buffering=0)
            _write_all(self.slot, b"{}"+b" "*(65536-2)); os.fsync(self.slot.fileno())
            ticket = _reserve(storage, directory/"monitor.jsonl", 0, phase="guardian.monitor_create")
            self.stream = (directory/"monitor.jsonl").open("xb", buffering=0)
            if storage is not None:
                storage.validate_append(directory/"monitor.jsonl", self.stream)
            _commit(storage, ticket, 0)
        self.sequence = 0
    def event(self, kind, detail, *, durable=False):
        payload = _json({"event_id": self.sequence+1, "kind": kind, "at_utc": datetime.now(timezone.utc).isoformat(), "detail": detail})
        with _transaction(self.storage, "guardian.monitor_append"):
            ticket = _reserve(self.storage, self.stream.name, len(payload), phase="guardian.monitor_append")
            if self.storage is not None:
                self.storage.validate_append(self.stream.name, self.stream)
            try:
                _write_all(self.stream, payload)
                if durable:
                    os.fsync(self.stream.fileno())
            except OSError as exc:
                raise _io_error(self.stream.name, "guardian.monitor_append", exc) from exc
            if self.storage is not None:
                self.storage.record_append(self.stream.name, payload, self.stream)
            _commit(self.storage, ticket, len(payload))
            self.sequence += 1
    def finish(self, report):
        _write_stop(self.slot, report, storage=self.storage)
    def close(self):
        with _transaction(self.storage, "guardian.recorder_close"):
            primary = None
            for stream in (self.stream, self.slot):
                try:
                    if not getattr(stream, "closed", False):
                        os.fsync(stream.fileno())
                    stream.close()
                except BaseException as exc:
                    if primary is None:
                        primary = exc
                    else:
                        primary.add_note("Secondary recorder close failure: "+type(exc).__name__+": "+str(exc))
            if primary is not None:
                raise primary


def _write_stop(slot, report, *, storage=None):
    payload = _json(report)
    if len(payload) > 65536:
        raise ValueError("Guardian stop record exceeds its preallocated slot")
    # The full fixed slot is reserved before any work. A failed scan must not
    # prevent recording its own failure through this retained emergency handle.
    with _transaction(storage, "guardian.stop_record"):
        try:
            slot.seek(0); _write_all(slot, payload+b" "*(65536-len(payload)))
            os.fsync(slot.fileno())
        except OSError as exc:
            raise _io_error(report.get("stop_record", str(slot.name)), "guardian.stop_record", exc) from exc


class _OutputPump:
    """Drain a child PIPE in bounded chunks; all log growth uses the same lock."""
    def __init__(self, source, path, storage):
        if source is None:
            raise RuntimeError("Guarded child did not expose its requested stdout PIPE")
        self.source, self.path, self.storage = source, Path(path), storage
        self.error = None
        with _transaction(storage, "guardian.stdout_create"):
            ticket = _reserve(storage, self.path, 0, phase="guardian.stdout_create")
            self.stream = self.path.open("xb", buffering=0)
            if storage is not None:
                storage.validate_append(self.path, self.stream)
            _commit(storage, ticket, 0)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    def _run(self):
        phase = "guardian.stdout_read"
        try:
            while True:
                phase = "guardian.stdout_read"
                chunk = self.source.read(8192)
                if not chunk:
                    return
                if not isinstance(chunk, bytes) or len(chunk) > 8192:
                    raise ValueError("Child stdout must be bounded binary PIPE chunks")
                with _transaction(self.storage, "guardian.stdout_append"):
                    phase = "guardian.stdout_append"
                    ticket = _reserve(self.storage, self.path, len(chunk), phase="guardian.stdout_append")
                    if self.storage is not None:
                        self.storage.validate_append(self.path, self.stream)
                    _write_all(self.stream, chunk)
                    if self.storage is not None:
                        self.storage.record_append(self.path, chunk, self.stream)
                    _commit(self.storage, ticket, len(chunk))
        except OSError as exc:
            self.error = _io_error(self.path, phase, exc)
        except BaseException as exc:
            self.error = exc
    def check(self):
        if self.error is not None:
            raise RuntimeError("guardian_stdout_pump_failed: "+type(self.error).__name__+": "+str(self.error)) from self.error
    def close(self):
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise TimeoutError("Child stdout PIPE did not reach EOF after process-tree cleanup")
        try:
            with _transaction(self.storage, "guardian.stdout_close"):
                for stream in (self.stream, self.source):
                    try:
                        if stream is self.stream and not getattr(stream, "closed", False):
                            os.fsync(stream.fileno())
                        stream.close()
                    except BaseException as exc:
                        if self.error is None:
                            self.error = exc
                        else:
                            self.error.add_note("Secondary stdout stream close failure: "+type(exc).__name__+": "+str(exc))
        except BaseException as exc:
            if self.error is None:
                self.error = exc
            else:
                self.error.add_note("Secondary stdout storage close failure: "+type(exc).__name__+": "+str(exc))
        self.check()


def plan(command, *, output_dir, lock_path=DEFAULT_LOCK, deadline_seconds=1800):
    """Read-only declaration; never creates a lock/output or samples resources."""
    if not isinstance(command, (list, tuple)) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ValueError("Target command must be a nonempty string argv, excluding the Python interpreter")
    if not _finite(deadline_seconds) or not 0 < deadline_seconds <= LIMITS["wall_seconds"]:
        raise ValueError("Guardian deadline must be positive and at most 1800 seconds")
    return {"kind": "tdyssey.wm_compat_guard_plan.v1", "execution_authorized_by_plan": False,
            "command": list(command), "output_dir": str(Path(output_dir).resolve()),
            "lock_path": str(Path(lock_path).resolve()), "deadline_seconds": deadline_seconds,
            "limits": dict(LIMITS), "bootstrap": "stdlib nonce/PID ACK after exact Windows Job Object ownership"}


def supervise(command, *, output_dir, lock_path=DEFAULT_LOCK, python_executable=None,
              deadline_seconds=1800, provider=None, owner_factory=None, launcher_factory=None,
              poll_seconds=.2, telemetry_dirs=(), ledger_dir=None, attempt_registry=None):
    """Guard an explicitly selected target; caller must supply execution authority.

    Provider exposes startup(output), start(), runtime(pid, output), close(). All
    injected providers/owners are reported as test injections, never native proof.
    No shell is used. The command is target-script argv without an interpreter.
    """
    began = time.monotonic()
    declaration = plan(command, output_dir=output_dir, lock_path=lock_path, deadline_seconds=deadline_seconds)
    if not _finite(poll_seconds) or not 0 < poll_seconds <= 1:
        raise ValueError("Resource poll interval must be in (0,1] seconds")
    registry = None if attempt_registry is None else _new_namespace(attempt_registry)
    if registry is not None and registry not in telemetry_dirs:
        telemetry_dirs = (*telemetry_dirs, registry)
    output, ledger, telemetry_dirs, lock = validate_paths(output_dir, ledger_dir=ledger_dir,
        telemetry_dirs=telemetry_dirs, lock_path=lock_path)
    if Path(command[0]).resolve() == ROOT/"scripts/validate_wm_compat.py":
        identity = _launch_identity(command)
        validate_execution_identity(ROOT, output, ledger, identity["authorization_id"])
        if registry != ROOT/"artifacts"/REGISTRY:
            raise ValueError("Production execution requires the fixed v3 permanent registry")
    if ledger is not None and ledger not in telemetry_dirs:
        telemetry_dirs = (*telemetry_dirs, ledger)
    nonce = uuid.uuid4().hex
    launch_seal = None
    if registry is not None:
        identity = _launch_identity(command)
        if ledger is None:
            raise ValueError("Sealed compatibility execution requires an explicit ledger")
        if registry.is_relative_to(ledger) or ledger.is_relative_to(registry):
            raise ValueError("Launch registry and ledger must be disjoint")
        seal = dict(kind="tdyssey.wm_compat_launch_seal.v1", attempt_id=ATTEMPT_ID,
            nonce=nonce, guard_pid=os.getpid(), output_dir=str(output), ledger_dir=str(ledger),
            started_monotonic=began, deadline_monotonic=began+deadline_seconds, deadline_seconds=deadline_seconds,
            command=list(command), command_sha256=hashlib.sha256(_json(list(command))).hexdigest(),
            resume_allowed=False, created_utc=datetime.now(timezone.utc).isoformat(), **identity)
        if len(_json(seal)) > 65536:
            raise ValueError("Permanent launch seal exceeds its bounded setup metadata")
        registry.mkdir(parents=True, exist_ok=True)
        seal_path = registry/"launcher-attempt.json"
        try:
            _exclusive(seal_path, seal)
        except FileExistsError as exc:
            raise RuntimeError("Compatibility launcher attempt is permanently sealed; no retry or replacement") from exc
        launch_seal = dict(path=str(seal_path), sha256=hashlib.sha256(_json(seal)).hexdigest(), nonce=nonce)
    output.mkdir(parents=True, exist_ok=True)
    directory = output/"guardian"/nonce
    directory.mkdir(parents=True)
    from .wm_compat_storage import Coordinator, create_lock_file
    storage_policy = _storage_policy(output, ledger, directory, telemetry_dirs)
    create_lock_file(storage_policy["lock_path"])
    storage = Coordinator(**storage_policy)
    if launch_seal is not None:
        path = Path(launch_seal["path"])
        ticket = _reserve(storage, path, path.stat().st_size, phase="guardian.setup_seal_accounting")
        row = storage.remember_file(path)
        _commit(storage, ticket, row.size)
    recorder = _Recorder(directory, storage)
    provider_injected, owner_injected = provider is not None, owner_factory is not None
    provider = provider or ResourceProvider()
    owner_factory, launcher_factory = owner_factory or WindowsWorker, launcher_factory or subprocess.Popen
    lock_acquired = False
    child = owned = None
    pump = None
    job_empty_verified = False
    reason = error = worker_exit = launcher_exit = initial = last_status = last_resource = last_storage = None
    storage_error = None
    journal_audit = final_audit = None
    loading_began, loading_finished = None, False
    maximum_loads = 0
    peaks = {name: 0 for name in ("working_set_bytes", "private_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes")}
    minima = {name: None for name in ("available_ram_bytes", "commit_headroom_bytes", "gpu_free_bytes", "disk_free_bytes")}
    samples = status_unavailable_samples = 0
    try:
        with storage.local_transaction("guardian.single_model_lock"):
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock_value = {"nonce": nonce, "guard_pid": os.getpid(), "output_dir": str(output)}
            if len(_json(lock_value)) > 65536:
                raise ValueError("Global single-model lock exceeds its reserved 64KiB")
            _exclusive(lock, lock_value)
        lock_acquired = True
        recorder.event("lock_acquired", declaration, durable=True)
        initial = provider.startup(output)
        reason = resource_reason(initial, startup=True)
        recorder.event("startup", initial, durable=True)
        if reason:
            raise RuntimeError(reason)
        launch = [str(python_executable or sys.executable), "-B", "-m", "tdyssey.wm_compat_guard", "--bootstrap",
            "--handshake", str(directory/"worker.json"), "--ack", str(directory/"worker-ack.json"),
            "--nonce", nonce, "--", *command]
        environment = os.environ.copy()
        environment.update({ENV_DIRECTORY: str(directory), ENV_NONCE: nonce, ENV_PARENT: str(os.getpid())})
        environment[ENV_STORAGE] = json.dumps(storage_policy, sort_keys=True)
        # These apply before the child imports NumPy/SciPy or the lazy backend.
        environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
            "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE"})
        with nullcontext():
            child = launcher_factory(launch, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            pump = _OutputPump(getattr(child, "stdout", None), directory/"worker-output.log", storage)
            handshake_began = time.monotonic()
            while not (directory/"worker.json").exists():
                pump.check()
                if child.poll() is not None or time.monotonic()-handshake_began >= min(10, deadline_seconds):
                    reason = "bootstrap_handshake_failed_or_expired"; raise RuntimeError(reason)
                time.sleep(.02)
            identity = _locked_read(directory/"worker.json", storage)
            if identity.get("nonce") != nonce:
                reason = "bootstrap_nonce_mismatch"; raise RuntimeError(reason)
            storage.remember_file(directory/"worker.json")
            owned = owner_factory(identity["pid"], child.pid)
            marker = {"status": "owned_before_ack", "guard_pid": os.getpid(), "launcher_pid": child.pid,
                "actual_worker_pid": owned.pid, "nonce": nonce, "job_memory_bytes": LIMITS["job_memory_bytes"],
                "kill_on_job_close": True, "output_dir": str(output), "limits": dict(LIMITS),
                "ledger_dir": None if ledger is None else str(ledger),
                "launch_seal": launch_seal,
                "storage_lock": storage_policy["lock_path"], "storage_policy": storage_policy,
                "storage_telemetry_roots": storage_policy["telemetry_roots"],
                "started_monotonic": began, "deadline_monotonic": began+deadline_seconds,
                "deadline_seconds": deadline_seconds}
            _immutable_json(directory/"guardian.json", marker, storage)
            recorder.event("actual_worker_owned", marker, durable=True)
            # Recheck available resources after ownership and before heavy work.
            last_resource = provider.runtime(owned.pid, output)
            reason = resource_reason(last_resource, startup=True) or resource_reason(last_resource)
            if reason:
                raise RuntimeError(reason)
            if time.monotonic()-began >= deadline_seconds:
                reason = "deadline_before_ack"; raise RuntimeError(reason)
            _immutable_json(directory/"worker-ack.json", {"pid": owned.pid, "nonce": nonce}, storage)
            acknowledged = time.monotonic()
            provider.start()
            status_path = directory/"worker-status.json"
            while not owned.poll_exit():
                pump.check()
                now = time.monotonic()
                if now-began >= deadline_seconds:
                    reason = "deadline_exceeded"; break
                last_resource = provider.runtime(owned.pid, output)
                reason = resource_reason(last_resource)
                if reason:
                    if owned.poll_exit() and reason in ("missing_or_invalid_resource:working_set_bytes", "missing_or_invalid_resource:private_bytes"):
                        reason = None
                    break
                for key in minima:
                    value = last_resource.get(key)
                    if type(value) is int:
                        minima[key] = value if minima[key] is None else min(minima[key], value)
                for key in ("working_set_bytes", "private_bytes"):
                    if type(last_resource.get(key)) is int:
                        peaks[key] = max(peaks[key], last_resource[key])
                if _finite(last_resource.get("process_peak_working_set_bytes")):
                    peaks["working_set_bytes"] = max(peaks["working_set_bytes"], last_resource["process_peak_working_set_bytes"])
                previous_status = last_status
                last_status = _latest_worker_status(status_path, storage, last_status)
                if last_status is previous_status:
                    status_unavailable_samples += 1
                if last_status is None:
                    if now-acknowledged > LIMITS["worker_status_max_age_seconds"]:
                        reason = "worker_status_initial_missing"; break
                else:
                    declared_loads = last_status.get("model_loads")
                    if type(declared_loads) is int:
                        if declared_loads < maximum_loads:
                            reason = "model_load_counter_moved_backward"; break
                        maximum_loads = max(maximum_loads, declared_loads)
                    if last_status.get("phase") == "model_loading":
                        if loading_finished:
                            reason = "model_loading_reentered"; break
                        loading_began = now if loading_began is None else loading_began
                    else:
                        loading_finished = loading_finished or loading_began is not None
                        loading_began = None
                    reason = worker_status_reason(last_status, pid=owned.pid, nonce=nonce,
                        loading_elapsed=None if loading_began is None else now-loading_began)
                    if reason:
                        break
                    for key in ("cuda_allocated_bytes", "cuda_reserved_bytes"):
                        if type(last_status.get(key)) is int:
                            peaks[key] = max(peaks[key], last_status[key])
                last_storage = storage_usage(output, telemetry_dirs, storage=storage)
                for key in ("output_bytes", "telemetry_bytes"):
                    amount = last_storage[key]
                    reservation_key = "outstanding_"+key
                    outstanding = 0 if last_status is None else last_status.get(reservation_key, 0)
                    if type(outstanding) is not int or outstanding < 0:
                        reason = "invalid_worker_storage_reservation:"+reservation_key; break
                    if amount+outstanding > LIMITS[key]:
                        reason = "storage_limit:"+key; break
                if reason:
                    break
                recorder.event("sample", {"elapsed_seconds": now-began, "resources": last_resource,
                    "phase": None if last_status is None else last_status.get("phase"), "storage": last_storage})
                samples += 1
                time.sleep(poll_seconds)
            if reason is None:
                pump.check()
                worker_exit = owned.exit_code()
                final_status = _latest_worker_status(status_path, storage, None)
                if final_status is not None:
                    last_status = final_status
                    reason = worker_status_reason(last_status, pid=owned.pid, nonce=nonce)
                else:
                    reason = "final_worker_status_missing"
                if worker_exit != 0:
                    reason = reason or "worker_exit_nonzero_or_unconfirmed"
                elif reason is None and last_status.get("phase") != "complete":
                    reason = "worker_did_not_publish_complete"
    except FileExistsError as exc:
        reason = reason or "single_model_lock_or_evidence_collision"
        error = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:
        reason = reason or (str(exc) if str(exc).startswith("storage_limit:") else
            "stdout_pump_failure" if str(exc).startswith("guardian_stdout_pump_failed:") else
            "storage_failure" if hasattr(exc, "details") else "io_failure" if isinstance(exc, OSError) else "guardian_exception")
        error = f"{type(exc).__name__}: {exc}"
        storage_error = _storage_error_details(exc)
    finally:
        cleanup_errors = []
        if owned is not None:
            try:
                if not owned.poll_exit():
                    owned.terminate_and_wait(timeout=10)
                worker_exit = owned.exit_code()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                active = owned.quiesce(timeout=10)
                if type(active) is not int or active != 0:
                    raise RuntimeError("Owned Job cleanup did not verify zero active processes")
                job_empty_verified = True
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                owned.close()  # Kill all descendants even when the worker exited first.
            except BaseException as exc:
                cleanup_errors.append(exc)
        if child is not None:
            try:
                try:
                    launcher_exit = child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        launcher_exit = child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill(); launcher_exit = child.wait(timeout=10)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if pump is not None:
            try:
                pump.close()
            except BaseException as exc:
                reason = reason or "stdout_pump_failure"
                error = error or f"{type(exc).__name__}: {exc}"
                storage_error = storage_error or _storage_error_details(exc)
                cleanup_errors.append(exc)
        try:
            provider.close()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            reason = reason or "process_tree_cleanup_failed"
            error = error or "; ".join(f"{type(exc).__name__}: {exc}" for exc in cleanup_errors)
        try:
            if lock_acquired and not cleanup_errors:
                with storage.local_transaction("guardian.single_model_unlock"):
                    if _read(lock).get("nonce") != nonce:
                        raise RuntimeError("Single-model lock owner changed")
                    lock.unlink()
        except BaseException as exc:
            reason = reason or "single_model_lock_release_failed"
            error = error or f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic()-began
        if elapsed >= deadline_seconds:
            reason = reason or "deadline_exceeded"
        if child is not None and launcher_exit != 0:
            reason = reason or "launcher_exit_nonzero_or_unconfirmed"
        try:
            expected, journal_audit = _completed_output_manifest(ledger, output,
                require_terminal=reason is None and ledger is not None)
            last_storage = storage.audit(expected=expected)
            final_audit = {"passed":True, "journal":journal_audit}
            for key in ("output_bytes", "telemetry_bytes"):
                amount = last_storage[key]
                if amount > LIMITS[key]:
                    reason = reason or "storage_limit:"+key
        except BaseException as exc:
            reason = reason or ("storage_failure" if hasattr(exc, "details") else "final_storage_measurement_failed")
            error = error or f"{type(exc).__name__}: {exc}"
            storage_error = storage_error or _storage_error_details(exc)
            final_audit = {"passed":False, "error":f"{type(exc).__name__}: {exc}", "journal":journal_audit}
        report = {"kind": "tdyssey.wm_compat_guard_report.v1", "status": "stopped" if reason else "exited",
            "guard_reason": reason, "error": error, "initial": initial, "last_resource": last_resource,
            "storage_error": storage_error,
            "final_audit": final_audit,
            "last_worker_status": last_status, "guard_pid": os.getpid(),
            "launcher_pid": None if child is None else child.pid, "actual_worker_pid": None if owned is None else owned.pid,
            "worker_exit_code": worker_exit, "launcher_exit_code": launcher_exit,
            "kill_on_job_close": owned is not None, "job_memory_bytes": LIMITS["job_memory_bytes"],
            "elapsed_seconds": elapsed, "deadline_seconds": deadline_seconds, "samples": samples,
            "worker_status_unavailable_samples": status_unavailable_samples,
            "started_monotonic": began, "deadline_monotonic": began+deadline_seconds,
            "observed_peaks": peaks, "sampled_minima": minima, "storage": last_storage,
            "maximum_observed_model_loads": maximum_loads,
            "launch_seal": launch_seal,
            "storage_lock": storage_policy["lock_path"], "storage_policy": storage_policy,
            "job_empty_verified_before_close": job_empty_verified,
            "single_model_lock_retained_after_cleanup_failure": bool(lock_acquired and cleanup_errors),
            "provider_injected": provider_injected, "owner_injected": owner_injected,
            "directory": str(directory), "stop_record": str(recorder.path),
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "sampling_limitations": "Floors and peaks are sampled. Job memory/kill-on-close apply to the owned process tree only."}
        # Keep an independent descriptor until journal/slot close have succeeded:
        # close can flush and fail after the ordinary slot was already closed.
        emergency = None
        def final_io_failure(exc):
            report["status"] = "stopped"
            report["guard_reason"] = report["guard_reason"] or "io_failure"
            detail = f"{type(exc).__name__}: {exc}"
            report["error"] = report["error"] or detail
            report["storage_error"] = report["storage_error"] or _storage_error_details(exc)
            report.setdefault("finalization_errors", []).append(detail)
        try:
            emergency = os.fdopen(os.dup(recorder.slot.fileno()), "r+b", buffering=0)
        except BaseException as exc:
            final_io_failure(exc)
        try:
            recorder.event("stopped" if reason else "exited", {"guard_reason": reason}, durable=True)
        except BaseException as exc:
            final_io_failure(exc)
        try:
            recorder.finish(report)
        except BaseException as exc:
            final_io_failure(exc)
        try:
            recorder.close()
        except BaseException as exc:
            final_io_failure(exc)
        try:
            storage.close()
        except BaseException as exc:
            final_io_failure(exc)
        if emergency is not None:
            try:
                _write_stop(emergency, report, storage=storage)
            except BaseException as exc:
                final_io_failure(exc)
                report["stop_record_persistence_failed"] = True
            finally:
                try:
                    emergency.close()
                except BaseException as exc:
                    final_io_failure(exc)
    return report


if __name__ == "__main__":
    if sys.argv[1:2] != ["--bootstrap"]:
        raise SystemExit("Only the parent guardian may select the --bootstrap entry")
    raise SystemExit(bootstrap_child(sys.argv[2:]))
