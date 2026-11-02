"""Incremental quota accounting and explicit full integrity audits.

Only shared byte counters use the process mutex. Content IO and owner journals
run outside it. Fixed telemetry is prepaid; failed reservations stay charged.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import sys
import threading
import time
import uuid

MIB = 1024**2
ATOMIC_BYTES = 65536
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class StorageError(RuntimeError):
    def __init__(self, path, phase, error=None, *, message=None):
        error = error if isinstance(error, BaseException) else ValueError(message or str(error or "Invalid storage"))
        self.details = dict(path=str(path), phase=str(phase), exception_type=type(error).__name__,
                            errno=getattr(error, "errno", None),winerror=getattr(error,"winerror",None),message=str(error))
        super().__init__("storage_failure: " + "; ".join(f"{key}={value}" for key, value in self.details.items()))


def _invalid(path, phase, message, kind=ValueError):
    raise StorageError(path, phase, kind(message))


def _lstat(path, phase):
    try:
        value = os.lstat(path)
    except OSError as error:
        raise StorageError(path, phase, error) from error
    if stat.S_ISLNK(value.st_mode) or getattr(value, "st_file_attributes", 0) & REPARSE:
        _invalid(path, phase, "Symbolic links and Windows reparse points are forbidden")
    return value


def unlinked_path(value):
    """Absolute lexical path; reject links before canonicalizing their targets."""
    if ".." in Path(value).parts:
        _invalid(value, "path.traversal", "Parent traversal is forbidden in storage paths")
    path = Path(os.path.abspath(value))
    for part in reversed((path, *path.parents)):
        try:
            _lstat(part, "path.lstat")
        except StorageError as error:
            if error.details["exception_type"] == "FileNotFoundError":
                continue  # Path validation alone permits an uncreated tail.
            raise
    return path


def _identity(value):
    return value.st_dev, value.st_ino


def _signature(value):
    return _identity(value), value.st_size, value.st_mtime_ns, getattr(value, "st_nlink", 1)


@dataclass(frozen=True)
class FileRecord:
    size: int
    mtime_ns: int
    sha256: str
    identity: tuple
    prefix_sha256: str | None = None
    change_token: int | None = None


_WIN_INFORMATION = None
_WIN_INFORMATION_LOCK = threading.Lock()


def _change_token(fd, snapshot, path, phase):
    """Use inode change time, not Windows' creation-time st_ctime surrogate."""
    if os.name != "nt":
        return getattr(snapshot, "st_ctime_ns", None)
    global _WIN_INFORMATION
    with _WIN_INFORMATION_LOCK:
        if _WIN_INFORMATION is None:
            import ctypes
            from ctypes import wintypes
            class BasicInfo(ctypes.Structure):
                _fields_ = [(name, ctypes.c_longlong) for name in
                            ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")]+[("FileAttributes", wintypes.DWORD)]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            query = kernel.GetFileInformationByHandleEx
            query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            query.restype = wintypes.BOOL
            _WIN_INFORMATION = query, BasicInfo
    import ctypes
    import msvcrt
    query, structure = _WIN_INFORMATION
    value = structure()
    if not query(msvcrt.get_osfhandle(fd), 0, ctypes.byref(value), ctypes.sizeof(value)):
        error = ctypes.WinError(ctypes.get_last_error())
        raise StorageError(path, phase+".change_time", error) from error
    return value.ChangeTime or None


def fingerprint(path, *, phase="fingerprint", prefix_size=None, cached=None, _parents_checked=False):
    """Hash a stable ordinary file; verify path/descriptor identity both sides."""
    path = Path(path) if _parents_checked else unlinked_path(path)
    before = _lstat(path, phase + ".lstat")
    if not stat.S_ISREG(before.st_mode) or getattr(before, "st_nlink", 1) != 1:
        _invalid(path, phase + ".type", "Only single-link ordinary files are allowed")
    if prefix_size is not None and (type(prefix_size) is not int or not 0 <= prefix_size <= before.st_size):
        _invalid(path, phase + ".prefix", "Append-only file shrank")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = None
    try:
        try:
            fd = os.open(path, flags)
            opened = os.fstat(fd)
        except OSError as error:
            raise StorageError(path, phase + ".open", error) from error
        if not stat.S_ISREG(opened.st_mode) or _signature(opened) != _signature(before):
            _invalid(path, phase + ".identity", "File changed between lstat and open")
        change = _change_token(fd, opened, path, phase)
        digest, prefix, consumed = hashlib.sha256(), hashlib.sha256(), 0
        reuse = (cached is not None and change is not None and change == cached.change_token
                 and cached.identity == _identity(opened) and cached.size == opened.st_size
                 and cached.mtime_ns == opened.st_mtime_ns
                 and (prefix_size is None or prefix_size == cached.size))
        try:
            while not reuse:
                block = os.read(fd, MIB)
                if not block:
                    break
                digest.update(block)
                if prefix_size is not None and consumed < prefix_size:
                    prefix.update(block[:prefix_size-consumed])
                consumed += len(block)
            after_handle = os.fstat(fd)
        except OSError as error:
            raise StorageError(path, phase + ".read", error) from error
        after_path = _lstat(path, phase + ".restat")
        if (_signature(before) != _signature(after_handle) or _signature(before) != _signature(after_path)
                or (not reuse and consumed != before.st_size)
                or _change_token(fd, after_handle, path, phase) != change):
            _invalid(path, phase + ".consistency", "File changed during its storage snapshot")
        return FileRecord(before.st_size, before.st_mtime_ns, cached.sha256 if reuse else digest.hexdigest(), _identity(before),
                          (cached.sha256 if reuse else prefix.hexdigest()) if prefix_size is not None else None,
                          change)
    finally:
        primary = sys.exc_info()[1]
        if fd is not None:
            try:
                os.close(fd)
            except OSError as error:
                detail = StorageError(path, phase + ".close", error)
                if primary is None:
                    raise detail from error
                primary.add_note(str(detail))


def scan_files(root, *, phase="scan", prefix_sizes=None, allow_missing_root=False, _lock_record=None, _cache=None):
    """Strict tree snapshot. Only an explicitly allowed initial root may be absent."""
    root = unlinked_path(root)
    try:
        initial = _lstat(root, phase + ".root")
    except StorageError as error:
        if allow_missing_root and error.details["exception_type"] == "FileNotFoundError":
            return {}
        raise
    if not stat.S_ISDIR(initial.st_mode):
        _invalid(root, phase + ".root", "Storage root is not a directory")
    result, prefix_sizes = {}, prefix_sizes or {}

    def walk(folder):
        before = _lstat(folder, phase + ".directory")
        if not stat.S_ISDIR(before.st_mode):
            _invalid(folder, phase + ".directory", "Directory replaced by another file type")
        try:
            with os.scandir(folder) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError as error:
            raise StorageError(folder, phase + ".enumerate", error) from error
        for name in names:
            path = folder / name
            value = _lstat(path, phase + ".entry")
            if stat.S_ISDIR(value.st_mode):
                walk(path)
            elif stat.S_ISREG(value.st_mode):
                if _lock_record is not None and path == _lock_record[0]:
                    row = _lock_record[1]
                    if value.st_size != 1 or _identity(value) != row.identity or value.st_mtime_ns != row.mtime_ns:
                        _invalid(path, phase + ".lock_identity", "Storage lock changed")
                    result[str(path)] = row
                else:
                    result[str(path)] = fingerprint(path, phase=phase + ".file", prefix_size=prefix_sizes.get(str(path)),
                        cached=(_cache or {}).get(str(path)), _parents_checked=True)
            else:
                _invalid(path, phase + ".type", "Only ordinary files and directories are allowed")
        after = _lstat(folder, phase + ".directory_restat")
        if _identity(before) != _identity(after) or before.st_mtime_ns != after.st_mtime_ns:
            _invalid(folder, phase + ".directory_consistency", "Directory changed during its storage snapshot")
    walk(root)
    return result


def create_lock_file(path):
    """Create one persistent byte exclusively; never reset an existing lock."""
    path, fd = unlinked_path(path), None
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)
        if os.write(fd, b"0") != 1:
            raise OSError("Short storage-lock write")
        os.fsync(fd)
    except OSError as error:
        raise StorageError(path, "lock.create", error) from error
    finally:
        if fd is not None:
            os.close(fd)
    return path


_LOCAL_GUARD = threading.Lock()
_LOCAL_LOCKS = {}
_HELD = threading.local()


class _StorageAudit:
    """Bounded shared lock with immutable, append-only and exact mutable slots."""
    def __init__(self, lock_path=None, *, result_roots=(), telemetry_roots=(), atomic_targets=(),
                 append_only=(), fixed_slots=(), pending_roots=None, pending_reserve=0,
                 telemetry_reserve=0, emergency_headroom=0, timeout=.5, limits=None):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 5:
            raise ValueError("Storage lock timeout must be finite and in (0,5] seconds")
        if type(pending_reserve) is not int or not 0 <= pending_reserve <= 2*MIB:
            raise ValueError("Only one chunk destination/temporary reserve is allowed")
        if type(telemetry_reserve) is not int or not 0 <= telemetry_reserve <= ATOMIC_BYTES:
            raise ValueError("External lock telemetry reserve must fit 64KiB")
        if type(emergency_headroom) is not int or not 0 <= emergency_headroom <= ATOMIC_BYTES:
            raise ValueError("Emergency telemetry headroom must be integer bytes in [0,64KiB]")
        self.timeout, self.pending_reserve, self.telemetry_reserve = timeout, pending_reserve, telemetry_reserve
        self.emergency_headroom = emergency_headroom
        self.lock_path = None if lock_path is None else unlinked_path(lock_path)
        self.result_roots = tuple(dict.fromkeys(unlinked_path(p) for p in result_roots))
        self.telemetry_roots = tuple(dict.fromkeys(unlinked_path(p) for p in telemetry_roots))
        self.pending_roots = tuple(unlinked_path(p) for p in (self.result_roots if pending_roots is None else pending_roots))
        items = atomic_targets.items() if hasattr(atomic_targets, "items") else ((p, ATOMIC_BYTES) for p in atomic_targets)
        self.atomic_targets = {unlinked_path(p): n for p,n in items}
        items = fixed_slots.items() if hasattr(fixed_slots, "items") else fixed_slots
        self.fixed_slots = {unlinked_path(p): n for p,n in items}
        if any(type(n) is not int or not 0 < n <= ATOMIC_BYTES for n in (*self.atomic_targets.values(), *self.fixed_slots.values())):
            raise ValueError("Atomic/fixed slots must fit 64KiB")
        self.append_only = frozenset(unlinked_path(p) for p in append_only)
        self.limits = dict(output_bytes=256*MIB, telemetry_bytes=16*MIB)
        if limits:
            if set(limits)-set(self.limits) or any(type(n) is not int or n <= 0 for n in limits.values()):
                raise ValueError("Invalid storage limits")
            self.limits.update(limits)
        for path in (*self.atomic_targets, *self.fixed_slots, *self.append_only):
            if self._category(path) != "telemetry_bytes":
                raise ValueError("Mutable telemetry must be inside declared telemetry roots")
        if set(self.atomic_targets) & (set(self.fixed_slots) | set(self.append_only)) or set(self.fixed_slots) & set(self.append_only):
            raise ValueError("A path cannot have conflicting mutable policies")
        self._key = os.path.normcase(str(self.lock_path)) if self.lock_path is not None else "local-"+uuid.uuid4().hex
        with _LOCAL_GUARD:
            self._local_lock = _LOCAL_LOCKS.setdefault(self._key, threading.RLock())
        self._observed, self._append_seen, self._fixed_seen, self._atomic_seen, self._root_seen = {}, {}, {}, set(), {}

    def _category(self, path):
        if any(path.is_relative_to(root) for root in self.telemetry_roots):
            return "telemetry_bytes"
        if any(path.is_relative_to(root) for root in self.result_roots):
            return "output_bytes"
        _invalid(path, "policy.root", "Path is outside declared storage roots")

    @contextmanager
    def transaction(self, phase="transaction"):
        began = time.monotonic()
        if not self._local_lock.acquire(timeout=self.timeout):
            raise StorageError(self.lock_path or self._key, phase+".lock", TimeoutError("Storage thread lock deadline"))
        if not hasattr(_HELD, "depths"):
            _HELD.depths, _HELD.fds = {}, {}
        held, fds = _HELD.depths, _HELD.fds
        fd, os_locked, entered = None, False, False
        try:
            if self._key not in held and self.lock_path is not None:
                path = unlinked_path(self.lock_path)
                before = _lstat(path, phase+".lock_lstat")
                if not stat.S_ISREG(before.st_mode) or before.st_size != 1 or getattr(before, "st_nlink", 1) != 1:
                    _invalid(path, phase+".lock_type", "Storage lock must be a one-byte ordinary file")
                try:
                    fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
                    if _signature(os.fstat(fd)) != _signature(before):
                        _invalid(path, phase+".lock_identity", "Storage lock path changed")
                    while True:
                        try:
                            os.lseek(fd, 0, os.SEEK_SET)
                            if os.name == "nt":
                                import msvcrt
                                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                            else:
                                import fcntl
                                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            os_locked = True
                            break
                        except OSError as error:
                            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                                raise StorageError(path, phase+".lock_acquire", error) from error
                            if time.monotonic()-began >= self.timeout:
                                raise StorageError(path, phase+".lock_acquire", TimeoutError("Storage process lock deadline")) from error
                            time.sleep(.005)
                    # The byte is a mutex position, not stored application data.
                    # Its content needs no read (Windows locks also deny reads).
                    fds[self._key] = fd
                except OSError as error:
                    raise StorageError(path, phase+".lock_open", error) from error
            held[self._key] = held.get(self._key, 0)+1
            entered = True
            yield self
        finally:
            primary = sys.exc_info()[1]
            if entered:
                held[self._key] -= 1
                if not held[self._key]:
                    del held[self._key]
                    fds.pop(self._key, None)
            try:
                if fd is not None:
                    try:
                        if os_locked:
                            os.lseek(fd, 0, os.SEEK_SET)
                            if os.name == "nt":
                                import msvcrt
                                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                            else:
                                import fcntl
                                fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        os.close(fd)
            except OSError as error:
                detail = StorageError(self.lock_path, phase+".lock_release", error)
                if primary is None:
                    raise detail from error
                primary.add_note(str(detail))
            finally:
                self._local_lock.release()

    def _atomic_group(self, path):
        if path in self.atomic_targets:
            return path
        for target in self.atomic_targets:
            if path.parent == target.parent and path.name.startswith(target.name+".") and path.name.endswith(".tmp"):
                if not re.fullmatch(re.escape(target.name)+r"\.[0-9a-f]{32}\.tmp", path.name):
                    _invalid(path, "policy.atomic_name", "Undeclared atomic temporary name")
                return target
        return None

    def _audit_files(self, *, pending_files=None, full_verify=True):
        with self._audit_lock:
            pending = {str(unlinked_path(p)): n for p,n in (pending_files or {}).items()}
            for name,n in pending.items():
                path = Path(name)
                if (type(n) is not int or not 0 < n <= MIB or name in self._observed
                        or not any(path.is_relative_to(root) for root in self.pending_roots)):
                    _invalid(path, "policy.pending", "Invalid bounded pending-output declaration")
            roots = tuple(dict.fromkeys((*self.result_roots, *self.telemetry_roots)))
            minimal = [p for p in roots if not any(p != other and p.is_relative_to(other) for other in roots)]
            files, root_seen = {}, dict(self._root_seen)
            prefixes = {name: row.size for name,row in self._append_seen.items()}
            lock_record = None
            for root in minimal:
                try:
                    value = _lstat(root, "storage.root")
                except StorageError as error:
                    if error.details["exception_type"] != "FileNotFoundError" or str(root) in root_seen:
                        raise
                    continue
                if str(root) in root_seen and root_seen[str(root)] != _identity(value):
                    _invalid(root, "storage.root_identity", "Observed root was replaced")
                root_seen[str(root)] = _identity(value)
                files.update(scan_files(root, phase="storage.scan", prefix_sizes=prefixes,
                                        allow_missing_root=False, _lock_record=lock_record,
                                        _cache={} if full_verify else {
                                            **self._observed, **self._append_seen, **self._fixed_seen}))
            current = set(files)
            for name in (*self._observed, *self._append_seen, *self._fixed_seen):
                if name not in current:
                    _invalid(name, "policy.required_missing", "Observed storage file disappeared", FileNotFoundError)
            for path in self._atomic_seen:
                if str(path) not in current:
                    _invalid(path, "policy.atomic_missing", "Observed atomic target disappeared", FileNotFoundError)
            observed, append_seen = dict(self._observed), dict(self._append_seen)
            fixed_seen, atomic_seen = dict(self._fixed_seen), set(self._atomic_seen)
            atomic_sizes = {p: [] for p in self.atomic_targets}
            totals = dict(output_observed_bytes=0, telemetry_observed_bytes=0)
            for name,row in files.items():
                path = Path(name)
                category = self._category(path)
                totals[category.replace("_bytes", "_observed_bytes")] += row.size
                group = self._atomic_group(path)
                if group is not None:
                    if row.size > self.atomic_targets[group]:
                        _invalid(path, "policy.atomic_size", "Atomic file exceeds its fixed slot")
                    atomic_sizes[group].append((path, row.size))
                    if path == group:
                        atomic_seen.add(path)
                elif path in self._internal_controls:
                    if row.size != self._internal_controls[path]:
                        _invalid(path, "policy.control_size", "Accounting control size changed")
                elif path in self.fixed_slots:
                    if row.size > self.fixed_slots[path] or (name in fixed_seen and row.identity != fixed_seen[name].identity):
                        _invalid(path, "policy.fixed_slot", "Fixed slot changed identity or exceeded its bound")
                    fixed_seen[name] = row
                elif path in self.append_only:
                    previous = append_seen.get(name)
                    if previous and (row.identity != previous.identity or row.size < previous.size or row.prefix_sha256 != previous.sha256):
                        _invalid(path, "policy.append_prefix", "Append-only identity/history changed or shrank")
                    append_seen[name] = row
                elif name in pending:
                    if row.size > pending[name]:
                        _invalid(path, "policy.pending_size", "Pending file exceeds its reservation")
                else:
                    previous = observed.get(name)
                    if previous and (row.size, row.mtime_ns, row.sha256, row.identity) != (
                            previous.size, previous.mtime_ns, previous.sha256, previous.identity):
                        _invalid(path, "policy.immutable", "Immutable storage content or identity changed")
                    observed[name] = row
            pending_delta = dict(output_bytes=0, telemetry_bytes=0)
            for name, maximum in pending.items():
                category = self._category(Path(name))
                pending_delta[category] += max(0, maximum-(files[name].size if name in files else 0))
            charged = totals["telemetry_observed_bytes"]+self.telemetry_reserve+pending_delta["telemetry_bytes"]
            for target, group in atomic_sizes.items():
                if sum(path != target for path,size in group) > 1:
                    _invalid(target, "policy.atomic_count", "More than one atomic temporary is present")
                charged += 2*self.atomic_targets[target]-sum(size for path,size in group)
            for path, maximum in self.fixed_slots.items():
                charged += maximum-files[str(path)].size if str(path) in files else maximum
            totals.update(output_bytes=totals["output_observed_bytes"]+max(self.pending_reserve, pending_delta["output_bytes"]),
                          telemetry_bytes=charged,
                          output_reserved_bytes=max(self.pending_reserve, pending_delta["output_bytes"]),
                          telemetry_reserved_bytes=charged-totals["telemetry_observed_bytes"])
            self._observed, self._append_seen = observed, append_seen
            self._fixed_seen, self._atomic_seen, self._root_seen = fixed_seen, atomic_seen, root_seen
            return totals


@dataclass
class Reservation:
    owner: object
    path: Path
    category: str
    maximum: int
    temporary: int
    sequence: int
    result_slot: bool = False
    committed: bool = False


class Coordinator(_StorageAudit):
    """Incremental quota: only counter updates hold the shared process lock.

    Mapped counters are admission state, never a resumable or durable ledger.
    File IO and content verification use owner-local serialization. Complete
    filesystem verification is an explicit quiescent audit, not a quota query.
    """
    _PAGE, _OFFSET, _MAGIC = 4096, 40, b"WMCOUNT1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import struct
        self._struct = struct.Struct("<8Q")
        self._audit_lock, self._io_lock = threading.RLock(), threading.RLock()
        self._append_hashes, self._owned_fixed = {}, set()
        self._tickets = {}
        self._counter_fd = self._mapping = None
        self._closed = False
        self._internal_controls, self._local_state = {}, [0]*8
        self._fixed_paths = tuple(sorted(set(self.atomic_targets)|set(self.fixed_slots),key=str))
        if len(self._fixed_paths)>128:
            raise ValueError("Too many fixed telemetry slots")
        self.counter_path = None
        policy = repr((self.result_roots,self.telemetry_roots,self.pending_roots,
            sorted(self.atomic_targets.items()),sorted(self.fixed_slots.items()),
            sorted(self.append_only),self.pending_reserve,self.telemetry_reserve,
            self.emergency_headroom,sorted(self.limits.items()))).encode()
        self._policy_digest = hashlib.sha256(policy).digest()
        if self.lock_path is not None:
            if self._totals(self._local_state)["telemetry_bytes"]+self._PAGE+1>self.limits["telemetry_bytes"]:
                _invalid(self.lock_path,"accounting.setup.capacity","Prepaid control/telemetry slots exceed cap")
            self._open_counters()

    def _open_counters(self):
        import mmap
        self.counter_path = self.lock_path.with_name(self.lock_path.name+".counters")
        self._internal_controls = {self.lock_path:1,self.counter_path:self._PAGE}
        with self.transaction("accounting.setup"):
            flags = os.O_RDWR|getattr(os,"O_BINARY",0)|getattr(os,"O_NOFOLLOW",0)
            created, fd = False, None
            try:
                try:
                    fd = os.open(unlinked_path(self.counter_path),flags|os.O_CREAT|os.O_EXCL,0o600)
                    created = True
                except FileExistsError:
                    fd = os.open(unlinked_path(self.counter_path),flags)
                if created:
                    os.ftruncate(fd,self._PAGE)
                value = _lstat(self.counter_path,"accounting.control")
                if (not stat.S_ISREG(value.st_mode) or value.st_size!=self._PAGE
                        or getattr(value,"st_nlink",1)!=1 or _identity(value)!=_identity(os.fstat(fd))):
                    _invalid(self.counter_path,"accounting.control","Invalid accounting control file")
                mapping = mmap.mmap(fd,self._PAGE)
                if created:
                    mapping[:40] = self._MAGIC+self._policy_digest
                    mapping.flush()
                    os.fsync(fd)  # Setup only; admissions never flush this page.
                if mapping[:40]!=self._MAGIC+self._policy_digest:
                    mapping.close()
                    _invalid(self.counter_path,"accounting.policy","Accounting policy differs")
                self._counter_fd,self._mapping = fd,mapping
            except BaseException as error:
                if fd is not None:
                    os.close(fd)
                if isinstance(error,OSError):
                    raise StorageError(self.counter_path,"accounting.setup",error) from error
                raise

    @contextmanager
    def transaction(self, phase="accounting"):
        # Check the one retained control descriptor before acquiring the mutex.
        # No result/telemetry tree is opened or enumerated here.
        fd = getattr(self,"_counter_fd",None)
        if fd is not None:
            path = unlinked_path(self.counter_path)
            try:
                value,opened = _lstat(path,phase+".control"),os.fstat(fd)
            except OSError as error:
                raise StorageError(path,phase+".control",error) from error
            if (not stat.S_ISREG(value.st_mode) or getattr(value,"st_nlink",1)!=1
                    or value.st_size!=self._PAGE or _identity(value)!=_identity(opened)):
                _invalid(path,phase+".control","Accounting control identity changed")
        with super().transaction(phase):
            yield self

    @contextmanager
    def local_transaction(self, phase="writer"):
        """Owner-local IO serialization independent of the quota mutex."""
        key = self._key+"\0io:"+phase
        with _LOCAL_GUARD:
            lock = _LOCAL_LOCKS.setdefault(key,threading.RLock())
        with lock:
            yield self

    @property
    def accounting_lock_held(self):
        return bool(getattr(_HELD,"depths",{}).get(self._key))

    def _read_state(self):
        if self._closed:
            _invalid(self.counter_path or self._key,"accounting.closed","Closed accounting cannot be reused")
        if self._mapping is None:
            return list(self._local_state)
        if self._mapping[:40]!=self._MAGIC+self._policy_digest:
            _invalid(self.counter_path,"accounting.header","Accounting header changed")
        return list(self._struct.unpack_from(self._mapping,self._OFFSET))

    def _write_state(self, state):
        # No flush/fsync: durable claims/journals forbid continuation after crash.
        if self._mapping is None:
            self._local_state[:] = state
        else:
            self._struct.pack_into(self._mapping,self._OFFSET,*state)

    def _totals(self, state):
        fixed = (sum(2*n for n in self.atomic_targets.values())+sum(self.fixed_slots.values())
            +self.telemetry_reserve+self.emergency_headroom+sum(self._internal_controls.values()))
        return dict(output_bytes=state[0]+self.pending_reserve,telemetry_bytes=state[1]+fixed,
            output_observed_bytes=state[2],telemetry_observed_bytes=state[3],
            output_reserved_bytes=state[0]-state[2]+self.pending_reserve,
            telemetry_reserved_bytes=state[1]-state[3]+fixed,accounting_sequence=state[4],
            pending_result_publications=state[5],accounting_commits=state[6],
            accounting_max_hold_ns=state[7],
            observed_bytes_kind="committed ordinary bytes; fixed slots measured at audit")

    def usage(self, *, pending_files=None, full_verify=False):
        if full_verify:
            return self.audit()
        with self.transaction("accounting.snapshot"):
            return self._totals(self._read_state())

    def check_growth(self, path, additional_bytes, *, phase="write", reserve=0,pending_files=None):
        path = unlinked_path(path)
        if type(additional_bytes)is not int or additional_bytes<0 or type(reserve)is not int or reserve<0:
            _invalid(path,phase,"Growth/reserve must be nonnegative integer bytes")
        category = self._category(path)
        extra = max(0,reserve-self.emergency_headroom) if category=="telemetry_bytes" else reserve
        with self.transaction(phase+".admission"):
            totals = self._totals(self._read_state())
            if totals[category]+additional_bytes+extra>self.limits[category]:
                _invalid(path,phase+".capacity",category+" limit exceeded including reservations")
            return totals

    def reserve_bytes(self, path, max_bytes, *, temporary_bytes=0,phase="write"):
        path = unlinked_path(path)
        if (type(max_bytes)is not int or max_bytes<0 or type(temporary_bytes)is not int
                or not 0<=temporary_bytes<=MIB):
            _invalid(path,phase,"Invalid bounded write reservation")
        category = self._category(path)
        if path in self.atomic_targets or path in self.fixed_slots:
            _invalid(path,phase,"Fixed telemetry uses its prepaid slot")
        result_slot = category=="output_bytes" and temporary_bytes>0 and self.pending_reserve>0
        extra = 0 if result_slot else temporary_bytes
        if result_slot and (max_bytes>MIB or temporary_bytes>self.pending_reserve):
            _invalid(path,phase,"Publication exceeds prepaid scratch")
        with self.transaction(phase+".reserve"):
            began = time.perf_counter_ns()
            state = self._read_state()
            if result_slot and state[5]:
                _invalid(path,phase,"Another result publication is outstanding")
            index = 0 if category=="output_bytes" else 1
            if self._totals(state)[category]+max_bytes+extra>self.limits[category]:
                _invalid(path,phase+".capacity",category+" limit exceeded including reservation")
            state[index] += max_bytes+extra
            state[4] += 1
            state[5] += int(result_slot)
            state[7] = max(state[7],time.perf_counter_ns()-began)
            self._write_state(state)
            ticket = Reservation(self,path,category,max_bytes,extra,state[4],result_slot)
            self._tickets[ticket.sequence] = ticket
            return ticket

    def commit(self, ticket, actual_bytes):
        if (not isinstance(ticket,Reservation) or ticket.owner is not self or ticket.committed
                or type(actual_bytes)is not int or not 0<=actual_bytes<=ticket.maximum):
            _invalid(getattr(ticket,"path","reservation"),"accounting.commit","Invalid write completion")
        with self.transaction("accounting.commit"):
            if self._tickets.get(ticket.sequence) is not ticket or ticket.committed:
                _invalid(ticket.path,"accounting.commit","Reservation already committed or unknown")
            began = time.perf_counter_ns()
            state = self._read_state()
            index = 0 if ticket.category=="output_bytes" else 1
            state[index] -= ticket.temporary  # Only successful scratch is released.
            state[index+2] += actual_bytes
            state[5] -= int(ticket.result_slot)
            state[6] += 1
            state[7] = max(state[7],time.perf_counter_ns()-began)
            self._write_state(state)
            ticket.committed = True
            del self._tickets[ticket.sequence]

    def claim_fixed(self, path):
        if self._closed:
            _invalid(path,"accounting.closed","Closed accounting cannot publish")
        path = unlinked_path(path)
        if path not in self._fixed_paths:
            _invalid(path,"accounting.fixed","Undeclared fixed slot")
        if path not in self._owned_fixed:
            with self.transaction("accounting.fixed_owner"):
                if self._mapping is not None:
                    import struct
                    offset = self._OFFSET+self._struct.size+8*self._fixed_paths.index(path)
                    owner = struct.unpack_from("<Q",self._mapping,offset)[0]
                    if owner not in (0,os.getpid()):
                        _invalid(path,"accounting.fixed_owner","Fixed slot has another writer")
                    struct.pack_into("<Q",self._mapping,offset,os.getpid())
            self._owned_fixed.add(path)
        return path

    def remember_file(self, path):
        path = unlinked_path(path)
        row = fingerprint(path,phase="publication.file")
        self._observed[str(path)] = row
        return row

    def validate_append(self, path, stream):
        path = unlinked_path(path)
        if path not in self.append_only:
            _invalid(path,"append.policy","Path is not declared append-only")
        value,opened = _lstat(path,"append.path"),os.fstat(stream.fileno())
        if (not stat.S_ISREG(value.st_mode) or getattr(value,"st_nlink",1)!=1
                or _signature(value)!=_signature(opened)):
            _invalid(path,"append.identity","Append descriptor/path changed")
        previous = self._append_seen.get(str(path))
        change = _change_token(stream.fileno(),opened,path,"append")
        if previous is None:
            if value.st_size:
                _invalid(path,"append.initial","New append stream must be empty")
            self._append_hashes[str(path)] = hashlib.sha256()
            self._append_seen[str(path)] = FileRecord(0,value.st_mtime_ns,
                self._append_hashes[str(path)].hexdigest(),_identity(value),change_token=change)
        elif ((_identity(value),value.st_size,value.st_mtime_ns)!=(previous.identity,previous.size,previous.mtime_ns)
                or change is None or change!=previous.change_token):
            _invalid(path,"append.changed","Append history changed outside its writer")

    def record_append(self, path, payload, stream):
        path = unlinked_path(path)
        previous = self._append_seen.get(str(path))
        if previous is None:
            self.validate_append(path,stream)
            previous = self._append_seen[str(path)]
        value,opened = _lstat(path,"append.complete"),os.fstat(stream.fileno())
        if (_signature(value)!=_signature(opened) or _identity(value)!=previous.identity
                or value.st_size!=previous.size+len(payload)):
            _invalid(path,"append.complete","Append size/identity differs from admitted payload")
        digest = self._append_hashes[str(path)]
        digest.update(payload)
        self._append_seen[str(path)] = FileRecord(value.st_size,value.st_mtime_ns,digest.hexdigest(),
            _identity(value),change_token=_change_token(stream.fileno(),opened,path,"append.complete"))

    def audit(self, *, expected=None):
        """Full physical-byte/content check after all producers are quiescent."""
        for name,row in (expected or {}).items():
            if isinstance(row,dict):
                row = FileRecord(row.get("size",row.get("bytes")),row["mtime_ns"],row["sha256"],tuple(row["identity"]))
            self._observed[str(unlinked_path(name))] = row
        physical = self._audit_files(full_verify=True)
        totals = self.usage()
        for category in ("output_bytes","telemetry_bytes"):
            observed = category.replace("_bytes","_observed_bytes")
            if physical[observed]>totals[category]:
                _invalid(self.lock_path or category,"audit.accounting","Physical bytes exceed charged quota")
            totals[observed] = physical[observed]
            totals[category.replace("_bytes","_reserved_bytes")] = totals[category]-physical[observed]
        totals["observed_bytes_kind"] = "quiescent full filesystem audit"
        return totals

    def close(self):
        self._closed = True
        if hasattr(self,"_tickets"):
            self._tickets.clear()
        if getattr(self,"_mapping",None) is not None:
            self._mapping.close()
            self._mapping = None
        if getattr(self,"_counter_fd",None) is not None:
            os.close(self._counter_fd)
            self._counter_fd = None

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass
