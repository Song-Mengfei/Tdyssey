"""Eval-only append status, bounded incremental reads, no rename/delete/retry.

Writer success means write-all, flush and fsync succeeded. Reader visibility of
a complete checked frame does not prove concurrent fsync has returned; the
existing heartbeat first-error slot must be checked before accepting new status.
"""
import ctypes
from copy import deepcopy
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import threading
import time

from .wm_compat_guard import _write_all

STATUS_NAME = 'evaluation-status.journal'
MAX_PAYLOAD_BYTES = 1536
MAX_RECORD_BYTES = 65536
MAX_RECORDS = 10000
MAX_BYTES = 16*1024**2
MAX_INCREMENT_BYTES = 65536
_HEADER = struct.Struct('>8sIQ')
_MAGIC, _COMMIT = b'GPCEVS01', b'DONE'
FRAME_OVERHEAD = _HEADER.size+32+len(_COMMIT)
MAX_ENCODED_RECORD_BYTES = FRAME_OVERHEAD+MAX_PAYLOAD_BYTES
_FORBIDDEN = ('environment_creations', 'resets', 'restores', 'controls',
    'physics_substeps', 'real_renders', 'policy_sampling')
_KERNEL = None


class StatusJournalError(RuntimeError):
    pass


class _FileInfo(ctypes.Structure):
    _fields_ = [('attributes', wintypes.DWORD), ('created', wintypes.FILETIME),
        ('accessed', wintypes.FILETIME), ('modified', wintypes.FILETIME),
        ('volume', wintypes.DWORD), ('size_high', wintypes.DWORD),
        ('size_low', wintypes.DWORD), ('links', wintypes.DWORD),
        ('index_high', wintypes.DWORD), ('index_low', wintypes.DWORD)]


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FileInfo)]
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.GetFileType.argtypes, kernel.GetFileType.restype = [wintypes.HANDLE], wintypes.DWORD
    kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL
    _KERNEL = kernel
    return kernel


def _parents(path):
    for parent in reversed(path.parents):
        value = os.lstat(parent)
        if (not stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode)
                or getattr(value, 'st_file_attributes', 0) & 0x400):
            raise StatusJournalError('Status ancestors must be ordinary non-reparse directories')


def _path(value):
    if '..' in Path(value).parts:
        raise StatusJournalError('Status parent traversal is forbidden')
    path = Path(os.path.abspath(value))
    if path.name != STATUS_NAME:
        raise StatusJournalError('Only the fixed evaluation status filename is allowed')
    _parents(path)
    return path


def _open_file(path, *, create=False):
    """Persistent Win32 READ|WRITE sharing; append requires no DELETE sharing."""
    if sys.platform != 'win32':
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND if create else os.O_RDONLY
        fd = os.open(path, flags | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
        return os.fdopen(fd, 'ab' if create else 'rb', buffering=0)
    import msvcrt
    kernel = _kernel()
    handle = kernel.CreateFileW(str(path), 0x40000080 if create else 0x80000000,
        3, None, 1 if create else 3, 0x02200000, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    fd = None
    try:
        flags = (os.O_WRONLY | os.O_APPEND) if create else os.O_RDONLY
        fd = msvcrt.open_osfhandle(int(handle), flags | os.O_BINARY | os.O_NOINHERIT)
        handle = None
        stream = os.fdopen(fd, 'ab' if create else 'rb', buffering=0)
        fd = None
        return stream
    finally:
        primary = sys.exc_info()[1]
        try:
            if fd is not None:
                os.close(fd)
            elif handle is not None and not kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException as secondary:
            if primary is None:
                raise
            primary.add_note('Secondary status-open close failure: '+repr(secondary))


def _snapshot(stream):
    if sys.platform == 'win32':
        import msvcrt
        kernel = _kernel();handle = msvcrt.get_osfhandle(stream.fileno());value = _FileInfo()
        if not kernel.GetFileInformationByHandle(handle, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        if kernel.GetFileType(handle) != 1 or value.attributes & (0x400 | 0x10) or value.links != 1:
            raise StatusJournalError('Status must be an ordinary single-link non-reparse file')
        identity = int(value.volume), int(value.index_high)<<32 | int(value.index_low)
        return identity, int(value.size_high)<<32 | int(value.size_low)
    value = os.fstat(stream.fileno())
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise StatusJournalError('Status must be an ordinary single-link file')
    return (value.st_dev, value.st_ino), value.st_size


def _path_snapshot(path):
    _parents(path)
    with _open_file(path) as stream:
        return _snapshot(stream)


def _identity(pid, nonce):
    if type(pid) is not int or pid <= 0 or not isinstance(nonce, str) or not 0 < len(nonce) <= 128:
        raise ValueError('A positive worker PID and bounded nonce are required')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _status(value, pid, nonce, *, raw_fulltrain=False):
    if not isinstance(value, dict) or type(value.get('actual_worker_pid')) is not int or value['actual_worker_pid'] != pid or value.get('nonce') != nonce:
        raise StatusJournalError('Status worker PID/nonce differs')
    if not all(_finite(value.get(key)) for key in ('at_unix', 'at_monotonic')):
        raise StatusJournalError('Status time is missing or nonfinite')
    phase = value.get('phase')
    phases = ('setup','model_loading','training','evaluation','export','complete','error') if raw_fulltrain else ('setup','model_loading','evaluation','export','complete','error')
    if phase not in phases:
        raise StatusJournalError('Status phase is not eval-only')
    caps = (dict(model_instances=1,model_loads=4,optimizer_updates=2328,forecasts=48,
        euler_steps=144,eval_batches=48,eval_examples=48,**dict.fromkeys(_FORBIDDEN,0)) if raw_fulltrain else
        dict(model_instances=1,model_loads=2,optimizer_updates=0,forecasts=3996,
        euler_steps=11988,eval_batches=32,eval_examples=32,**dict.fromkeys(_FORBIDDEN,0)))
    for key, maximum in caps.items():
        if type(value.get(key)) is not int or not 0 <= value[key] <= maximum:
            raise StatusJournalError('Eval-only status counter missing/invalid/over cap: '+key)
    if type(value.get('training_complete')) is not bool:
        raise StatusJournalError('Status training_complete must be explicit bool')
    if raw_fulltrain:
        if (value.get('raw_fulltrain') is not True or value.get('evaluation_only') is not False
                or value.get('training_applicable') is not True or value.get('source_checkpoint_update') is not None):
            raise StatusJournalError('Raw full-training contract marker differs')
        if value['training_complete'] and value['optimizer_updates']!=2328:
            raise StatusJournalError('Raw training completion requires2328 updates')
        if phase=='training' and value['model_instances']!=1:
            raise StatusJournalError('Raw training requires exactly one live model')
    if phase == 'error':
        return
    names = ('cuda_allocated_bytes', 'cuda_reserved_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes')
    stamps = ('cuda_sample_at_unix', 'cuda_sample_at_monotonic')
    if value.get('cuda_sample_state') == 'unavailable':
        if phase not in ('setup', 'model_loading') or any(value.get(key) is not None for key in (*names, *stamps)):
            raise StatusJournalError('Unavailable CUDA fields are inconsistent')
    elif value.get('cuda_sample_state') == 'fresh':
        for key, maximum in zip(names, (512*1024**2, 1024**3, 512*1024**2, 1024**3)):
            if type(value.get(key)) is not int or not 0 <= value[key] <= maximum:
                raise StatusJournalError('CUDA status field missing/invalid/over cap: '+key)
        if not all(_finite(value.get(key)) for key in stamps):
            raise StatusJournalError('CUDA sample time is missing/nonfinite')
        if raw_fulltrain and (not -.5<=value['at_unix']-value['cuda_sample_at_unix']<=5
                or not 0<=value['at_monotonic']-value['cuda_sample_at_monotonic']<=5):
            raise StatusJournalError('Raw CUDA sample was not fresh at publication')
    else:
        raise StatusJournalError('CUDA sample state is missing/invalid')


def validate(value, *, raw_fulltrain=False):
    """Validate one complete record at publication time, never against now.

    Entry callbacks may reuse this without importing the guardian or recursing
    through Reader. Live heartbeat freshness remains the parent's separate5s
    check. No file, native or tensor operation is performed here.
    """
    if not isinstance(value,dict):raise StatusJournalError('Status must be a JSON object')
    _identity(value.get('actual_worker_pid'),value.get('nonce'))
    _status(value,value['actual_worker_pid'],value['nonce'],raw_fulltrain=raw_fulltrain)


def _frame(sequence, payload):
    header = _HEADER.pack(_MAGIC, len(payload), sequence)
    return header+payload+hashlib.sha256(header+payload).digest()+_COMMIT


class Writer:
    """Exclusive one-PID writer; failed reservations/partial bytes never refund."""
    def __init__(self, path, *, pid, nonce, max_records=MAX_RECORDS, max_bytes=MAX_BYTES,
                 clock=time.time, monotonic=time.monotonic, raw_fulltrain=False):
        _identity(pid, nonce)
        if type(raw_fulltrain) is not bool:raise ValueError('raw_fulltrain must be explicit bool')
        self.raw_fulltrain = raw_fulltrain
        if pid != os.getpid():
            raise ValueError('Status writer must own the actual worker PID')
        if (type(max_records) is not int or not 0 < max_records <= MAX_RECORDS
                or type(max_bytes) is not int or not 0 < max_bytes <= MAX_BYTES):
            raise ValueError('Status limits may only tighten the fixed caps')
        self.path, self.pid, self.nonce = _path(path), pid, nonce
        self.max_records, self.max_bytes = max_records, max_bytes
        self.clock, self.monotonic = clock, monotonic
        self.reserved_records = self.committed_records = self.reserved_bytes = self.committed_bytes = 0
        self.first_error = None;self._last_monotonic = None;self._lock = threading.RLock()
        self._stream = _open_file(self.path, create=True)
        try:
            self.file_id, size = _snapshot(self._stream)
            if size != 0:
                raise StatusJournalError('New exclusive status file is not empty')
        except BaseException as error:
            try:self._stream.close()
            except BaseException as secondary:error.add_note('Secondary initial status close: '+repr(secondary))
            self._stream = None
            raise

    @property
    def sequence(self):return self.committed_records

    def _fail(self, error, phase):
        if self.first_error is None:self.first_error = error
        if not getattr(error, 'heartbeat_step', None):error.heartbeat_step = phase
        error.journal_path = str(self.path)

    def publish(self, payload):
        with self._lock:
            if self.first_error is not None:
                raise StatusJournalError('Status writer permanently failed; no retry') from self.first_error
            phase = 'status.append.identity'
            try:
                if self._stream is None or os.getpid() != self.pid:
                    raise StatusJournalError('Status writer closed or belongs to another PID')
                identity, size = _snapshot(self._stream);path_identity, path_size = _path_snapshot(self.path)
                if identity != self.file_id or path_identity != identity or size != self.committed_bytes or path_size != size:
                    raise StatusJournalError('Writer path/file identity or append length changed')
                phase = 'status.append.encode';value = dict(payload)
                if ('actual_worker_pid' in value and value['actual_worker_pid'] != self.pid
                        or 'nonce' in value and value['nonce'] != self.nonce):
                    raise StatusJournalError('Publisher attempted to change PID/nonce')
                value.update(actual_worker_pid=self.pid, nonce=self.nonce, at_unix=self.clock(),
                    at_monotonic=self.monotonic(), at_utc=datetime.now(timezone.utc).isoformat())
                _status(value, self.pid, self.nonce, raw_fulltrain=self.raw_fulltrain)
                if self._last_monotonic is not None and value['at_monotonic'] < self._last_monotonic:
                    raise StatusJournalError('Status monotonic time moved backward')
                raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
                if len(raw) > MAX_PAYLOAD_BYTES:
                    raise StatusJournalError('Compact status exceeds its 1536-byte payload bound')
                framed = _frame(self.reserved_records+1, raw)
                if (len(framed) > MAX_RECORD_BYTES or self.reserved_records >= self.max_records
                        or self.reserved_bytes+len(framed) > self.max_bytes):
                    raise StatusJournalError('Status record/16MiB lifetime allowance exhausted')
                self.reserved_records += 1;self.reserved_bytes += len(framed)
                phase = 'status.append.write';_write_all(self._stream, framed)
                phase = 'status.append.flush';self._stream.flush()
                phase = 'status.append.fsync';os.fsync(self._stream.fileno())
                phase = 'status.append.commit'
                identity, size = _snapshot(self._stream);path_identity, path_size = _path_snapshot(self.path)
                if identity != self.file_id or path_identity != identity or size != self.reserved_bytes or path_size != size:
                    raise StatusJournalError('Status identity/length changed during append commit')
                self.committed_records = self.reserved_records;self.committed_bytes = self.reserved_bytes
                self._last_monotonic = value['at_monotonic']
                return value
            except BaseException as error:
                self._fail(error, phase)
                raise

    def close(self):
        with self._lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:stream.close()
                except BaseException as error:
                    if self.first_error is not None:self.first_error.add_note('Secondary status close failure: '+repr(error))
                    else:self._fail(error, 'status.append.close')
                    raise

    def __enter__(self):return self
    def __exit__(self, kind, value, tb):
        try:self.close()
        except BaseException as secondary:
            if value is None:raise
            value.add_note('Secondary status writer close failure: '+repr(secondary))


def _reject_constant(value):
    raise StatusJournalError('Nonfinite JSON constant in status: '+value)


class Reader:
    """One persistent handle, bounded new-byte reads, every frame validated.

    poll returns only a newly observed complete status (otherwise None). It
    never changes timestamps. validate(status) may raise or return a failure
    reason/False; None or True means success. This callback runs for EVERY new
    complete frame, including intermediate frames in one poll. It must not
    apply historical-record age limits; the parent checks latest age normally.
    """
    def __init__(self, path, *, pid, nonce, validate=None, max_read_bytes=MAX_INCREMENT_BYTES, raw_fulltrain=False):
        _identity(pid, nonce)
        if type(raw_fulltrain) is not bool:raise ValueError('raw_fulltrain must be explicit bool')
        self.raw_fulltrain = raw_fulltrain
        if type(max_read_bytes) is not int or not 0 < max_read_bytes <= MAX_INCREMENT_BYTES:
            raise ValueError('Incremental reads may only tighten the 64KiB bound')
        self.path, self.pid, self.nonce = _path(path), pid, nonce
        self.validate, self.max_read_bytes = validate, max_read_bytes
        self.committed_records = self.committed_bytes = self.read_offset = self._seen_size = 0
        self.first_error = None;self._latest = None;self._last_monotonic = None
        self._buffer = b'';self._lock = threading.RLock();self._stream = _open_file(self.path)
        try:
            self.file_id, _ = _snapshot(self._stream)
            self._verify()
        except BaseException as error:
            try:self._stream.close()
            except BaseException as secondary:error.add_note('Secondary initial reader close: '+repr(secondary))
            self._stream = None
            raise

    @property
    def sequence(self):return self.committed_records
    @property
    def partial_bytes(self):return len(self._buffer)
    @property
    def bytes_seen(self):return self._seen_size
    @property
    def latest(self):return deepcopy(self._latest)

    def _verify(self):
        if self._stream is None:
            raise StatusJournalError('Status reader is closed')
        identity, size = _snapshot(self._stream);path_identity, path_size = _path_snapshot(self.path)
        if identity != self.file_id or path_identity != identity:
            raise StatusJournalError('Status handle/path fileID changed')
        if size < self._seen_size or path_size < self._seen_size:
            raise StatusJournalError('Append-only status file was truncated')
        size = max(size, path_size)
        if size > MAX_BYTES:
            raise StatusJournalError('Status file exceeds its 16MiB limit')
        self._seen_size = size
        return size

    def poll(self):
        with self._lock:
            if self.first_error is not None:
                raise StatusJournalError('Status reader permanently failed; no retry') from self.first_error
            phase = 'status.read.identity'
            try:
                size = self._verify();amount = min(self.max_read_bytes, size-self.read_offset)
                if amount:
                    phase = 'status.read.increment'
                    raw = self._stream.read(amount)
                    if not isinstance(raw, bytes) or not raw or len(raw) > amount:
                        raise OSError('Status incremental read made no/invalid progress')
                    self.read_offset += len(raw);self._buffer += raw
                consumed = count = 0;latest = None;last_time = self._last_monotonic
                while len(self._buffer)-consumed >= _HEADER.size:
                    phase = 'status.read.frame'
                    header = self._buffer[consumed:consumed+_HEADER.size]
                    magic, length, sequence = _HEADER.unpack(header)
                    if magic != _MAGIC or not 0 < length <= MAX_PAYLOAD_BYTES:
                        raise StatusJournalError('Status frame magic/length is invalid')
                    total = FRAME_OVERHEAD+length
                    if total > MAX_RECORD_BYTES:
                        raise StatusJournalError('Status frame exceeds 64KiB')
                    if len(self._buffer)-consumed < total:
                        break  # A partial tail is retained; it cannot refresh status.
                    if sequence != self.committed_records+count+1 or sequence > MAX_RECORDS:
                        raise StatusJournalError('Duplicate/out-of-order/over-limit status sequence')
                    body_start = consumed+_HEADER.size;body_end = body_start+length
                    payload = self._buffer[body_start:body_end]
                    checksum = self._buffer[body_end:body_end+32]
                    if (self._buffer[body_end+32:consumed+total] != _COMMIT
                            or hashlib.sha256(header+payload).digest() != checksum):
                        raise StatusJournalError('Complete status checksum/commit marker is invalid')
                    phase = 'status.read.schema'
                    value = json.loads(payload.decode('utf-8'), parse_constant=_reject_constant)
                    _status(value, self.pid, self.nonce, raw_fulltrain=self.raw_fulltrain)
                    if last_time is not None and value['at_monotonic'] < last_time:
                        raise StatusJournalError('Status monotonic time moved backward')
                    if self.validate is not None:
                        result = self.validate(deepcopy(value))
                        if result is not None and result is not True:
                            raise StatusJournalError('Additional status schema rejected frame: '+str(result))
                    last_time = value['at_monotonic'];latest = value
                    consumed += total;count += 1
                phase = 'status.read.identity_after';self._verify()
                if count:
                    self.committed_records += count;self.committed_bytes += consumed
                    self._buffer = self._buffer[consumed:];self._latest = latest;self._last_monotonic = last_time
                    return deepcopy(latest)
                return None
            except BaseException as error:
                self.first_error = error;error.journal_phase = phase;error.journal_path = str(self.path)
                raise

    def close(self):
        with self._lock:
            stream, self._stream = self._stream, None
            if stream is not None:
                try:stream.close()
                except BaseException as error:
                    if self.first_error is None:self.first_error = error
                    else:self.first_error.add_note('Secondary status reader close failure: '+repr(error))
                    raise

    def __enter__(self):return self
    def __exit__(self, kind, value, tb):
        try:self.close()
        except BaseException as secondary:
            if value is None:raise
            value.add_note('Secondary status reader close failure: '+repr(secondary))
