"""Stdlib-only Windows memory observations; importing never loads a Win32 DLL.

Failures are explicit records, never zero-valued resource observations. GPU
queries are separate and optional so a frequent host monitor need not run them.
"""

import ctypes
from ctypes import wintypes
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in ("total_physical", "available_physical", "total_pagefile",
        "available_pagefile", "total_virtual", "available_virtual", "extended")]


class _PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
        "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
        (name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


class _ProcessCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
        "PeakPagefileUsage", "PrivateUsage")]


class _ProcessEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]


class _JobBasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount",
                "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _JobExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _JobBasicLimits), ("IoInfo", _IOCounters)] + [
        (name, ctypes.c_size_t) for name in ("ProcessMemoryLimit", "JobMemoryLimit",
                                           "PeakProcessMemoryUsed", "PeakJobMemoryUsed")]


def _libraries():
    if sys.platform != "win32":
        raise OSError("Windows resource APIs are unavailable on this platform")
    return ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)


def _api_error(name):
    code = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else None
    return OSError(f"{name} failed (Win32 error {code})")


def _last_error_value():
    return ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else None


def _process_parents(kernel):
    """One Toolhelp snapshot, including error checking and snapshot-handle closure."""
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
        function.restype = wintypes.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
    if not snapshot or snapshot == ctypes.c_void_p(-1).value:
        raise _api_error("CreateToolhelp32Snapshot")
    try:
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        if not kernel.Process32FirstW(snapshot, ctypes.byref(entry)):
            raise _api_error("Process32FirstW")
        parents = {}
        while True:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            if not kernel.Process32NextW(snapshot, ctypes.byref(entry)):
                if _last_error_value() != 18:  # ERROR_NO_MORE_FILES
                    raise _api_error("Process32NextW")
                return parents
    finally:
        if not kernel.CloseHandle(snapshot):
            raise _api_error("CloseHandle(process snapshot)")


def _descends_from(pid, ancestor, parents):
    seen = set()
    while pid not in seen and pid > 0:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = parents.get(pid, 0)
    return False


class OwnedProcess:
    """Own an acknowledged worker through its exact retained Windows handle.

    Construct while the lightweight worker is blocked waiting for the parent's
    acknowledgement. The worker must belong to this process's launcher tree.
    A noninherited Job Object kills the worker on close or parent death. Failure
    to verify identity or attach the job never permits the heavy-work ACK.
    """

    def __init__(self, pid, launcher_pid):
        for label, value in (("pid", pid), ("launcher_pid", launcher_pid)):
            if type(value) is not int or not 0 < value <= 0xFFFFFFFF:
                raise ValueError(f"{label} must be a positive Windows DWORD PID")
        self.pid, self.launcher_pid = pid, launcher_pid
        self._kernel, _ = _libraries()
        self._handle = self._job = None
        self._verified = False
        self._closed_exit_code = None
        kernel = self._kernel
        signatures = {
            "OpenProcess": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            "GetProcessId": ([wintypes.HANDLE], wintypes.DWORD),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "WaitForSingleObject": ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            "GetExitCodeProcess": ([wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            "TerminateProcess": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(kernel, name)
            function.argtypes, function.restype = args, result
        try:
            # QUERY_LIMITED_INFORMATION | TERMINATE | SYNCHRONIZE | SET_QUOTA.
            self._handle = kernel.OpenProcess(0x00101101, False, pid)
            if not self._handle:
                raise _api_error("OpenProcess(worker)")
            if kernel.GetProcessId(self._handle) != pid:
                raise OSError("Worker handle identity does not match supplied PID")
            if self.poll_exit():
                raise OSError("Worker exited before ownership was established")
            parents = _process_parents(kernel)
            current = os.getpid()
            if (launcher_pid == current or pid == current
                    or not _descends_from(launcher_pid, current, parents)
                    or not _descends_from(pid, launcher_pid, parents)):
                raise OSError("Worker/launcher is outside this parent's process tree")
            self._verified = True
            if self.poll_exit():
                raise OSError("Worker exited during ownership verification")
            self._job = kernel.CreateJobObjectW(None, None)
            if not self._job:
                raise _api_error("CreateJobObjectW")
            limits = _JobExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self._job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise _api_error("SetInformationJobObject")
            if not kernel.AssignProcessToJobObject(self._job, self._handle):
                raise _api_error("AssignProcessToJobObject")
            if self.poll_exit():
                raise OSError("Worker exited before job ownership completed")
        except BaseException:
            # Only a verified member of our own tree may be terminated here.
            if self._verified and self._handle:
                try:
                    self.terminate_and_wait(timeout=10.0)
                except (OSError, TimeoutError):
                    pass
            self.close()
            raise

    def _require_open(self):
        if not self._handle:
            raise ValueError("OwnedProcess is closed")

    def poll_exit(self):
        """True only when the retained process handle is signaled."""
        self._require_open()
        result = self._kernel.WaitForSingleObject(self._handle, 0)
        if result == 0:
            return True
        if result == 258:
            return False
        raise _api_error("WaitForSingleObject")

    def exit_code(self):
        """Return the actual worker's exit code, including a signaled code 259."""
        if not self._handle:
            if self._closed_exit_code is not None:
                return self._closed_exit_code
            raise ValueError("OwnedProcess is closed without a recorded exit code")
        if not self.poll_exit():
            return None
        code = wintypes.DWORD()
        if not self._kernel.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            raise _api_error("GetExitCodeProcess")
        return int(code.value)

    def terminate_and_wait(self, timeout=10.0):
        """Terminate only the retained worker; never reopen or kill by PID."""
        self._require_open()
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0 or timeout > 4294967:
            raise ValueError("Termination timeout must be finite positive seconds below Win32 INFINITE")
        if self.poll_exit():
            return
        if not self._verified:
            raise OSError("Cannot terminate an unverified worker")
        if not self._kernel.TerminateProcess(self._handle, 1):
            failure = _api_error("TerminateProcess")
            # TerminateProcess can report access denied while another owner or
            # the job is already terminating this exact process asynchronously.
            status = self._kernel.WaitForSingleObject(self._handle, math.ceil(timeout * 1000))
            if status == 0:
                return
            if status != 258:
                raise _api_error("WaitForSingleObject(termination race)")
            raise failure
        status = self._kernel.WaitForSingleObject(self._handle, math.ceil(timeout * 1000))
        if status == 258:
            raise TimeoutError("Owned worker did not exit before termination timeout")
        if status != 0:
            raise _api_error("WaitForSingleObject(termination)")

    def close(self):
        """Close the kill-on-close job before releasing the exact worker handle."""
        error = None
        job_closed = False
        if self._job:
            job, self._job = self._job, None
            if not self._kernel.CloseHandle(job):
                error = _api_error("CloseHandle(worker job)")
            else:
                job_closed = True
        if self._handle:
            try:
                # Job closure requests termination; retain the process until it exits.
                if job_closed:
                    status = self._kernel.WaitForSingleObject(self._handle, 10000)
                    if status == 258:
                        raise TimeoutError("Owned worker did not exit after kill-on-close job closure")
                    if status != 0:
                        raise _api_error("WaitForSingleObject(job closure)")
                elif self._verified and not self.poll_exit():
                    self.terminate_and_wait(timeout=10.0)
                self._closed_exit_code = self.exit_code()
            except (OSError, TimeoutError, ValueError) as close_error:
                error = error or close_error
            handle, self._handle = self._handle, None
            if not self._kernel.CloseHandle(handle):
                error = error or _api_error("CloseHandle(worker)")
        if error:
            raise error

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _base(source):
    return {"ok": False, "at_utc": datetime.now(timezone.utc).isoformat(), "source": source}


def _error(error):
    result = {"type": type(error).__name__, "message": str(error)}
    if getattr(error, "winerror", None) is not None:
        result["winerror"] = error.winerror
    return result


def system_memory():
    """Physical RAM and actual system commit; commit page counts become bytes."""
    started = time.perf_counter()
    result = _base("Win32 GlobalMemoryStatusEx and PSAPI GetPerformanceInfo")
    try:
        kernel, psapi = _libraries()
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatus)]
        kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise _api_error("GlobalMemoryStatusEx")
        result["physical"] = {"total_bytes": int(status.total_physical),
                              "available_bytes": int(status.available_physical), "load_percent": int(status.load)}
        info = _PerformanceInfo()
        info.cb = ctypes.sizeof(info)
        psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(_PerformanceInfo), wintypes.DWORD]
        psapi.GetPerformanceInfo.restype = wintypes.BOOL
        if not psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            raise _api_error("GetPerformanceInfo")
        page = int(info.PageSize)
        total, limit, peak = (int(getattr(info, name)) * page for name in ("CommitTotal", "CommitLimit", "CommitPeak"))
        if (page <= 0 or status.total_physical <= 0 or status.available_physical > status.total_physical
                or limit <= 0 or total > limit):
            raise ValueError("Windows returned inconsistent physical/commit memory counters")
        result["commit"] = {"total_bytes": total, "limit_bytes": limit, "peak_bytes": peak,
                            "headroom_bytes": limit - total, "page_size_bytes": page,
                            "raw_pages": {name: int(getattr(info, name)) for name in ("CommitTotal", "CommitLimit", "CommitPeak")}}
        result["ok"] = True
    except (OSError, ValueError, AttributeError) as error:
        result["error"] = _error(error)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def process_memory(pid):
    """Query one PID with a short-lived handle; peaks are OS process counters."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("pid must be a positive integer")
    started = time.perf_counter()
    result = {**_base("Win32 OpenProcess and PSAPI GetProcessMemoryInfo"), "pid": pid}
    handle = None
    kernel = None
    try:
        kernel, psapi = _libraries()
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL
        handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)  # QUERY_LIMITED_INFORMATION | VM_READ
        if not handle:
            raise _api_error("OpenProcess")
        counters = _ProcessCounters()
        counters.cb = ctypes.sizeof(counters)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessCounters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise _api_error("GetProcessMemoryInfo")
        result.update(ok=True, working_set_bytes=int(counters.WorkingSetSize),
                      peak_working_set_bytes=int(counters.PeakWorkingSetSize),
                      private_bytes=int(counters.PrivateUsage), pagefile_bytes=int(counters.PagefileUsage),
                      peak_pagefile_bytes=int(counters.PeakPagefileUsage), page_fault_count=int(counters.PageFaultCount))
    except (OSError, ValueError, AttributeError) as error:
        result["error"] = _error(error)
    finally:
        if handle and kernel is not None:
            try:
                if not kernel.CloseHandle(handle):
                    raise _api_error("CloseHandle")
            except (OSError, ValueError, AttributeError) as error:
                result.update(ok=False, close_error=_error(error))
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def gpu_memory(timeout=5.0):
    """Optional installed nvidia-smi query, with a finite independent timeout."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("GPU query timeout must be finite and positive")
    started = time.perf_counter()
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]
    result = {**_base("installed NVIDIA nvidia-smi; current driver memory, not process allocator peaks"),
              "source_command": command}
    try:
        query = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        result.update(exit_code=query.returncode, raw_stdout=query.stdout, raw_stderr=query.stderr)
        if query.returncode:
            raise RuntimeError(f"nvidia-smi exited with code {query.returncode}")
        devices = []
        for fields in csv.reader(query.stdout.strip().splitlines(), skipinitialspace=True):
            index, name, total, free, used = (part.strip() for part in fields)
            row = {"index": int(index), "name": name}
            for label, value in (("total", total), ("free", free), ("used", used)):
                row[label + "_mib"] = int(value)
                row[label + "_bytes"] = int(value) * 1024**2
            if row["total_mib"] <= 0 or not 0 <= row["free_mib"] <= row["total_mib"] or not 0 <= row["used_mib"] <= row["total_mib"]:
                raise ValueError("Invalid driver GPU memory counters")
            devices.append(row)
        if not devices:
            raise ValueError("nvidia-smi returned no GPU devices")
        result.update(ok=True, devices=devices)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result["error"] = _error(error)
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def disk_snapshot(path):
    """Read the nearest existing directory's volume; never create an output."""
    result = _base("Python shutil.disk_usage / Windows GetDiskFreeSpaceExW")
    try:
        requested = Path(path).resolve()
        existing = requested
        while not existing.is_dir() and existing != existing.parent:
            existing = existing.parent
        usage = shutil.disk_usage(existing)
        result.update(ok=True, requested_path=str(requested), queried_path=str(existing),
                      total_bytes=int(usage.total), free_bytes=int(usage.free), used_bytes=int(usage.used))
    except (OSError, ValueError, TypeError) as error:
        result["error"] = _error(error)
    return result


def guard_reason(snapshot, minimum_available_bytes=None, minimum_commit_bytes=None, *,
                 min_available_bytes=None, min_commit_headroom_bytes=None):
    """Return None only for a successful, complete snapshot above both floors.

    The short keyword aliases are accepted for the load child; conflicting
    thresholds are rejected rather than silently choosing the weaker guard.
    """
    pairs = ((minimum_available_bytes, min_available_bytes), (minimum_commit_bytes, min_commit_headroom_bytes))
    floors = []
    for canonical, alias in pairs:
        if canonical is not None and alias is not None and canonical != alias:
            raise ValueError("Conflicting memory guard threshold aliases")
        value = canonical if canonical is not None else alias
        if type(value) is not int or value < 0:
            raise ValueError("Both memory guard floors must be nonnegative integer bytes")
        floors.append(value)
    if not isinstance(snapshot, dict) or snapshot.get("ok") is not True:
        return f"System memory measurement failed or unavailable: {snapshot.get('error') if isinstance(snapshot, dict) else 'invalid snapshot'}"
    for group, key, floor, label in (("physical", "available_bytes", floors[0], "physical available RAM"),
                                     ("commit", "headroom_bytes", floors[1], "system commit headroom")):
        values = snapshot.get(group)
        observed = values.get(key) if isinstance(values, dict) else None
        if type(observed) is not int or observed < 0:
            return f"Missing or invalid {label}; refusing to infer zero or sufficient capacity"
        if observed < floor:
            return f"Insufficient {label}: {observed} bytes below required {floor} bytes"
    return None
