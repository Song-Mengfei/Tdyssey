"""Fake Win32/subprocess resource checks; never query a GPU, PID or real memory."""

import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tdyssey import policy_memory as memory


GIB = 1024**3


def fake_libraries():
    def physical(pointer):
        status = ctypes.cast(pointer, ctypes.POINTER(memory._MemoryStatus)).contents
        status.total_physical, status.available_physical, status.load = 16 * GIB, 5 * GIB, 68
        return True

    def system(pointer, size):
        info = ctypes.cast(pointer, ctypes.POINTER(memory._PerformanceInfo)).contents
        info.CommitTotal, info.CommitLimit, info.CommitPeak, info.PageSize = 10, 20, 30, 4096
        return True

    def process(handle, pointer, size):
        counters = ctypes.cast(pointer, ctypes.POINTER(memory._ProcessCounters)).contents
        counters.WorkingSetSize, counters.PeakWorkingSetSize = 1000, 2000
        counters.PrivateUsage, counters.PagefileUsage, counters.PeakPagefileUsage = 3000, 4000, 5000
        counters.PageFaultCount = 6
        return True

    kernel = SimpleNamespace(GlobalMemoryStatusEx=Mock(side_effect=physical),
                             OpenProcess=Mock(return_value=123), CloseHandle=Mock(return_value=True))
    psapi = SimpleNamespace(GetPerformanceInfo=Mock(side_effect=system), GetProcessMemoryInfo=Mock(side_effect=process))
    return kernel, psapi


def fake_owned_kernel(*, worker_parent=202, assign=True, exit_code=0, async_job=False):
    """Synthetic process tree, retained handle and asynchronous job termination."""
    state = {"exited": False, "assigned": False, "job_closing": False, "entry": 0, "closed": []}
    entries = [(101, 1), (202, 101), (303, worker_parent), (404, 1)]

    def entry(pointer):
        result = ctypes.cast(pointer, ctypes.POINTER(memory._ProcessEntry)).contents
        result.th32ProcessID, result.th32ParentProcessID = entries[state["entry"]]
        return True

    def first(handle, pointer):
        state["entry"] = 0
        return entry(pointer)

    def next_entry(handle, pointer):
        state["entry"] += 1
        return entry(pointer) if state["entry"] < len(entries) else False

    def wait(handle, timeout):
        if timeout and state["job_closing"]:
            state["exited"] = True
        return 0 if state["exited"] else 258

    def terminate(handle, code):
        state["exited"] = True
        return True

    def get_code(handle, pointer):
        ctypes.cast(pointer, ctypes.POINTER(memory.wintypes.DWORD)).contents.value = exit_code if state["exited"] else 259
        return True

    def set_limits(job, kind, pointer, size):
        limits = ctypes.cast(pointer, ctypes.POINTER(memory._JobExtendedLimits)).contents
        state["job_flag"] = limits.BasicLimitInformation.LimitFlags
        state["job_kind"] = kind
        return True

    def assign_job(job, handle):
        state["assigned"] = assign
        return assign

    def close(handle):
        state["closed"].append(handle)
        if handle == 900 and state["assigned"]:
            state["job_closing"] = True
            if not async_job:
                state["exited"] = True
        return True

    kernel = SimpleNamespace(OpenProcess=Mock(return_value=800), GetProcessId=Mock(return_value=303),
        CloseHandle=Mock(side_effect=close), WaitForSingleObject=Mock(side_effect=wait),
        GetExitCodeProcess=Mock(side_effect=get_code), TerminateProcess=Mock(side_effect=terminate),
        CreateJobObjectW=Mock(return_value=900), SetInformationJobObject=Mock(side_effect=set_limits),
        AssignProcessToJobObject=Mock(side_effect=assign_job), CreateToolhelp32Snapshot=Mock(return_value=700),
        Process32FirstW=Mock(side_effect=first), Process32NextW=Mock(side_effect=next_entry))
    return kernel, state


class PolicyMemoryChecks(unittest.TestCase):
    def test_import_is_portable_stdlib_without_loading_windows_or_torch(self):
        code = ("import ctypes,sys\n"
                "def forbidden(*a,**k): raise RuntimeError('DLL loaded during import')\n"
                "ctypes.WinDLL=forbidden\n"
                "import tdyssey.policy_memory\n"
                "assert 'torch' not in sys.modules\n"
                "print('portable-import')\n")
        root = Path(__file__).resolve().parents[1]
        child = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=root,
                               capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.strip(), "portable-import")

    def test_system_commit_pages_are_bytes_and_historical_peak_is_not_current_limit(self):
        kernel, psapi = fake_libraries()
        with patch.object(memory, "_libraries", return_value=(kernel, psapi)):
            result = memory.system_memory()
        self.assertTrue(result["ok"])
        self.assertEqual(result["physical"]["available_bytes"], 5 * GIB)
        self.assertEqual(result["commit"]["total_bytes"], 10 * 4096)
        self.assertEqual(result["commit"]["limit_bytes"], 20 * 4096)
        self.assertEqual(result["commit"]["headroom_bytes"], 10 * 4096)
        self.assertEqual(result["commit"]["peak_bytes"], 30 * 4096)
        self.assertTrue(result["at_utc"].endswith("+00:00"))
        json.dumps(result, allow_nan=False)

    def test_system_failures_are_explicit_and_never_invent_zero_resources(self):
        kernel, psapi = fake_libraries()
        psapi.GetPerformanceInfo.return_value = False
        psapi.GetPerformanceInfo.side_effect = None
        with patch.object(memory, "_libraries", return_value=(kernel, psapi)):
            partial = memory.system_memory()
        self.assertFalse(partial["ok"])
        self.assertEqual(partial["physical"]["available_bytes"], 5 * GIB)
        self.assertNotIn("commit", partial)
        self.assertIn("GetPerformanceInfo", partial["error"]["message"])
        with patch.object(memory, "_libraries", side_effect=OSError("no Windows API")):
            unavailable = memory.system_memory()
        self.assertFalse(unavailable["ok"])
        self.assertNotIn("physical", unavailable)
        self.assertNotIn("commit", unavailable)

    def test_process_fields_map_to_exact_psapi_counters_and_close_the_handle(self):
        kernel, psapi = fake_libraries()
        with patch.object(memory, "_libraries", return_value=(kernel, psapi)):
            result = memory.process_memory(456)
        self.assertTrue(result["ok"])
        self.assertEqual(result["pid"], 456)
        self.assertEqual(result["working_set_bytes"], 1000)
        self.assertEqual(result["peak_working_set_bytes"], 2000)
        self.assertEqual(result["private_bytes"], 3000)
        self.assertEqual(result["pagefile_bytes"], 4000)
        self.assertEqual(result["peak_pagefile_bytes"], 5000)
        kernel.CloseHandle.assert_called_once_with(123)
        psapi.GetProcessMemoryInfo.assert_called_once()
        for pid in (0, -1, True):
            with self.subTest(pid=pid), self.assertRaises(ValueError):
                memory.process_memory(pid)

    def test_failed_process_queries_close_only_acquired_handles_and_report_failure(self):
        kernel, psapi = fake_libraries()
        psapi.GetProcessMemoryInfo.side_effect, psapi.GetProcessMemoryInfo.return_value = None, False
        with patch.object(memory, "_libraries", return_value=(kernel, psapi)):
            result = memory.process_memory(456)
        self.assertFalse(result["ok"])
        self.assertNotIn("private_bytes", result)
        kernel.CloseHandle.assert_called_once_with(123)
        kernel.OpenProcess.return_value = 0
        kernel.CloseHandle.reset_mock()
        with patch.object(memory, "_libraries", return_value=(kernel, psapi)):
            denied = memory.process_memory(456)
        self.assertFalse(denied["ok"])
        kernel.CloseHandle.assert_not_called()
        kernel.OpenProcess.return_value = 123
        kernel.CloseHandle.return_value = False
        with patch.object(memory, "_libraries", return_value=(kernel, psapi)):
            close_failed = memory.process_memory(456)
        self.assertFalse(close_failed["ok"])
        self.assertIn("close_error", close_failed)

    def test_gpu_query_is_optional_timed_and_parses_mib_without_assuming_sums(self):
        query = SimpleNamespace(returncode=0, stdout='0, "NVIDIA, Fake", 8192, 6909, 1116\n', stderr="")
        with patch.object(memory.subprocess, "run", return_value=query) as run:
            result = memory.gpu_memory(timeout=.25)
        self.assertTrue(result["ok"])
        self.assertEqual(result["devices"][0]["free_bytes"], 6909 * 1024**2)
        self.assertEqual(result["devices"][0]["name"], "NVIDIA, Fake")
        self.assertEqual(run.call_args.kwargs["timeout"], .25)
        with patch.object(memory.subprocess, "run", side_effect=subprocess.TimeoutExpired("nvidia-smi", .25)):
            failure = memory.gpu_memory(timeout=.25)
        self.assertFalse(failure["ok"])
        self.assertEqual(failure["error"]["type"], "TimeoutExpired")
        self.assertNotIn("devices", failure)

    def test_guard_rejects_missing_failed_or_low_measurements_and_accepts_equal_floors(self):
        snapshot = {"ok": True, "physical": {"available_bytes": GIB}, "commit": {"headroom_bytes": 2 * GIB}}
        self.assertIsNone(memory.guard_reason(snapshot, min_available_bytes=GIB, min_commit_headroom_bytes=2 * GIB))
        self.assertIsNone(memory.guard_reason(snapshot, minimum_available_bytes=GIB, minimum_commit_bytes=2 * GIB))
        for bad in (None, {"ok": False, "error": {"message": "denied"}},
                    {"ok": True, "physical": None, "commit": {}},
                    {"ok": True, "physical": {"available_bytes": GIB - 1}, "commit": {"headroom_bytes": 2 * GIB}},
                    {"ok": True, "physical": {"available_bytes": GIB}, "commit": {"headroom_bytes": 2 * GIB - 1}}):
            with self.subTest(snapshot=bad):
                self.assertIsInstance(memory.guard_reason(bad, GIB, 2 * GIB), str)
        with self.assertRaises(ValueError):
            memory.guard_reason(snapshot, GIB, 2 * GIB, min_available_bytes=0)

    def test_disk_snapshot_does_not_create_paths_and_reports_failure_without_zero(self):
        usage = SimpleNamespace(total=100, free=70, used=30)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-created" / "future.json"
            with patch.object(memory.shutil, "disk_usage", return_value=usage):
                result = memory.disk_snapshot(target)
            self.assertTrue(result["ok"])
            self.assertEqual(result["free_bytes"], 70)
            self.assertEqual(result["queried_path"], str(Path(directory).resolve()))
            self.assertFalse(target.parent.exists())
            with patch.object(memory.shutil, "disk_usage", side_effect=OSError("fake failure")):
                failure = memory.disk_snapshot(target)
            self.assertFalse(failure["ok"])
            self.assertNotIn("free_bytes", failure)

    def test_owned_worker_attaches_noninherited_kill_job_and_uses_retained_handle(self):
        kernel, state = fake_owned_kernel()
        with patch.object(memory, "_libraries", return_value=(kernel, None)), \
                patch.object(memory.os, "getpid", return_value=101), \
                patch.object(memory, "_last_error_value", return_value=18):
            owner = memory.OwnedProcess(303, 202)
            self.assertFalse(owner.poll_exit())
            self.assertIsNone(owner.exit_code())
            self.assertEqual(state["job_flag"], 0x2000)
            self.assertEqual(state["job_kind"], 9)
            kernel.CreateJobObjectW.assert_called_once_with(None, None)
            kernel.AssignProcessToJobObject.assert_called_once_with(900, 800)
            kernel.OpenProcess.assert_called_once_with(0x00101101, False, 303)
            owner.terminate_and_wait(timeout=.2)
            kernel.TerminateProcess.assert_called_once_with(800, 1)
            self.assertTrue(owner.poll_exit())
            self.assertEqual(owner.exit_code(), 0)
            owner.close()
            self.assertEqual(owner.exit_code(), 0)
            self.assertEqual(state["closed"], [700, 900, 800])
            owner.close()
            with self.assertRaises(ValueError):
                owner.poll_exit()

    def test_owned_worker_rejects_unrelated_process_without_terminating_it(self):
        kernel, state = fake_owned_kernel(worker_parent=404)
        with patch.object(memory, "_libraries", return_value=(kernel, None)), \
                patch.object(memory.os, "getpid", return_value=101), \
                patch.object(memory, "_last_error_value", return_value=18):
            with self.assertRaisesRegex(OSError, "outside this parent's process tree"):
                memory.OwnedProcess(303, 202)
        kernel.TerminateProcess.assert_not_called()
        kernel.CreateJobObjectW.assert_not_called()
        self.assertEqual(state["closed"], [700, 800])

    def test_owned_worker_job_assignment_failure_is_fail_closed(self):
        kernel, state = fake_owned_kernel(assign=False)
        with patch.object(memory, "_libraries", return_value=(kernel, None)), \
                patch.object(memory.os, "getpid", return_value=101), \
                patch.object(memory, "_last_error_value", return_value=18):
            with self.assertRaisesRegex(OSError, "AssignProcessToJobObject"):
                memory.OwnedProcess(303, 202)
        kernel.TerminateProcess.assert_called_once_with(800, 1)
        self.assertTrue(state["exited"])
        self.assertEqual(state["closed"], [700, 900, 800])

    def test_owned_worker_close_waits_for_async_job_and_preserves_signaled_exit_259(self):
        kernel, state = fake_owned_kernel(async_job=True, exit_code=259)
        with patch.object(memory, "_libraries", return_value=(kernel, None)), \
                patch.object(memory.os, "getpid", return_value=101), \
                patch.object(memory, "_last_error_value", return_value=18):
            owner = memory.OwnedProcess(303, 202)
            self.assertIsNone(owner.exit_code())
            owner.close()
            self.assertEqual(owner.exit_code(), 259)
        kernel.TerminateProcess.assert_not_called()
        self.assertIn(unittest.mock.call(800, 10000), kernel.WaitForSingleObject.call_args_list)
        self.assertEqual(state["closed"], [700, 900, 800])

    def test_owned_worker_concurrent_termination_waits_and_timeout_remains_failure(self):
        kernel, state = fake_owned_kernel()
        with patch.object(memory, "_libraries", return_value=(kernel, None)), \
                patch.object(memory.os, "getpid", return_value=101), \
                patch.object(memory, "_last_error_value", return_value=18):
            owner = memory.OwnedProcess(303, 202)
            kernel.TerminateProcess.side_effect = lambda handle, code: state.update(job_closing=True) or False
            owner.terminate_and_wait(timeout=.2)
            self.assertTrue(owner.poll_exit())
            owner.close()
        kernel, state = fake_owned_kernel()
        with patch.object(memory, "_libraries", return_value=(kernel, None)), \
                patch.object(memory.os, "getpid", return_value=101), \
                patch.object(memory, "_last_error_value", return_value=18):
            owner = memory.OwnedProcess(303, 202)
            kernel.TerminateProcess.side_effect = None
            kernel.TerminateProcess.return_value = True
            with self.assertRaises(TimeoutError):
                owner.terminate_and_wait(timeout=.01)
            owner.close()


if __name__ == "__main__":
    unittest.main()
