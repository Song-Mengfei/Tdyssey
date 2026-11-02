"""Synthetic publisher regressions; Win32 calls are injected, never executed.

These check publication contracts and failure handling. Actual Windows handle
sharing/concurrency belongs to the separately authorized native diagnostic.
"""
from contextlib import ExitStack, contextmanager
import errno
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation_guard as gpc
from tdyssey import wm_compat_guard as shared


EVIDENCE = gpc.ROOT/"artifacts/gpc-status-publication-review-20261026/synthetic-fixtures"


class FakeWindowsCall:
    """Record either candidate primitive, without prescribing its flags."""
    def __init__(self, backend, name):
        self.backend, self.name = backend, name

    def __call__(self, *arguments):
        if self.name == "ReplaceFileW":
            target, source = arguments[:2]
        elif self.name == "MoveFileExW":
            source, target = arguments[:2]
        else:
            raise AssertionError("Unreviewed fake Windows API: "+self.name)
        self.backend.calls.append((self.name, str(source), str(target)))
        if self.backend.failure is not None:
            return 0
        self.backend.real_replace(source, target)
        return 1


class FakePublicationBackend:
    def __init__(self):
        self.real_rename, self.real_replace = os.rename, os.replace
        self.calls, self.failure = [], None
        self.kernel = SimpleNamespace(ReplaceFileW=FakeWindowsCall(self, "ReplaceFileW"),
                                      MoveFileExW=FakeWindowsCall(self, "MoveFileExW"))

    def rename(self, source, target, *args, **kwargs):
        self.calls.append(("rename", str(source), str(target)))
        if self.failure is not None:
            raise self.failure
        return self.real_rename(source, target, *args, **kwargs)

    def replace(self, source, target, *args, **kwargs):
        self.calls.append(("replace", str(source), str(target)))
        if self.failure is not None:
            raise self.failure
        return self.real_replace(source, target, *args, **kwargs)

    def winerror(self, code):
        if self.failure is None or getattr(self.failure, "winerror", None) != code:
            raise AssertionError("Unexpected injected Windows error code")
        return self.failure


class NoTickThread:
    """Deterministic lifecycle only; periodic heartbeat behavior has old tests."""
    def __init__(self, *, target, daemon):
        self.target, self.alive = target, False

    def start(self):
        self.alive = True

    def join(self, timeout):
        if timeout != 3 or not self.target.__self__.stop.is_set():
            raise AssertionError("Heartbeat must stop and join with its original bound")
        self.alive = False

    def is_alive(self):
        return self.alive


class FixedSlotStorage:
    """Only the prepaid slot/local-transaction contract; no real account files."""
    def __init__(self, target):
        self.atomic_targets = {target: shared.STATUS_BYTES}
        self.active, self.phases, self.claims = False, [], []

    @contextmanager
    def local_transaction(self, phase):
        if self.active:
            raise AssertionError("Unexpected nested fixture transaction")
        self.active = True; self.phases.append(("enter", phase))
        try:
            yield
        finally:
            self.active = False; self.phases.append(("exit", phase))

    def claim_fixed(self, path):
        if not self.active or path not in self.atomic_targets:
            raise AssertionError("Undeclared/unserialized fixture status claim")
        self.claims.append(path)


class StatusPublicationTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="fake-publication-", dir=EVIDENCE)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.handoff = self.root/"handoff.json"
        self.target = self.root/"worker-status.json"
        self.handoff.write_text(json.dumps(dict(status_path=str(self.target),
            actual_worker_pid=os.getpid(), nonce="synthetic-only")), encoding="utf-8")
        self.backend = FakePublicationBackend()
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        # Inject every possible Win32 publication call; no native DLL is opened.
        self.stack.enter_context(patch.object(shared.sys, "platform", "win32"))
        self.stack.enter_context(patch.object(shared.ctypes, "WinDLL", return_value=self.backend.kernel, create=True))
        self.stack.enter_context(patch.object(shared.ctypes, "get_last_error",
            side_effect=lambda: getattr(self.backend.failure, "winerror", 0), create=True))
        self.stack.enter_context(patch.object(shared.ctypes, "WinError", side_effect=self.backend.winerror, create=True))
        self.stack.enter_context(patch.object(shared.os, "rename", side_effect=self.backend.rename))
        self.stack.enter_context(patch.object(shared.os, "replace", side_effect=self.backend.replace))
        # Keep both actual production publisher bodies. Only the native boundary
        # and the handoff/emergency reader are replaced with ordinary-file fakes.
        self.stack.enter_context(patch.object(gpc, "_publish_json", shared._publish_json))
        self.stack.enter_context(patch.object(gpc, "_read", side_effect=self.read_owned_json))
        self.stack.enter_context(patch.object(gpc.threading, "Thread", NoTickThread))
        self.stderr = io.StringIO()
        self.stack.enter_context(patch("sys.stderr", self.stderr))

    def read_owned_json(self, path):
        path = Path(path)
        if not path.is_relative_to(self.root):
            raise AssertionError("Fixture tried to open a non-owned JSON")
        return json.loads(path.read_bytes())

    @contextmanager
    def snapshot_reader(self, path):
        # This is an immutable in-memory generation, not a Windows HANDLE test.
        with io.BytesIO(Path(path).read_bytes()) as stream:
            yield stream

    @staticmethod
    def injected_1175():
        error = OSError(errno.EINVAL, "synthetic status publication failure")
        error.winerror = 1175
        return error

    def publish(self, generation):
        return gpc.publish_status(self.handoff, dict(gpc.status_template(), generation=generation,
            whole=dict(marker="generation-"+str(generation), values=list(range(16)))))

    def test_actual_publishers_and_json_reader_keep_complete_old_and_new_generations(self):
        old = self.publish(1)
        with patch.object(shared, "_open_shared_read", self.snapshot_reader):
            self.assertEqual(shared._read(self.target), old)
            with shared._open_shared_read(self.target) as held_old:
                new = self.publish(2)
                self.assertEqual(json.loads(held_old.read()), old)
                self.assertEqual(shared._read(self.target), new)
        self.assertEqual(new["whole"]["values"], list(range(16)))
        self.assertEqual(new["actual_worker_pid"], os.getpid())
        self.assertEqual(new["nonce"], "synthetic-only")
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(list(self.root.glob("worker-status.json.*.tmp")), [])

    def test_oversized_gpc_payload_is_refused_before_temporary_or_publication(self):
        with self.assertRaisesRegex(ValueError, "fixed slot"):
            gpc.publish_status(self.handoff, dict(gpc.status_template(), oversized="x"*gpc.SLOT))
        self.assertEqual(self.backend.calls, [])
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.root.glob("worker-status.json.*.tmp")), [])

    def test_existing_shared_target_over_slot_or_hardlinked_is_refused_before_primitive(self):
        temporary = self.root/(self.target.name+"."+"a"*32+".tmp")
        temporary.write_bytes(b'{"new":true}\n')
        for kind in ("oversized", "hardlink"):
            with self.subTest(kind=kind):
                self.target.write_bytes(b"x"*(shared.STATUS_BYTES+1) if kind == "oversized" else b'{"old":true}\n')
                old = self.target.read_bytes()
                lstat = Path.lstat
                def target_info(path, *args, **kwargs):
                    value = lstat(path, *args, **kwargs)
                    if path == self.target and kind == "hardlink":
                        return SimpleNamespace(st_mode=value.st_mode, st_nlink=2,
                            st_file_attributes=getattr(value, "st_file_attributes", 0), st_size=value.st_size)
                    return value
                with patch.object(Path, "lstat", target_info):
                    with self.assertRaises((ValueError, RuntimeError)):
                        shared._publish_json(temporary, self.target, shared_status=True)
                self.assertEqual(self.target.read_bytes(), old)
                self.assertEqual(temporary.read_bytes(), b'{"new":true}\n')
        self.assertEqual(self.backend.calls, [])

    def test_source_target_and_ancestor_links_or_reparse_are_refused_before_primitive(self):
        temporary = self.root/(self.target.name+"."+"b"*32+".tmp")
        temporary.write_bytes(b'{"new":true}\n'); self.target.write_bytes(b'{"old":true}\n')
        lstat = os.lstat
        for affected in (temporary, self.target, self.root):
            for kind in ("symlink", "reparse"):
                def linked_info(path, *args, **kwargs):
                    value = lstat(path, *args, **kwargs)
                    if Path(path) == affected:
                        return SimpleNamespace(st_mode=stat.S_IFLNK if kind == "symlink" else value.st_mode,
                            st_file_attributes=0x400 if kind == "reparse" else 0)
                    return value
                with self.subTest(path=affected.name, kind=kind), patch.object(os, "lstat", linked_info):
                    with self.assertRaisesRegex(RuntimeError, "links|reparse"):
                        shared._publish_json(temporary, self.target, shared_status=True)
        self.assertEqual(self.backend.calls, [])

    def test_shared_atomic_json_1175_preserves_old_and_failed_temp_and_original_cause_once(self):
        from tdyssey.wm_compat_storage import StorageError
        storage = FixedSlotStorage(self.target)
        shared.atomic_json(self.target, {"old": True}, storage=storage)
        old = self.target.read_bytes()
        primary = self.injected_1175(); self.backend.failure = primary
        with patch.object(shared.time, "sleep", side_effect=AssertionError("1175 publication must not retry")):
            with self.assertRaises(StorageError) as caught:
                shared.atomic_json(self.target, {"new": True}, storage=storage)
        self.assertIs(caught.exception.__cause__, primary)
        self.assertEqual(caught.exception.details["phase"], "atomic.replace")
        self.assertEqual(caught.exception.details["exception_type"], "OSError")
        self.assertEqual((caught.exception.details["errno"], caught.exception.details["winerror"]),
                         (errno.EINVAL, 1175))
        self.assertEqual(len(self.backend.calls), 2)  # One initial success, one 1175 call.
        self.assertEqual(self.target.read_bytes(), old)
        failed = list(self.root.glob("worker-status.json.*.tmp"))
        self.assertEqual(len(failed), 1)
        self.assertEqual(json.loads(failed[0].read_bytes()), {"new": True})
        self.assertFalse(storage.active)
        self.assertEqual(len(storage.claims), 2)
        self.assertEqual(storage.phases, [(state, "guardian.atomic_status")
            for state in ("enter", "exit", "enter", "exit")])
        with self.assertRaisesRegex(StorageError, "Orphaned temporary"):
            shared.atomic_json(self.target, {"retry": "forbidden"}, storage=storage)
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(failed, list(self.root.glob("worker-status.json.*.tmp")))
        self.assertEqual(self.target.read_bytes(), old)

    def test_shared_source_must_be_ordinary_single_link_and_bounded_for_first_and_next_publication(self):
        lstat, real_stat = os.lstat, os.stat
        for existing in (False, True):
            for kind in ("directory", "hardlink", "oversized"):
                with self.subTest(existing=existing, source=kind):
                    directory = self.root/(str(existing)+"-"+kind); directory.mkdir()
                    target = directory/"worker-status.json"
                    temporary = directory/(target.name+"."+"c"*32+".tmp")
                    if existing:
                        target.write_bytes(b'{"old":true}\n')
                    if kind == "directory":
                        temporary.mkdir()
                    else:
                        temporary.write_bytes(b"x"*(shared.STATUS_BYTES+1) if kind == "oversized" else b'{"new":true}\n')

                    def source_info(call):
                        def declared(path, *args, **kwargs):
                            value = call(path, *args, **kwargs)
                            if Path(path) == temporary and kind == "hardlink":
                                fields = {name: getattr(value, name) for name in dir(value) if name.startswith("st_")}
                                fields["st_nlink"] = 2
                                return SimpleNamespace(**fields)
                            return value
                        return declared

                    before_calls = len(self.backend.calls)
                    with patch.object(os, "lstat", source_info(lstat)), patch.object(os, "stat", source_info(real_stat)):
                        with self.assertRaises((ValueError, RuntimeError)):
                            shared._publish_json(temporary, target, shared_status=True)
                    self.assertEqual(len(self.backend.calls), before_calls,
                        "Invalid source reached the publication primitive")
                    self.assertTrue(temporary.exists())
                    if existing:
                        self.assertEqual(target.read_bytes(), b'{"old":true}\n')
                    else:
                        self.assertFalse(target.exists())

    def test_injected_1175_has_one_attempt_keeps_old_and_failed_temp_and_never_retries(self):
        self.publish(1); old = self.target.read_bytes()
        primary = self.injected_1175(); self.backend.failure = primary
        with self.assertRaises(OSError) as caught:
            self.publish(2)
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.heartbeat_step, "status.atomic.publish")
        self.assertEqual(len(self.backend.calls), 2)  # Initial success, one failed publication.
        self.assertEqual(self.target.read_bytes(), old)
        temporary = Path(primary.heartbeat_temporary_path)
        self.assertEqual(json.loads(temporary.read_bytes())["generation"], 2)
        with self.assertRaisesRegex(RuntimeError, "Previous status publication"):
            self.publish(3)
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(self.target.read_bytes(), old)

    def test_heartbeat_1175_is_durable_first_cause_and_later_update_refuses_before_publish(self):
        primary = self.injected_1175()
        with self.assertRaises(OSError) as caught:
            with gpc.Heartbeat(self.handoff, gpc.status_template()) as heartbeat:
                self.backend.failure = primary
                try:
                    heartbeat.update(generation=2)
                except OSError:
                    count = len(self.backend.calls)
                    first = dict(heartbeat.first_error)
                    with self.assertRaisesRegex(RuntimeError, "heartbeat"):
                        heartbeat.update(forbidden_retry=True)
                    heartbeat.record_error(OSError(errno.ENOSPC, "secondary"), step="fixture.secondary")
                    self.assertEqual(heartbeat.first_error, first)
                    self.assertEqual(len(self.backend.calls), count)
                    self.assertNotIn("forbidden_retry", heartbeat.payload)
                    raise
        self.assertIs(caught.exception, primary)
        record = gpc.heartbeat_failure(self.handoff)
        self.assertEqual((record["errno"], record["winerror"], record["step"]),
                         (errno.EINVAL, 1175, "status.atomic.publish"))
        self.assertIn("synthetic status publication failure", record["traceback"])
        self.assertEqual((self.root/"heartbeat-error.json").stat().st_size, gpc.SLOT)
        self.assertFalse(heartbeat.thread.is_alive())

    def test_partial_write_retains_bytes_and_first_errno_without_cleanup_mask_or_retry(self):
        active, cleanup_attempts = [False], []
        original_write, original_unlink = gpc._write_all, Path.unlink
        primary = OSError(errno.EIO, "synthetic partial status write")
        def partial(stream, data):
            if active[0] and Path(stream.name).name.startswith("worker-status.json."):
                stream.write(data[:7]); raise primary
            return original_write(stream, data)
        def cleanup(path, *args, **kwargs):
            if active[0] and path.name.startswith("worker-status.json."):
                cleanup_attempts.append(str(path)); raise PermissionError(errno.EACCES, "secondary cleanup")
            return original_unlink(path, *args, **kwargs)
        with patch.object(gpc, "_write_all", side_effect=partial), patch.object(Path, "unlink", cleanup):
            with self.assertRaises(OSError) as caught:
                with gpc.Heartbeat(self.handoff, gpc.status_template()) as heartbeat:
                    active[0] = True
                    heartbeat.update(generation=2)
        self.assertIs(caught.exception, primary)
        record = gpc.heartbeat_failure(self.handoff)
        self.assertEqual(record["errno"], errno.EIO)
        self.assertEqual(record["step"], "status.temporary.write")
        self.assertEqual(len(Path(record["temporary_path"]).read_bytes()), 7)
        self.assertEqual(cleanup_attempts, [])
        self.assertEqual(len(self.backend.calls), 1)
        self.assertFalse(heartbeat.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "heartbeat"):
            heartbeat.update(forbidden_retry=True)
        self.assertEqual(len(self.backend.calls), 1)

    def test_five_second_status_and_allocator_age_limits_remain_independent(self):
        binding = gpc._clocks({}, 100., 10.)
        payload = dict(gpc.status_template(), phase="evaluation", training_complete=True,
            optimizer_updates=500, model_instances=1, cuda_allocated_bytes=0, cuda_reserved_bytes=0,
            cuda_peak_allocated_bytes=0, cuda_peak_reserved_bytes=0, cuda_sample_state="fresh",
            cuda_sample_at_unix=100., cuda_sample_at_monotonic=10.)
        with patch.object(gpc.time, "time", return_value=105.), patch.object(gpc.time, "monotonic", return_value=15.):
            value = gpc.publish_status(self.handoff, payload)
        self.assertEqual((value["cuda_sample_at_unix"], value["cuda_sample_at_monotonic"]), (100., 10.))
        self.assertIsNone(gpc.monitor_reason(value, binding, pid=os.getpid(), nonce="synthetic-only",
            now_unix=105., now_monotonic=15.))
        fresh_heartbeat = dict(value, at_unix=105.000001)
        self.assertEqual(gpc.monitor_reason(fresh_heartbeat, binding, pid=os.getpid(), nonce="synthetic-only",
            now_unix=105.000001, now_monotonic=15.000001), "cuda_sample_stale_or_invalid")
        stale_status = dict(value, at_unix=100.)
        self.assertEqual(gpc.monitor_reason(stale_status, binding, pid=os.getpid(), nonce="synthetic-only",
            now_unix=105.000001, now_monotonic=15.000001), "worker_status_stale_or_invalid")


if __name__ == "__main__":
    unittest.main()
