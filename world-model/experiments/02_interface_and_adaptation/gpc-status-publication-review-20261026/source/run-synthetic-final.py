"""One-off, stdlib-only regression runner; no model/native/process work."""
import ctypes
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[2]
OUT = Path(__file__).absolute().parent
sys.path.insert(0, str(ROOT))
mode = sys.argv[1]
if mode not in ("baseline", "fixed", "final"):
    raise ValueError("Only two frozen synthetic modes")
result_path = OUT / ("synthetic-"+mode+".json")
if result_path.exists():
    raise FileExistsError(result_path)
started = time.monotonic()
inputs = ["tdyssey/wm_compat_guard.py", "tdyssey/gpc_adaptation_guard.py", "tests/test_gpc_status_publication.py"]
before = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
timer = threading.Timer(55, lambda: os._exit(124))
timer.daemon = True
timer.start()
blocked = []


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "numpy", "pymunk", "pygame", "gpc_code"):
            blocked.append(fullname)
            raise AssertionError("Real runtime forbidden: "+fullname)


def forbidden(*args, **kwargs):
    blocked.append("native/process/network")
    raise AssertionError("Native/process/network forbidden")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/"tests"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EVIDENCE = OUT/"synthetic-fixtures"
    if hasattr(module, "RACE_EVIDENCE"):
        module.RACE_EVIDENCE = OUT/"synthetic-fixtures"
    return module


sys.meta_path.insert(0, NoRuntime())
from tdyssey import wm_compat_guard as shared


@contextmanager
def owned_fake_reader(path):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(OUT):
        raise AssertionError("Synthetic reader outside current review root")
    with path.open("rb") as stream:
        yield stream


capture = io.StringIO()
suite = unittest.TestSuite()
with patch.object(ctypes, "WinDLL", side_effect=forbidden, create=True), \
        patch.object(subprocess, "Popen", side_effect=forbidden), \
        patch.object(socket, "socket", side_effect=forbidden), \
        patch.object(shared, "_open_shared_read", owned_fake_reader):
    publication = load("test_gpc_status_publication")
    if mode == "baseline":
        names = [name for name in dir(publication.StatusPublicationTests)
                 if name.startswith("test_shared_source_") or name.startswith("test_shared_atomic_")]
        if not names:
            raise AssertionError("Contract gap cases missing")
        suite.addTests(publication.StatusPublicationTests(name) for name in names)
    else:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(publication.StatusPublicationTests))
        wm = load("test_wm_compat_guard")
        suite.addTests(wm.StorageRaceEngineeringChecks(name) for name in (
            "test_atomic_publication_diagnostics_preserve_primary_and_retain_orphan_bytes",
            "test_atomic_only_retries_explicit_windows_sharing_or_lock_errors"))
        bridge = load("test_gpc_adaptation_guard")
        suite.addTests(bridge.BridgeTests(name) for name in (
            "test_owned_ack_success_and_durable_report",
            "test_first_resource_failure_survives_cleanup_failure",
            "test_final_close_time_is_in_original_deadline_and_durable_stop"))
        observer = load("test_gpc_startup_observability")
        suite.addTests(observer.StartupObservabilityChecks(name) for name in (
            "test_heartbeat_freshness_limit_remains_exactly_five_seconds",
            "test_refresh_failure_is_durable_and_thread_cleanup_keeps_first_cause",
            "test_partial_status_write_and_cleanup_error_preserve_original_error",
            "test_publication_sharing_error_is_recorded_once_without_retry",
            "test_cuda_sample_age_and_state_cannot_be_replaced_by_fresh_heartbeat"))
    result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
record = dict(mode=mode, tests=result.testsRun, failures=len(result.failures),
              errors=len(result.errors), successful=result.wasSuccessful(),
              seconds=time.monotonic()-started, forbidden_attempts=blocked,
              source_sha256=before, source_unchanged=before == {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs},
              real_models=0, sdk=0, native=0, network=0, physical_operations=0,
              output=capture.getvalue())
with result_path.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
timer.cancel()
print(json.dumps({k:v for k,v in record.items() if k != "output"}))
print(capture.getvalue())
sys.exit(0 if result.wasSuccessful() else 1)

