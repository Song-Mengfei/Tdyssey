"""Deterministic stdlib-only old scanner/atomic heartbeat race in a temp repo."""

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tdyssey import wm_compat_budget as budget
from tdyssey import wm_compat_guard as guard


def reproduce():
    # Capture the actual old callable before production integration changes it.
    old_files = budget._files
    old_source = inspect.getsource(old_files)
    atomic_source = inspect.getsource(guard.atomic_json)
    events, producer_errors = [], []
    ready, enumerated, published = (threading.Event() for _ in range(3))
    original_walk, original_replace = os.walk, os.replace
    with tempfile.TemporaryDirectory(prefix="fake-repo-", dir=Path(__file__).parent) as directory:
        fake_repo = Path(directory)
        guardian = fake_repo / "artifacts/wm-input-compat-fake-output/guardian" / ("b"*32)
        guardian.mkdir(parents=True)
        status = guardian / "worker-status.json"
        # The first heartbeat is a legal atomic publication without an old
        # target, avoiding this sandbox's unrelated replace-existing denial.
        assert not status.exists()
        immutable = guardian / "guardian.json"
        immutable.write_bytes(b'{"fake_only":true}\n')
        immutable_before = hashlib.sha256(immutable.read_bytes()).hexdigest()
        temporary = status.with_name(status.name+"."+"a"*32+".tmp")

        def replace_after_enumeration(source, destination):
            assert Path(source) == temporary and Path(destination) == status
            events.append("producer_temp_fsynced_and_closed")
            ready.set()
            if not enumerated.wait(5):
                raise RuntimeError("fixture enumeration barrier timed out")
            original_replace(source, destination)
            events.append("producer_atomic_replace_completed")
            published.set()

        def walk_after_publication(root, *args, **kwargs):
            if not ready.wait(5):
                raise RuntimeError("fixture producer barrier timed out")
            for folder, directories, names in original_walk(root, *args, **kwargs):
                if Path(folder) == guardian:
                    assert temporary.name in names
                    events.append("scanner_enumerated_temporary_name")
                    enumerated.set()
                    if not published.wait(5):
                        raise RuntimeError("fixture publication barrier timed out")
                    events.append("scanner_validates_stale_enumeration")
                yield folder, directories, names

        def producer():
            try:
                guard.atomic_json(status, {"generation":1})
            except BaseException as exc:
                producer_errors.append(type(exc).__name__+": "+str(exc))
                published.set()

        with patch.object(guard.uuid, "uuid4", return_value=SimpleNamespace(hex="a"*32)), \
                patch.object(guard.os, "replace", side_effect=replace_after_enumeration), \
                patch.object(budget.os, "walk", side_effect=walk_after_publication):
            thread = threading.Thread(target=producer, daemon=True)
            thread.start()
            scanner_error = None
            try:
                old_files(guardian)
            except budget.BudgetError as exc:
                scanner_error = type(exc).__name__+": "+str(exc)
            thread.join(5)
        assert not thread.is_alive() and not producer_errors, producer_errors
        assert scanner_error == "BudgetError: Only ordinary output files are allowed", scanner_error
        assert json.loads(status.read_text()) == {"generation":1}
        assert not temporary.exists()
        assert hashlib.sha256(immutable.read_bytes()).hexdigest() == immutable_before
        return dict(kind="old_budget_atomic_heartbeat_race.v1",
            created_utc=datetime.now(timezone.utc).isoformat(),
            source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            old_files_source=old_source, old_files_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
            atomic_json_source_sha256=hashlib.sha256(atomic_source.encode()).hexdigest(),
            events=events, scanner_error=scanner_error,
            producer_succeeded=True, initial_status_absent=True,
            final_status_generation=1, temporary_absent=True,
            immutable_fixture_unchanged=True, fake_repo_removed_on_exit=True,
            model_loads=0, forwards=0, decodes=0, transforms=0,
            resource_queries=0, physical_operations=0, production_claims=0,
            explanation="os.walk returned a valid temporary name; atomic_json renamed it before is_file, making the old scanner reject a legal publication")


if __name__ == "__main__":
    result = reproduce()
    output = Path(__file__).with_name("old-scanner-race.json")
    with output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                         scanner_error=result["scanner_error"],producer_succeeded=True),indent=2))
