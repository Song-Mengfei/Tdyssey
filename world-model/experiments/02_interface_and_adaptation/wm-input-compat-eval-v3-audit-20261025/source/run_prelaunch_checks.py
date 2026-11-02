"""Fake/native-process engineering checks only; never launch production."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
loader = unittest.TestLoader()
suite = unittest.TestSuite(loader.discover(str(ROOT/"tests"), pattern=name) for name in (
    "test_wm_compat_budget.py", "test_wm_compat_guard.py", "test_wm_compat_evaluation.py"))
def ids(item):
    if isinstance(item, unittest.TestSuite):
        return [name for child in item for name in ids(child)]
    return [item.id()]
test_ids = ids(suite)
started, cpu = time.monotonic(), time.process_time()
result = unittest.TextTestRunner(verbosity=1).run(suite)
forbidden = [name for name in ("torch", "pymunk", "pygame", "diffusion_policy",
    "tdyssey.gpc_world_model", "tdyssey.wm_image_measurement_v2") if name in sys.modules]
record = dict(kind="v3_prelaunch_fake_checks",created_utc=datetime.now(timezone.utc).isoformat(),
    tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,test_ids=test_ids,
    forbidden_runtime_modules_imported=forbidden,formal_runs=0,
    passed=result.wasSuccessful() and not forbidden)
path = Path(__file__).with_name("prelaunch-focused-tests.json")
with path.open("x",encoding="utf-8",newline="\n") as stream:
    json.dump(record,stream,indent=2,sort_keys=True);stream.write("\n")
print(json.dumps({key:record[key] for key in ("passed","tests","failures","errors","skipped","wall_seconds")}))
raise SystemExit(0 if record["passed"] else 1)
