"""CPU/fake verification with the original guard fixture's clock origin fixed."""
import argparse
from contextlib import ExitStack
from functools import wraps
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/"tests"))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report",type=Path,required=True)
args=parser.parse_args()
folder=Path(__file__).resolve().parent
names=["test_gpc_adaptation_data","test_gpc_supervised_adaptation","test_gpc_adaptation_runtime",
       "test_gpc_adaptation","test_gpc_adaptation_reporting","test_gpc_adaptation_guard"]
spec=importlib.util.spec_from_file_location("ranking_repair_cpu_harness",ROOT/"scripts/check_gpc_adaptation.py")
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
with ExitStack() as stack:
    for name in names:
        module=importlib.import_module(name)
        if hasattr(module,"EVIDENCE"):
            stack.enter_context(patch.object(module,"EVIDENCE",folder/"fixtures"/name))
    guard_tests=importlib.import_module("test_gpc_adaptation_guard")
    original=guard_tests.BridgeTests.test_final_close_time_is_in_original_deadline_and_durable_stop
    @wraps(original)
    def exact_clock_fixture(test):
        with patch.object(guard_tests.guard.time,"time",return_value=1000000.),patch.object(guard_tests.guard.time,"monotonic",return_value=1000.):
            return original(test)
    stack.enter_context(patch.object(guard_tests.BridgeTests,"test_final_close_time_is_in_original_deadline_and_durable_stop",exact_clock_fixture))
    code=entry.main(["--report",str(args.report)])
sys.exit(code)
