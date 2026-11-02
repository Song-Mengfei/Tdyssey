"""Run only the adaptation's tiny CPU and fake-resource tests in one process."""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"tests"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv)
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("Use a fresh CPU test process")
    modules=["test_gpc_adaptation_data","test_gpc_supervised_adaptation","test_gpc_adaptation_runtime",
        "test_gpc_adaptation","test_gpc_adaptation_reporting","test_gpc_adaptation_guard"]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    identifiers=[]
    def visit(test):
        if isinstance(test,unittest.TestSuite):
            for child in test:visit(child)
        else:identifiers.append(test.id())
    visit(suite)
    stream=io.StringIO();began=time.perf_counter()
    with patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("No CUDA initialization in CPU tests")), \
         patch("tdyssey.gpc_world_model.load_world_model_source",side_effect=AssertionError("No real GPC source import")):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    if torch.cuda.is_initialized():raise AssertionError("CPU test initialized CUDA")
    report=dict(kind="tdyssey.gpc_adaptation_engineering_checks.v1",passed=result.wasSuccessful(),
        tests=result.testsRun,test_ids=identifiers,seconds=time.perf_counter()-began,
        failures=[dict(test=test.id(),traceback=tb) for test,tb in result.failures],
        errors=[dict(test=test.id(),traceback=tb) for test,tb in result.errors],
        skipped=[dict(test=test.id(),reason=reason) for test,reason in result.skipped],
        cuda_initialized=False,real_gpc_source_imports=0,real_model_loads=0,real_forwards=0,
        real_predictions=0,real_optimizer_updates=0,physics_operations=0,native_preflight_executed=False,
        scope="Small synthetic arrays, two-scalar CPU denoisers, atomic checkpoint faults and fake native ownership/resources")
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open("x",encoding="utf-8") as out:json.dump(report,out,indent=2)
    print(stream.getvalue());print(json.dumps({key:report[key] for key in ("passed","tests","seconds","cuda_initialized")}))
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__=="__main__":
    raise SystemExit(main())
