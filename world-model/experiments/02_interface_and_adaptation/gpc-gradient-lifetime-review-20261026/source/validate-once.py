"""One bounded stdlib/marker validation command; no production attempt."""
from contextlib import ExitStack
import importlib.abc
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
REVIEW=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
started=time.monotonic()
timeout=threading.Timer(55,lambda:os._exit(124))
timeout.daemon=True
timeout.start()
blocked=[]

class NoRealRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','numpy','pygame','pymunk','gpc','diffusion_policy'}:
            blocked.append(fullname)
            raise AssertionError('Real runtime import forbidden: '+fullname)

sys.meta_path.insert(0,NoRealRuntime())
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
from tests import test_gpc_gradient_lifetime as lifetime
from tests import test_gpc_training_schedule as schedule
from tests import test_gpc_adaptation_runtime as budget
from tdyssey import policy_memory

names=[
    'test_500_marker_operations_consume_manifest_order_and_checkpoint_cursor',
    'test_resume_binds_order_and_cumulative_cursor_without_replaying_examples',
    'test_pending_failure_spends_cursor_and_missing_schedule_cannot_use_legacy_path',
    'test_checkpoint_restore_checks_schedule_cursor_before_marker_state_mutation',
]
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(lifetime)])
suite.addTests(schedule.ScheduleChecks(name) for name in names)
suite.addTests(budget.RuntimeBudgetTests(name) for name in (
    'test_reservation_is_durable_before_entry_and_failed_cost_is_not_refunded',
    'test_fsync_failure_keeps_reservation_and_original_failure_durable'))
temporary=REVIEW/'temporary-fixtures'
temporary.mkdir(exist_ok=True)
assert temporary.resolve().is_relative_to(REVIEW)
attempts=[]

def denied(name):
    def fail(*args,**kwargs):
        attempts.append(name)
        raise AssertionError('Real native/process/network operation forbidden: '+name)
    return fail

stream=io.StringIO()
try:
    with ExitStack() as stack:
        stack.enter_context(patch.object(tempfile,'tempdir',str(temporary)))
        stack.enter_context(patch.object(budget,'EVIDENCE',REVIEW/'budget-fixtures'))
        stack.enter_context(patch.object(policy_memory,'_libraries',side_effect=denied('native_resource_libraries')))
        stack.enter_context(patch('subprocess.Popen',side_effect=denied('process_launch')))
        stack.enter_context(patch.object(socket,'socket',side_effect=denied('network_socket')))
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    log=stream.getvalue()
    assert len(log.encode())<=256*1024
    evidence=dict(kind='gpc_gradient_lifetime_validation.v1',test_command_number=1,
        command_limit=3,per_command_cap_seconds=60,hard_exit_after_seconds=55,
        cumulative_test_cap_seconds=180,seconds=time.monotonic()-started,
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        passed=result.wasSuccessful(),blocked_runtime_import_attempts=blocked,native_process_network_attempts=attempts,
        real_model_or_sdk_or_physics_operations=0,
        comparison=getattr(lifetime.GradientLifetimeTests,'comparison_evidence',None),
        temporary_allowance_bytes=4*1024**2,evidence_cap_bytes=16*1024**2)
    with (REVIEW/'validation-01.log').open('x',encoding='utf-8') as out:out.write(log)
    before=sum(p.stat().st_size for p in REVIEW.rglob('*') if p.is_file())
    evidence['retained_review_bytes_before_result']=before
    assert before+4*1024**2+65536<=16*1024**2
    with (REVIEW/'validation-01.json').open('x',encoding='utf-8') as out:
        json.dump(evidence,out,sort_keys=True,indent=2);out.write('\n')
    print(json.dumps(evidence))
    if not result.wasSuccessful():print(log)
    code=0 if result.wasSuccessful() and not blocked and not attempts else 1
finally:
    timeout.cancel()
raise SystemExit(code)
