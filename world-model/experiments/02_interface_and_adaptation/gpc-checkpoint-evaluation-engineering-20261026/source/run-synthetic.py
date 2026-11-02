"""Bounded engineering tests; no SDK, native process, device or physics."""
from contextlib import contextmanager
import ctypes
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

HERE=Path(__file__).absolute().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
number=int(sys.argv[1])
if number not in (1,2,3,4):raise ValueError('Four synthetic command identities only')
out=HERE/('test-command-'+str(number)+'.json')
if out.exists():raise FileExistsError(out)
started=time.monotonic();timer=threading.Timer(55,lambda:os._exit(124));timer.daemon=True;timer.start()
blocked=[]
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','numpy','scipy','pygame','pymunk','gpc_code'):
            blocked.append(fullname);raise AssertionError('Forbidden real runtime: '+fullname)
def forbidden(*args,**kwargs):
    blocked.append('native/process/network');raise AssertionError('Forbidden native/process/network')
sys.meta_path.insert(0,NoRuntime())
from tdyssey import wm_compat_guard as shared
@contextmanager
def owned_reader(path):
    path=Path(os.path.abspath(path))
    if not path.is_relative_to(HERE):raise AssertionError('Synthetic reader outside current review')
    with path.open('rb') as stream:yield stream
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    if hasattr(module,'EVIDENCE'):module.EVIDENCE=HERE/'unit-fixtures'
    if hasattr(module,'RACE_EVIDENCE'):module.RACE_EVIDENCE=HERE/'unit-fixtures'
    return module
paths=['tdyssey/gpc_evaluation_status.py','tdyssey/gpc_evaluation_assets.py','tdyssey/gpc_checkpoint_evaluation.py',
    'tdyssey/gpc_evaluation_entry.py','tdyssey/gpc_adaptation_guard.py','tdyssey/gpc_supervised_adaptation.py',
    'scripts/evaluate_gpc_checkpoint500.py','configs/gpc_checkpoint500_evaluation.json']
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
capture=io.StringIO();suite=unittest.TestSuite()
with patch.object(ctypes,'WinDLL',side_effect=forbidden,create=True),patch.object(subprocess,'Popen',side_effect=forbidden), \
        patch.object(socket,'socket',side_effect=forbidden),patch.object(shared,'_open_shared_read',owned_reader):
    old=load('test_gpc_adaptation_guard')
    for name in ('test_owned_ack_success_and_durable_report','test_first_resource_failure_survives_cleanup_failure',
                 'test_final_close_time_is_in_original_deadline_and_durable_stop'):
        suite.addTest(old.BridgeTests(name))
    observer=load('test_gpc_startup_observability')
    for name in ('test_heartbeat_freshness_limit_remains_exactly_five_seconds',
                 'test_refresh_failure_is_durable_and_thread_cleanup_keeps_first_cause',
                 'test_partial_status_write_and_cleanup_error_preserve_original_error',
                 'test_publication_sharing_error_is_recorded_once_without_retry',
                 'test_cuda_sample_age_and_state_cannot_be_replaced_by_fresh_heartbeat'):
        suite.addTest(observer.StartupObservabilityChecks(name))
    for name in ('test_gpc_evaluation_status','test_gpc_evaluation_assets','test_gpc_checkpoint_evaluation','test_gpc_evaluation_entry'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(load(name)))
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(suite)
    from tdyssey import gpc_evaluation_entry as entry
    before_output=(ROOT/entry.OUTPUT).exists()
    try:
        prepared=entry.plan()
        plan=dict(kind=prepared['kind'],ready=prepared['ready'],gaps=prepared['gaps'],mode=prepared['mode'],
            config_sha256=prepared['config_sha256'],assets_sha256=prepared['assets_sha256'],
            execution_authorized=prepared['execution_authorized'],real_operations=prepared['real_operations'],
            native_resource_queries=prepared['native_resource_queries'],
            production_output_before=before_output,production_output_after=(ROOT/entry.OUTPUT).exists())
        if plan['production_output_before'] or plan['production_output_after'] or not plan['ready']:
            raise AssertionError('Read-only evaluation plan was not ready or created production output')
    except BaseException as error:
        import traceback
        plan=dict(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
record=dict(command=number,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    successful=result.wasSuccessful() and 'error' not in plan,seconds=time.monotonic()-started,forbidden_attempts=blocked,
    source_sha256=hashes,source_unchanged=hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
    read_only_plan=plan,real_models=0,sdk=0,native_apis=0,process_launches=0,network=0,physics=0,output=capture.getvalue())
with out.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
timer.cancel();print(json.dumps({k:v for k,v in record.items() if k!='output'}));print(capture.getvalue())
sys.exit(0 if record['successful'] else 1)
