"""One of three bounded CPU checks; real Torch/physics are forbidden."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent.parent
sys.path.insert(0,str(ROOT))
modules=(['tests.test_gpc_conditioning_contract','tests.test_gpc_conditioning_data',
    'tests.test_gpc_conditioning_diagnostic','tests.test_gpc_conditioning_entry',
    'tests.test_gpc_adaptation_guard','tests.test_gpc_checkpoint_evaluation'] if sys.argv[1]=='integration'
    else ['tests.test_gpc_conditioning_entry','tests.test_gpc_conditioning_diagnostic'])
suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
result=unittest.TextTestRunner(verbosity=1).run(suite)
if not result.wasSuccessful():raise SystemExit(1)
if 'torch' in sys.modules:raise AssertionError('Synthetic check imported real Torch')
if sys.argv[1]=='final':
    for command in ([sys.executable,'-S','-B',str(ROOT/'scripts/check_repo.py')],
            [sys.executable,'-S','-B',str(ROOT/'scripts/check_messages.py'),'--commit-message',str(BASE/'commit-message.txt')]):
        checked=subprocess.run(command,cwd=ROOT,timeout=15,capture_output=True,text=True)
        print(checked.stdout,end='');print(checked.stderr,end='',file=sys.stderr)
        if checked.returncode:raise SystemExit(checked.returncode)
    from tdyssey.gpc_conditioning_entry import plan
    prepared=plan()
    if not prepared['ready']:raise ValueError(prepared['gaps'])
    with (BASE/'prepared.json').open('x',encoding='utf-8') as stream:
        json.dump(prepared,stream,indent=2,allow_nan=False)
        stream.write('\n')
    order=[dict(role=role,window_id=spec['window_id'],cursor=spec['cursor'],shape=spec['shape'],
        arm=arm,sigma_cond=None if arm=='A' else .002,seed=5478,horizon=1)
        for role in ('frozen','adapted') for spec in prepared['inputs']['windows'] for arm in ('A','B')]
    with (BASE/'run-plan.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(kind='conditioning_frozen_run_plan.v1',order=order,config=prepared['config'],
            inputs_sha256=prepared['inputs_sha256'],assets_sha256=prepared['assets_sha256'],
            upstream='a96146e568826ef46134f73c22fdf59a675e7264',attempts=1,
            zero_operations=['training','loss','backward','physics','render','reset','collection','policy'],
            failure='preserve and stop; no resume/retry/grid',interpretation='mechanism screen only'),stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({key:prepared[key] for key in ('ready','config_sha256','inputs_sha256','assets_sha256')}))
