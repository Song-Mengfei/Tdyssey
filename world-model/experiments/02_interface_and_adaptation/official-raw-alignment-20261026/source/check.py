"""Focused bounded engineering check; no real Torch, weights, simulator or SDK."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent.parent
sys.path.insert(0,str(ROOT))
names=['tests.test_official_world_model','tests.test_official_contact_region',
    'tests.test_official_contact_trainer','tests.test_official_contact_parity',
    'tests.test_manual_official_profile','tests.test_manual_world_model',
    'tests.test_manual_world_model_ui','tests.test_manual_prediction_usage','tests.test_gpc_foundation_contract']
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
if not result.wasSuccessful():raise SystemExit(1)
if 'torch' in sys.modules:raise AssertionError('Engineering check imported actual Torch')
if sys.argv[1]=='repair':
    probes=[]
    for args,expected in ((['scripts/visualize_world_model.py','--help'],0),
            (['scripts/visualize_world_model.py','--check-assets'],2),
            (['scripts/train_official_contact.py','--help'],0),
            (['scripts/train_official_contact.py','--datasets','missing-no-read','--output','missing-no-create'],0)):
        checked=subprocess.run([sys.executable,'-S','-B',*args],cwd=ROOT,capture_output=True,text=True,timeout=10)
        probes.append(dict(arguments=args,exit_code=checked.returncode,stdout=checked.stdout,stderr=checked.stderr))
        if checked.returncode!=expected:raise AssertionError('Read-only CLI probe failed: '+str(probes[-1]))
    with (BASE/'cli-probes.json').open('x',encoding='utf-8') as stream:json.dump(probes,stream,indent=2)
if sys.argv[1]=='final':
    for command in ([sys.executable,'-S','-B',str(ROOT/'scripts/check_repo.py')],
            [sys.executable,'-S','-B',str(ROOT/'scripts/check_messages.py'),'--commit-message',str(BASE/'commit-message.txt')]):
        r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=15)
        print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
        if r.returncode:raise SystemExit(r.returncode)
