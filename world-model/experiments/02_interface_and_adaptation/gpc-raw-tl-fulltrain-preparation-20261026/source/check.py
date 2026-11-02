"""Bounded synthetic/source preparation; no real SDK, weights or image reads."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent.parent
sys.path.insert(0,str(ROOT))
stage=sys.argv[1]
names=['tests.'+path.stem for path in sorted((ROOT/'tests').glob('test_gpc_raw_fulltrain*.py'))]
names += ['tests.test_official_world_model','tests.test_manual_official_profile',
    'tests.test_gpc_foundation_contract','tests.test_manual_prediction_usage']
if stage != 'first':
    names += ['tests.test_gpc_adaptation_data','tests.test_gpc_adaptation_guard','tests.test_gpc_evaluation_status']
if stage == 'final':
    names += ['tests.test_manual_world_model_ui']
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
if not result.wasSuccessful():raise SystemExit(1)
if 'torch' in sys.modules:raise AssertionError('Synthetic checks imported actual Torch')
from tdyssey import gpc_raw_fulltrain_entry as entry
from tdyssey.gpc_raw_fulltrain_data import plan_data,schedule_bytes
if stage=='first':
    cfg=entry.configuration_template()
    with entry.CONFIG.open('x',encoding='utf-8') as stream:json.dump(cfg,stream,indent=2);stream.write('\n')
    data=plan_data(ROOT/cfg['data_manifest'],ROOT/cfg['prefix_schedule']['path'])
    for key in ('full_schedule','development'):
        with (ROOT/cfg[key]).open('xb') as stream:stream.write(schedule_bytes(data[key]))
prepared=entry.plan()
if not prepared['ready']:raise AssertionError(prepared['gaps'])
compact={key:prepared[key] for key in ('kind','ready','gaps','config_sha256','inputs_sha256','plan_identities')}
compact['train_coverage']=prepared['inputs']['full_schedule']['coverage']
compact['development_coverage']=prepared['inputs']['development']['coverage']
with (BASE/('plan-'+stage+'.json')).open('x',encoding='utf-8') as stream:json.dump(compact,stream,indent=2)
print(json.dumps(compact))
if stage=='final':
    for name in ('check_repo','check_messages'):
        spec=importlib.util.spec_from_file_location('_raw_'+name,ROOT/'scripts'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if name=='check_messages':
            errors=module.check_commit((BASE/'commit-message.txt').read_text(encoding='utf-8'))
            if errors:raise AssertionError(errors)
        else:
            old=sys.argv;sys.argv=[str(ROOT/'scripts/check_repo.py')]
            try:
                code=module.main()
                if code:raise SystemExit(code)
            finally:sys.argv=old
if 'torch' in sys.modules:raise AssertionError('Actual Torch appeared')
