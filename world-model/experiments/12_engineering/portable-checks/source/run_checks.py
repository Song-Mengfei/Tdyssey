"""Local bounded fake/array checks; never a neural, simulator or GUI run."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','pygame','pymunk','cv2','gym'}:
            raise ImportError('Fake/CPU check forbids real runtime: '+fullname)

sys.meta_path.insert(0,NoRuntime())
names=['tests.test_official_world_model_portable','tests.test_manual_world_model_portable',
    'tests.test_raw_training_guard']
if (ROOT/'tests/test_raw_world_model_training.py').exists():names.append('tests.test_raw_world_model_training')
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
if not result.wasSuccessful():raise SystemExit(1)
assert not any(sys.modules.get(name) is not None for name in ('torch','pygame','pymunk','cv2','gym'))
for name in ('official_world_model','manual_world_model','raw_training_guard'):
    module=sys.modules['tdyssey.'+name]
    assert Path(module.__file__).resolve().is_relative_to(ROOT)
print('PASS: scoped modules originate in the main-based checkout; real runtime imports0')
