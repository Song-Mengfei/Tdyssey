import hashlib,io,json,sys,time,unittest
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
repo=out.parents[4];sys.path.insert(0,str(repo))
from tdyssey.original_contact_pilot_storage import pack_telemetry,unpack_telemetry,TELEMETRY_STORAGE_KEYS
start=time.perf_counter();stream=io.StringIO()
suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_original_contact_pilot_storage')
test_start=time.perf_counter();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);test_seconds=time.perf_counter()-test_start
assert result.wasSuccessful(),stream.getvalue()
with np.load(out/'first-complete-T-raw-v2-column-major-comparison.npz',allow_pickle=False)as saved:
    storage={key:saved[key]for key in saved.files if key in TELEMETRY_STORAGE_KEYS}
payload=unpack_telemetry(storage)
new_pack=pack_telemetry(payload['telemetry'],payload['decisions'])
assert set(storage)==set(new_pack)
for key in storage:
    assert storage[key].dtype==new_pack[key].dtype and storage[key].shape==new_pack[key].shape
    assert storage[key].tobytes()==new_pack[key].tobytes(),key
assert new_pack['telemetry_microstates_float64'].flags.f_contiguous and new_pack['telemetry_microstate_refs'].flags.f_contiguous
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
record=dict(kind='final-lossless-helper-pure-evidence-check.v2',passed=True,tests_run=result.testsRun,
    failures=len(result.failures),errors=len(result.errors),tests_seconds=test_seconds,
    actual_T_final_pack_arrays_match_compared_column_major_archive=True,
    whole_final_check_seconds=time.perf_counter()-start,
    helper_sha256=sha(repo/'tdyssey/original_contact_pilot_storage.py'),tests_sha256=sha(repo/'tests/test_original_contact_pilot_storage.py'),
    model_calls=0,physics_controls=0,GPU_calls=0,optimizer_updates=0,Torch_imported='torch'in sys.modules,Pymunk_imported='pymunk'in sys.modules,
    current_collector_integrated=False,current_report_modified_by_this_task=False,
    scope='Pure saved-file/array tests. Does not claim physics, model or collection runtime acceptance.')
(out/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
(out/'final-check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
