"""Persist actual stdlib audit output and the stable source hash it tested."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys,time,unittest

HERE=Path(__file__).resolve().parent
PHASE=HERE.parents[1]/'background/phase.py'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    output=HERE/a.label;output.mkdir(exist_ok=False)
    before=sha(PHASE);started=time.perf_counter()
    spec=importlib.util.spec_from_file_location('phase_actual_audit_tests',HERE/'test_phase.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    with(output/'test-output.txt').open('w',encoding='utf-8')as stream:
        actual=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    after=sha(PHASE)
    value=dict(kind='strict-original-background-stdlib-audit.v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=actual.wasSuccessful()and before==after,
        tests_run=actual.testsRun,failures=[test.id()for test,_ in actual.failures],errors=[test.id()for test,_ in actual.errors],skipped=len(actual.skipped),
        phase_source_sha256_before=before,phase_source_sha256_after=after,source_stable=before==after,test_source_sha256=sha(HERE/'test_phase.py'),
        wall_seconds=time.perf_counter()-started,actual_subprocesses_started=0,physics_steps=0,model_calls=0,GPU_calls=0,optimizer_updates=0,
        technique='Mocked Popen/Windows process and launcher primitives; no collector, physics, GPU or worker was launched')
    (output/'receipt.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value));return 0 if value['passed']else 1
if __name__=='__main__':raise SystemExit(main())
