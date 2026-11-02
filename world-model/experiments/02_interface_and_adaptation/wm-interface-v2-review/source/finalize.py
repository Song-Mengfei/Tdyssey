"""Bind final local source and review outputs with standard-library hashes only."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

root=Path(__file__).resolve().parents[2]
review=Path(__file__).resolve().parent
main=root/'artifacts/wm-interface-v2-development'
manifest=json.loads((main/'output-manifest.json').read_text(encoding='utf-8'))
assert set(manifest['files'])=={p.relative_to(main).as_posix() for p in main.rglob('*') if p.is_file()}-{'output-manifest.json'}
for name,record in manifest['files'].items():
    data=(main/name).read_bytes()
    assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
assert manifest['total_output_bytes']==sum(p.stat().st_size for p in main.rglob('*') if p.is_file())
assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
sources=('tdyssey/wm_image_measurement_v2.py','tdyssey/wm_interface_diagnostics.py',
         'tdyssey/wm_measurement_contract.py','configs/wm_image_measurement_v2.json',
         'scripts/diagnose_wm_interface_v2.py','docs/WM_INTERFACE_V2_PROTOCOL.md',
         'docs/WM_INTERFACE_V2_RESULTS.md')
record=dict(final_local_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    measured_local_commit='ab01b9e',
    final_changes='reporting metadata and documentation only; saved numeric measurements unchanged',
    source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
    focused_tests=dict(passed=30,wall_seconds=2.390,command="python -B -m unittest discover -s tests -p test_wm_*.py -q"),
    protected_old_files=595,protected_old_bytes=186686009,protected_bc_files=150,protected_bc_bytes=97364187,
    after_audit_sha256=hashlib.sha256((root/'artifacts/wm-interface-v2-audit/after.json').read_bytes()).hexdigest(),
    all_original_v2_output_hashes_verified=True,real_operations=0,learned_model_calls=0,
    additional_cpu_timing_scope='main measurement/aggregation only; report-only scripts and tests excluded')
with (review/'final-source.json').open('x',encoding='utf-8') as stream:
    json.dump(record,stream,indent=2,sort_keys=True)
    stream.write('\n')
make_manifest=runpy.run_path(str(root/'scripts/diagnose_wm_interface_v2.py'))['output_manifest']
for directory in (review,root/'artifacts/wm-interface-v2-audit'):
    info=make_manifest(directory)
    print(json.dumps(dict(directory=str(directory),total_output_bytes=info['total_output_bytes'])))
directories=[main,review,root/'artifacts/wm-interface-v2-audit']
print(json.dumps(dict(commit=record['final_local_commit'],all_new_output_bytes=sum(
    p.stat().st_size for directory in directories for p in directory.rglob('*') if p.is_file()),
    original_v2_output_preserved=True,git_clean=True)))
