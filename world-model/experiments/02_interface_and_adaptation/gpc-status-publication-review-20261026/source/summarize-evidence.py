"""Read completed evidence only; no publisher, runtime, fixture or native call."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[1]
digest = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_bytes())
records = []
for mode in ("original", "fixed"):
    base = HERE/("native-"+mode)
    parent = read(base/"parent-report.json")
    worker = parent["worker_result"]
    cases = []
    for case in worker["cases"]:
        directory = base/case["case"]
        copies = []
        for item in case["copies"]:
            if item.get("missing"):
                copies.append(item)
                continue
            content = (directory/item["copy"]).read_bytes()
            original = (directory/item["source"]).read_bytes()
            if content != original or len(content) != item["bytes"] or digest(content) != item["sha256"]:
                raise ValueError("Retained native evidence changed")
            payload = json.loads(content)
            copies.append(dict(source=item["source"], bytes=len(content), generation=payload["generation"],
                               mirror=payload["mirror"], sha256=digest(content)))
        cases.append(dict(case=case["case"], status=case["status"], attempted=case["attempted"],
            success=case["success"], first_error=case["error"], retained=copies,
            old_reader_complete=case.get("old_reader_complete"), readers=case.get("readers")))
    records.append(dict(mode=mode, parent_status=parent["status"], seconds=parent["wall_seconds"],
        outer=read(HERE/("native-"+mode+"-outer.json")), counts=worker["counts"], cases=cases,
        job_active=parent["job_active_processes_after_cleanup"], cleanup_errors=parent["cleanup_errors"]))

final = (ROOT/"tdyssey/wm_compat_guard.py").read_bytes()
replace = b'''            kernel.ReplaceFileW.argtypes = [wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.LPCWSTR,
                wintypes.DWORD,ctypes.c_void_p,ctypes.c_void_p]
            kernel.ReplaceFileW.restype = wintypes.BOOL
            if not kernel.ReplaceFileW(str(path),str(temporary),None,0,None,None):
'''
move = b'''            kernel.MoveFileExW.argtypes = [wintypes.LPCWSTR,wintypes.LPCWSTR,wintypes.DWORD]
            kernel.MoveFileExW.restype = wintypes.BOOL
            # Same-directory rename: REPLACE_EXISTING | WRITE_THROUGH. Select
            # once, without ReplaceFileW's metadata merge/name-gap behavior.
            if not kernel.MoveFileExW(str(temporary),str(path),0x1|0x8):
'''
if final.count(replace) != 1:
    raise ValueError("Cannot reconstruct the exact rejected candidate")
candidate = final.replace(replace, move)
expected = read(HERE/"plan-fixed.json")["source_sha256"]["tdyssey/wm_compat_guard.py"]
if digest(candidate) != expected:
    raise ValueError("Candidate reconstruction does not match executed SHA")
with (HERE/"wm_compat_guard.rejected-candidate.py").open("xb") as stream:
    stream.write(candidate)
blob = subprocess.check_output(["git", "show", "f9287e8d96392e823accef487b51e20d3eafa542:tdyssey/wm_compat_guard.py"], cwd=ROOT)
expected = read(HERE/"plan.json")["source_sha256"]["tdyssey/wm_compat_guard.py"]
original = next((b for b in (blob, blob.replace(b"\n", b"\r\n")) if digest(b) == expected), None)
if original is None:
    raise ValueError("Original source archive differs from executed source")
with (HERE/"wm_compat_guard.original.py").open("xb") as stream:
    stream.write(original)
summary = dict(native=records, total_attempted=sum(r["counts"]["attempted"] for r in records),
    total_success=sum(r["counts"]["success"] for r in records),
    native_outer_seconds=sum(r["outer"]["wall_seconds"] for r in records),
    native_guarded_seconds=sum(r["seconds"] for r in records),
    synthetic=[read(HERE/("synthetic-"+m+".json")) for m in ("baseline", "fixed", "final")],
    final_source_sha256=digest(final), rejected_candidate_sha256=digest(candidate),
    original_source_sha256=digest(original), native_runs_exhausted=True,
    original_1175_reproduced=False, publication_problem_resolved=False,
    real_model_sdk_physics_operations=0)
with (HERE/"summary.json").open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ("native", "synthetic")}))
