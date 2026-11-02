"""Byte/native-resource preflight only: no image decode/model or production writes."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tdyssey.wm_compat_budget import canonical_plan, DEFAULT_LIMITS, EXACT_CHECKPOINT
from tdyssey.wm_compat_identity import ATTEMPT_ID, REGISTRY, OUTPUT, LEDGER, AUTHORIZATION_ID
from tdyssey.wm_compat_guard import DEFAULT_LOCK, LIMITS, ResourceProvider, resource_reason

record=dict(kind="v3_native_read_only_prelaunch",created_utc=datetime.now(timezone.utc).isoformat(),
    formal_runs=0,model_loads=0,decoder_calls=0,transform_calls=0,passed=False)
try:
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    assert head==sys.argv[1],"Exact reviewed commit differs"
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"Dirty source"
    paths={"registry":ROOT/"artifacts"/REGISTRY,"output":ROOT/"artifacts"/OUTPUT,"ledger":ROOT/"artifacts"/LEDGER}
    assert all(not os.path.lexists(path) for path in paths.values()),"Production path already exists"
    assert not os.path.lexists(DEFAULT_LOCK),"Global model lock already exists"
    spec=importlib.util.spec_from_file_location("v3_readonly_entry",ROOT/"scripts/validate_wm_compat.py")
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    run=ROOT/"artifacts/frozen-wm-validation-v1-20261024"
    checkpoint=ROOT/"artifacts/frozen-wm-assets/denoiser.pth"
    assets=[run/"report.json",checkpoint]
    assets.extend(run/f"cal-{shape}-{i:02d}.npz" for shape in ("L","T") for i in range(12))
    for branch in canonical_plan()["branches"]:
        if branch["status"]=="missing":continue
        prefix=branch["anchor"]+"-"+branch["candidate"]
        assets.append(run/(prefix+"-physical.npz"))
        assets.extend(run/(prefix+f"-noise-{seed}.npz") for seed in (5478,5479,5480))
    assert len(assets)==278
    identities=entry.verify_frozen_assets(assets,run,checkpoint)
    upstream=ROOT.parent/"Tdyssey/third_party/gpc_code"
    manifest=json.loads((ROOT/"manifests/gpc_world_model.json").read_text())
    upstream_head=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    assert upstream_head==manifest["source_commit"]
    for name,expected in manifest["source_sha256"].items():
        assert entry.digest(upstream/name)==expected,name
    protocol=entry.digest(ROOT/"docs/GPC_INPUT_COMPATIBILITY_PROPOSAL.md")
    assert protocol=="2bfff44111d41f5c54a626b717ec2ea35f4bee5051b9fbeb19926f8fa4dbacf2"
    scientific=["scripts/validate_wm_compat.py","scripts/run_guarded_wm_compat.py",
        "tdyssey/wm_compat_storage.py","tdyssey/wm_compat_engine.py","tdyssey/wm_compat_evaluation.py",
        "tdyssey/wm_compat_transform.py","tdyssey/wm_compat_reporting.py","tdyssey/wm_compat_diagnostics.py",
        "tdyssey/wm_image_measurement_v2.py","tdyssey/wm_measurement_contract.py","configs","manifests",
        "pyproject.toml","uv.lock",".python-version","docs/GPC_INPUT_COMPATIBILITY_PROPOSAL.md"]
    subprocess.run(["git","diff","--exit-code","df229186c386ac78db31e9bcb3da73d06ccb0067","--",*scientific],cwd=ROOT,check=True)
    assert DEFAULT_LIMITS["seconds"]==1800 and DEFAULT_LIMITS["images"]==2268 and DEFAULT_LIMITS["euler_steps"]==6804
    assert LIMITS["output_bytes"]==256*1024**2 and LIMITS["telemetry_bytes"]==16*1024**2
    provider=ResourceProvider()
    try: resources=provider.startup(ROOT)
    finally: provider.close()
    reason=resource_reason(resources,startup=True)
    record.update(reviewed_commit=head,attempt_id=ATTEMPT_ID,authorization_id=AUTHORIZATION_ID,
        absent_production_paths={name:str(path) for name,path in paths.items()},global_lock_absent=True,
        checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=EXACT_CHECKPOINT,
        verified_asset_paths=len(identities),asset_sha256=identities,upstream_commit=upstream_head,
        upstream_files=len(manifest["source_sha256"]),protocol_sha256=protocol,
        resource_snapshot=resources,resource_stop_reason=reason,budget_limits=DEFAULT_LIMITS,guard_limits=LIMITS)
    assert reason is None,reason
    forbidden=[name for name in ("torch","pymunk","pygame","diffusion_policy","tdyssey.wm_image_measurement_v2") if name in sys.modules]
    assert not forbidden,forbidden
    record.update(passed=True,forbidden_runtime_modules_imported=forbidden)
except BaseException as error:
    record["error"]=f"{type(error).__name__}: {error}"
    raise
finally:
    with Path(__file__).with_name("preflight.json").open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(record,stream,sort_keys=True,indent=2);stream.write("\n")
print(json.dumps({key:record[key] for key in ("passed","reviewed_commit","verified_asset_paths","resource_snapshot")}))
