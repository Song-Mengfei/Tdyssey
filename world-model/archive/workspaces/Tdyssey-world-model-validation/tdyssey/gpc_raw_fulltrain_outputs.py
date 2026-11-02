"""Fixed raw T+L exports and display plates; no training/decoder implementation.

Import is standard-library-only. Root owns execution and the aggregate run
budget. This module has exactly two64MiB weight slots, sixteen256KiB plates,
one1MiB overview and five128KiB metadata slots, all immutable and below1GiB /
16MiB metadata. It never scans mutable guardian/ledger directories for quota.
"""
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

from .official_world_model import INITIAL_SHA256, RAW_CONTRACT, model_profile

OUTPUT_NAME = "gpc-raw-tl-fulltrain-20261026"
UPDATES = (500, 2328)
WEIGHT_MAX_BYTES = 64 * 1024**2
JSON_MAX_BYTES = 128 * 1024
PANEL_MAX_BYTES = 256 * 1024
OVERVIEW_MAX_BYTES = 1024**2
OUTPUT_MAX_BYTES = 1024**3
METADATA_MAX_BYTES = 16 * 1024**2
MAX_MODULE_BYTES = 2*WEIGHT_MAX_BYTES + 16*PANEL_MAX_BYTES + OVERVIEW_MAX_BYTES + 5*JSON_MAX_BYTES
assert MAX_MODULE_BYTES <= OUTPUT_MAX_BYTES and 5*JSON_MAX_BYTES <= METADATA_MAX_BYTES


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _ordinary_path(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise ValueError("Output aliases are not accepted")
    return path


def _root(output_root):
    root = _ordinary_path(output_root)
    if root.name != OUTPUT_NAME:
        raise ValueError("Use the fixed dedicated raw T+L output root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _identity(path):
    path = _ordinary_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Output must be an ordinary single-link file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    after = path.stat()
    if any(getattr(before,key) != getattr(after,key) for key in ("st_dev","st_ino","st_size","st_mtime_ns")):
        raise OSError("Output changed while sealing its identity")
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def _exclusive(path):
    path = _ordinary_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,"O_BINARY",0), 0o600)
    return os.fdopen(descriptor, "wb")


def _write_json(path, value):
    payload = _json(value)
    if len(payload) > JSON_MAX_BYTES:
        raise ValueError("Fixed JSON metadata slot exceeded")
    return _write_bytes(path, payload, JSON_MAX_BYTES)


def _write_bytes(path, payload, maximum):
    if len(payload) > maximum:
        raise ValueError("Fixed output slot exceeded before publication")
    with _exclusive(path) as stream:
        if stream.write(payload) != len(payload):
            raise OSError("Short immutable output write; partial file remains spent")
        stream.flush()
        os.fsync(stream.fileno())
    return _identity(path)


class _BoundedFile:
    """Bound every serialization write, including seek-back writes."""
    def __init__(self, stream, maximum):
        self.stream, self.maximum = stream, maximum
    def write(self, value):
        if self.stream.tell()+len(value) > self.maximum:
            raise OSError("Model-only checkpoint exceeded its fixed64MiB slot")
        count = self.stream.write(value)
        if count != len(value):
            raise OSError("Short checkpoint write; no retry or deletion")
        return count
    def seek(self, offset, whence=0):
        if whence != 0 or not 0 <= offset <= self.maximum:
            raise OSError("Checkpoint seek outside fixed output slot")
        return self.stream.seek(offset)
    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()


def _provenance(value):
    if (not isinstance(value,dict) or any(not isinstance(value.get(key),dict) or not value[key]
            for key in ("data_identity","list_identity","source_identity"))):
        raise ValueError("Export needs bound data/list/source identity dictionaries")
    if value.get("init_sha256",INITIAL_SHA256) != INITIAL_SHA256:
        raise ValueError("Raw T+L initialization identity differs")
    result = deepcopy(value)
    if len(_json(result)) > 32768:
        raise ValueError("Export provenance exceeds32KiB")
    return result


def _names(root, updates):
    return root/f"raw-tl-step-{updates:04d}.pt", root/f"raw-tl-step-{updates:04d}.json"


def _export_info(root, updates):
    checkpoint, sidecar = _names(root,updates)
    if sidecar.stat().st_size > JSON_MAX_BYTES:
        raise ValueError("Saved sidecar exceeds its fixed slot")
    metadata = json.loads(sidecar.read_bytes())
    weight = _identity(checkpoint)
    if (metadata["updates"] != updates or metadata["init_sha256"] != INITIAL_SHA256
            or metadata["checkpoint"] != weight or metadata["raw_contract"] != RAW_CONTRACT
            or metadata["train_shapes"] != ["T","L"]):
        raise ValueError("Existing fixed checkpoint/sidecar identity differs")
    side = _identity(sidecar)
    return dict(updates=updates,checkpoint=weight,sidecar=side,
        profile=model_profile("official",checkpoint,expected_sha256=weight["sha256"],
            sidecar=sidecar,expected_sidecar_sha256=side["sha256"],label=f"Raw T+L / step{updates}"))


def export_checkpoint(model, output_root, updates, provenance):
    """Future execution: two immutable plain state_dict files, never an optimizer bundle.

    The2328 manifest is published immediately after that export, independently
    of later diagnostic success. No checkpoint is selected using its scores.
    A serialization/IO failure leaves the claimed file in place and propagates.
    """
    if type(updates) is not int or updates not in UPDATES:
        raise ValueError("Only the fixed500 and2328 checkpoints may be exported")
    provenance = _provenance(provenance)
    root = _root(output_root)
    checkpoint, sidecar = _names(root,updates)
    if checkpoint.exists() or sidecar.exists() or (updates==2328 and (root/"raw-tl-final.json").exists()):
        raise FileExistsError("Fixed checkpoint slot is already spent")
    import torch
    values = model.state_dict()
    if (not isinstance(values,dict) or not values or any(not isinstance(key,str) or not key.startswith("inner_model.")
            or not isinstance(value,torch.Tensor) for key,value in values.items())):
        raise ValueError("Export accepts only the pinned model state_dict")
    if sum(value.numel()*value.element_size() for value in values.values()) > WEIGHT_MAX_BYTES:
        raise ValueError("Model tensors exceed the fixed checkpoint slot")
    state = {name:value.detach().cpu() for name,value in values.items()}
    try:
        if any(not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("Nonfinite model-only checkpoint")
        with _exclusive(checkpoint) as stream:
            stream.flush();os.fsync(stream.fileno())  # Permanent, exclusive slot before serialization.
            torch.save(state,_BoundedFile(stream,WEIGHT_MAX_BYTES))
            stream.flush();os.fsync(stream.fileno())
    finally:
        state.clear()
    weight = _identity(checkpoint)
    _write_json(sidecar,dict(kind="gpc_raw_tl_model_only.v1",updates=updates,init_sha256=INITIAL_SHA256,
        train_shapes=["T","L"],input_transform="identity",input_contract=RAW_CONTRACT["version"],
        raw_contract=deepcopy(RAW_CONTRACT),t1=False,optimizer_state=False,rng_state=False,
        checkpoint=weight,provenance=provenance,selection="fixed step; no score-based checkpoint selection"))
    exported = _export_info(root,updates)
    if updates==2328:
        earlier = _export_info(root,500)
        first_metadata = json.loads(Path(earlier["sidecar"]["path"]).read_bytes())
        if any(first_metadata["provenance"][key] != provenance[key] for key in
               ("data_identity","list_identity","source_identity")):
            raise ValueError("The500 and2328 exports have different data/list/source bindings")
        exported["final_manifest"] = _write_json(root/"raw-tl-final.json",dict(kind="gpc_raw_tl_fixed_final.v1",
            default_updates=2328,final_updates=2328,selection="fixed2328; not validation selection",
            diagnostic_status="unknown; inspect the separate diagnostic outputs",
            initial_sha256=INITIAL_SHA256,train_shapes=["T","L"],raw_contract=deepcopy(RAW_CONTRACT),
            exports={"500":earlier,"2328":deepcopy(exported)}))
    return exported


def write_panels(output_root, cases, results):
    """Future execution: sixteen target/initial/raw500/raw2328 panels and overview.

    cases is a fixed ordered list of16 {case_id,target,...}; results maps each
    case_id to {initial,raw500,raw2328,metrics(optional)}. Metrics are supplied by
    root; this function computes neither MAE/loss nor decoder/pose/coverage.
    Clipping here affects previews only and never the supplied arrays.
    """
    if not isinstance(cases,(list,tuple)) or len(cases)!=16 or not isinstance(results,dict):
        raise ValueError("Exactly the fixed sixteen diagnostic cases are required")
    ids = [case.get("case_id") for case in cases]
    if (any(not isinstance(key,str) or re.fullmatch(r"[A-Za-z0-9_.-]{1,64}",key) is None for key in ids)
            or len(set(ids))!=16 or set(results)!=set(ids)):
        raise ValueError("Fixed diagnostic case identities differ")
    metrics = {key:deepcopy(results[key].get("metrics",{})) for key in ids}
    if len(_json(metrics)) > 65536:
        raise ValueError("Supplied diagnostic metrics exceed64KiB")
    import numpy as np
    from PIL import Image, ImageDraw
    for case in cases:
        row = results[case["case_id"]]
        if set(row)-{"initial","raw500","raw2328","metrics"} or not {"initial","raw500","raw2328"} <= set(row):
            raise ValueError("Diagnostic needs exactly the three declared model predictions")
        for key,frame in (("target",case["target"]), *((name,row[name]) for name in ("initial","raw500","raw2328"))):
            if (not isinstance(frame,np.ndarray) or frame.dtype!=np.float32 or frame.shape!=(3,96,96)
                    or not np.isfinite(frame).all()):
                raise ValueError("Diagnostic arrays must be finite float32 CHW96")
            if key=="target" and ((frame<0).any() or (frame>1).any()):
                raise ValueError("Diagnostic target must be original real RGB [0,1]")
    root = _root(output_root)
    folder = root/"raw-diagnostic-panels"
    # This immutable claim makes a failed/partial plate export non-retryable.
    _write_json(folder/"slot.json",dict(kind="gpc_raw_tl_panel_slot.v1",case_ids=ids,
        maximum_panels=16,panel_max_bytes=PANEL_MAX_BYTES,overview_max_bytes=OVERVIEW_MAX_BYTES))
    overview = Image.new("RGB",(4*192,4*60),"white")
    records = []
    for index,case in enumerate(cases):
        key,row = case["case_id"],results[case["case_id"]]
        sheet = Image.new("RGB",(4*96,120),"white")
        draw = ImageDraw.Draw(sheet)
        for column,(label,frame) in enumerate((("target",case["target"]),("initial",row["initial"]),
                                              ("raw500",row["raw500"]),("raw2328",row["raw2328"]))):
            pixels = np.rint(np.clip(frame,0,1)*255).astype(np.uint8).transpose(1,2,0)
            sheet.paste(Image.fromarray(pixels),(column*96,0))
            draw.text((column*96+2,98),label,fill="black")
        draw.text((2,109),key,fill="black")
        buffer = io.BytesIO();sheet.save(buffer,format="PNG")
        identity = _write_bytes(folder/f"{index:02d}-{key}.png",buffer.getvalue(),PANEL_MAX_BYTES)
        overview.paste(sheet.resize((192,60)),((index%4)*192,(index//4)*60))
        records.append(dict(case_id=key,panel=identity,metrics=metrics[key]))
    buffer = io.BytesIO();overview.save(buffer,format="PNG")
    overview_image = _write_bytes(folder/"overview.png",buffer.getvalue(),OVERVIEW_MAX_BYTES)
    report = dict(kind="gpc_raw_tl_fixed_diagnostic_panels.v1",cases=16,columns=["target","initial","raw500","raw2328"],
        original_colour=True,t1=False,display_clip_only=True,overview=overview_image,rows=records,
        measurements="Only root-supplied raw MAE/loss; no decoder, pose, coverage or policy acceptance")
    identity = _write_json(folder/"overview.json",report)
    return dict(panels=records,overview_image=overview_image,overview=identity,diagnostic_only=True)


def launch_demo(repo_root, updates=2328, *, runner=None):
    """Future double-click entry: declared trained weights only, never a fallback.

    The small final manifest supplies actual file hashes. Relative source
    discovery uses this checkout / its existing sibling only; no author path.
    """
    if type(updates) is not int or updates not in UPDATES:
        raise ValueError("Select the fixed2328 final or explicit500 diagnostic")
    repo = Path(repo_root).absolute()
    root = _ordinary_path(repo/"artifacts"/OUTPUT_NAME)
    manifest_path = _ordinary_path(root/"raw-tl-final.json")
    if not manifest_path.is_file() or manifest_path.stat().st_size > JSON_MAX_BYTES:
        raise FileNotFoundError("Raw T+L final manifest is missing/invalid; no Initial fallback")
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get("kind") != "gpc_raw_tl_fixed_final.v1" or manifest.get("default_updates") != 2328
            or manifest.get("final_updates") != 2328 or manifest.get("initial_sha256") != INITIAL_SHA256
            or set(manifest.get("exports",{})) != {"500","2328"}):
        raise ValueError("Not the fixed raw T+L final manifest")
    selected = manifest["exports"][str(updates)]
    if selected.get("updates") != updates:
        raise ValueError("Requested checkpoint step differs from its final manifest")
    checkpoint, sidecar = _names(root,updates)
    for path,label in ((checkpoint,"checkpoint"),(sidecar,"sidecar")):
        actual,declared = _identity(path),selected[label]
        if any(actual[key] != declared[key] for key in ("bytes","sha256")):
            raise ValueError("Raw T+L "+label+" changed; no fallback")
    def source(env, candidates):
        declared = os.environ.get(env)
        choices = [Path(declared)] if declared else candidates
        found = next((path for path in choices if path.is_dir()),None)
        if found is None:raise FileNotFoundError("Existing source missing; set "+env)
        return found.absolute()
    gpc = source("TDYSSEY_GPC_ROOT",[repo/"third_party/gpc",repo/"third_party/gpc_code",repo.parent/"Tdyssey/third_party/gpc_code"])
    official = source("TDYSSEY_OFFICIAL_SOURCE",[repo/"third_party/diffusion_policy",repo.parent/"Tdyssey/third_party/diffusion_policy"])
    import sys
    arguments = [sys.executable,"-B",str(repo/"scripts/visualize_world_model.py"),
        "--checkpoint",str(checkpoint),"--checkpoint-sha256",selected["checkpoint"]["sha256"],
        "--sidecar",str(sidecar),"--sidecar-sha256",selected["sidecar"]["sha256"],
        "--checkpoint-kind","official","--checkpoint-source",f"Raw T+L fixed step{updates}; diagnostics separate",
        "--model-label","Raw T+L",
        "--gpc-root",str(gpc),"--source-root",str(official)]
    if runner is None:
        import subprocess
        runner = subprocess.run
    return runner(arguments,cwd=repo,check=False).returncode


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Launch the fixed raw T+L checkpoint; never trains",allow_abbrev=False)
    parser.add_argument("--launch-step",type=int,choices=UPDATES,default=2328)
    args = parser.parse_args(argv)
    try:
        return launch_demo(Path(__file__).resolve().parents[1],args.launch_step)
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
