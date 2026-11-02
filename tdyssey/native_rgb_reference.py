"""Hash-bound zero-copy formal TRAIN/seen_val view. No model or physics step."""

from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

from .native_rgb_env import CONTROLLER,PROTOCOL_SHA256,STATE_COLUMNS

KIND="native-rgb-supervised-reference-data-v1"
TRAIN_VIEW_SHA256="23865f38e19eb3b3fddf14b17c96f06586468bd04516e2403aa743f42c557d7e"
EXPECTED_COHORT_COUNTS={"2A":162,"2B":234,"2C":162}
REFERENCE_LETTERS=tuple("ABCEFHIJLMOPSTUVXY")
PREFIX={"2A":"stage2a/data","2B":"stage2b/data","2C":"stage2c/data"}
SPLIT={"2A":"train","2B":"train","2C":"seen_val"}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")


def _safe(base,relative,prefix):
    path=Path(relative)
    if path.is_absolute():raise ValueError("Absolute reference path forbidden")
    target=(base/path).resolve();allowed=(base/prefix).resolve()
    if not allowed.is_relative_to(base) or not target.is_relative_to(allowed):raise ValueError("Reference escapes approved cohort/root")
    return target


def _base(joint_root,manifest):
    joint=Path(joint_root).resolve();base=joint.parents[1]
    if joint.name!="joint-data" or joint.parent.name!="stage2c" or Path(manifest["base_root"]).resolve()!=base:
        raise ValueError("Joint reference root must be supervised_root/stage2c/joint-data")
    return base


def resolve_reference_entry(joint_root,manifest,entry,key="file"):
    base=_base(joint_root,manifest);cohort=entry.get("cohort")
    if cohort not in PREFIX or entry.get("stage")!=cohort or entry.get("split")!=SPLIT[cohort]:
        raise ValueError("Reference split/stage/cohort mismatch")
    return _safe(base,entry[key],PREFIX[cohort])


def _verify_source(base,cohort,source):
    prefix=PREFIX[cohort];folder=prefix.split("/")[0]
    expected=dict(manifest_file=prefix+"/manifest.json",audit_file=folder+"/physics-audit.json",source_identity_file=folder+"/source-identity.json")
    for key,relative in expected.items():
        if source.get(key)!=relative or sha(_safe(base,relative,folder))!=source.get(key.replace("_file","_sha256")):
            raise ValueError("Source manifest/audit/identity changed")
    manifest=_read(base/source["manifest_file"]);audit=_read(base/source["audit_file"])
    if manifest.get("split")!=SPLIT[cohort] or manifest.get("stage")!=cohort:raise ValueError("Forbidden source split/cohort")
    if manifest.get("source_identity_sha256")!=source["source_identity_sha256"]:raise ValueError("Collector source identity not bound")
    if (manifest.get("controller")!=CONTROLLER or manifest.get("protocol_sha256")!=PROTOCOL_SHA256
            or list(manifest.get("state_columns",()))!=list(STATE_COLUMNS)):
        raise ValueError("Fixed physical protocol/state columns changed")
    if audit.get("manifest_sha256",audit.get("dataset_manifest_sha256"))!=source["manifest_sha256"]:
        raise ValueError("Original physical audit identifies another manifest")
    if audit.get("threshold_world_px")!=2 or audit.get("threshold_relaxed") is not False:raise ValueError("Physical threshold not frozen")
    identity=_read(base/source["source_identity_file"])
    active_physics=Path(manifest["source_root"]).resolve()
    for row in identity["files"]:
        snapshot=_safe(base,(Path(folder)/row["snapshot"]).as_posix(),folder)
        if sha(snapshot)!=row["sha256"] or snapshot.stat().st_size!=row["bytes"]:raise ValueError("Immutable collector snapshot changed")
        active=Path(row["source"]).resolve()
        if active.is_relative_to(active_physics) and sha(active)!=row["sha256"]:raise ValueError("Active official physics source changed")
    if len(manifest["entries"])!=EXPECTED_COHORT_COUNTS[cohort]:raise ValueError("Formal source cohort incomplete")
    return manifest,audit


def _validate_arrays(path,entry,metadata):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        states,actions,valid=z["states"],z["actions"],z["valid_transition"];n=len(actions)
        if n!=64 or entry.get("controls")!=n or states.shape!=(n+1,15) or actions.shape!=(n,2) or valid.shape!=(n,):
            raise ValueError("Recorded timeline/control alignment changed")
        sub,pen,contact=z["substep_penetrations"],z["penetrations"],z["contact"]
        if sub.shape!=(n,10,3) or pen.shape!=(n+1,3) or contact.shape!=(n,):raise ValueError("Physical evidence alignment changed")
        if not all(np.isfinite(v).all() for v in (states,actions,sub,pen,contact)) or (actions<0).any() or (actions>511).any() or (contact<0).any():
            raise ValueError("Nonfinite/invalid recorded state or action")
        if not np.array_equal(sub.max(axis=1),pen[1:]):raise ValueError("All100Hz substep evidence must match saved maxima")
        computed=sub.max(axis=(1,2))<=2
        if valid.dtype!=np.bool_ or not np.array_equal(computed,valid):raise ValueError("Physical validity flags differ from substeps")
        if sha_bytes(states[0].tobytes())!=entry["initial_scene_hash"]:raise ValueError("Full initial scene identity changed")
        if not np.array_equal(np.asarray(metadata.get("scene",{}).get("initial_state"),dtype=np.float64),states[0]):
            raise ValueError("Recorded metadata initial state differs from NPZ")
        if len(metadata.get("steps",[]))!=n:raise ValueError("Metadata timeline differs")
        request=metadata.get("request",{})
        if request.get("split")!=entry["split"] or request.get("seed")!=entry["seed"]:raise ValueError("Metadata split/seed differs")
        return dict(controls=n,invalid=int((~valid).sum()),h1=sum(bool(valid[t-3:t+1].all()) for t in range(3,n)),
            h4=sum(bool(valid[t-3:t+4].all()) for t in range(3,n-3)))


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def _inspect(base,manifest,require_gate):
    from .native_rgb_data import TRAIN_LETTERS,UNSEEN_LETTERS
    if (base/"stage2c/COLLECTION_RUNNING.lock").exists():raise RuntimeError("Validation collection still active; no hot data reads")
    train_file="stage2b/report/merged-train-manifest.json"
    if manifest.get("train_view_file")!=train_file or sha(_safe(base,train_file,"stage2b/report"))!=manifest.get("train_view_sha256"):
        raise ValueError("Frozen TRAIN reference view changed")
    if TRAIN_VIEW_SHA256 is not None and manifest["train_view_sha256"]!=TRAIN_VIEW_SHA256:raise ValueError("TRAIN differs from parent-frozen2B view")
    train_view=_read(base/train_file)
    if train_view.get("kind")!="native-rgb-formal-train-reference-view.v1" or Path(train_view["base_root"]).resolve()!=base:
        raise ValueError("Wrong TRAIN reference root/kind")
    originals={};source_audits={};physics_root=None
    for cohort in PREFIX:
        source,audit=_verify_source(base,cohort,manifest["reference_sources"][cohort]);source_audits[cohort]=audit
        originals[cohort]={entry["trajectory_id"]:entry for entry in source["entries"]}
        if len(originals[cohort])!=len(source["entries"]):raise ValueError("Duplicate original trajectory")
        current=Path(source["source_root"]).resolve()
        if physics_root is not None and current!=physics_root:raise ValueError("Cohort physics roots differ")
        physics_root=current
    if Path(manifest.get("physics_source_root","")).resolve()!=physics_root:raise ValueError("Explicit physics source root differs")
    if set(manifest.get("train_letters",()))!=set(TRAIN_LETTERS) or set(manifest.get("unseen_letters",()))!=set(UNSEEN_LETTERS):
        raise ValueError("Explicit seen/unseen shape directory changed")
    if manifest.get("controller")!=CONTROLLER or manifest.get("protocol_sha256")!=PROTOCOL_SHA256 or list(manifest.get("state_columns",()))!=list(STATE_COLUMNS):
        raise ValueError("Joint physical protocol/header differs")
    train_expected={(e["cohort"],e["trajectory_id"]):e for e in train_view["entries"]}
    seen_ids=set();scenes=set();seeds=set();summaries=[];seen_keys=set()
    for entry in manifest["entries"]:
        cohort=entry.get("cohort")
        if cohort not in PREFIX or entry.get("stage")!=cohort or entry.get("split")!=SPLIT[cohort] or entry.get("shape") not in TRAIN_LETTERS:
            raise ValueError("Probe/validation-to-TRAIN/unseen contamination forbidden")
        original=originals[cohort].get(entry["trajectory_id"])
        if original is None:raise ValueError("Unknown reference trajectory")
        for key in ("shape","seed","initial_scene_hash","split","stage","controls","sha256","metadata_sha256"):
            if entry.get(key)!=original.get(key):raise ValueError("Reference identity differs from source: "+key)
        for key in ("file","metadata_file"):
            if entry[key]!=(Path(PREFIX[cohort])/original[key]).as_posix():raise ValueError("Reference path differs from original")
        if cohort in ("2A","2B"):
            old=train_expected.get((cohort,entry["trajectory_id"]))
            if old is None or any(entry.get(key)!=old.get(key) for key in ("file","metadata_file","sha256","metadata_sha256","selector_version","split")):
                raise ValueError("TRAIN entry differs from frozen merged view")
        for key,seen in (("trajectory_id",seen_ids),("initial_scene_hash",scenes),("seed",seeds)):
            value=entry.get(key)
            if value is None or value in seen:raise ValueError("TRAIN/seen_val duplicate trajectory,seed or full scene")
            seen.add(value)
        path=_safe(base,entry["file"],PREFIX[cohort]);meta_path=_safe(base,entry["metadata_file"],PREFIX[cohort])
        if sha(path)!=entry["sha256"] or sha(meta_path)!=entry["metadata_sha256"]:raise ValueError("Original NPZ/metadata hash changed")
        metadata=_read(meta_path);version=metadata.get("scene",{}).get("selection_version")
        version=version or ("motion-room-actual-hit-prior-v1" if cohort=="2A" else None)
        if entry.get("selector_version")!=version:raise ValueError("Selector version differs from recorded scene")
        value=_validate_arrays(path,entry,metadata);summaries.append(dict(trajectory_id=entry["trajectory_id"],split=entry["split"],cohort=cohort,**value))
        seen_keys.add((cohort,entry["trajectory_id"]))
    expected={(c,t) for c,entries in originals.items() for t in entries}
    if seen_keys!=expected or set(train_expected)!={(c,t) for c,t in expected if c in ("2A","2B")}:
        raise ValueError("Joint view omitted or added original records")
    for cohort,audit in source_audits.items():
        selected=[row for row in summaries if row["cohort"]==cohort];controls=sum(row["controls"] for row in selected)
        if (audit.get("controls")!=controls or audit.get("audited_substeps")!=10*controls
                or audit.get("valid_h1_candidates")!=sum(row["h1"] for row in selected)
                or audit.get("valid_h4_candidates")!=sum(row["h4"] for row in selected)):
            raise ValueError("Original audit counts differ from actual recorded timeline")
    for split,planned in (("train",EXPECTED_COHORT_COUNTS["2A"]+EXPECTED_COHORT_COUNTS["2B"]),("seen_val",EXPECTED_COHORT_COUNTS["2C"])):
        actual=Counter(e["shape"] for e in manifest["entries"] if e["split"]==split)
        if actual!={letter:planned//len(REFERENCE_LETTERS) for letter in REFERENCE_LETTERS}:raise ValueError("Formal split letter quotas incomplete")
    passed=all(audit.get("passed") is True and audit.get("physics_collection_gate_passed") is True and audit.get("invalid_controls")==0 for audit in source_audits.values()) and not any(row["invalid"] for row in summaries)
    if require_gate and not passed:raise RuntimeError("Original physical gate failed; reference TRAIN prohibited")
    declared={name:sorted(e["trajectory_id"] for e in manifest["entries"] if e["split"]==name) for name in ("train","seen_val","unseen_test")}
    if manifest.get("splits")!=declared:raise ValueError("Explicit split directory differs from entries")
    return dict(physics_collection_gate_passed=passed,dataset_npz_sha256_verified=True,metadata_sha256_verified=True,
        source_snapshots_verified=True,split_isolation_verified=True,invalid_controls=sum(v["invalid"] for v in summaries),
        actual_controls=sum(v["controls"] for v in summaries),
        valid_h1_by_split={name:sum(v["h1"] for v in summaries if v["split"]==name) for name in ("train","seen_val")},
        valid_h4_by_split={name:sum(v["h4"] for v in summaries if v["split"]==name) for name in ("train","seen_val")},
        source_audit_sha256={cohort:manifest["reference_sources"][cohort]["audit_sha256"] for cohort in PREFIX},
        physics_controls_executed=0,model_calls=0,index_draws=0)


def verify_reference_dataset(joint_root,manifest=None,require_gate=True):
    joint=Path(joint_root).resolve();manifest=_read(joint/"manifest.json") if manifest is None else manifest
    if manifest.get("kind")!=KIND or manifest.get("adapter_source_sha256")!=sha(__file__):raise ValueError("Unknown/changed reference adapter identity")
    base=_base(joint,manifest);audit=_read(joint/"physics-audit.json")
    if audit.get("dataset_manifest_sha256")!=sha(joint/"manifest.json"):raise ValueError("Joint physical audit is stale")
    computed=_inspect(base,manifest,require_gate)
    if any(audit.get(key)!=value for key,value in computed.items()):raise ValueError("Joint identity/physical audit differs from original evidence")
    return manifest,audit


def build_joint_reference(supervised_root,output_root=None):
    """Call only after root confirms completed2C. Writes JSON only, zero NPZ copies."""
    from .native_rgb_data import TRAIN_LETTERS,UNSEEN_LETTERS
    base=Path(supervised_root).resolve();joint=Path(output_root or base/"stage2c/joint-data").resolve()
    if joint!=base/"stage2c/joint-data":raise ValueError("Joint output must be stage2c/joint-data")
    if (base/"stage2c/COLLECTION_RUNNING.lock").exists():raise RuntimeError("Validation collection still active")
    view_file="stage2b/report/merged-train-manifest.json";view=_read(_safe(base,view_file,"stage2b/report"));entries=[dict(e) for e in view["entries"]]
    sources={};physics_root=None
    for cohort,prefix in PREFIX.items():
        folder=prefix.split("/")[0];manifest_file=prefix+"/manifest.json";source=_read(_safe(base,manifest_file,folder))
        sources[cohort]=dict(manifest_file=manifest_file,manifest_sha256=sha(base/manifest_file),audit_file=folder+"/physics-audit.json",
            audit_sha256=sha(base/folder/"physics-audit.json"),source_identity_file=folder+"/source-identity.json",
            source_identity_sha256=sha(base/folder/"source-identity.json"))
        physics_root=source["source_root"]
        if cohort=="2C":
            for original in source["entries"]:
                entry=dict(original,cohort=cohort)
                entry["file"]=(Path(prefix)/original["file"]).as_posix();entry["metadata_file"]=(Path(prefix)/original["metadata_file"]).as_posix()
                metadata=_read(_safe(base,entry["metadata_file"],prefix));entry["selector_version"]=metadata.get("scene",{}).get("selection_version")
                entries.append(entry)
    manifest=dict(kind=KIND,base_root=str(base),entries=entries,reference_sources=sources,train_view_file=view_file,train_view_sha256=sha(base/view_file),
        controller=CONTROLLER,protocol_sha256=PROTOCOL_SHA256,state_columns=list(STATE_COLUMNS),physics_source_root=str(Path(physics_root).resolve()),
        source_root=str(Path(physics_root).resolve()),source_root_semantics="legacy collector physics source, never GPC model source",
        train_letters=list(TRAIN_LETTERS),unseen_letters=list(UNSEEN_LETTERS),
        splits={name:sorted(e["trajectory_id"] for e in entries if e["split"]==name) for name in ("train","seen_val","unseen_test")},
        adapter_source_sha256=sha(__file__),reference_only=True,npz_files_copied=0,training_consumption_authorized=False)
    report=_inspect(base,manifest,True)
    joint.mkdir(parents=True,exist_ok=False);_write(joint/"manifest.json",manifest)
    audit=dict(kind="native-rgb-joint-reference-audit-v1",dataset_manifest_sha256=sha(joint/"manifest.json"),**report)
    _write(joint/"physics-audit.json",audit)
    _write(joint/"identity-audit.json",dict(manifest_sha256=sha(joint/"manifest.json"),audit_sha256=sha(joint/"physics-audit.json"),
        source_manifest_sha256={cohort:sources[cohort]["manifest_sha256"] for cohort in PREFIX},**report))
    return dict(root=str(joint),manifest=manifest,audit=audit)


def validate_cpu_samples(joint_root,output,selections):
    """At most four predeclared direct Dataset items. No training iterator/model."""
    import numpy as np
    from .native_rgb_data import NativeRGBDataset
    selections=list(selections)
    if not 1<=len(selections)<=4:raise ValueError("CPU sample check allows1..4 predeclared windows")
    start=time.monotonic();datasets={};evidence=[];batches=defaultdict(list)
    for choice in selections:
        split=choice["split"]
        if split not in ("train","seen_val"):raise ValueError("Only formal train/seen_val CPU samples")
        if split not in datasets:datasets[split]=NativeRGBDataset(joint_root,split,cache_bytes=1024**2)
        data=datasets[split];i=next(i for i,e in enumerate(data.entries) if e["trajectory_id"]==choice["trajectory_id"]);anchor=int(choice["anchor"])
        index=data.windows.index((i,anchor));item=data[index];entry,arrays=data.trajectory(i)
        expected=arrays["actions"][anchor-3:anchor+1].astype(np.float32)/511*2-1
        if item["image"].shape!=(5,3,96,96) or item["action"].shape!=(4,2) or not np.array_equal(item["action"],expected):raise ValueError("Original GPC timeline/layout changed")
        if item["image"].dtype!=np.float32 or item["action"].dtype!=np.float32 or not np.isfinite(item["image"]).all() or item["image"].min()<0 or item["image"].max()>1:raise ValueError("CPU image dtype/range invalid")
        batches[split].append(item);evidence.append(dict(**choice,history_state_indices=list(range(anchor-3,anchor+1)),history_action_indices=list(range(anchor-3,anchor)),
            current_action_index=anchor,target_state_index=anchor+1,npz_sha256=entry["sha256"],metadata_sha256=entry["metadata_sha256"],cache_bytes=data._image_bytes,
            image_sha256=sha_bytes(item["image"].tobytes()),action_sha256=sha_bytes(item["action"].tobytes())))
    shapes={name:dict(image=list(np.stack([v["image"] for v in values]).shape),action=list(np.stack([v["action"] for v in values]).shape)) for name,values in batches.items()}
    result=dict(samples=evidence,batches=shapes,wall_seconds=time.monotonic()-start,physics_controls=0,model_calls=0,optimizer_updates=0,index_draws=0,
        access="predeclared direct Dataset indices, no training sampler/iterator",network_inputs_only=["image","action"])
    _write(output,result);return result
