"""Concrete saved-data IO and guarded lazy truth for formal DEV36 validation.

Preparation/load never construct NativeRGBDataset, render, or load a model.
Only a future authorized callback can invoke the native predictor; H4 truth is
rendered separately, after raw rollout persistence, and never enters feedback.
"""
from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SUPERVISED = ROOT / "artifacts/native-rgb-supervised-20261029"
PINNED = dict(
    manifest="d5cb1be08f1bdb46f03e856277a08478fea20c33899a70c6c448dedfd8f88e3e",
    audit="dbe53432c13bf32a0efb4e4704d93319063d1096a0e05bb54ed9bd72d1e3e1d4",
    development_windows="8f3e13bf7a76fb7b77da430ff11c82f5f5ac7f0f80ee3d219059f1547c3156de",
    paired_batches="cf4553fd2001a7878f8c423e969b8c0b4f799e5ccca901fc918d6d0058666843",
    cached_truth="13cb15e4f8652d070d793cb44713c4279fb08c2acb7d566b89efbedbe5b6be8c",
    cached_h1="60e337188ea1a1aaee6897164f44aac2019c83e035aa5f4b1878d68df44c91b2",
    cached_h1_metadata="fe4b2df41b87929aaf4c3b5d78f296a4d990929df8d1f9655fcda0475db33b3c")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def _state_hash(state):
    import numpy as np
    value = np.ascontiguousarray(state)
    return _hash(dict(dtype=str(value.dtype), shape=list(value.shape), bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest()))


def _reference_path(manifest, entry, key):
    base = Path(manifest["base_root"]).resolve()
    relative = Path(entry[key])
    path = (base / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(base / "stage2c/data"):
        raise ValueError("Validation reference escapes the fixed2C cohort")
    return path


def _metadata_reference(joint_root, *, expected_manifest_sha256, expected_audit_sha256):
    """Lightweight .entries/.windows/.trajectory interface, no renderer method."""
    import numpy as np
    root = Path(joint_root).resolve()
    if _sha(root/"manifest.json") != expected_manifest_sha256 or _sha(root/"physics-audit.json") != expected_audit_sha256:
        raise ValueError("Joint validation manifest/audit hash differs")
    manifest, audit = _json(root/"manifest.json"), _json(root/"physics-audit.json")
    if (manifest.get("kind") != "native-rgb-supervised-reference-data-v1"
            or audit.get("physics_collection_gate_passed") is not True
            or audit.get("dataset_manifest_sha256") != expected_manifest_sha256
            or audit.get("dataset_npz_sha256_verified") is not True):
        raise ValueError("Only the passed hash-bound formal reference data is allowed")
    entries = [entry for entry in manifest["entries"] if entry["split"] == "seen_val"]
    windows = []
    for i, entry in enumerate(entries):
        if entry.get("stage") != "2C" or entry.get("cohort") != "2C":
            raise ValueError("Seen validation must contain only independent2C trajectories")
        for key, digest in (("file", "sha256"), ("metadata_file", "metadata_sha256")):
            if _sha(_reference_path(manifest, entry, key)) != entry[digest]:
                raise ValueError("Validation trajectory/metadata hash differs")
        with np.load(_reference_path(manifest, entry, "file"), allow_pickle=False) as source:
            valid = source["valid_transition"]
            windows.extend((i, anchor) for anchor in range(3, len(valid)) if valid[anchor-3:anchor+1].all())
    cache = {}

    def trajectory(index):
        if type(index) is not int or not 0 <= index < len(entries):
            raise ValueError("Validation trajectory index outside frozen catalogue")
        if cache.get("index") != index:
            entry = entries[index]
            path = _reference_path(manifest, entry, "file")
            if _sha(path) != entry["sha256"]:
                raise ValueError("Validation compact trajectory changed")
            with np.load(path, allow_pickle=False) as source:
                arrays = {key:source[key].copy() for key in ("states", "actions", "valid_transition")}
            cache.clear()
            cache.update(index=index, arrays=arrays)
        return entries[index], cache["arrays"]

    return SimpleNamespace(root=root, manifest=manifest, audit=audit, split="seen_val", entries=entries,
                           windows=windows, trajectory=trajectory, cache_scope="one compact trajectory; no RGB")


def _cached_truth(path, window_ids):
    import numpy as np
    with np.load(path, allow_pickle=False) as source:
        truth = {key:source[key].copy() for key in source.files}
    if list(truth["window_ids"]) != list(window_ids):
        raise ValueError("Cached real histories/H1 truth order differs")
    if (truth["histories"].shape != (36,4,3,96,96) or truth["true"].shape != (36,3,96,96)
            or not np.array_equal(truth["last"], truth["histories"][:,-1])
            or not np.array_equal(truth["hold"], truth["last"])):
        raise ValueError("Original real image/hold timeline differs")
    for key in ("histories", "last", "true", "hold", "last_full_mask", "true_full_mask"):
        value = truth[key]
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("Cached real RGB/full mask must stay finite in[0,1]")
    return truth


def prepare_validation_inputs(output, *, joint_root=None, development_windows_path=None,
        paired_batches_path=None, cached_truth_path=None, cached_h1_path=None):
    """CPU-only preparation of frozen refs/actions7; does not render any image."""
    import numpy as np
    from .native_rgb_formal_validation import build_validation_plan, load_h1_cache
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    joint = Path(joint_root or SUPERVISED/"stage2c/joint-data").resolve()
    files = dict(manifest=joint/"manifest.json", audit=joint/"physics-audit.json",
        development_windows=Path(development_windows_path or SUPERVISED/"stage3c/dev36-windows.json").resolve(),
        paired_batches=Path(paired_batches_path or SUPERVISED/"stage3c/paired-batches.json").resolve(),
        cached_truth=Path(cached_truth_path or SUPERVISED/"stage3b/truth/dev36.npz").resolve(),
        cached_h1=Path(cached_h1_path or SUPERVISED/"stage3b/predictions/frozen-dev36.npz").resolve())
    files["cached_h1_metadata"] = files["cached_h1"].with_suffix(".json")
    for name, path in files.items():
        if _sha(path) != PINNED[name]:
            raise ValueError("Original fixed validation input changed: "+name)
    if (output/"validation-inputs.json").exists() or (output/"validation-plan.json").exists() or (output/"actions7.npz").exists():
        raise FileExistsError("Never overwrite frozen validation IO preparation")
    windows, paired = _json(files["development_windows"]), _json(files["paired_batches"])
    ids = [row["window_id"] for row in windows]
    truth = _cached_truth(files["cached_truth"], ids)
    dataset = _metadata_reference(joint, expected_manifest_sha256=PINNED["manifest"], expected_audit_sha256=PINNED["audit"])
    plan = build_validation_plan(dataset, windows, paired, real_histories=truth["histories"])
    if len(dataset.entries) != 162 or len(dataset.windows) != 9882 or not np.array_equal(plan["absolute_actions"][:,:4], truth["actions_absolute"]):
        raise ValueError("Original seen_val catalogue/first four actions changed")
    metadata = []
    for row in plan["window_metadata"]:
        entry, arrays = dataset.trajectory(row["trajectory_index"])
        target = deepcopy(row)
        target.pop("h4_target_state")
        target.update(trajectory_id=entry["trajectory_id"], source_file=entry["file"],
            h4_target_state_sha256=_state_hash(arrays["states"][row["h4_target_state_index"]]))
        metadata.append(target)
    h1 = load_h1_cache(files["cached_h1"], expected_sha256=PINNED["cached_h1"], window_ids=ids, batches=plan["batches"])
    h1_metadata = _json(files["cached_h1_metadata"])
    if (h1_metadata["updates"] != 0 or h1_metadata["postprocessing"] is not False
            or h1_metadata["unmodified_native_RGB"] is not True or h1_metadata["horizon"] != 1
            or h1_metadata["paired_batches_sha256"] != PINNED["paired_batches"]
            or any(any(row[key] != planned[key] for key in ("batch_index","batch_seed","batch_size","positions","window_ids"))
                   for row, planned in zip(h1_metadata["batch_records"],plan["batches"]))):
        raise ValueError("Published cached H1 model/batch/seed provenance differs")
    with (output/"actions7.npz").open("xb") as stream:
        np.savez_compressed(stream, absolute_actions=plan["absolute_actions"], window_ids=np.asarray(ids))
    saved = dict(kind="native-rgb-formal-dev36-io-plan.v1", windows=deepcopy(windows), window_ids=ids,
        window_metadata=metadata, batches=deepcopy(plan["batches"]), actions_file="actions7.npz",
        gallery_template=deepcopy(plan["gallery_template"]), scoring_targets_are_deferred=True,
        future_truth_in_feedback=False, hold_all_horizons="original cached s[a]", preparation_render_calls=0,
        preparation_physics_controls=0, validation_trajectories=162, validation_h1_windows=9882)
    _write(output/"validation-plan.json", saved)
    references = {name:dict(path=str(path),sha256=PINNED[name]) for name,path in files.items()}
    identity = dict(plan_sha256=_sha(output/"validation-plan.json"), actions7_sha256=_sha(output/"actions7.npz"),
        references=references, window_ids_sha256=_hash(ids), original_h1_model_updates=0,
        original_h1_parameter_digest=h1_metadata["parameter_digest"], no_new_H1_baseline_prediction=True,
        no_truth_render_before_future_authorized_rollout=True)
    _write(output/"validation-inputs.json", dict(kind="native-rgb-formal-dev36-io-inputs.v1",
        input_identity=identity, identity_sha256=_hash(identity), real_H1_baseline_status_counts=dict(Counter(h1["status"].tolist()))))
    return dict(output=str(output), identity_sha256=_hash(identity), windows=36, actions_shape=list(plan["absolute_actions"].shape),
        h4_target_state_references=36, cached_h1_referenced=True, preparation_render_calls=0, dataset_instances=0,
        model_calls=0, native_sampler_calls=0, physics_controls=0)


def load_inputs(output, *, expected_identity=None):
    """Load hash-bound caches and action metadata; no dataset/model/rendering."""
    import numpy as np
    from .native_rgb_formal_validation import load_h1_cache
    output = Path(output).resolve()
    binding, saved = _json(output/"validation-inputs.json"), _json(output/"validation-plan.json")
    identity = binding["input_identity"]
    if _hash(identity) != binding["identity_sha256"] or (expected_identity is not None and binding["identity_sha256"] != expected_identity):
        raise ValueError("Frozen validation input identity differs")
    if _sha(output/"validation-plan.json") != identity["plan_sha256"] or _sha(output/"actions7.npz") != identity["actions7_sha256"]:
        raise ValueError("Prepared validation plan/actions changed")
    if set(identity["references"]) != set(PINNED):
        raise ValueError("Original validation reference set differs")
    for name, reference in identity["references"].items():
        if reference["sha256"] != PINNED[name] or _sha(reference["path"]) != reference["sha256"]:
            raise ValueError("Original validation reference changed: "+name)
    truth = _cached_truth(identity["references"]["cached_truth"]["path"], saved["window_ids"])
    with np.load(output/"actions7.npz", allow_pickle=False) as source:
        actions, ids = source["absolute_actions"].copy(), list(source["window_ids"])
    if actions.shape != (36,7,2) or ids != saved["window_ids"] or not np.array_equal(actions[:,:4], truth["actions_absolute"]):
        raise ValueError("Frozen seven-action/real-history order changed")
    plan = dict(saved, real_histories=truth["histories"], absolute_actions=actions)
    h1 = load_h1_cache(identity["references"]["cached_h1"]["path"], expected_sha256=PINNED["cached_h1"],
                       window_ids=ids, batches=plan["batches"])
    return dict(plan=plan, cached_h1=h1, cached_truth=truth, input_identity=deepcopy(identity),
                identity_sha256=binding["identity_sha256"])


def make_runtime_bridges(inputs, *, execution_guard, raw_writer):
    """Concrete callbacks for formal_validation.make_validation_callback."""
    if not callable(execution_guard) or not callable(raw_writer):
        raise ValueError("Future authorization guard and raw persistence are mandatory")
    plan, truth = inputs["plan"], inputs["cached_truth"]
    state = dict(context=None, update=None, rollout=None, guard_calls=0, h4_truth_render_calls=0)

    def window_entry(dataset, row):
        trajectory_index, anchor = dataset.windows[row["dataset_index"]]
        entry = dataset.entries[trajectory_index]
        if (dataset.split != "seen_val" or entry["split"] != "seen_val"
                or entry["trajectory_id"] != row["trajectory_id"] or anchor != row["anchor"]
                or entry["sha256"] != row["npz_sha256"] or entry["metadata_sha256"] != row["metadata_sha256"]):
            raise ValueError("Future validation source trajectory/anchor differs")
        return trajectory_index, entry

    def guard(context):
        state.update(context=None, update=None, rollout=None)
        execution_guard(context)
        for row in plan["window_metadata"]:
            window_entry(context["validation"], row)
        state.update(context=context, update=context["progress"]["updates"], rollout=None, guard_calls=state["guard_calls"]+1)

    def authorized(context):
        if state["context"] is not context or state["update"] != context["progress"]["updates"]:
            raise PermissionError("Truth/predictor access requires this callback's passed execution guard")

    def predictor_factory(context):
        authorized(context)
        from .native_rgb_formal_validation import native_predictor_factory
        return native_predictor_factory(context)

    def writer(result, context):
        authorized(context)
        if result["window_ids"] != plan["window_ids"] or result["batches"] != plan["batches"]:
            raise ValueError("Completed raw rollout identity differs")
        artifact = raw_writer(result, context)
        state["rollout"] = dict(evaluation_complete=result["evaluation_complete"], fatal_error=deepcopy(result["fatal_error"]))
        return artifact

    def truth_provider(horizon, positions, window_ids, *, context, rollout_token):
        import numpy as np
        authorized(context)
        token, result = rollout_token, state["rollout"]
        if (result is None or token.get("authorized") is not True or token.get("rollout_finished") is not True
                or token.get("update") != state["update"] or token.get("window_ids") != plan["window_ids"]
                or token.get("evaluation_complete") != result["evaluation_complete"] or token.get("fatal_error") != result["fatal_error"]):
            raise PermissionError("Offline truth is available only after authorized raw rollout persistence")
        if type(horizon) is not int or horizon not in (1,4) or not positions or len(set(positions)) != len(positions) or any(type(i) is not int or not 0 <= i < 36 for i in positions):
            raise ValueError("Invalid fixed truth horizon/positions")
        if list(window_ids) != [plan["window_ids"][i] for i in positions]:
            raise ValueError("Truth-provider order differs")
        if horizon == 1:
            return dict(horizon=1,window_ids=list(window_ids),rgb=truth["true"][positions].copy(),
                full_object_mask=truth["true_full_mask"][positions].copy(), last_full_mask=truth["last_full_mask"][positions].copy())
        dataset = context["validation"]
        rgb, masks = [], []
        for i in positions:
            row = plan["window_metadata"][i]
            trajectory_index, entry = window_entry(dataset, row)
            entry, arrays = dataset.trajectory(trajectory_index)
            saved_state = np.asarray(arrays["states"])[row["h4_target_state_index"]].copy()
            if _state_hash(saved_state) != row["h4_target_state_sha256"]:
                raise ValueError("Saved H4 target state differs from the frozen reference")
            layers = dataset.render_layers(entry, saved_state)
            image, mask = np.asarray(layers["rgb"]), np.asarray(layers["full_object_mask"])
            if (image.shape != (3,96,96) or image.dtype != np.uint8 or mask.shape != (96,96)
                    or not np.isfinite(mask).all() or mask.min()<0 or mask.max()>1):
                raise ValueError("Truth renderer returned an invalid original RGB/fullmask layer")
            rgb.append(image.astype(np.float32)/255)
            masks.append(mask.astype(np.float32))
            state["h4_truth_render_calls"] += 1
        return dict(horizon=4,window_ids=list(window_ids),rgb=np.stack(rgb),full_object_mask=np.stack(masks),
                    last_full_mask=truth["last_full_mask"][positions].copy())

    return dict(execution_guard=guard,predictor_factory=predictor_factory,truth_provider=truth_provider,raw_writer=writer,
        io_stats=lambda:dict(guard_calls=state["guard_calls"],h4_truth_render_calls=state["h4_truth_render_calls"],
                             offline_truth_after_rollout_only=True))
