"""Explicit metadata-only recovery across a proven atomic-JSON I/O-only patch.

Never changes model, optimizer, scheduler, scaler, RNG, sampler, progress or
config. The original checkpoint remains untouched and all spent cost remains.
"""
import argparse
import ast
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAINER = "tdyssey/native_rgb_training.py"
PROTECTED = ("model", "optimizer", "scaler", "scheduler", "rng", "order", "progress", "config")


def without_atomic_writer(source):
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "atomic_json"]
    if len(selected) != 1: raise ValueError("Exactly one atomic_json function is required")
    tree.body = [node for node in tree.body if node not in selected]
    return ast.dump(tree, include_attributes=False)


def state_digest(value):
    """Hash typed state values, including every CPU tensor byte."""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor" + str(item.dtype).encode() + repr(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy" + item.dtype.str.encode() + repr(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr): visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + str(len(item)).encode())
            for entry in item: visit(entry)
        else: digest.update(type(item).__name__.encode() + b":" + repr(item).encode() + b"\0")
    visit(value)
    return digest.hexdigest()


def migrate(checkpoint, source_snapshot, output, artifact_root):
    import torch
    from tdyssey.native_rgb_training import (CHECKPOINT_KIND, atomic_json, sha256,
                                             source_identity, load_checkpoint_payload)
    checkpoint, source_snapshot = Path(checkpoint).resolve(), Path(source_snapshot).resolve()
    output, artifact_root = Path(output).resolve(), Path(artifact_root).resolve()
    if not checkpoint.is_relative_to(artifact_root) or not output.is_relative_to(artifact_root):
        raise ValueError("Recovery paths must remain inside the explicitly named experiment")
    if output.exists() or output.with_suffix(".json").exists(): raise FileExistsError("Never overwrite a recovery")
    started = time.perf_counter()
    payload = load_checkpoint_payload(checkpoint)
    if payload.get("kind") != CHECKPOINT_KIND: raise ValueError("Expected an owned complete native checkpoint")
    old_identity, current = deepcopy(payload["identity"]), source_identity()
    previous = old_identity["source"]
    if previous["files"].keys() != current["files"].keys(): raise ValueError("Source inventory changed")
    changed = [name for name in previous["files"] if previous["files"][name] != current["files"][name]]
    if changed != [TRAINER]: raise ValueError("Only atomic-writer module identity may change: " + repr(changed))
    old_path, new_path = source_snapshot / TRAINER, ROOT / TRAINER
    if sha256(old_path) != previous["files"][TRAINER]: raise ValueError("Original source snapshot is not intact")
    old_text, new_text = old_path.read_text(encoding="utf-8"), new_path.read_text(encoding="utf-8")
    if without_atomic_writer(old_text) != without_atomic_writer(new_text):
        raise ValueError("A change outside atomic_json would alter training; refuse I/O migration")
    before = {key: state_digest(payload[key]) for key in PROTECTED}
    payload["identity"]["source"] = current
    migration = dict(kind="native-rgb-io-only-checkpoint-migration.v1",
        original_checkpoint=str(checkpoint), original_checkpoint_sha256=sha256(checkpoint),
        from_source_sha256=previous["files"][TRAINER], to_source_sha256=current["files"][TRAINER],
        unchanged_AST_except="atomic_json", reason="bounded PermissionError retry for Windows atomic replace",
        protected_state_sha256=before, optimizer_updates=0, budget_refunded=False,
        script_sha256=sha256(__file__), original_identity=old_identity)
    payload["io_compatibility_migration"] = migration
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream: torch.save(payload, stream)
    saved = load_checkpoint_payload(output)
    after = {key: state_digest(saved[key]) for key in PROTECTED}
    if before != after: raise RuntimeError("Protected mathematical training state changed during I/O migration")
    migration.update(migrated_checkpoint=str(output), migrated_checkpoint_sha256=sha256(output),
        all_protected_states_identical=True, protected_state_sha256_after=after,
        elapsed_seconds=time.perf_counter()-started, model_updates=saved["progress"]["updates"])
    atomic_json(output.with_suffix(".json"), migration)
    (output.parent / "atomic-writer-only.diff").write_text("".join(difflib.unified_diff(
        old_text.splitlines(True), new_text.splitlines(True), fromfile="original", tofile="retry")), encoding="utf-8")
    return migration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = migrate(args.checkpoint, args.source_snapshot, args.output, args.artifact_root)
    print(json.dumps({key:result[key] for key in ("model_updates","all_protected_states_identical","optimizer_updates","budget_refunded")}))


if __name__ == "__main__": main()
