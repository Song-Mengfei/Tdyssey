"""Audit frozen dynamics assets without datasets, physics, or model execution.

The only optional network operation fetches the one manifest-pinned checkpoint.
Existing files are never overwritten. Tensor inspection uses CPU weights_only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_policy_assets import fetch_checkpoint, file_hash


def file_record(path, expected=None):
    path = Path(path)
    record = {"path": str(path.resolve()), "exists": path.is_file()}
    if record["exists"]:
        record.update(bytes=path.stat().st_size, sha256=file_hash(path))
    if expected is not None:
        record["expected"] = expected
        record["matches"] = (record["exists"]
                             and all(record.get(key) == value for key, value in expected.items()))
    return record


def discover_assets(roots):
    """Inspect filenames only; never open a catalog or held-out dataset."""
    found = {"gpc_checkpoints": [], "custom_checkpoints": [], "catalogs": []}
    excluded = {".git", ".venv", "venv", "__pycache__", "node_modules"}
    for root in roots:
        for base, dirs, files in os.walk(Path(root)):
            dirs[:] = sorted(name for name in dirs
                             if name not in excluded
                             and "model_test" not in name.lower()
                             and "model-test" not in name.lower())
            for name in sorted(files):
                lower = name.lower()
                path = str((Path(base) / name).resolve())
                if "denoiser" in lower and lower.endswith(".pth"):
                    found["gpc_checkpoints"].append(path)
                elif lower == "best.pt" or ("world" in lower and "model" in lower
                                             and lower.endswith((".pt", ".pth"))):
                    found["custom_checkpoints"].append(path)
                elif "catalog" in lower and lower.endswith(".json"):
                    found["catalogs"].append(path)
    return {key: sorted(set(value)) for key, value in found.items()}


def tensor_summary(path):
    """Return canonical state fingerprints, independent of torch serialization."""
    import torch

    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Expected a nonempty weights-only state dictionary")
    entries = {}
    fingerprint = hashlib.sha256()
    for name, tensor in sorted(payload.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Checkpoint has non-tensor state entries")
        if not torch.isfinite(tensor).all():
            raise ValueError("Checkpoint has non-finite weights")
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        metadata = {"shape": list(value.shape), "dtype": str(value.dtype),
                    "elements": value.numel(), "sha256": digest}
        entries[name] = metadata
        fingerprint.update(json.dumps([name, metadata], sort_keys=True,
                                      separators=(",", ":")).encode("utf-8"))
    return {"loader": "torch.load(map_location='cpu', weights_only=True, mmap=True)",
            "state_entries": len(entries),
            "tensor_elements": sum(value["elements"] for value in entries.values()),
            "state_fingerprint_sha256": fingerprint.hexdigest(), "tensors": entries}


def compare_summaries(left, right):
    left_entries, right_entries = left["tensors"], right["tensors"]
    common = sorted(left_entries.keys() & right_entries.keys())
    missing = sorted(left_entries.keys() - right_entries.keys())
    extra = sorted(right_entries.keys() - left_entries.keys())
    changed = [name for name in common if left_entries[name] != right_entries[name]]
    equal = not missing and not extra and not changed
    return {"same_state_keys": not missing and not extra,
            "left_only_keys": missing, "right_only_keys": extra,
            "equal_tensor_entries": len(common) - len(changed),
            "changed_tensor_entries": len(changed), "changed_keys": changed,
            "identical_tensor_state": equal,
            "conclusion": ("same tensor state; file hash mismatch is serialization-level"
                           if equal else "different tensor state; not a serialization-only mismatch")}


def source_audit(source_root, spec):
    source_root = Path(source_root).resolve()
    command = ["git", "-c", f"safe.directory={source_root.as_posix()}",
               "-C", str(source_root), "rev-parse", "HEAD"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    revision = result.stdout.strip() if result.returncode == 0 else None
    files = [file_record(source_root / name, {"sha256": expected})
             for name, expected in spec["source_sha256"].items()]
    return {"path": str(source_root), "source_repository": spec["source_repository"],
            "expected_commit": spec["source_commit"], "actual_commit": revision,
            "revision_matches": revision == spec["source_commit"],
            "git_error": result.stderr.strip() or None,
            "files": files, "all_source_hashes_match": all(row["matches"] for row in files)}


def tracked_provenance(path, source_root):
    """Identify a bundled upstream weight without changing Git configuration."""
    path, source_root = Path(path).resolve(), Path(source_root).resolve()
    try:
        relative = path.relative_to(source_root).as_posix()
    except ValueError:
        return None
    command = ["git", "-c", f"safe.directory={source_root.as_posix()}", "-C", str(source_root)]
    tracked = subprocess.run(command + ["ls-files", "--stage", "--", relative],
                             capture_output=True, text=True, check=False)
    committed = subprocess.run(command + ["rev-parse", "HEAD:" + relative],
                               capture_output=True, text=True, check=False)
    blob = subprocess.run(command + ["hash-object", "--", relative],
                          capture_output=True, text=True, check=False)
    fields = tracked.stdout.split()
    committed_blob = committed.stdout.strip() if committed.returncode == 0 else None
    return {"source_relative_path": relative, "tracked_at_source_commit": committed_blob is not None,
            "tracked_git_blob": fields[1] if len(fields) >= 2 else None,
            "head_git_blob": committed_blob,
            "actual_git_blob": blob.stdout.strip() or None,
            "matches_tracked_blob": (len(fields) >= 2 and fields[1] == blob.stdout.strip()),
            "matches_head_blob": committed_blob is not None and committed_blob == blob.stdout.strip()}


def run_audit(search_roots, source_root, output, download=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "asset-audit.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior audit report: {report_path}")
    spec = json.loads((ROOT / "manifests/gpc_world_model.json").read_text())
    checkpoint = spec["checkpoint"]
    snapshot_dir = ROOT / "docs/media/world-model-2026-10-19"
    snapshot = json.loads((snapshot_dir / "manifest.json").read_text())
    curated = [file_record(snapshot_dir / entry["path"],
                           {"bytes": entry["bytes"], "sha256": entry["sha256"]})
               for entry in snapshot["files"]]
    inventory = discover_assets(search_roots)
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "frozen assets only; no dataset contents, physics, model inference, or training",
              "search_roots": [str(Path(path).resolve()) for path in search_roots],
              "search_exclusions": [".git", ".venv", "venv", "__pycache__", "node_modules",
                                    "directories containing model_test or model-test"],
              "inventory": inventory,
              "custom_L": {"available": False,
                           "candidate_checkpoint_and_catalog_found": bool(inventory["custom_checkpoints"]
                                                                          and inventory["catalogs"]),
                           "conclusion": "historical comparison unavailable unless exact old checkpoint and catalog are found",
                           "catalog_contents_read": False},
              "curated_snapshot": {"files": curated, "count": len(curated),
                                   "bytes": sum(entry.get("bytes", 0) for entry in curated),
                                   "budget_bytes": snapshot["budget_bytes"],
                                   "all_files_match": all(entry["matches"] for entry in curated)},
              "source": source_audit(source_root, spec),
              "pinned_checkpoint": checkpoint, "checkpoint_records": [],
              "download": {"requested": download, "max_bytes": 40 * 1024**2,
                           "maximum_asset_bytes": checkpoint["bytes"],
                           "url": f"https://huggingface.co/{checkpoint['repo_id']}/resolve/{checkpoint['revision']}/{checkpoint['path']}"},
              "optimizer_updates": {"world_model": 0, "policy": 0}}
    expected = {"bytes": checkpoint["bytes"], "sha256": checkpoint["sha256"]}
    existing = [file_record(path, expected) for path in inventory["gpc_checkpoints"]]
    exact = next((entry["path"] for entry in existing if entry["matches"]), None)
    destination = output / checkpoint["filename"]
    if exact:
        report["download"]["status"] = "existing_exact_asset_reused"
    elif download:
        try:
            report["download"]["status"] = fetch_checkpoint(checkpoint, destination, 40 * 1024**2)
            exact = str(destination.resolve())
        except Exception as error:
            report["download"].update(status="failed", error_type=type(error).__name__,
                                      error=str(error))
    else:
        report["download"]["status"] = "not_requested"
    if exact and exact not in inventory["gpc_checkpoints"]:
        existing.append(file_record(exact, expected))
    summaries = {}
    for entry in existing:
        before = file_record(entry["path"])
        entry["git_provenance"] = tracked_provenance(entry["path"], source_root)
        try:
            summary = tensor_summary(entry["path"])
            summaries[entry["path"]] = summary
            entry["tensor_state"] = summary
        except Exception as error:
            entry["tensor_error"] = {"type": type(error).__name__, "error": str(error)}
        after = file_record(entry["path"])
        entry["hash_unchanged_by_audit"] = before["sha256"] == after["sha256"]
    report["checkpoint_records"] = existing
    report["resolved_exact_checkpoint"] = exact
    report["comparisons"] = []
    if exact in summaries:
        # Pinned blocks.py: FourierFeatures.register_buffer('weight', ...).
        # These two 128-element buffers are state entries, not parameters.
        buffers = ["inner_model.noise_emb.weight", "inner_model.noise_cond_emb.weight"]
        exact_summary = summaries[exact]
        buffer_elements = sum(exact_summary["tensors"][name]["elements"] for name in buffers)
        report["gpc_tensor_accounting"] = {
            "parameter_count": exact_summary["tensor_elements"] - buffer_elements,
            "buffer_elements": buffer_elements, "buffer_keys": buffers,
            "state_entries": exact_summary["state_entries"],
            "basis": "verified pinned diffusion/blocks.py FourierFeatures.register_buffer and checkpoint tensor inventory"}
        for path, summary in summaries.items():
            if path != exact:
                report["comparisons"].append({"left": path, "right": exact,
                                               **compare_summaries(summary, summaries[exact])})
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/frozen-wm-assets")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    report_path, report = run_audit(args.search_root, args.source_root, args.output, args.download)
    print(json.dumps({"report": str(report_path.resolve()),
                      "exact_checkpoint": report["resolved_exact_checkpoint"],
                      "download": report["download"],
                      "curated_files_match": report["curated_snapshot"]["all_files_match"],
                      "comparisons": [{key: row[key] for key in
                                       ("left", "right", "same_state_keys", "equal_tensor_entries",
                                        "changed_tensor_entries", "identical_tensor_state", "conclusion")}
                                      for row in report["comparisons"]]}, indent=2))


if __name__ == "__main__":
    main()
