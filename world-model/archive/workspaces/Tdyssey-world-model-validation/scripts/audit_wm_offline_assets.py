"""Hash existing world-model evidence before/after offline interface diagnostics.

Standard library only. Files are read as bytes: no NPZ/pickle/model loading,
simulator imports, rendering, physical work, downloads, or changes to old assets.
New artifacts/wm-interface-v2-* directories are explicitly allowed; files inside
pre-existing protected directories remain immutable, including additions.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_DIAGNOSTIC_PREFIX = "wm-interface-v2-"


def record_file(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Protected asset cannot be a symbolic link: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Protected asset must be an ordinary file: {path}")
    first = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    last = path.stat()
    if (first.st_size, first.st_mtime_ns) != (last.st_size, last.st_mtime_ns):
        raise ValueError(f"Protected asset changed while hashing: {path}")
    return {"path": str(path), "bytes": last.st_size, "sha256": digest.hexdigest()}


def tree_files(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Missing protected directory: {directory}")
    files = []
    for base, dirs, names in os.walk(directory, followlinks=False):
        for name in dirs:
            if (Path(base) / name).is_symlink():
                raise ValueError("Protected evidence cannot contain directory symlinks")
        dirs.sort()
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError("Protected evidence cannot contain file symlinks")
            files.append(path.resolve())
    return files


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _same(actual, expected):
    return actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"]


def validate_new_output(path, repo):
    """Audit reports themselves may only create files in the new diagnostic area."""
    path, artifacts = Path(path).resolve(), Path(repo).resolve() / "artifacts"
    if not path.is_relative_to(artifacts):
        raise ValueError("Audit output must be inside artifacts/wm-interface-v2-*")
    parts = path.relative_to(artifacts).parts
    if not parts or not parts[0].startswith(NEW_DIAGNOSTIC_PREFIX):
        raise ValueError("Audit output must be inside artifacts/wm-interface-v2-*")


def take_before(repo, audit_dir, bc_baseline, legacy_checkpoint):
    repo, audit_dir = Path(repo).resolve(), Path(audit_dir).resolve()
    validate_new_output(audit_dir, repo)
    before_path = audit_dir / "before.json"
    if before_path.exists():
        raise FileExistsError("Before inventory already exists; it cannot be replaced")
    artifacts = repo / "artifacts"
    roots = sorted(p.resolve() for p in artifacts.glob("frozen-wm*") if p.is_dir())
    if not roots:
        raise ValueError("No existing frozen-wm evidence directories found")
    roots.append((repo / "docs/media/world-model-2026-10-19").resolve())
    if any(audit_dir == path or audit_dir.is_relative_to(path) for path in roots):
        raise ValueError("New audit output must be outside protected evidence roots")
    protected = {}
    for directory in roots:
        for path in tree_files(directory):
            protected[str(path)] = {**record_file(path), "category": "existing_world_model_evidence"}
    legacy_checkpoint = Path(legacy_checkpoint).resolve()
    protected[str(legacy_checkpoint)] = {**record_file(legacy_checkpoint), "category": "upstream_bundled_checkpoint"}
    bc_baseline = Path(bc_baseline).resolve()
    baseline = json.loads(bc_baseline.read_text(encoding="utf-8"))
    bc_root = Path(baseline["protected_checkout"]).resolve()
    bc_checks = []
    for expected in baseline["files"]:
        path = (bc_root / expected["path"]).resolve()
        if not path.is_relative_to(bc_root):
            raise ValueError("B/C baseline contains a path outside its declared checkout")
        actual = record_file(path)
        bc_checks.append({**actual, "matches_prior_inventory": _same(actual, expected)})
        protected[str(path)] = {**actual, "category": "existing_B_C_asset"}
    if not bc_checks or not all(row["matches_prior_inventory"] for row in bc_checks):
        raise ValueError("B/C assets already differ from their prior protected inventory")
    manifest = json.loads((repo / "manifests/gpc_world_model.json").read_text(encoding="utf-8"))
    exact = repo / "artifacts/frozen-wm-assets" / manifest["checkpoint"]["filename"]
    exact_check = record_file(exact)
    if not _same(exact_check, manifest["checkpoint"]):
        raise ValueError("Exact GPC checkpoint differs from manifest bytes/hash")
    media = repo / "docs/media/world-model-2026-10-19"
    publication = json.loads((media / "manifest.json").read_text(encoding="utf-8"))
    curated = []
    for row in publication["files"]:
        actual = record_file(media / row["path"])
        curated.append({**actual, "matches_manifest": _same(actual, row)})
    if not all(row["matches_manifest"] for row in curated):
        raise ValueError("Curated world-model evidence differs from its manifest")
    value = {"schema_version": 1, "kind": "world_model_offline_assets_before",
             "created_utc": datetime.now(timezone.utc).isoformat(), "repo": str(repo),
             "method": "streaming bytes/size/SHA256; JSON scope metadata only; no NPZ/image/model deserialization or model/simulator imports",
             "protected_roots": [str(p) for p in roots],
             "protected_files": sorted(protected.values(), key=lambda row: row["path"]),
             "existing_artifact_directories": sorted(p.name for p in artifacts.iterdir() if p.is_dir()),
             "allowed_new_diagnostic_directories": f"{artifacts}/{NEW_DIAGNOSTIC_PREFIX}*",
             "new_files_inside_protected_roots_allowed": False,
             "excluded": ["active offline-analysis scripts/tests and their output directory in the separate DPPO checkout",
                          "unlisted files outside the declared roots, including disposable caches/test-temp"],
             "B_C_baseline": record_file(bc_baseline), "B_C_checks": bc_checks,
             "exact_checkpoint": exact_check, "upstream_bundled_checkpoint": record_file(legacy_checkpoint),
             "curated_manifest_checks": curated, "all_prechecks_pass": True,
             "summary": {"protected_root_count": len(roots), "protected_file_count": len(protected),
                         "protected_bytes": sum(r["bytes"] for r in protected.values()),
                         "B_C_existing_file_count": len(bc_checks), "curated_file_count": len(curated)}}
    write_exclusive(before_path, value)
    return before_path, value


def check_after(before_path, output_path):
    before_path, output_path = Path(before_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError("After report already exists; choose a new audit report name")
    before = json.loads(before_path.read_text(encoding="utf-8"))
    validate_new_output(output_path, before["repo"])
    expected = {row["path"]: row for row in before["protected_files"]}
    rows, missing, changed, errors = [], [], [], []
    for name, prior in expected.items():
        path = Path(name)
        if not path.is_file():
            missing.append(name)
            continue
        try:
            actual = record_file(path)
            rows.append(actual)
            if not _same(actual, prior):
                changed.append({"before": prior, "after": actual})
        except (OSError, ValueError) as exc:
            errors.append({"path": name, "error": str(exc)})
    discovered = set()
    for name in before["protected_roots"]:
        try:
            discovered.update(str(p) for p in tree_files(name))
        except (OSError, ValueError) as exc:
            errors.append({"path": name, "error": str(exc)})
    added_inside = sorted(discovered - expected.keys())
    artifacts = Path(before["repo"]) / "artifacts"
    current_dirs = {p.name for p in artifacts.iterdir() if p.is_dir()}
    new_dirs = current_dirs - set(before["existing_artifact_directories"])
    allowed = sorted(name for name in new_dirs if name.startswith(NEW_DIAGNOSTIC_PREFIX))
    unexpected = sorted(new_dirs - set(allowed))
    passed = not (missing or changed or errors or added_inside or unexpected)
    value = {"schema_version": 1, "kind": "world_model_offline_assets_after",
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "before_inventory": record_file(before_path), "all_old_assets_unchanged": passed,
             "protected_file_count": len(expected), "verified_file_count": len(rows),
             "missing": missing, "changed": changed, "errors": errors,
             "added_inside_protected_roots": added_inside,
             "allowed_new_diagnostic_directories": allowed, "unexpected_new_artifact_directories": unexpected,
             "files": rows, "physical_operations": 0, "model_loads": 0, "downloads": 0}
    write_exclusive(output_path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    before = commands.add_parser("before")
    before.add_argument("--repo", type=Path, default=ROOT)
    before.add_argument("--audit-dir", type=Path, default=ROOT / "artifacts/wm-interface-v2-audit")
    before.add_argument("--bc-baseline", type=Path, default=ROOT / "artifacts/frozen-wm-assets/protected-inventory-before.json")
    before.add_argument("--legacy-checkpoint", type=Path, required=True)
    after = commands.add_parser("after")
    after.add_argument("--before", type=Path, default=ROOT / "artifacts/wm-interface-v2-audit/before.json")
    after.add_argument("--output", type=Path, default=ROOT / "artifacts/wm-interface-v2-audit/after.json")
    args = parser.parse_args()
    if args.command == "before":
        path, result = take_before(args.repo, args.audit_dir, args.bc_baseline, args.legacy_checkpoint)
        print(json.dumps({"before": str(path), **result["summary"], "all_prechecks_pass": True}, indent=2))
        return 0
    result = check_after(args.before, args.output)
    print(json.dumps({"after": str(args.output.resolve()), "all_old_assets_unchanged": result["all_old_assets_unchanged"],
                      "protected_file_count": result["protected_file_count"],
                      "changed": len(result["changed"]), "missing": len(result["missing"]),
                      "added_inside_protected_roots": len(result["added_inside_protected_roots"]),
                      "allowed_new_diagnostic_directories": result["allowed_new_diagnostic_directories"]}, indent=2))
    return 0 if result["all_old_assets_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
