"""Build a verified small from-scratch native GPC upload, never a resume bundle.

The default archive contains only original published initialization, source,
configuration/dependencies and bounded curated evidence. It never includes local
trained checkpoints, raw trajectories, environments or image caches.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "native-gpc-rgb-from-scratch.zip"
MAX_ZIP_BYTES = 500*1024**2
MAX_EXPERIMENT_BYTES = 3*1024**3
MAX_OPTIONAL_BYTES = 64*1024**2
MAX_OPTIONAL_FILES = 1024
EXCLUDED_MODELS = {"tdyssey/world_model_data.py", "tdyssey/world_model_training.py",
                   "tdyssey/world_model_protocol.py"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")


def safe_target(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Unsafe archive target: "+repr(value))
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("Archive targets must be canonical relative paths: "+value)
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("Unsafe archive target: "+value)
    if any(part in (".git", ".venv", "__pycache__") or part.startswith(".venv-") for part in path.parts):
        raise ValueError("Excluded environment/history/cache target: "+value)
    return value


def disk_bytes(root):
    root = Path(root)
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _under(root, relative):
    relative = safe_target(relative)
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError("Missing or escaped required source: "+str(path))
    return path


def python_dependency_closure(root, seeds):
    """Resolve repository-local Python imports without importing any modules."""
    root = Path(root).resolve()
    pending, found = list(seeds), set()

    def module_file(module):
        if not module or module.split(".")[0] not in ("tdyssey", "scripts"):
            return None
        stem = module.replace(".", "/")
        for candidate in (stem+".py", stem+"/__init__.py"):
            if (root/candidate).is_file():
                return candidate
        return None

    while pending:
        relative = safe_target(str(pending.pop()).replace("\\", "/"))
        if relative in found:
            continue
        if relative.startswith("tdyssey/models/") or relative in EXCLUDED_MODELS:
            raise ValueError("Native dependency closure includes an old custom motion model: "+relative)
        source = _under(root, relative)
        found.add(relative)
        for parent in PurePosixPath(relative).parents:
            initializer = (parent/"__init__.py").as_posix()
            if initializer != "__init__.py" and (root/initializer).is_file():
                pending.append(initializer)
        module = relative[:-3].replace("/", ".")
        package = module[:-9] if module.endswith(".__init__") else module.rpartition(".")[0]
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"), filename=relative)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".") if package else []
                    if node.level > len(parts):
                        raise ValueError("Relative import leaves package in "+relative)
                    prefix = ".".join(parts[:len(parts)-node.level+1])
                    base = ".".join(part for part in (prefix, node.module) if part)
                else:
                    base = node.module or ""
                modules = [base]+[base+"."+alias.name for alias in node.names if alias.name != "*"]
            for imported in modules:
                candidate = module_file(imported)
                if candidate and candidate not in found:
                    pending.append(candidate)
    return sorted(found)


def _curated_files(experiment):
    explicit = ("local-report.json", "REPORT.md", "preflight.json", "budget.json", "run-config.json",
        "pilot-effective.json", "native-rgb-cost-ledger.json", "resource-probe.json", "physics-probe/report.json",
        "data/manifest.json", "data/physics-audit.json", "engineering/report.json",
        "microfit/result.json", "microfit/validation.json", "microfit/history.jsonl",
        "training/result.json", "training/validation.json", "training/history.jsonl",
        "training-interrupted-winerror5/error.json", "training-interrupted-winerror5/history.jsonl",
        "training-interrupted-winerror5/validation.json", "training-recovery-plan.json", "budget-stop.json",
        "frozen-renderer-calibration.json", "physical-measurement-calibration-final.json", "data-qualification.json",
        "renderer-calibration-run.json", "evaluation/current-seen-val.json", "training-final-audit.json",
        "focused-native-tests-final.txt", "focused-native-tests-final.json", "repository-check.txt",
        "repository-check-final.txt", "training-curve.png", "validation-physical-errors.png",
        "retained-gpc-source-shape-tests-final.txt",
        "retained-gpc-source-shape-tests-final.json",
        "parallel-implementation.json", "parallel-focused-final-rerun.json", "server-sampling-policy-prepared.json",
        "server-probe-real-pipeline-selftest/selftest-result.json")
    found = {name for name in explicit if (experiment/name).is_file()}
    allowed = {".svg", ".csv", ".json", ".png", ".jpg", ".jpeg", ".webp", ".md", ".txt"}
    excluded_trees = {"cache", "caches", "source-snapshot", "source_snapshot", "source-snapshots",
                      "source-at-start", "collection-source", "data-review-source", "__pycache__"}
    trees = {directory: allowed for directory in ("data/coverage", "evaluation/seen_val", "report-coverages",
             "renderer-calibration-run", "parallel-implementation", "server-sampling-policy-prepared")}
    trees["recovery-io-retry"] = {".json", ".diff"}
    trees["report-diagnostics"] = {".json", ".diff", ".md", ".txt"}
    for directory, extensions in trees.items():
        root = experiment/directory
        if root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in extensions:
                    relative = path.relative_to(experiment).as_posix()
                    if not any(part in excluded_trees for part in path.relative_to(experiment).parts):
                        found.add(relative)
    for pattern in ("server-probe-cpu*/selftest-result.json", "server-probe-cpu*/**/selftest-result.json"):
        for path in experiment.glob(pattern):
            if path.is_file() and not any(part in excluded_trees for part in path.relative_to(experiment).parts):
                found.add(path.relative_to(experiment).as_posix())
    return sorted(found, key=lambda name: (name != "local-report.json", name))


def _snapshot_dependencies(repo):
    """Training also hashes a few runtime assets named as string literals."""
    source = repo/"tdyssey/native_rgb_training.py"
    if not source.is_file():
        return []
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "source_identity":
            for statement in ast.walk(node):
                if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "files" for target in statement.targets):
                    values = ast.literal_eval(statement.value)
                    if not isinstance(values, (tuple, list)) or any(not isinstance(value, str) for value in values):
                        raise ValueError("Training source identity must expose a static file list")
                    return [safe_target(value) for value in values]
            raise ValueError("Cannot identify required training source-snapshot assets")
    return []


def _source_plan(repo, experiment, gpc_root, physics_root, init_checkpoint, include_reports):
    rows, generated, targets = [], [], set()

    def add(source, target, role, required=True, version=None, expected=None):
        target = safe_target(target)
        source = Path(source).resolve()
        if target in targets:
            raise ValueError("Duplicate archive target: "+target)
        if not source.is_file():
            if required:
                raise FileNotFoundError("Required upload source missing: "+str(source))
            return
        digest = sha256(source)
        if expected is not None and digest != expected:
            raise ValueError("Pinned source/hash mismatch: "+str(source))
        if source.suffix.lower() in (".pt", ".pth", ".npz", ".npy", ".safetensors") and role != "original_published_initialization":
            raise ValueError("Trained weights/raw arrays are excluded from the default upload")
        targets.add(target)
        rows.append(dict(source=str(source), target=target, bytes=source.stat().st_size,
                         sha256=digest, version=version or "workspace-sha256:"+digest,
                         required=bool(required), role=role))

    required = ("docs/NATIVE_RGB_WORLD_MODEL.md", "docs/STATUS.md", "docs/DECISIONS.md",
        "configs/native_rgb_local.json", "configs/native_rgb_server.json",
        "requirements/native-rgb-server-torch.txt", "requirements/native-rgb-server.txt",
        "run_native_rgb_local.ps1", "run_native_rgb_server.sh")
    for name in required:
        add(_under(repo, name), name, "run_guide" if name.startswith("docs/") else "configuration_or_launcher")
    seeds = sorted({path.relative_to(repo).as_posix() for pattern in ("scripts/native_rgb*.py", "tdyssey/native_rgb*.py")
                    for path in repo.glob(pattern) if path.is_file()})
    snapshots = _snapshot_dependencies(repo)
    seeds += [name for name in snapshots if name.endswith(".py")]
    if not seeds or not (repo/"scripts/native_rgb_world_model.py").is_file():
        raise FileNotFoundError("Native RGB CLI sources are required")
    for name in python_dependency_closure(repo, seeds):
        add(_under(repo, name), name, "native_python_dependency")
    for name in snapshots:
        if name not in targets and not name.startswith("manifests/"):
            add(_under(repo, name), name, "runtime_source_snapshot_dependency")
    for pattern in ("configs/native_rgb*", "requirements/native-rgb-*", "run_native_*.ps1", "run_native_*.sh"):
        for path in sorted(repo.glob(pattern)):
            if path.is_file() and path.relative_to(repo).as_posix() not in targets:
                add(path, path.relative_to(repo).as_posix(), "configuration_or_launcher")
    for path in sorted((repo/"manifests").glob("*.json")):
        add(path, path.relative_to(repo).as_posix(), "pinned_project_manifest")
    gpc = json.loads(_under(repo, "manifests/gpc_world_model.json").read_text(encoding="utf-8"))
    physics = json.loads(_under(repo, "manifests/official_dp.json").read_text(encoding="utf-8"))
    pinned = gpc["checkpoint"]
    add(init_checkpoint, "assets/denoiser.pth", "original_published_initialization",
        version=pinned["repo_id"]+"@"+pinned["revision"], expected=pinned["sha256"])
    if Path(init_checkpoint).stat().st_size != pinned["bytes"]:
        raise ValueError("Original initialization size differs from its published manifest")
    for label, root, spec in (("gpc", gpc_root, gpc), ("official", physics_root, physics)):
        if not spec.get("source_sha256") or not spec.get("source_commit"):
            raise ValueError("Pinned source identity is incomplete: "+label)
        for name, digest in sorted(spec["source_sha256"].items()):
            add(_under(root, name), "vendor/"+label+"/"+name, "unchanged_pinned_upstream_source",
                version=spec["source_commit"], expected=digest)
        provenance = dict(kind="tdyssey.pinned-source-export.v1", source_commit=spec["source_commit"],
                          source_sha256=spec["source_sha256"])
        generated.append(dict(label=label, target="vendor/"+label+"/tdyssey-source-export.json",
                              value=provenance, version=spec["source_commit"]))
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
            path = Path(root)/name
            target = "vendor/"+label+"/"+name
            if path.is_file() and target not in targets:
                add(_under(root, name), target, "upstream_license", required=False, version=spec["source_commit"])
    skipped, optional_bytes, optional_count = [], 0, 0
    if include_reports:
        for name in _curated_files(experiment):
            source = _under(experiment, name)
            size = source.stat().st_size
            if size > 10*1024**2 or optional_bytes+size > MAX_OPTIONAL_BYTES or optional_count >= MAX_OPTIONAL_FILES:
                if name == "local-report.json":
                    raise ValueError("The local qualification report needs a bounded curated representation")
                skipped.append(dict(source=str(source), reason="bounded_curated_evidence_limit"))
                continue
            add(source, "reports/local/"+name, "curated_local_evidence", required=False)
            optional_bytes += size
            optional_count += 1
    return rows, generated, skipped


def verify_archive(path, manifest):
    """Stream-decompress and compare every file. Nothing is extracted to disk."""
    expected = {safe_target(row["target"]): row for row in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Inventory contains duplicate targets")
    verified_bytes = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            safe_target(name)
        if len(names) != len(set(names)) or set(names) != set(expected)|{"upload-manifest.json"}:
            raise ValueError("ZIP contents do not exactly match the inventory")
        embedded = json.loads(archive.read("upload-manifest.json"))
        if embedded != manifest:
            raise ValueError("Embedded/external upload inventories differ")
        for name, row in expected.items():
            info = archive.getinfo(name)
            if info.file_size != row["bytes"] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP size/type mismatch: "+name)
            digest, size = hashlib.sha256(), 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    digest.update(block)
                    size += len(block)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("ZIP decompressed content mismatch: "+name)
            verified_bytes += size
    return dict(verified_files=len(expected), decompressed_bytes=verified_bytes, complete=True)


def _manifest_markdown(manifest):
    lines = ["# Native GPC RGB from-scratch upload inventory", "",
        "This bundle starts from the original published GPC weights. It contains no local trained weights or raw trajectories.",
        "", "The JSON inventory records exact absolute local sources, server-relative targets, sizes, hashes, versions and roles.",
        "", "| Server target | Bytes | SHA256 | Required | Local source |", "|---|---:|---|---|---|"]
    for row in manifest["files"]:
        lines.append(f"| `{row['target']}` | {row['bytes']} | `{row['sha256']}` | {row['required']} | `{row['source']}` |")
    return ("\n".join(lines)+"\n").encode("utf-8")


def build_upload(output, experiment_root, gpc_root, physics_root, init_checkpoint, *, repo_root=ROOT,
                 include_reports=True, max_zip_bytes=MAX_ZIP_BYTES, max_experiment_bytes=MAX_EXPERIMENT_BYTES):
    repo, experiment, output = Path(repo_root).resolve(), Path(experiment_root).resolve(), Path(output).resolve()
    delivery = (experiment/"delivery").resolve()
    if not delivery.is_relative_to(experiment) or not output.is_relative_to(delivery):
        raise ValueError("Upload output must resolve inside this experiment's delivery directory")
    if not 0 < max_zip_bytes <= MAX_ZIP_BYTES or not 0 < max_experiment_bytes <= MAX_EXPERIMENT_BYTES:
        raise ValueError("Upload caps cannot exceed500MiB ZIP/3GiB whole experiment")
    archive_path = output/ZIP_NAME
    external_json, external_md, summary_path = (output/name for name in ("upload-manifest.json", "upload-manifest.md", "archive-summary.json"))
    if any(path.exists() for path in (archive_path, external_json, external_md, summary_path)):
        raise FileExistsError("Upload slots already exist; use a fresh delivery output directory")
    rows, generated, skipped = _source_plan(repo, experiment, Path(gpc_root).resolve(), Path(physics_root).resolve(),
                                             Path(init_checkpoint).resolve(), include_reports)
    generated_bytes = sum(len(json_bytes(item["value"])) for item in generated)
    estimated_archive = sum(row["bytes"] for row in rows)+generated_bytes
    estimated_archive += estimated_archive//100+4*1024**2
    if estimated_archive > max_zip_bytes:
        raise ValueError("Conservative upload size estimate exceeds the ZIP cap")
    if disk_bytes(experiment)+estimated_archive+generated_bytes+1024**2 > max_experiment_bytes:
        raise ValueError("Upload would exceed the aggregate native experiment disk budget")
    output.mkdir(parents=True, exist_ok=True)
    for item in generated:
        source = output/"source-exports"/(item["label"]+"-tdyssey-source-export.json")
        if not source.resolve().is_relative_to(output):
            raise ValueError("Generated source-export path escapes its upload directory")
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open("xb") as stream:
            stream.write(json_bytes(item["value"]))
        rows.append(dict(source=str(source.resolve()), target=item["target"], bytes=source.stat().st_size,
            sha256=sha256(source), version=item["version"], required=True, role="verified_source_export_provenance"))
    rows.sort(key=lambda row: row["target"])
    manifest = dict(kind="native-gpc-rgb-upload.v1", mode="from_scratch", files=rows,
        required_bytes=sum(row["bytes"] for row in rows if row["required"]),
        optional_bytes=sum(row["bytes"] for row in rows if not row["required"]),
        original_initialization_only=True, local_trained_checkpoints_included=False, raw_trajectories_included=False,
        source_git_history_included=False, optional_skipped=skipped,
        inventory_self="upload-manifest.json is excluded from files to avoid a self-hash cycle",
        server_command="bash run_native_rgb_server.sh prepare --setup")
    owns_archive = False
    try:
        with archive_path.open("xb") as target:
            owns_archive = True
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=False) as archive:
                archive.writestr("upload-manifest.json", json_bytes(manifest))
                for row in rows:
                    digest, size = hashlib.sha256(), 0
                    info = zipfile.ZipInfo(row["target"])
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = (0o100755 if row["target"].endswith(".sh") else 0o100644) << 16
                    with Path(row["source"]).open("rb") as source, archive.open(info, "w") as destination:
                        for block in iter(lambda: source.read(1024**2), b""):
                            digest.update(block)
                            size += len(block)
                            destination.write(block)
                    if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                        raise ValueError("Upload source changed while writing ZIP: "+row["source"])
                    if target.tell() > max_zip_bytes:
                        raise ValueError("ZIP cap reached while writing")
        if archive_path.stat().st_size > max_zip_bytes:
            raise ValueError("Final ZIP exceeds its cap")
        verified = verify_archive(archive_path, manifest)
        with external_json.open("xb") as stream:
            stream.write(json_bytes(manifest))
        with external_md.open("xb") as stream:
            stream.write(_manifest_markdown(manifest))
        summary = dict(kind="native-gpc-rgb-upload-archive.v1", archive=str(archive_path),
            archive_bytes=archive_path.stat().st_size, archive_sha256=sha256(archive_path),
            manifest=str(external_json), manifest_sha256=sha256(external_json),
            files=len(rows), required_bytes=manifest["required_bytes"], optional_bytes=manifest["optional_bytes"],
            verification=verified, max_zip_bytes=max_zip_bytes, max_experiment_bytes=max_experiment_bytes,
            experiment_bytes_before_summary=disk_bytes(experiment), gpu_runs=0, optimizer_updates=0)
        with summary_path.open("xb") as stream:
            stream.write(json_bytes(summary))
        if disk_bytes(experiment) > max_experiment_bytes:
            raise ValueError("Aggregate experiment disk cap exceeded")
        return summary
    except BaseException:
        if owns_archive:
            # Only this invocation's exclusively-created archive is removed.
            archive_path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory inside EXPERIMENT/delivery/")
    parser.add_argument("--experiment-root", type=Path, default=ROOT/"artifacts/native-rgb-20261028")
    parser.add_argument("--gpc-root", type=Path, required=True)
    parser.add_argument("--physics-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--no-reports", action="store_true")
    args = parser.parse_args(argv)
    result = build_upload(args.output, args.experiment_root, args.gpc_root, args.physics_root,
                          args.init_checkpoint, include_reports=not args.no_reports)
    print(json.dumps({key: result[key] for key in ("archive", "archive_bytes", "archive_sha256", "files")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
