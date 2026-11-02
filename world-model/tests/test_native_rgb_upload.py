"""Temporary CPU-only archives; never package actual experiment assets in tests."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.native_rgb_upload import (ZIP_NAME, build_upload, disk_bytes, json_bytes,
    python_dependency_closure, safe_target, sha256, verify_archive)

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def fixture():
    sandbox = (ROOT/"artifacts").resolve()
    sandbox.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="native-rgb-upload-cpu-test-", dir=sandbox)
    base = Path(temporary.name).resolve()
    if not base.is_relative_to(sandbox):
        raise AssertionError("Test fixture escaped its owned workspace sandbox")

    def put(relative, data):
        path = base/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
        return path

    try:
        for name in ("NATIVE_RGB_WORLD_MODEL", "STATUS", "DECISIONS"):
            put("repo/docs/"+name+".md", "# Native RGB fixture\n")
        for name in ("local", "server"):
            put("repo/configs/native_rgb_"+name+".json", "{}\n")
        put("repo/requirements/native-rgb-server-torch.txt", "torch==2.10.0\n")
        put("repo/requirements/native-rgb-server.txt", "numpy==1.26.4\n")
        put("repo/run_native_rgb_local.ps1", "Write-Output 'fixture'\n")
        put("repo/run_native_rgb_server.sh", "#!/bin/sh\nexit 0\n")
        put("repo/scripts/native_rgb_world_model.py", "from tdyssey.native_rgb_training import source_identity\n")
        put("repo/tdyssey/__init__.py", "")
        put("repo/tdyssey/shared.py", "VALUE=1\n")
        put("repo/tdyssey/official.py", "ORIGINAL_PHYSICS=True\n")
        put("repo/tdyssey/native_rgb_training.py",
            "from .shared import VALUE\ndef source_identity():\n    files=('tdyssey/official.py',)\n    return files\n")
        checkpoint = put("original/denoiser.pth", b"original-published-GPC-fixture")
        gpc_source = put("gpc/gpc_rank_evaluation/diffusion/denoiser.py", "# unchanged GPC\n")
        physics_source = put("official/diffusion_policy/env/pusht/pusht_env.py", "# unchanged physics\n")
        gpc = dict(source_commit="a"*40,
                   source_sha256={"gpc_rank_evaluation/diffusion/denoiser.py": sha256(gpc_source)},
                   checkpoint=dict(repo_id="publisher/model", revision="b"*40, bytes=checkpoint.stat().st_size,
                                   sha256=sha256(checkpoint)))
        physics = dict(source_commit="c"*40,
                       source_sha256={"diffusion_policy/env/pusht/pusht_env.py": sha256(physics_source)})
        put("repo/manifests/gpc_world_model.json", json_bytes(gpc))
        put("repo/manifests/official_dp.json", json_bytes(physics))
        put("gpc/LICENSE", "original license\n")
        put("experiment/local-report.json", json_bytes(dict(physics_passed=True, qualified_for_server_scaleup=False)))
        put("experiment/data/coverage/A.svg", "<svg/>\n")
        put("experiment/data/raw.npz", b"raw data must stay local")
        put("experiment/training/best.pt", b"trained weights must stay local")
        put("experiment/training/history.jsonl", b'{"update":2}\n')
        put("experiment/cache/images.npy", b"float image cache must stay local")
        put("repo/.venv/python.exe", b"environment must stay local")
        put("repo/tdyssey/models/visual_dynamics.py", "# old custom motion model must stay out\n")
        yield dict(base=base, repo=base/"repo", experiment=base/"experiment", gpc=base/"gpc",
                   physics=base/"official", checkpoint=checkpoint, output=base/"experiment/delivery/release")
    finally:
        if not base.resolve().is_relative_to(sandbox):
            raise AssertionError("Refusing cleanup of an escaped test fixture")
        temporary.cleanup()


def package(value, **kwargs):
    return build_upload(value["output"], value["experiment"], value["gpc"], value["physics"],
                        value["checkpoint"], repo_root=value["repo"], **kwargs)


class NativeUploadChecks(unittest.TestCase):
    def test_verified_inventory_and_pinned_exports_exclude_training_assets(self):
        with fixture() as value:
            result = package(value)
            manifest = json.loads((value["output"]/"upload-manifest.json").read_text())
            self.assertEqual(result["archive_sha256"], sha256(value["output"]/ZIP_NAME))
            self.assertEqual(result["verification"]["verified_files"], len(manifest["files"]))
            targets = {row["target"] for row in manifest["files"]}
            self.assertIn("assets/denoiser.pth", targets)
            self.assertIn("tdyssey/shared.py", targets)
            self.assertIn("tdyssey/official.py", targets)  # A string-named source-snapshot asset.
            self.assertIn("reports/local/local-report.json", targets)
            self.assertIn("reports/local/data/coverage/A.svg", targets)
            self.assertFalse(any(name.endswith((".pt", ".npz", ".npy", ".exe")) for name in targets))
            self.assertFalse(any("visual_dynamics" in name for name in targets))
            self.assertIn("reports/local/training/history.jsonl", targets)
            for row in manifest["files"]:
                self.assertTrue(Path(row["source"]).is_absolute())
                self.assertEqual(Path(row["source"]).stat().st_size, row["bytes"])
                self.assertEqual(sha256(row["source"]), row["sha256"])
                self.assertTrue(row["version"] and row["role"])
            with zipfile.ZipFile(result["archive"]) as archive:
                for label, spec in (("gpc", "gpc_world_model"), ("official", "official_dp")):
                    provenance = json.loads(archive.read("vendor/"+label+"/tdyssey-source-export.json"))
                    original = json.loads((value["repo"]/("manifests/"+spec+".json")).read_text())
                    self.assertEqual(provenance["kind"], "tdyssey.pinned-source-export.v1")
                    self.assertEqual(provenance["source_commit"], original["source_commit"])
                    self.assertEqual(provenance["source_sha256"], original["source_sha256"])

    def test_existing_archive_is_not_overwritten(self):
        with fixture() as value:
            result = package(value)
            before = sha256(result["archive"])
            with self.assertRaises(FileExistsError):
                package(value)
            self.assertEqual(before, sha256(result["archive"]))

    def test_missing_guide_or_tampered_original_fails_before_zip(self):
        with fixture() as value:
            (value["repo"]/"docs/NATIVE_RGB_WORLD_MODEL.md").unlink()
            with self.assertRaises(FileNotFoundError):
                package(value)
            self.assertFalse((value["output"]/ZIP_NAME).exists())
        with fixture() as value:
            (value["gpc"]/"gpc_rank_evaluation/diffusion/denoiser.py").write_text("changed upstream bytes")
            with self.assertRaisesRegex(ValueError, "Pinned source/hash mismatch"):
                package(value)
            self.assertFalse((value["output"]/ZIP_NAME).exists())
        with fixture() as value:
            value["checkpoint"].write_bytes(b"T/L adapted checkpoint")
            with self.assertRaises(ValueError):
                package(value)
            self.assertFalse((value["output"]/ZIP_NAME).exists())

    def test_output_and_size_limits_are_checked_before_creation(self):
        with fixture() as value:
            with self.assertRaises(ValueError):
                build_upload(value["base"]/"outside", value["experiment"], value["gpc"], value["physics"],
                             value["checkpoint"], repo_root=value["repo"])
            with self.assertRaisesRegex(ValueError, "ZIP cap"):
                package(value, max_zip_bytes=1)
            with self.assertRaisesRegex(ValueError, "aggregate"):
                package(value, max_experiment_bytes=disk_bytes(value["experiment"])+100)
            self.assertFalse((value["output"]/ZIP_NAME).exists())

    def test_traversal_duplicates_and_changed_decompressed_bytes_are_rejected(self):
        for target in ("../outside", "/absolute", "C:/absolute", "x/../y", "x\\..\\y", "x//y", ".git/config"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                safe_target(target)
        with fixture() as value:
            result = package(value)
            manifest = json.loads((value["output"]/"upload-manifest.json").read_text())
            malicious = value["output"]/"traversal.zip"
            with zipfile.ZipFile(malicious, "x") as archive:
                archive.writestr("../outside", "bad")
            with self.assertRaises(ValueError):
                verify_archive(malicious, {"files": []})
            changed = value["output"]/"changed.zip"
            with zipfile.ZipFile(result["archive"]) as source, zipfile.ZipFile(changed, "x") as destination:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "assets/denoiser.pth":
                        data = b"X"+data[1:]
                    destination.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "decompressed content mismatch"):
                verify_archive(changed, manifest)

    def test_old_motion_model_is_a_dependency_error_not_silently_omitted(self):
        with fixture() as value:
            script = value["repo"]/"scripts/native_rgb_world_model.py"
            script.write_text("from tdyssey.models.visual_dynamics import Model\n")
            with self.assertRaisesRegex(ValueError, "old custom motion model"):
                python_dependency_closure(value["repo"], ["scripts/native_rgb_world_model.py"])

    def test_final_coverage_calibration_recovery_and_test_evidence_are_curated(self):
        with fixture() as value:
            paths = ["report-coverages/REPORT.md", "report-coverages/summary.csv", "report-coverages/overview.png",
                "report-coverages/report-sources.json", "frozen-renderer-calibration.json",
                "physical-measurement-calibration-final.json", "renderer-calibration-run/metrics.json",
                "renderer-calibration-run/gallery/case.png", "data-qualification.json", "training-recovery-plan.json",
                "recovery-io-retry/migrated-step-750.json", "recovery-io-retry/atomic-io-only.diff",
                "training-interrupted-winerror5/error.json", "training-interrupted-winerror5/history.jsonl",
                "training-interrupted-winerror5/validation.json", "microfit/history.jsonl", "budget-stop.json",
                "focused-native-tests-final.txt", "repository-check.txt", "parallel-implementation.json",
                "server-sampling-policy-prepared.json", "server-probe-cpu-selftest-resumed/selftest-result.json",
                "renderer-calibration-run.json", "evaluation/current-seen-val.json", "training-final-audit.json",
                "focused-native-tests-final.json", "parallel-focused-final-rerun.json",
                "server-probe-real-pipeline-selftest/selftest-result.json", "training-curve.png",
                "retained-gpc-source-shape-tests-final.txt", "retained-gpc-source-shape-tests-final.json",
                "repository-check-final.txt", "validation-physical-errors.png",
                "report-diagnostics/shape-gate-source-migration.json", "report-diagnostics/shape-gate-source-only.diff"]
            excluded = ["recovery-io-retry/migrated-step-750.pt", "renderer-calibration-run/raw.npz",
                "report-coverages/cache/cache.json", "recovery-io-retry/source_snapshot/identity.json",
                "renderer-calibration-run/source-snapshot/source.py", "training-interrupted-winerror5/last.pt"]
            for name in paths+excluded:
                path = value["experiment"]/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"bounded fixture evidence\n")
            result = package(value)
            with zipfile.ZipFile(result["archive"]) as archive:
                names = set(archive.namelist())
            for name in paths:
                self.assertIn("reports/local/"+name, names)
            for name in excluded:
                self.assertNotIn("reports/local/"+name, names)


if __name__ == "__main__":
    unittest.main()
