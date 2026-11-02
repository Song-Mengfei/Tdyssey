"""Three explicit pilot identities; stdlib metadata and injected Git responses only."""
from copy import deepcopy
from contextlib import ExitStack
import builtins
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import gpc_adaptation as protocol


PILOTS = (
    ("gpc-tl-supervised-one-step-01", protocol.AUTHORIZATION_01,
     "artifacts/gpc-supervised-adaptation-pilot-20261025",
     protocol.ROOT/"configs/gpc_supervised_adaptation.json"),
    ("gpc-tl-supervised-one-step-02", protocol.AUTHORIZATION_02,
     "artifacts/gpc-supervised-adaptation-pilot02-20261026",
     protocol.ROOT/"configs/gpc_supervised_adaptation_pilot02.json"),
    ("gpc-tl-supervised-one-step-03", protocol.AUTHORIZATION_03,
     "artifacts/gpc-supervised-adaptation-pilot03-20261026",
     protocol.ROOT/"configs/gpc_supervised_adaptation_pilot03.json"),
)
EVIDENCE = protocol.ROOT/"artifacts/gpc-supervised-pilot03-review-20261026/identity-fixtures"


class Pilot02IdentityTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        original_import = builtins.__import__

        def safe_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "pygame", "pymunk", "diffusion_policy"}:
                raise AssertionError("Tensor/model/physics import forbidden: "+name)
            return original_import(name, *args, **kwargs)

        self.stack.enter_context(patch("builtins.__import__", side_effect=safe_import))
        self.git = self.stack.enter_context(patch.object(protocol.subprocess, "check_output",
            side_effect=AssertionError("No real Git/process operation in this fixture")))
        self.stack.enter_context(patch.object(protocol.subprocess, "Popen",
            side_effect=AssertionError("No process launch in this fixture")))

    @staticmethod
    def cfg(pilot):
        return dict(candidate_id=pilot[0], output_dir=pilot[2])

    @staticmethod
    def prepared(pilot):
        return dict(ready=True, gaps=[], config_path=str(pilot[3]),
            config=Pilot02IdentityTests.cfg(pilot), config_sha256="c"*64,
            data=dict(sha256="d"*64), benchmark=dict(sha256="e"*64))

    def test_original_bytes_preserved_and_only_two_scientific_identity_fields_differ(self):
        old_bytes = PILOTS[0][3].read_bytes()
        self.assertEqual(hashlib.sha256(old_bytes).hexdigest(), protocol.BASE_CONFIG_SHA256)
        self.assertEqual(hashlib.sha256(PILOTS[1][3].read_bytes()).hexdigest(),
            "bd43e43e28e0f54fe8965bdd6c558f21dcbf5d7b20128b7520a1d3594c8ca0e8")
        old = json.loads(old_bytes)
        for pilot in PILOTS[1:]:
            new_bytes = pilot[3].read_bytes(); new = json.loads(new_bytes)
            self.assertEqual({key for key in set(old) | set(new) if old.get(key) != new.get(key)},
                {"candidate_id", "output_dir"})
            expected = old_bytes.replace(PILOTS[0][0].encode(), pilot[0].encode()).replace(
                PILOTS[0][2].encode(), pilot[2].encode())
            self.assertEqual(new_bytes, expected)
        for pilot in PILOTS:
            cfg, sha = protocol.read_config(pilot[3])
            self.assertEqual((cfg["candidate_id"], cfg["output_dir"]), (pilot[0], pilot[2]))
            self.assertEqual(sha, hashlib.sha256(pilot[3].read_bytes()).hexdigest())

    def test_planning_allows_identity_copy_but_execution_requires_fixed_config_path(self):
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            path = Path(directory)/"planning-copy.json"
            for pilot in PILOTS:
                path.write_bytes(pilot[3].read_bytes())
                cfg, _ = protocol.read_config(path)
                self.assertEqual(protocol.execution_identity(path, cfg, strict_path=False)["candidate_id"], pilot[0])
                with self.assertRaises(PermissionError):
                    protocol.execution_identity(path, cfg, dict(authorization_id=pilot[1]))

    def test_planning_rejects_changed_science_extra_fields_and_unpaired_output(self):
        changed = []
        for pilot in PILOTS[1:]:
            cfg = json.loads(pilot[3].read_bytes())
            mutation = deepcopy(cfg); mutation["limits"]["forecasts"] += 1; changed.append(mutation)
            mutation = deepcopy(cfg); mutation["extra_science"] = True; changed.append(mutation)
            mutation = deepcopy(cfg); mutation["output_dir"] = PILOTS[0][2]; changed.append(mutation)
        with tempfile.TemporaryDirectory(dir=EVIDENCE) as directory:
            path = Path(directory)/"invalid.json"
            for mutation in changed:
                with self.subTest(mutation=mutation):
                    path.write_text(json.dumps(mutation), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        protocol.read_config(path)

    def test_exact_identity_pairs_and_relative_canonical_paths(self):
        for pilot in PILOTS:
            with self.subTest(candidate=pilot[0]):
                relative = pilot[3].relative_to(protocol.ROOT)
                identity = protocol.execution_identity(relative, self.cfg(pilot),
                    dict(authorization_id=pilot[1], candidate_id=pilot[0], output_dir=pilot[2],
                         config_path=str(relative)))
                self.assertEqual(identity, dict(candidate_id=pilot[0], authorization_id=pilot[1],
                    output_dir=pilot[2], config_path=str(pilot[3].resolve())))

    def test_crossed_candidate_output_authorization_and_config_paths_are_rejected(self):
        for pilot in PILOTS:
            for other in PILOTS:
                if other == pilot:
                    continue
                wrong_output = self.cfg(pilot); wrong_output["output_dir"] = other[2]
                with self.assertRaises(ValueError):
                    protocol.execution_identity(pilot[3], wrong_output)
                with self.assertRaises(PermissionError):
                    protocol.execution_identity(pilot[3], self.cfg(pilot), dict(authorization_id=other[1]))
                with self.assertRaises(PermissionError):
                    protocol.execution_identity(other[3], self.cfg(pilot), dict(authorization_id=pilot[1]))
            for authorization in ("arbitrary-unregistered-authorization", ""):
                with self.subTest(candidate=pilot[0], authorization=authorization):
                    with self.assertRaises(PermissionError):
                        protocol.execution_identity(pilot[3], self.cfg(pilot), dict(authorization_id=authorization))
        with self.assertRaises(ValueError):
            protocol.execution_identity(PILOTS[2][3], dict(candidate_id="gpc-tl-supervised-one-step-04",
                output_dir="artifacts/gpc-supervised-adaptation-pilot04-20261026"))

    def test_explicit_binding_fields_cannot_override_the_selected_identity(self):
        for pilot in PILOTS:
            for other in PILOTS:
                if other == pilot:
                    continue
                for key, value in (("candidate_id", other[0]), ("output_dir", other[2]), ("config_path", str(other[3]))):
                    binding = dict(authorization_id=pilot[1]); binding[key] = value
                    with self.subTest(candidate=pilot[0], other=other[0], field=key):
                        with self.assertRaises(PermissionError):
                            protocol.execution_identity(pilot[3], self.cfg(pilot), binding)

    def test_verify_binds_both_pilots_with_fake_clean_reviewed_source(self):
        for pilot in PILOTS:
            with self.subTest(candidate=pilot[0]), patch.object(protocol.subprocess, "check_output",
                    side_effect=["a"*40+"\n", ""]) as fake_git:
                result = protocol.verify_execution(self.prepared(pilot), authorization_id=pilot[1],
                    reviewed_commit="a"*40, authorized_config_sha256="c"*64)
                self.assertEqual(fake_git.call_count, 2)
                self.assertEqual((result["candidate_id"], result["authorization_id"], result["output_dir"],
                    result["config_path"]), (pilot[0], pilot[1], pilot[2], str(pilot[3].resolve())))
                self.assertEqual(result["config_sha256"], "c"*64)

    def test_verify_rejects_crossed_or_missing_identity_before_process_work(self):
        for pilot in PILOTS:
            malformed = []
            prepared = self.prepared(pilot); del prepared["config_path"]; malformed.append((prepared, pilot[1]))
            for other in PILOTS:
                if other == pilot:
                    continue
                prepared = self.prepared(pilot); prepared["config_path"] = str(other[3]); malformed.append((prepared, pilot[1]))
                prepared = self.prepared(pilot); prepared["config"]["output_dir"] = other[2]; malformed.append((prepared, pilot[1]))
                malformed.append((self.prepared(pilot), other[1]))
            for prepared, authorization in malformed:
                with self.subTest(candidate=pilot[0], prepared=prepared, authorization=authorization):
                    with self.assertRaises((PermissionError, ValueError)):
                        protocol.verify_execution(prepared, authorization_id=authorization,
                            reviewed_commit="a"*40, authorized_config_sha256="c"*64)
        self.git.assert_not_called()

    def test_plan_carries_the_selected_config_path_without_opening_source_payloads(self):
        data = dict(sha256="d"*64, episodes=[], readiness=dict(ready=True, gaps=[], shapes={}))
        for pilot in PILOTS:
            cfg = self.cfg(pilot); cfg.update(data_manifest="artifacts/fake/data.json",
                benchmark_manifest="artifacts/fake/benchmark.json", evaluation=dict(loss_examples_per_role=16))
            with patch.object(protocol, "read_config", return_value=(cfg, "c"*64)), \
                 patch("tdyssey.gpc_adaptation_data.read_adaptation_manifest", return_value=data), \
                 patch.object(protocol, "read_benchmark", return_value=dict(ready=True, gaps=[])):
                prepared = protocol.plan(pilot[3], claimed=True)
            self.assertTrue(prepared["ready"])
            self.assertEqual(prepared["config_path"], str(pilot[3].absolute()))


if __name__ == "__main__":
    unittest.main()
