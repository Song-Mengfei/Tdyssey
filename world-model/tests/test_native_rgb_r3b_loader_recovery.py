"""No-process regressions for the confirmed Windows redirector handle repair."""
import json
import multiprocessing as mp
from multiprocessing import spawn
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import native_rgb_r3b_loader_diagnostics as diagnostic
from tdyssey import native_rgb_r3b_loader_recovery as recovery


@unittest.skipUnless(os.name == 'nt', 'Confirmed Windows-specific spawn boundary')
class LoaderSpawnRepairTests(unittest.TestCase):
    def _environment(self, root, *, confirmed=True):
        base = root / 'base-python'; venv = root / 'existing-venv'
        actual = base / 'python.exe'; alias = venv / 'Scripts/python.exe'
        (root / 'diagnostics').mkdir()
        (root / 'diagnostics/root-cause.json').write_text(json.dumps(dict(
            root_cause_confirmed=confirmed, category='venv_redirector_wrong_handle_target',
            evidence_bindings=[], root_cause='Fixture: synchronization handles target the redirector PID.')),
            encoding='utf-8')
        identity = SimpleNamespace(executable=str(alias), prefix=str(venv), base_prefix=str(base))
        return actual, alias, identity

    def test_confirmed_alias_targets_actual_pe_and_preserves_existing_venv(self):
        original = spawn.get_executable()
        try:
            with tempfile.TemporaryDirectory(dir=recovery.ROOT / 'artifacts') as temporary:
                root = Path(temporary); actual, alias, identity = self._environment(root)
                # No subprocess/worker/model is involved: inspect public spawn state.
                with patch.object(recovery, 'ATTEMPT', root), patch.object(recovery, 'sys', identity), \
                     patch.object(diagnostic, 'actual_pe', return_value=str(actual)), \
                     patch.object(diagnostic, 'identity', return_value={'fixture': True}), \
                     patch.dict(os.environ, {'__PYVENV_LAUNCHER__': 'wrong-previous-alias'}), \
                     patch('multiprocessing.process.BaseProcess.start', side_effect=AssertionError('No process allowed')):
                    proof = recovery.apply_spawn_repair()
                    self.assertEqual(Path(os.fsdecode(spawn.get_executable())), actual.resolve())
                    self.assertEqual(Path(os.environ['__PYVENV_LAUNCHER__']), alias.resolve())
                    self.assertTrue(proof['passed'])
                    self.assertEqual(proof['venv_python'], str(alias.resolve()))
        finally:
            mp.set_executable(original)

    def test_unconfirmed_cause_cannot_change_spawn_or_venv(self):
        before = spawn.get_executable()
        with tempfile.TemporaryDirectory(dir=recovery.ROOT / 'artifacts') as temporary:
            root = Path(temporary); _, _, identity = self._environment(root, confirmed=False)
            with patch.object(recovery, 'ATTEMPT', root), patch.object(recovery, 'sys', identity), \
                 patch.dict(os.environ, {'__PYVENV_LAUNCHER__': 'retained-venv'}), \
                 patch('multiprocessing.process.BaseProcess.start', side_effect=AssertionError('No process allowed')):
                with self.assertRaisesRegex(RuntimeError, 'Confirmed redirector/handle evidence required'):
                    recovery.apply_spawn_repair()
                self.assertEqual(spawn.get_executable(), before)
                self.assertEqual(os.environ['__PYVENV_LAUNCHER__'], 'retained-venv')


if __name__ == '__main__':
    unittest.main()
