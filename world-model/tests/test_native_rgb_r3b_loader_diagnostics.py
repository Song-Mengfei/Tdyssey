"""Diagnostic hooks preserve failures and native Windows spawn arguments."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tdyssey import native_rgb_r3b_loader_diagnostics as diag

ROOT=Path(__file__).resolve().parents[1]

class LoaderDiagnosticsTests(unittest.TestCase):
    def events(self,root):
        return [json.loads(line)for path in Path(root).glob('bootstrap-*.jsonl')for line in path.read_text(encoding='utf-8').splitlines()]

    def test_initializer_records_original_exception_and_propagates(self):
        from tdyssey import native_rgb_r3a_data as data
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts')as temporary, \
                patch.dict(os.environ,TDYSSEY_LOADER_DIAGNOSTIC_DIR=temporary), \
                patch.object(data,'_initialize_worker',side_effect=ValueError('unique initializer canary')):
            with self.assertRaisesRegex(ValueError,'unique initializer canary'):
                diag.diagnostic_initializer({'unused':True})
            records=self.events(temporary)
            self.assertEqual([r['step']for r in records],['data_initializer_entered','data_module_imported','data_initializer_failed'])
            self.assertEqual(records[-1]['exception_type'],'ValueError')
            self.assertIn('unique initializer canary',records[-1]['traceback'])

    @unittest.skipUnless(os.name=='nt','Windows spawn boundary')
    def test_CreateProcess_observer_preserves_original_arguments_and_limits_env(self):
        import _winapi
        environment={'__PYVENV_LAUNCHER__':'venv/python.exe','PYTHONPATH':'diagnostic-only','UNRELATED_SECRET':'must-not-record'}
        arguments=('real/python.exe','"real/python.exe" -c spawn',None,None,False,0,environment,None,None)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts')as temporary, \
                patch.dict(os.environ,TDYSSEY_LOADER_DIAGNOSTIC_DIR=temporary), \
                patch.object(_winapi,'CreateProcess',return_value=(11,12,13,14))as actual:
            before=len(diag.HOOK_RESTORE)
            try:
                diag.install_spawn_observation()
                self.assertEqual(_winapi.CreateProcess(*arguments),(11,12,13,14))
                actual.assert_called_once_with(*arguments)
                records=self.events(temporary)
                self.assertEqual([r['step']for r in records],['CreateProcess_requested','CreateProcess_returned'])
                self.assertNotIn('UNRELATED_SECRET',json.dumps(records))
                self.assertNotIn('must-not-record',json.dumps(records))
                self.assertEqual(records[0]['application'],'real/python.exe')
                self.assertEqual(records[0]['creation_flags'],0)
            finally:
                for restore in reversed(diag.HOOK_RESTORE[before:]):restore()
                del diag.HOOK_RESTORE[before:]

if __name__=='__main__':unittest.main()
