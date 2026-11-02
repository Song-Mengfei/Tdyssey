"""Failure containment and storage migration; no physical/model construction."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

STAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(STAGE))
import common
import worker

class OwnerTests(unittest.TestCase):
    def test_selection_failure_never_starts_training(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as tmp:
            root=Path(tmp); (root/'background').mkdir(); calls=[]
            def child(c,a,name):
                calls.append(name)
                if name=='selecting': raise ValueError('bad action timeline')
                return {'completed':True}
            c={'run_id':'fixture','parent_thread_id':'parent'}
            from types import SimpleNamespace
            a=SimpleNamespace(config='fixture',config_sha='fixture',launcher_pid=1)
            with patch.object(worker,'ROOT',root),patch.object(worker,'validate',return_value=c),\
                 patch.object(worker,'child_phase',side_effect=child),\
                 patch.object(worker.detached,'actual_python',return_value='fixture'),\
                 patch.object(worker.detached,'in_job',return_value=False),\
                 patch.object(worker.detached,'wait_launcher_exit',return_value={'exited':True}):
                self.assertEqual(worker.run(c,a),1)
            self.assertEqual(calls,['collecting','selecting'])
            self.assertTrue(common.read(root/'background/completion.json')['failed'])

    def test_shared_cap_migration_preserves_reserve_plus_next_write(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as tmp:
            root=Path(tmp); cfg={'shared_cap_bytes':10_000_000_000,'shared_storage_roots':[]}
            # A real old-library size above3GiB is permitted by the newly approved cap.
            with patch.object(common,'snapshot',side_effect=[{'bytes':100},{'bytes':7_000_000_000}]),\
                 patch.object(common.shutil,'disk_usage',return_value=type('D',(),{'free':20_000_000_000})()):
                self.assertTrue(common.check_storage(root,cfg,full=True)['passed'])
            common.atomic_json(root/'storage-baseline.json',{'external_bytes':9_600_000_000})
            with patch.object(common,'snapshot',return_value={'bytes':0}),\
                 patch.object(common.shutil,'disk_usage',return_value=type('D',(),{'free':20_000_000_000})()):
                self.assertTrue(common.check_storage(root,cfg)['passed'])
                with self.assertRaises(RuntimeError): common.check_storage(root,cfg,reserve_bytes=100_000_000)

if __name__=='__main__': unittest.main()
