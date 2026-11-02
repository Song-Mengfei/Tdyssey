import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from scripts import native_rgb_supervised_stage1b as runner


class FixedSlotTests(unittest.TestCase):
    def test_independent_audit_rejects_changed_initial_state_even_with_true_logged_flag(self):
        import numpy as np
        frozen=json.loads((runner.ROOT/'artifacts/native-rgb-supervised-20261029/stage1b/frozen-slots.json').read_text())
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'artifacts') as temporary:
            root=Path(temporary);data=root/'probe';data.mkdir();entries=[]
            (root/'frozen-slots.json').write_text(json.dumps(frozen))
            for slot in frozen['slots']:
                name=slot['slot_id']+'.npz';np.savez(data/name,states=np.asarray([slot['initial_state']],dtype=np.float64))
                entries.append(dict(slot_id=slot['slot_id'],file=name,initial_scene_hash=slot['initial_state_sha256'],
                    initial_state_hash_verified=True,baseline_trajectory_id=slot['baseline_trajectory_id'],role=slot['role'],seed=slot['seed']))
            (data/'manifest.json').write_text(json.dumps(dict(entries=entries,frozen_slots_sha256=runner.sha256(root/'frozen-slots.json'))))
            (root/'resource-ledger.json').write_text(json.dumps(dict(charged_controls=512,trajectories=list(range(8)))))
            (root/'baseline-preservation.json').write_text(json.dumps(dict(files=[])))
            with patch.object(runner,'audit_probe',side_effect=lambda _:dict(passed=True,issues=[])):
                self.assertTrue(runner.audit(root)['passed'])
                altered=np.asarray([frozen['slots'][0]['initial_state']],dtype=np.float64);altered[0,0]+=1
                np.savez(data/entries[0]['file'],states=altered)
                report=runner.audit(root)
                self.assertFalse(report['passed']);self.assertIn('initial_slot_binding',report['issues'])

    def test_paired_initial_hash_and_original_pick_are_fixed_before_all_eight_slots(self):
        baseline=runner.ROOT/'artifacts/native-rgb-supervised-20261029'
        manifest=json.loads((baseline/'probe/manifest.json').read_text())
        originals={e['trajectory_id']:e for e in manifest['entries']}
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'artifacts') as temporary:
            output=Path(temporary)
            with patch.object(runner,'authorize',return_value=(baseline,{})):
                result=runner.prepare(output)
                self.assertEqual((len(result['slots']),result['combined_planned_controls']),(8,3968))
                self.assertEqual([s['role'] for s in result['slots']],['paired']*6+['new']*2)
                for s in result['slots'][:6]:
                    old=originals[s['baseline_trajectory_id']]
                    self.assertEqual(s['initial_state_sha256'],old['initial_scene_hash'])
                    scene=json.loads((baseline/'probe'/old['metadata_file']).read_text())['scene']
                    self.assertEqual(s['scene']['staging_segment'],scene['staging_segment'])
                    self.assertEqual(s['scene']['pusher_start'],scene['pusher_start'])
                self.assertEqual([s['seed'] for s in result['slots'][6:]],[2026102991,2026103991])
                self.assertTrue(set(s['initial_state_sha256'] for s in result['slots'][6:]).isdisjoint(e['initial_scene_hash'] for e in originals.values()))
                with self.assertRaises(FileExistsError):runner.prepare(output)

    def test_cli_cannot_collect_without_execute_and_rejects_failed_audit(self):
        with patch.object(runner,'authorize'),redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'--execute'):
                runner.main(['collect','--physics-root','unused'])
            with patch.object(runner,'audit',return_value={'passed':False,'issues':['mock']}):
                self.assertEqual(runner.main(['audit','--physics-root','unused']),1)


if __name__=='__main__':unittest.main()
