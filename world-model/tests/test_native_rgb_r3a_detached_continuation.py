"""Cold handoff selector/actor ownership checks; no process stopping or physics."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
from scripts import native_rgb_r3a_collect_v3 as cli


class DetachedContinuationContracts(unittest.TestCase):
    def test_all_completed_scenes_skip_and_only_receipted_partial_gets_next_candidate(self):
        slot=dict(slot_id='fixture-T',attempts=[dict(attempt_index=i,seed=100+i) for i in range(3)])
        complete=dict(status='completed',successful_trajectory_id='fixture-T-a0',attempts=[dict(status='completed',trajectory_id='fixture-T-a0')])
        before=deepcopy(complete)
        self.assertEqual(cli.selected_candidates(slot,complete,'continue-unfinished',{}),[])
        self.assertEqual(complete,before)
        partial=dict(status='handoff_ready',attempts=[dict(status='failed_system_handoff',trajectory_id='fixture-T-a0')])
        receipt=dict(failed_slot_next_attempts={'fixture-T':1})
        self.assertEqual(cli.selected_candidates(slot,partial,'continue-unfinished',receipt),slot['attempts'][1:])
        with self.assertRaises(PermissionError):cli.selected_candidates(slot,partial,'continue-unfinished',{})
        fresh=dict(status='not_executed',attempts=[])
        self.assertEqual(cli.selected_candidates(slot,fresh,'continue-unfinished',receipt),slot['attempts'])
        exhausted=dict(status='handoff_ready',attempts=[dict(status='failed_system_handoff') for _ in range(3)])
        with self.assertRaisesRegex(RuntimeError,'two frozen replacements'):
            cli.selected_candidates(slot,exhausted,'continue-unfinished',{'failed_slot_next_attempts':{'fixture-T':3}})

    def test_handoff_receipt_sha_plan_and_previous_owner_are_required(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp);cli.atomic_json(root/'plan.json',{'fixture':'immutable_plan'})
            receipt=root/'handoff.json';cli.atomic_json(receipt,dict(authorized=True,mode='continue-unfinished',plan_sha256=cli.sha(root/'plan.json'),failed_slot_next_attempts={},previous_owner_pid=12345))
            config=dict(continue_unfinished_authorized=True,handoff_receipt=str(receipt),handoff_receipt_sha256=cli.sha(receipt))
            self.assertEqual(cli.handoff_for(config,root)['verified_previous_owner_pids'],[12345])
            wrong=dict(config,handoff_receipt_sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'SHA differs'):cli.handoff_for(wrong,root)
            value=cli.read(receipt);value['authorized']=False;cli.atomic_json(receipt,value);config['handoff_receipt_sha256']=cli.sha(receipt)
            with self.assertRaises(PermissionError):cli.handoff_for(config,root)

    def test_old_live_owner_refused_and_new_actor_has_exclusive_owner_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root=Path(tmp);config=dict(output=str(root));receipt=dict(verified_previous_owner_pids=[12345])
            with patch.object(cli,'read',return_value=config),patch.object(cli,'config_contract',return_value=True),patch.object(cli,'handoff_for',return_value=receipt),patch.object(cli,'process_is_running',return_value=True),patch.object(cli,'run_owned') as collect:
                with self.assertRaisesRegex(RuntimeError,'still running'):cli.run('fixture','continue-unfinished',execute=True)
                collect.assert_not_called();self.assertFalse((root/'collector-owner.lock').exists())
            def owned(*args,**kwargs):
                self.assertTrue((root/'collector-owner.lock').exists())
                with self.assertRaises(FileExistsError):
                    with cli.exclusive_owner(root,receipt):pass
                return 0
            with patch.object(cli,'config_contract',return_value=True),patch.object(cli,'handoff_for',return_value=receipt),patch.object(cli,'process_is_running',return_value=False),patch.object(cli,'run_owned',side_effect=owned):
                path=root/'config.json';cli.atomic_json(path,config)
                self.assertEqual(cli.run(path,'continue-unfinished',execute=True),0)
            self.assertFalse((root/'collector-owner.lock').exists())
            self.assertNotIn('pymunk',sys.modules);self.assertNotIn('torch',sys.modules)

    def test_readonly_current_pid_check_and_unchanged_physics_loop_AST(self):
        self.assertTrue(cli.process_is_running(os.getpid()))
        with self.assertRaises(ValueError):cli.process_is_running(0)
        old=ast.parse((ROOT/'scripts/native_rgb_r3a_collect_v2.py').read_text(encoding='utf-8'))
        new=ast.parse((ROOT/'scripts/native_rgb_r3a_collect_v3.py').read_text(encoding='utf-8'))
        def control_loop(tree):
            return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='control')
        self.assertEqual(ast.dump(control_loop(old)),ast.dump(control_loop(new)))
        def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        self.assertEqual(ast.dump(function(old,'save_npz')),ast.dump(function(new,'save_npz')))
        self.assertEqual(ast.dump(function(old,'replace_with_retry')),ast.dump(function(new,'replace_with_retry')))
        self.assertNotIn('pymunk',sys.modules);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
