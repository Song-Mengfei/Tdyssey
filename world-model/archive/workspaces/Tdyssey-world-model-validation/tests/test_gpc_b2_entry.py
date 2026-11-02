"""B2 metadata gates only: fake schedule/accounts, no native/simulator/model."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('gpc_b2_entry_under_test',ROOT/'scripts/collect_gpc_adaptation_assets.py')
ENTRY=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(ENTRY)
SHA256=hashlib.sha256
EVIDENCE=ROOT/'artifacts/wm-sampling-b-only-20261026/entry-tests'
STOP_HASHES={'A':'28c873d8f32ccd066938b9d1d65dea1f68b6405f48e1fc338365b016f95020be',
             'B':'e3ddffa52e7aec5308e8c737b59a7dde479069b8eb2cb672bc737adc8e85c100'}


class B2EntryTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(prefix='fixture-',dir=EVIDENCE)
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.original_read=Path.read_bytes;self.stop_anchors={};self.outputs={};self.registries={}
        for stage in ('A','B','B2'):
            output=self.root/'artifacts'/('gpc-data-stage-'+stage.lower()+'-20261026')
            registry=self.root/'artifacts/gpc-data-stage-20261026-registry'/(stage+'.json')
            self.outputs[stage]=output;self.registries[stage]=registry
            if stage=='B2':continue
            (output/'guardian').mkdir(parents=True);registry.parent.mkdir(parents=True,exist_ok=True)
            claim=self.encoded(dict(stage=stage,spent=True,controls=4));registry.write_bytes(claim)
            stop=self.encoded(dict(status='complete' if stage=='A' else 'stopped',native_integration_verified=stage=='A',
                claim=dict(path=str(registry),sha256=SHA256(claim).hexdigest())))
            (output/'guardian/stop-record.json').write_bytes(stop);self.stop_anchors[stop]=STOP_HASHES[stage]
        self.source=self.outputs['A']/'adaptation-data.json';self.source.write_bytes(self.encoded(dict(fixture='metadata only',episodes=[])))
        self.source_sha=SHA256(self.original_read(self.source)).hexdigest()
        self.schedule=self.root/'artifacts/wm-sampling-b-only-20261026/training-schedule.json'
        self.schedule.parent.mkdir(parents=True);self.schedule.write_bytes(self.encoded(dict(fixture='fake loader validates this reference')))
        self.schedule_sha=SHA256(self.original_read(self.schedule)).hexdigest()
        self.plan=self.root/'plan.json';self.plan.write_bytes(self.encoded(dict(fixture='fixed construction plan')))
        self.exclusion_path=self.root/'exclusions.json'
        self.exclusion_path.write_bytes(self.encoded(dict(reviewed_scene_sha256=[f'{n:064x}' for n in range(779)],excluded_reset_seeds=list(range(109)))))
        self.config=self.root/'config.json';self.cfg=json.loads((ROOT/'configs/gpc_supervised_adaptation.json').read_bytes())
        self.cfg['data_manifest']=str(self.source.relative_to(self.root));self.cfg['train']['order']='frozen_train_schedule_v1'
        self.cfg['training_schedule']=dict(path=str(self.schedule.relative_to(self.root)),sha256=self.schedule_sha);self.save_config()
        for key,value in dict(ROOT=self.root,EXCLUSIONS=self.exclusion_path,PLAN=self.plan,SCHEDULE=self.schedule,
                ADAPTATION_CONFIG=self.config,PLAN_SHA256=SHA256(self.original_read(self.plan)).hexdigest(),
                EXCLUSIONS_SHA256=SHA256(self.original_read(self.exclusion_path)).hexdigest(),A_MANIFEST_SHA256=self.source_sha).items():
            self.stack.enter_context(patch.object(ENTRY,key,value))
        self.stack.enter_context(patch.object(ENTRY,'stage_paths',side_effect=lambda stage:(self.outputs[stage],self.registries[stage])))
        loader=types.ModuleType('tdyssey.gpc_training_schedule');loader.ORDER='frozen_train_schedule_v1'
        loader.load_schedule=Mock(return_value=dict(source_manifest_sha256=self.source_sha))
        self.loader=loader.load_schedule;self.stack.enter_context(patch.dict(sys.modules,{'tdyssey.gpc_training_schedule':loader}))
        # Map only the two fake original byte strings to the reviewed anchors;
        # any mutation receives its real digest and must be rejected.
        def digest(payload=b'',*args,**kwargs):
            if isinstance(payload,bytes) and payload in self.stop_anchors:return types.SimpleNamespace(hexdigest=lambda:self.stop_anchors[payload])
            return SHA256(payload,*args,**kwargs)
        self.stack.enter_context(patch.object(ENTRY.hashlib,'sha256',side_effect=digest))
        self.launch=self.stack.enter_context(patch.object(ENTRY,'supervise',side_effect=AssertionError('Native launcher forbidden')))
        self.held={self.outputs['A']/'held.npz',self.outputs['A']/'held-record.json'}
        for path in self.held:path.write_bytes(b'never open held payload')
        def read(path,*args,**kwargs):
            if path in self.held:raise AssertionError('Held payload access forbidden')
            return self.original_read(path,*args,**kwargs)
        self.stack.enter_context(patch.object(Path,'read_bytes',read))
        self.old_paths=[self.source,*self.registries.values(),*(self.outputs[s]/'guardian/stop-record.json' for s in ('A','B'))]
        self.old_bytes={str(p):self.original_read(p) for p in self.old_paths if p.is_file()}

    @staticmethod
    def encoded(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    def save_config(self):self.config.write_bytes(self.encoded(self.cfg))
    def gate(self):return ENTRY.frozen_b2_inputs(ENTRY.read_exclusions())
    def main(self):
        with patch.object(sys,'argv',['collect','--execute','--stage','B2','--source-root',str(self.root/'official')]),patch.object(ENTRY.subprocess,'check_output',return_value='a'*40+'\n'),redirect_stdout(io.StringIO()):return ENTRY.main()

    def test_valid_metadata_gate_binds_config_schedule_preserves_old_accounts(self):
        result=self.gate()
        self.assertEqual(result,dict(training_schedule_sha256=self.schedule_sha,training_config_sha256=SHA256(self.original_read(self.config)).hexdigest()))
        self.assertEqual(self.loader.call_args.args,(self.schedule,self.schedule_sha))
        self.assertEqual({str(p):self.original_read(p) for p in self.old_paths if p.is_file()},self.old_bytes)
        self.launch.assert_not_called();self.assertFalse(self.registries['B2'].exists())

    def test_schedule_file_or_declared_path_mutation_rejected(self):
        original=self.original_read(self.schedule);self.schedule.write_bytes(original+b' ')
        with self.assertRaises(ValueError):self.gate()
        self.schedule.write_bytes(original);self.cfg['training_schedule']['path']='artifacts/other-schedule.json';self.save_config()
        with self.assertRaises(ValueError):self.gate()
        self.loader.assert_not_called();self.launch.assert_not_called()

    def test_schedule_loader_failure_cannot_reach_launch(self):
        self.loader.side_effect=ValueError('Not the complete frozen500 schedule')
        with self.assertRaisesRegex(ValueError,'frozen500'):self.main()
        self.launch.assert_not_called();self.assertFalse(self.registries['B2'].exists())

    def test_frozen_training_config_and_data_source_mutation_rejected(self):
        original=deepcopy(self.cfg)
        for label,change in [('updates',lambda c:c['train'].update(max_updates=499)),
                ('optimizer',lambda c:c['optimizer'].update(lr=.2)),
                ('source',lambda c:c.update(data_manifest='artifacts/other-data.json'))]:
            with self.subTest(field=label):
                self.cfg=deepcopy(original);change(self.cfg);self.save_config()
                with self.assertRaises(ValueError):self.gate()
        self.launch.assert_not_called();self.assertFalse(self.registries['B2'].exists())

    def test_schedule_for_another_source_manifest_rejected(self):
        self.loader.return_value=dict(source_manifest_sha256='f'*64)
        with self.assertRaises(ValueError):self.gate()
        self.launch.assert_not_called()

    def test_old_a_manifest_stop_or_claim_mutation_rejected(self):
        for path in (self.source,self.outputs['A']/'guardian/stop-record.json',self.outputs['B']/'guardian/stop-record.json',self.registries['A'],self.registries['B']):
            with self.subTest(path=str(path.relative_to(self.root))):
                original=self.original_read(path);path.write_bytes(original+b' ')
                try:
                    with self.assertRaises(ValueError):self.gate()
                finally:path.write_bytes(original)
        self.launch.assert_not_called();self.assertFalse(self.registries['B2'].exists())

    def test_main_binds_frozen_metadata_before_any_launcher_call(self):
        self.launch.side_effect=None;self.launch.return_value=dict(status='complete');self.main()
        args,kwargs=self.launch.call_args;self.assertEqual(args[0],'B2')
        self.assertEqual(args[1]['training_schedule_sha256'],self.schedule_sha)
        self.assertEqual(args[1]['training_config_sha256'],SHA256(self.original_read(self.config)).hexdigest())
        self.assertEqual(args[1]['source_commit'],'a'*40);self.assertIn('B-only',args[1]['authorization_id'])
        self.assertEqual(args[1]['stage_a_manifest_sha256'],self.source_sha)
        self.assertEqual({str(p):self.original_read(p) for p in self.old_paths if p.is_file()},self.old_bytes)
        self.assertFalse(self.outputs['B2'].exists());self.assertFalse(self.registries['B2'].exists())

    def test_main_refuses_changed_config_before_stage_claim(self):
        self.cfg['train']['max_updates']=499;self.save_config()
        with self.assertRaises(ValueError):self.main()
        self.launch.assert_not_called();self.assertFalse(self.registries['B2'].exists())

    def test_worker_rechecks_bound_config_before_environment_factory(self):
        import tdyssey.gpc_data_collection as collection
        budget=Mock();beat=Mock()
        class Heartbeat:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return beat
            def __exit__(self,*args):return False
        binding=dict(benchmark_plan_sha256=SHA256(self.original_read(self.plan)).hexdigest(),
            exclusions_sha256=SHA256(self.original_read(self.exclusion_path)).hexdigest(),
            training_schedule_sha256=self.schedule_sha,training_config_sha256='0'*64)
        handoff=dict(stage='B2',binding=binding,claim={'fake':True},worker_options={'source_root':'never-called'})
        with patch.object(sys,'meta_path',list(sys.meta_path)),patch.object(ENTRY,'verify_worker',return_value=handoff),patch.object(ENTRY.CollectionBudget,'create',return_value=budget),patch.object(ENTRY,'Heartbeat',Heartbeat),patch.object(ENTRY,'_verify_authority'),patch.object(collection,'real_environment_factory',side_effect=AssertionError('Simulator import/factory forbidden')) as factory:
            with self.assertRaisesRegex(ValueError,'schedule/config changed'):ENTRY.worker(self.root/'fake-handoff.json')
        factory.assert_not_called();budget.fail.assert_called_once();budget.close.assert_called_once()

    def test_adaptation_worker_rejects_each_schedule_identity_before_torch_import(self):
        import builtins
        import tdyssey.gpc_adaptation as adaptation
        import tdyssey.gpc_adaptation_guard as guard
        spec=importlib.util.spec_from_file_location('adaptation_entry_b2_identity_only',ROOT/'scripts/adapt_gpc_world_model.py')
        entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        identities=dict(sha256='1'*64,order_sha256='2'*64,rule_sha256='3'*64)
        prepared=dict(ready=True,config=deepcopy(self.cfg),config_sha256='4'*64,data={'sha256':'5'*64},benchmark={'sha256':'6'*64},training_schedule=identities)
        binding=dict(data_sha256='5'*64,benchmark_sha256='6'*64,config_sha256='4'*64,
            **{'training_schedule_'+k:v for k,v in identities.items()})
        real_import=builtins.__import__
        def no_models(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','diffusers','pygame','pymunk'):raise AssertionError('Forbidden runtime import reached')
            return real_import(name,*args,**kwargs)
        with patch.object(entry,'plan',return_value=prepared),patch.object(adaptation,'run_candidate',side_effect=AssertionError('Training forbidden')) as run,patch.object(builtins,'__import__',side_effect=no_models):
            for field in identities:
                with self.subTest(identity=field):
                    changed=dict(binding,**{'training_schedule_'+field:'f'*64})
                    with patch.object(guard,'verify_worker',return_value=dict(config_path='unused',binding=changed)):
                        with self.assertRaisesRegex(ValueError,'schedule identity/order/rule'):entry.worker('fake-handoff')
            run.assert_not_called()


if __name__=='__main__':unittest.main()
