"""Asset and reuse regressions; no Torch, GPU, server or physics construction."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from runtime_config import REPO,SERVICE,load_runtime,validate_assets,verify_checkpoint_payload
from start import matches


class RuntimeTests(unittest.TestCase):
    def test_external_config_resolves_against_its_location(self):
        with tempfile.TemporaryDirectory() as temp:
            config=Path(temp)/'runtime.local.json'
            config.write_text(json.dumps(dict(model='hmf500',asset_root='private-assets',gpc_root='gpc',physics_root='physics')),encoding='utf-8')
            selected=load_runtime(config_path=config)
            self.assertEqual(selected['checkpoint'],Path(temp)/'private-assets'/selected['profile']['checkpoint_relative'])
            self.assertEqual(selected['gpc_root'],Path(temp)/'gpc')
            other=load_runtime('hmf3000',config)
            self.assertEqual(other['profile']['successful_updates'],3000)
            self.assertNotEqual(other['checkpoint'],selected['checkpoint'])

    def test_missing_checkpoint_is_actionable_without_loading_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            config=load_runtime(config_path=Path(temp)/'runtime.local.json')
            config['checkpoint']=Path(temp)/'missing.pt'
            with self.assertRaisesRegex(FileNotFoundError,'Configure external assets'):
                validate_assets(config)

    def test_wrong_weight_hash_and_wrong_progress_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=load_runtime(config_path=root/'runtime.local.json');config['checkpoint']=root/'weights.pt';config['checkpoint'].write_bytes(b'bad')
            config['profile']=dict(config['profile'],checkpoint_bytes=3,checkpoint_sha256=hashlib.sha256(b'good').hexdigest())
            config['gpc_root']=root/'gpc';config['physics_root']=root/'physics'
            for key,relative in [('gpc_root','gpc_rank_evaluation/diffusion/denoiser.py'),('physics_root','diffusion_policy/env/pusht/pusht_env.py')]:
                p=config[key]/relative;p.parent.mkdir(parents=True);p.write_text('fixture',encoding='utf-8')
            with self.assertRaisesRegex(FileNotFoundError,'SHA-256'):validate_assets(config)
            with self.assertRaisesRegex(ValueError,'progress'):
                verify_checkpoint_payload(dict(kind=config['profile']['checkpoint_kind'],progress=dict(successful_updates=3000,lineage_updates=14367)),config['profile'])

    def test_health_reuse_requires_this_repository_model_and_hash(self):
        config=load_runtime();good=dict(ready=True,service=SERVICE,repository=str(REPO.resolve()),model_id=config['model_id'],checkpoint_sha256=config['profile']['checkpoint_sha256'])
        self.assertTrue(matches(good,config))
        for key,value in [('service','tdyssey-r3b-playground-v2'),('repository','another-checkout'),('model_id','old-model'),('checkpoint_sha256','0'*64)]:
            self.assertFalse(matches(dict(good,**{key:value}),config))


if __name__=='__main__':unittest.main()
