"""Prepare R3A collection settings and, separately, freeze execution sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

stage=Path(__file__).resolve().parent
repo=stage.parents[2]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_stage_budget import native_storage_paths


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    config_path=repo/'configs/native_rgb_r3a.json'
    if not a.freeze:
        if config_path.exists():raise FileExistsError('Do not replace R3A settings silently')
        old=read(repo/'configs/native_rgb_r2c3.json')
        sources=[q for folder,pattern in [('tdyssey','*r3a*.py'),('scripts','*r3a*.py'),('tests','*r3a*.py'),('configs','*r3a*.json')]for q in (repo/folder).glob(pattern)]
        sources.append(config_path)
        required=[repo/'configs/native_rgb_default.json',repo/'configs/native_rgb_selected_physics.json',stage/'PARENT_REVIEW.md',stage/'renderer-consistency/consistency.json',stage/'post-cleanup-storage-baseline.json']
        config=dict(kind='R3A-real-independent48-control-collection.v1',execution_authorized=True,training_authorized=False,
            repo_root=str(repo),output=str(stage),stage_state=str(stage.parent/'stage-state.json'),
            source_identity=str(stage/'source-identity.json'),input_hashes={str(q):sha(q)for q in required},
            physics_source=str(repo.parent/'Tdyssey/third_party/diffusion_policy'),
            storage_roots=[str(q)for q in native_storage_paths(repo,stage.parent,old['storage_roots'])]+[str(config_path)],
            new_stage_source_files=sorted(set(map(str,sources))),
            train_per_letter=128,val_per_letter=24,controls_per_scene=48,max_replacements=2,
            protocol_sha256='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6',
            seed=20261030,prior_supervised_controls=41976,prior_all_native_controls=76536,
            R3A_cap_bytes=96*1024**2,R3B_reserved_bytes=144*1024**2,atomic_reserve_bytes=256*1024**2,
            shared_cap_bytes=3*1024**3,model_time_policy='record_only_v2; no model work in this stage')
        write(config_path,config);print('R3A config prepared; source freeze pending focused contracts and static plan')
        return
    config=read(config_path)
    if Path(config['source_identity']).exists():raise FileExistsError('Execution identity already frozen')
    if not (stage/'plan.json').exists():raise FileNotFoundError('Freeze independent scene plan first')
    files=['tdyssey/native_rgb_r3a_collection.py','scripts/native_rgb_r3a_collect.py',
        'tdyssey/native_rgb_env.py','tdyssey/envs/official.py','tdyssey/native_rgb_data.py',
        'tdyssey/native_rgb_r2a_env.py','tdyssey/native_rgb_r2a2_env.py','tdyssey/native_rgb_selected_physics.py',
        'tdyssey/native_rgb_r2b_sampling.py','tdyssey/native_rgb_supervised_sampling.py',
        'tdyssey/native_rgb_r2c1r1_training.py','tdyssey/native_rgb_stage_budget.py',
        'configs/native_rgb_default.json','configs/native_rgb_selected_physics.json','manifests/official_dp.json']
    rows=[]
    for name in files:
        source=repo/name;snapshot=stage/'frozen-source'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,snapshot)
        rows.append(dict(source=str(source),snapshot=snapshot.relative_to(stage).as_posix(),sha256=sha(source)))
    for name in ('pusht_env.py','pymunk_override.py'):
        source=Path(config['physics_source'])/'diffusion_policy/env/pusht'/name
        snapshot=stage/'frozen-source/upstream-physics'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,snapshot)
        rows.append(dict(source=str(source),snapshot=snapshot.relative_to(stage).as_posix(),sha256=sha(source)))
    config['plan_sha256']=sha(stage/'plan.json')
    config['storage_roots']=list(dict.fromkeys(map(str,native_storage_paths(repo,stage.parent,config['storage_roots']))))
    config['new_stage_source_files']=sorted(set(str(q)for folder,pattern in [('tdyssey','*r3a*.py'),('scripts','*r3a*.py'),('tests','*r3a*.py'),('configs','*r3a*.json')]for q in (repo/folder).glob(pattern)))
    write(config_path,config)
    write(config['source_identity'],dict(kind='R3A-collection-frozen-source.v1',files=rows,
        plan_sha256=config['plan_sha256'],config_file=str(config_path),config_sha256=sha(config_path),
        historical_cleanup_receipt='authorized-cleanup-deletions.jsonl',old_audits_not_rewritten=True))
    print(json.dumps(dict(frozen_files=len(rows),plan_sha256=config['plan_sha256'],config_sha256=sha(config_path))))


if __name__=='__main__':main()
