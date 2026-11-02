"""Freeze versioned I/O-only repair while retaining the old execution identity."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

stage=Path(__file__).resolve().parent;repo=stage.parents[2];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_training import atomic_json
from tdyssey.native_rgb_stage_budget import native_storage_paths


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    path=repo/'configs/native_rgb_r3a.json';config=read(path)
    identity=stage/'source-identity-v2.json'
    if identity.exists():raise FileExistsError('Do not refreeze execution sources silently')
    contracts=read(stage/'collector-v2-contracts.json')
    if not contracts.get('passed'):raise ValueError('Focused repair contracts required')
    previous=read(stage/'source-identity.json');files=previous['files'][:]
    for name in ['scripts/native_rgb_r3a_collect_v2.py','tdyssey/native_rgb_training.py']:
        source=repo/name;snapshot=stage/'frozen-source-v2'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,snapshot)
        files.append(dict(source=str(source),snapshot=snapshot.relative_to(stage).as_posix(),sha256=sha(source)))
    config.update(source_identity=str(identity),entry_script='scripts/native_rgb_r3a_collect_v2.py',entry_version=2,
        repair_first_six_authorized=True,recovery_receipt=str(stage/'io-recovery.json'),recovery_receipt_sha256=sha(stage/'io-recovery.json'))
    config['storage_roots']=list(dict.fromkeys(map(str,native_storage_paths(repo,stage.parent,config['storage_roots']))))
    config['new_stage_source_files']=sorted(set(str(q)for folder,pattern in [('tdyssey','*r3a*.py'),('scripts','*r3a*.py'),('tests','*r3a*.py'),('configs','*r3a*.json')]for q in (repo/folder).glob(pattern)))
    atomic_json(path,config)
    atomic_json(identity,dict(kind='R3A-versioned-I/O-repair-source.v2',files=files,previous_identity='source-identity.json',
        original_plan_unchanged_sha256=sha(stage/'plan.json'),config_file=str(path),config_sha256=sha(path),
        collector_v2_contracts_sha256=sha(stage/'collector-v2-contracts.json'),recovery_receipt_sha256=config['recovery_receipt_sha256'],
        model_work=0,physics_or_controller_changed=False))
    print(json.dumps(dict(frozen_files=len(files),config_sha256=sha(path),identity_sha256=sha(identity))))


if __name__=='__main__':main()
