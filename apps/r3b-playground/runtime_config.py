"""Resolve external assets without loading Torch or touching model state."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

APP=Path(__file__).resolve().parent
REPO=APP.parents[1]
SERVICE='tdyssey-letter-world-model-playground-v3'


def load_runtime(model=None,config_path=None):
    manifest=json.loads((REPO/'manifests/playground-models.json').read_text(encoding='utf-8'))
    config_path=Path(config_path or os.environ.get('TDYSSEY_PLAYGROUND_CONFIG',APP/'runtime.local.json')).resolve()
    local=json.loads(config_path.read_text(encoding='utf-8-sig')) if config_path.is_file() else {}
    model=model or local.get('model',manifest['default_model'])
    if model not in manifest['models']:raise ValueError('Unknown model profile: '+str(model))
    profile=manifest['models'][model]
    def path(value,default):
        if value is None:return Path(default).resolve()
        p=Path(value).expanduser()
        return (config_path.parent/p).resolve() if not p.is_absolute() else p.resolve()
    assets=path(local.get('asset_root'),REPO)
    checkpoint=path(local.get('checkpoints',{}).get(model),assets/profile['checkpoint_relative'])
    return dict(model_id=model,profile=profile,config_path=config_path,asset_root=assets,checkpoint=checkpoint,
                gpc_root=path(local.get('gpc_root'),REPO/'third_party/gpc_code'),
                physics_root=path(local.get('physics_root'),REPO/'third_party/diffusion_policy'),
                report_root=path(local.get('report_root'),assets/'artifacts/high-motion-fit-20261101/report'),
                shared_storage_plan=path(local['shared_storage_plan'],REPO) if local.get('shared_storage_plan') else None)


def validate_assets(config,verify_hash=True):
    problems=[];checkpoint=config['checkpoint'];profile=config['profile']
    if not checkpoint.is_file():problems.append('Missing checkpoint: '+str(checkpoint))
    elif checkpoint.stat().st_size!=profile['checkpoint_bytes']:problems.append('Checkpoint size differs from selected '+config['model_id'])
    elif verify_hash:
        with checkpoint.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=profile['checkpoint_sha256']:problems.append('Checkpoint SHA-256 differs from selected '+config['model_id'])
    for key,relative in [('gpc_root','gpc_rank_evaluation/diffusion/denoiser.py'),('physics_root','diffusion_policy/env/pusht/pusht_env.py')]:
        if not (config[key]/relative).is_file():problems.append('Missing '+key+': '+str(config[key]))
    if config['shared_storage_plan'] is not None and not config['shared_storage_plan'].is_file():problems.append('Missing shared storage plan: '+str(config['shared_storage_plan']))
    if problems:
        raise FileNotFoundError('\n'.join(problems)+'\nConfigure external assets in '+str(config['config_path'])+' (see runtime.example.json). Weights stay outside Git.')
    return dict(model_id=config['model_id'],checkpoint_sha256=profile['checkpoint_sha256'],checkpoint_bytes=profile['checkpoint_bytes'],assets_verified=True,model_calls=0)


def verify_checkpoint_payload(payload,profile):
    if payload.get('kind')!=profile['checkpoint_kind']:raise ValueError('Unexpected checkpoint format')
    progress=payload['progress']
    if progress['successful_updates']!=profile['successful_updates'] or progress['lineage_updates']!=profile['lineage_updates']:
        raise ValueError('Checkpoint progress differs from selected model profile')
