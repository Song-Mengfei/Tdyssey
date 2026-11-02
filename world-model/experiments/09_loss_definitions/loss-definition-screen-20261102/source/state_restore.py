"""Lazy runtime helpers; one CPU parent and one active optimizer, no disk clones."""
from copy import deepcopy
import hashlib
from authorization import require_permit

def load_parent(permit):
    c=require_permit(permit)
    with open(c['parent'],'rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=c['parent_sha256']:raise RuntimeError('Parent hash mismatch')
    import torch
    parent=torch.load(c['parent'],map_location='cpu',weights_only=False,mmap=True)
    if parent['kind']!='oc_native_rgb_complete.v1' or not parent['progress']['epoch_complete'] or parent['progress']['lineage_updates']!=11367:
        raise RuntimeError('Original completed E2 boundary required')
    return parent

def restore_full(permit,model,parent,arm,io):
    require_permit(permit)
    import torch
    from tdyssey.raw_world_model_training import OPTIMIZER,make_generator
    from tdyssey.native_rgb_training import restore_rng,capture_rng
    from tdyssey.native_rgb_r3a_conditioning import make_conditioning_rngs
    from tdyssey.native_rgb_r2c1r1_training import _tree_digest
    digest=lambda x:_tree_digest(x,torch)
    model.load_state_dict(parent['model'],strict=True);model.requires_grad_(True)
    opt=torch.optim.AdamW(model.parameters(),**dict(OPTIMIZER,lr=1e-5));opt.load_state_dict(deepcopy(parent['optimizer']))
    scaler=torch.amp.GradScaler('cuda',enabled=True,init_scale=128);scaler.load_state_dict(deepcopy(parent['scaler']))
    generator=make_generator(5478,model.device);restore_rng(parent['global_rng'],generator)
    rngs=make_conditioning_rngs(**parent['conditioning_rngs']['seeds'],device=model.device);rngs.load_state_dict(parent['conditioning_rngs'])
    proof=dict(model=digest(model.state_dict())==digest(parent['model']),optimizer=digest(opt.state_dict())==digest(parent['optimizer']),
               scaler=digest(scaler.state_dict())==digest(parent['scaler']),global_rng=digest(capture_rng(generator))==digest(parent['global_rng']),
               conditioning_rng=digest(rngs.state_dict())==digest(parent['conditioning_rngs']))
    io.write_json('restore-'+arm+'.json',proof)
    if not all(proof.values()) or any(x['lr']!=1e-5 for x in opt.param_groups):raise RuntimeError('Independent full restore mismatch')
    return opt,scaler,generator,rngs

