"""Real-history native residual training only; no generation or region weights.
No Torch/model/GPU work occurs on import. Existing objective/noise math is reused.
"""
from __future__ import annotations
import math
from types import SimpleNamespace

def draw_noise(rngs,batch_size,device):
    """Exactly the original three-stream conditioning draws, without hot hashes."""
    import torch
    b=int(batch_size)
    uniform=float(torch.rand((),generator=rngs.branch,device='cpu',dtype=torch.float32))
    branch='clean'if uniform<.5 else'weak'
    def normal(shape):return torch.randn(shape,device=device,dtype=torch.float32,generator=rngs.target)
    sigma=(normal((b,))*1.2-1.2).exp().clamp(.002,20.)
    offset=.1*normal((b,3,1,1));epsilon=normal((b,3,96,96))
    if branch=='clean':sigma_cond=None;history_epsilon=None
    else:
        u=torch.rand((b,),device=device,dtype=torch.float32,generator=rngs.weak)
        sigma_cond=(math.log(.002)+u*(math.log(.05)-math.log(.002))).exp().clamp(.002,.05)
        history_epsilon=torch.randn((b,12,96,96),device=device,dtype=torch.float32,generator=rngs.weak)
    return SimpleNamespace(branch=branch,branch_uniform=uniform,sigma=sigma,target_offset=offset,target_epsilon=epsilon,
        sigma_cond=sigma_cond,history_epsilon=history_epsilon)

def residual_loss(model,history,target,actions,noise):
    """Native FP32 residual MSE; generated history has no artificial RGB range cap.

    A combined boolean tensor is checked at the optimizer boundary, reducing
    host synchronizations without accepting nonfinite forward/loss values.
    """
    import torch
    import torch.nn.functional as F
    b=len(history)
    if (history.shape!=(b,4,3,96,96) or target.shape!=(b,3,96,96) or actions.shape!=(b,4,2)
            or any(v.dtype!=torch.float32 or v.requires_grad for v in (history,target,actions))): raise ValueError('Split constant FP32 history/target/action contract differs')
    cfg=model.cfg
    if cfg.sigma_data!=.5 or cfg.sigma_offset_noise!=.1 or cfg.noise_previous_obs is not True or cfg.inner_model.num_steps_conditioning!=4: raise ValueError('Frozen native residual recipe required')
    flat=history.reshape(b,12,96,96)
    noisy_history=flat if noise.branch=='clean'else flat+noise.history_epsilon*noise.sigma_cond[:,None,None,None]
    noisy_target=target+noise.target_offset+noise.target_epsilon*noise.sigma[:,None,None,None]
    conditioners=model.compute_conditioners(noise.sigma,noise.sigma_cond)
    output=model.compute_model_output(noisy_target,noisy_history,actions,conditioners)
    if output.shape!=target.shape:raise ValueError('Native residual output shape mismatch')
    residual=(target-conditioners.c_skip*noisy_target)/conditioners.c_out
    loss=F.mse_loss(output.float(),residual.float())
    if loss.dtype!=torch.float32:raise ValueError('Residual loss must remain FP32 under AMP')
    finite=torch.stack([torch.isfinite(x).all()for x in (history,target,actions,output,loss)]).all()
    true_target_range=((target>=0)&(target<=1)).all()
    true_history_range=((history>=0)&(history<=1)).all()
    original_action_range=((actions>=-1)&(actions<=1)).all()
    return loss,finite&true_target_range&true_history_range&original_action_range

def microbatch_backward(model,images,actions,rngs,scaler,*,timer=None,observer=None):
    import torch
    if images.shape!=(16,5,3,96,96)or actions.shape!=(16,4,2):raise ValueError('Exactly16real-history H1 targets per optimizer attempt')
    flags=[];losses=[];branches=[]
    for micro in range(4):
        if observer:observer('supervised_start',micro,4)
        batch=images[micro*4:(micro+1)*4];act=actions[micro*4:(micro+1)*4]
        noise=draw_noise(rngs,4,images.device);stamp=timer.start()if timer else None
        model.train()
        with torch.autocast(device_type=images.device.type,dtype=torch.float16,enabled=images.device.type=='cuda'):
            loss,finite=residual_loss(model,batch[:,:4],batch[:,4],act,noise)
        if observer:observer('loss_complete',micro,4)
        if observer:observer('backward_attempt',micro,4)
        scaler.scale(loss*.25).backward()
        if observer:observer('backward_complete',micro,4)
        if timer:timer.stop('loss_forward_backward_seconds',stamp)
        flags.append(finite);losses.append(loss.detach());branches.append(noise.branch)
    return dict(loss=torch.stack(losses).mean(),forward_finite=torch.stack(flags).all(),branches=branches,
        supervised_batch_forwards=4,backward_calls=4,generated_history_frames=0)
