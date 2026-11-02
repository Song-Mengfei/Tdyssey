"""Four-target residual supervision with detached, native generated histories.

No model/GPU work occurs on import. The generation API has no target/state input.
Native sampler source/math are invoked directly; only its random stream is scoped.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
from types import SimpleNamespace

SAMPLER=dict(num_steps_denoising=3,sigma_min=.002,sigma_max=5.,rho=7,order=1,s_churn=0.,s_cond=0.)

@contextmanager
def independent_native_rng(generator,device):
    """Keep the upstream sampler unchanged and isolate its sole noise stream."""
    import torch
    if device.type=='cuda':
        previous=torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(generator.get_state(),device)
        try: yield
        finally:
            generator.set_state(torch.cuda.get_rng_state(device));torch.cuda.set_rng_state(previous,device)
    elif device.type=='cpu':
        previous=torch.get_rng_state();torch.set_rng_state(generator.get_state())
        try: yield
        finally: generator.set_state(torch.get_rng_state());torch.set_rng_state(previous)
    else: raise ValueError('Native CPU contract tests or authorized local CUDA only')

class NativeHistoryGenerator:
    """Original3-denoising, FP32/eval/no_grad sampling from pure noise only."""
    def __init__(self,model,source,generator):
        self.model=model;self.generator=generator
        self.sampler=source.DiffusionSampler(model,source.DiffusionSamplerConfig(**SAMPLER))
        self.batch_calls=0;self.generated_frames=0;self.denoiser_batch_forwards=0
    def generate(self,history,actions):
        import torch
        if history.shape[1:]!=(4,3,96,96) or actions.shape!=(len(history),4,2): raise ValueError('Pure history/action native generation contract')
        if history.requires_grad or actions.requires_grad or history.dtype!=torch.float32 or actions.dtype!=torch.float32: raise ValueError('Detached FP32 generation inputs required')
        mode=self.model.training
        try:
            self.model.eval()
            with torch.no_grad(),torch.autocast(device_type=history.device.type,enabled=False),independent_native_rng(self.generator,history.device):
                raw,_=self.sampler.sample(history,actions)
            # no_grad creates ordinary tensors (not inference_mode tensors).
            if raw.shape!=(len(history),3,96,96) or raw.dtype!=torch.float32: raise ValueError('Native generated RGB shape/dtype differs')
            self.batch_calls+=1;self.generated_frames+=len(history);self.denoiser_batch_forwards+=3
            return raw.detach()
        finally:self.model.train(mode)

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
    return loss,finite&true_target_range

def step_history(real_images,current,generated,h,arm):
    """h is zero-based; A uses each real prefix, B appends exactly the last P."""
    import torch
    if arm=='A':return real_images[:,h:h+4]
    if arm!='B':raise ValueError('Only the approved A/B arms')
    return torch.cat((current[:,1:],generated[:,None]),dim=1).detach()

def four_step_backward(model,images,actions,rngs,generator,scaler,arm,*,timer=None,observer=None):
    """Four equally weighted loss graphs; three detached Ps only in B.

    Parameters are unchanged throughout; the caller alone performs one Adam step.
    """
    import torch
    if images.shape[1:]!=(8,3,96,96)or actions.shape!=(len(images),7,2):raise ValueError('Eight images/seven actions required')
    history=images[:,:4];flags=[];losses=[];branches=[];generated=[]
    for h in range(4):
        if observer:observer('supervised_start',h,len(images))
        if arm=='A':history=images[:,h:h+4]
        action4=actions[:,h:h+4]
        noise=draw_noise(rngs,len(images),images.device)
        start=timer.start()if timer else None
        model.train()
        with torch.autocast(device_type=images.device.type,dtype=torch.float16,enabled=images.device.type=='cuda'):
            loss,finite=residual_loss(model,history,images[:,h+4],action4,noise)
        if observer:observer('loss_complete',h,len(images))
        if observer:observer('backward_attempt',h,len(images))
        scaler.scale(loss*.25).backward()
        if observer:observer('backward_complete',h,len(images))
        if timer:timer.stop('loss_forward_backward_seconds',start)
        flags.append(finite);losses.append(loss.detach());branches.append(noise.branch)
        if arm=='B'and h<3:
            if observer:observer('generation_start',h,len(images))
            start=timer.start()if timer else None
            p=generator.generate(history.detach(),action4.detach())
            if observer:observer('generation_complete',h,len(images))
            if timer:timer.stop('generation_seconds',start)
            flags.append(torch.isfinite(p).all());generated.append(p)
            history=step_history(images,history,p,h,arm)
    return dict(loss=torch.stack(losses).mean(),forward_finite=torch.stack(flags).all(),branches=branches,
        supervised_batch_forwards=4,backward_calls=4,generated=generated)
