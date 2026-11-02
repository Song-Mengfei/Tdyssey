"""Training-only native-forward quantizer with an explicit surrogate derivative.

No changes to upstream sampler/inference. Inputs contain only RGB/action/noise.
Every four-step rollout retains its whole graph, including generated histories.
"""
from contextlib import nullcontext
import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,d):
        ctx.save_for_backward(d)
        return d.clamp(-1,1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
    @staticmethod
    def backward(ctx,grad):
        (d,)=ctx.saved_tensors
        # Identity derivative inside clamp; zero outside. Byte quantization's
        # true derivative is zero almost everywhere. This is NOT an exact gradient.
        return grad*((d>=-1)&(d<=1))

def sample(model,history,actions,epsilon,sigmas,*,checkpointed=True,recompute_context=nullcontext):
    b=len(history);flat=history.reshape(b,12,96,96);x=epsilon
    for sigma,next_sigma in zip(sigmas[:-1],sigmas[1:]):
        # Pass sigma explicitly: no late-binding loop closure during backward.
        def forward(z,h,a,s):
            cs=model.compute_conditioners(s,None)
            return model.compute_model_output(z,h,a,cs)
        if checkpointed and torch.is_grad_enabled():
            with set_checkpoint_early_stop(False):
                output=checkpoint(forward,x,flat,actions,sigma,use_reentrant=False,
                                  context_fn=lambda:(nullcontext(),recompute_context()))
        else:output=forward(x,flat,actions,sigma)
        cs=model.compute_conditioners(sigma,None)
        denoised=QuantizeSTE.apply(cs.c_skip*x+cs.c_out*output)
        x=x+(x-denoised)/sigma*(next_sigma-sigma)
    return x

def rollout(model,history,actions,epsilons,sigmas,**kwargs):
    outputs=[]
    for k in range(4):
        pred=sample(model,history,actions[:,k:k+4].contiguous(),epsilons[k],sigmas,**kwargs)
        outputs.append(pred)
        history=torch.cat((history[:,1:],pred[:,None]),dim=1)
    return outputs

def objective(outputs,targets):
    # Four RGB targets, all pixels/channels equal. No masks, poses or templates.
    # lambda=1/sigma_data**2=4, fixed before any forward results.
    return 4*torch.stack([(p-y).square().mean() for p,y in zip(outputs,targets.unbind(1))]).mean()
