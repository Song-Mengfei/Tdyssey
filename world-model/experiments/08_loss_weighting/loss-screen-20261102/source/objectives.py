"""Frozen TEST_SPEC equations; no import-time Torch/model/asset work.
Sources: Hang ICCV2023 Appendix D.1.1; Mathieu ICLR2016 Eq6/7;
Jiang ICCV2021 and EndlessSora/focal-frequency-loss author implementation.
GPC adaptations/coefficient/unit choices are fixed in SOURCE_TEST_SPEC.txt.
"""

def gdl(q,v):
    dy=((q[:,:,1:,:]-q[:,:,:-1,:]).abs()-(v[:,:,1:,:]-v[:,:,:-1,:]).abs()).abs().mean()
    dx=((q[:,:,:,1:]-q[:,:,:,:-1]).abs()-(v[:,:,:,1:]-v[:,:,:,:-1]).abs()).abs().mean()
    return .5*(dy+dx)

def ffl(q,v,*,return_matrix=False):
    import torch
    with torch.autocast(device_type=q.device.type,enabled=False):
        delta=torch.fft.fft2(q.float(),norm='ortho')-torch.fft.fft2(v.float(),norm='ortho')
        magnitude=delta.abs();maximum=magnitude.amax(dim=(-2,-1),keepdim=True)
        safe_maximum=torch.where(maximum==0,torch.ones_like(maximum),maximum)
        matrix=(magnitude/safe_maximum).detach()
        loss=(matrix*(delta.real.square()+delta.imag.square())).mean()
    return (loss,matrix) if return_matrix else loss

def loss_for(arm,bundle):
    import torch
    with torch.autocast(device_type=bundle['target'].device.type,enabled=False):
        base=bundle['native_loss']
        if arm=='BASE':return base,dict(base=base.detach(),extra=None)
        if arm=='SNR':
            rho=.5**2/(bundle['sigma_draw'].square()+.1**2)
            weight=rho.clamp(max=5)/rho
            value=(bundle['residual_error'].square().mean((1,2,3))*weight).mean()
            return value,dict(base=base.detach(),extra=None,weight_min=weight.min().detach(),weight_max=weight.max().detach())
        q=bundle['reconstructed_unquantized']/.5;v=bundle['target']/.5
        extra=gdl(q,v) if arm=='GDL' else ffl(q,v) if arm=='FFL' else None
        if extra is None:raise ValueError('Only frozen BASE/SNR/GDL/FFL arms')
        return base+extra,dict(base=base.detach(),extra=extra.detach())

