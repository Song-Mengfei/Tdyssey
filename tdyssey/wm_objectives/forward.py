"""Single native H1 forward exposes losses without choosing candidate formulas."""

def forward_bundle(model,history,target,actions,noise):
    import torch
    import torch.nn.functional as F
    b=len(history);cfg=model.cfg
    if history.shape!=(b,4,3,96,96) or target.shape!=(b,3,96,96) or actions.shape!=(b,4,2):raise ValueError('Native input shape changed')
    if cfg.sigma_data!=.5 or cfg.sigma_offset_noise!=.1 or not cfg.noise_previous_obs:raise ValueError('Native preconditioning changed')
    flat=history.reshape(b,12,96,96)
    h=flat if noise.branch=='clean' else flat+noise.history_epsilon*noise.sigma_cond[:,None,None,None]
    z=target+noise.target_offset+noise.target_epsilon*noise.sigma[:,None,None,None]
    cs=model.compute_conditioners(noise.sigma,noise.sigma_cond)
    output=model.compute_model_output(z,h,actions,cs)
    residual=(target-cs.c_skip*z)/cs.c_out
    native_loss=F.mse_loss(output.float(),residual.float())
    reconstructed=cs.c_skip*z+cs.c_out*output.float()
    return dict(native_loss=native_loss,residual_error=output.float()-residual.float(),reconstructed_unquantized=reconstructed,
                target=target,sigma_draw=noise.sigma,sigma_effective=(noise.sigma.square()+cfg.sigma_offset_noise**2).sqrt(),
                conditioners=cs,noise_branch=noise.branch,additional_model_forwards=0,
                finite=torch.stack([torch.isfinite(x).all() for x in (history,target,actions,output,native_loss,reconstructed)]).all()
                    &((history>=0)&(history<=1)).all()&((target>=0)&(target<=1)).all()&((actions>=-1)&(actions<=1)).all())
