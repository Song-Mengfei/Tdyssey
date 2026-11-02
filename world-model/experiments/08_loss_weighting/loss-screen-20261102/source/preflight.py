"""Bounded equation/gradient checks; no optimizer steps or extra training data."""
import time
from authorization import require_permit

def tensor_checks(io):
    import torch
    from objectives import gdl,ffl
    start=time.perf_counter();device='cuda';gen=torch.Generator(device=device).manual_seed(2026110209)
    z=torch.zeros(1,1,2,2,device=device,requires_grad=True);zero=ffl(z,z.detach())
    zg=torch.autograd.grad(zero,z)[0]
    q=torch.tensor([[[[0.,1.],[0.,1.]]]],device=device,requires_grad=True)
    edge=gdl(q,torch.zeros_like(q));eg=torch.autograd.grad(edge,q)[0]
    v=torch.rand(2,3,96,96,device=device,generator=gen);q=torch.roll(v,1,-1).detach().requires_grad_()
    frequency,matrix=ffl(q,v,return_matrix=True);fg=torch.autograd.grad(frequency,q)[0]
    delta=torch.fft.fft2(q.detach(),norm='ortho')-torch.fft.fft2(v,norm='ortho')
    parseval=torch.allclose(delta.abs().square().mean(),(q.detach()-v).square().mean(),atol=1e-6,rtol=1e-5)
    same_amplitude=torch.allclose(torch.fft.fft2(q.detach(),norm='ortho').abs(),torch.fft.fft2(v,norm='ortho').abs(),atol=2e-5,rtol=2e-5)
    s=torch.tensor([.002,.1,1.,20.],device=device);rho=.25/(s.square()+.1**2);w=rho.clamp(max=5)/rho
    checks=dict(ffl_zero_exact=float(zero)==0,ffl_zero_gradient_finite=bool(torch.isfinite(zg).all()),ffl_zero_gradient_exact=bool((zg==0).all()),
        gdl_zero_exact=float(gdl(q,v))>=0 and float(gdl(v,v))==0,gdl_no_padding_2x2=float(edge)==.5,
        gdl_gradient_finite=bool(torch.isfinite(eg).all()),ffl_phase_sensitive=float(frequency)>0 and bool(same_amplitude),
        ffl_matrix_detached=not matrix.requires_grad,ffl_matrix_per_channel=bool(torch.all(matrix.amax((-2,-1))==1)),
        ffl_gradient_finite_nonzero=bool(torch.isfinite(fg).all()) and float(fg.abs().sum())>0,
        fft_parseval=bool(parseval),snr_weight_bounded=bool(((w>0)&(w<=1)).all()),snr_eff_offset_used=float(w[0])<.201 and float(w[1])>.399)
    torch.cuda.synchronize();record=dict(passed=all(checks.values()),checks=checks,wall_seconds=time.perf_counter()-start,
        tensor_only_gradient_calls=3,tensor_only_model_forwards=0,optimizer_updates=0,ffl_dtype='float32/complex64')
    io.write_json('tensor-only-preflight.json',record)
    if not record['passed']:raise RuntimeError('Tensor-only frozen-equation check failed')
    return record

def run(permit,model,native,dataset,schedule,rngs,ledger,io):
    require_permit(permit)
    import torch
    from native_forward import forward_bundle
    from objectives import loss_for
    start=time.perf_counter();tensor=tensor_checks(io)
    data=dataset.model_batch(schedule['native_h1_batches'][0][:4])
    image=torch.as_tensor(data['image'],device='cuda');actions=torch.as_tensor(data['action'],device='cuda')
    history=image[:,:4];target=image[:,4];noise=native.draw_noise(rngs,4,model.device)
    params=tuple(model.parameters());model.train();model.zero_grad(set_to_none=True)
    rows=[];reference=None;reference_loss=None
    for arm in ('REFERENCE','BASE','SNR','GDL','FFL'):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with ledger.phase('preflight/'+arm),torch.autocast('cuda',dtype=torch.float16):
                if arm=='REFERENCE':loss,finite=native.residual_loss(model,history,target,actions,noise);bundle=None
                else:
                    bundle=forward_bundle(permit,model,history,target,actions,noise);loss,_=loss_for(arm,bundle);finite=bundle['finite']
            if not bool(finite) or not bool(torch.isfinite(loss)):raise FloatingPointError('Model preflight loss/input nonfinite')
            ledger.grad_begin(arm,preflight=True)
            grads=torch.autograd.grad(loss,params,retain_graph=arm=='SNR',allow_unused=True)
            ledger.grad_end(arm,preflight=True)
            norm=torch.sqrt(sum(x.float().square().sum() for x in grads if x is not None))
            good=bool(torch.isfinite(norm)) and float(norm)>0 and all(bool(torch.isfinite(x).all()) for x in grads if x is not None)
            row=dict(arm=arm,loss=float(loss.detach()),gradient_norm=float(norm),finite_nonzero=good)
            if not good:raise FloatingPointError('Model candidate parameter gradient nonfinite/zero')
            if arm=='REFERENCE':reference=grads;reference_loss=loss.detach()
            elif arm=='BASE':
                row['loss_bitwise_equal']=torch.equal(reference_loss,loss.detach())
                row['parameter_gradients_bitwise_equal']=all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(reference,grads))
                if not row['loss_bitwise_equal'] or not row['parameter_gradients_bitwise_equal']:raise RuntimeError('BASE differs from original native loss/gradient')
                del reference
            elif arm=='SNR':
                rho=.25/(noise.sigma.square()+.1**2);w=rho.clamp(max=5)/rho
                equivalent=(((bundle['reconstructed_unquantized']-target)/bundle['conditioners'].c_out).square().mean((1,2,3))*w).mean()
                row['reconstruction_weighted_equivalence']=bool(torch.allclose(loss,equivalent,atol=1e-6,rtol=1e-5))
                ledger.grad_begin('SNR_equivalent',preflight=True);eqgrads=torch.autograd.grad(equivalent,params,allow_unused=True);ledger.grad_end('SNR_equivalent',preflight=True)
                row['equivalent_gradient_close']=all((a is None and b is None) or (a is not None and b is not None and torch.allclose(a,b,atol=1e-6,rtol=1e-4)) for a,b in zip(grads,eqgrads))
                del eqgrads
                if not row['reconstruction_weighted_equivalence'] or not row['equivalent_gradient_close']:raise RuntimeError('SNR equivalent objective differs')
            rows.append(row);io.write_json('preflight-progress.json',dict(rows=rows,ledger=ledger.snapshot()))
            del grads,loss,bundle,norm
    model.zero_grad(set_to_none=True);torch.cuda.synchronize()
    result=dict(passed=True,rows=rows,tensor_only=tensor,wall_seconds=time.perf_counter()-start,optimizer_updates=0,
                same_single_B4_noise_plan=True,model_image_forwards=20,model_gradient_calls=6,ledger=ledger.snapshot())
    io.write_json('preflight.json',result);return result

