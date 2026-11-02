"""Single16-training-sample gradient-unit calibration; no updates or val tuning."""
import statistics,time
from common import ROOT,read,write,sha
def norm(grads):
    import torch
    return torch.sqrt(sum(v.float().square().sum() for v in grads if v is not None))
def cosine(a,b):
    import torch
    dot=sum(x.float().mul(y.float()).sum() for x,y in zip(a,b) if x is not None and y is not None)
    den=norm(a)*norm(b)
    return None if not bool(torch.isfinite(den)) or float(den)==0 else float(dot/den)
def run(permit,model,native,dataset,rngs,ledger):
    import torch
    from native_forward import forward_bundle
    from objectives import parts,palette
    begin=time.perf_counter();params=tuple(model.parameters());indices=read(ROOT/'schedule.json')['native_h1_batches'][0]
    assert len(indices)==16
    batch=dataset.model_batch(indices);images=torch.as_tensor(batch['image'],device='cuda');actions=torch.as_tensor(batch['action'],device='cuda')
    rows=[];reference_grads=None;reference_value=None;ratios={'G':[],'T':[]};model.train();model.zero_grad(set_to_none=True);first_noise=None
    for j in range(4):
        x=images[j*4:j*4+4];a=actions[j*4:j*4+4];noise=native.draw_noise(rngs,4,model.device)
        if j==0:first_noise=noise
        if j==0:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                with ledger.phase('preflight/REFERENCE'),torch.autocast('cuda',dtype=torch.float16):reference,finite=native.residual_loss(model,x[:,:4],x[:,4],a,noise)
                if not bool(finite):raise FloatingPointError('Original native preflight nonfinite')
                ledger.grad_begin('REFERENCE',True);reference_grads=torch.autograd.grad(reference,params,allow_unused=True);ledger.grad_end('REFERENCE',True);reference_value=reference.detach()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with ledger.phase('preflight/CALIBRATION'),torch.autocast('cuda',dtype=torch.float16):bundle=forward_bundle(permit,model,x[:,:4],x[:,4],a,noise)
            if not bool(bundle['finite']):raise FloatingPointError('Native bundle input/loss nonfinite')
            q=parts(bundle);grads={};metrics={}
            for key in ['native','G','T']:
                ledger.grad_begin('CALIBRATION_'+key,True);gg=torch.autograd.grad(q[key],params,retain_graph=key!='T',allow_unused=True);ledger.grad_end('CALIBRATION_'+key,True)
                n=norm(gg);good=bool(torch.isfinite(n)) and float(n)>0 and all(bool(torch.isfinite(v).all()) for v in gg if v is not None)
                grads[key]=gg;metrics[key]=dict(loss=float(q[key].detach()),gradient_norm=float(n) if bool(torch.isfinite(n)) else None,finite_nonzero=good)
                if key=='native' and not good:raise FloatingPointError('Original native gradient unavailable')
            equal=None
            if j==0:
                equal=dict(loss_bitwise_equal=torch.equal(reference_value,q['native'].detach()),parameter_gradients_bitwise_equal=all((a is None and b is None) or(a is not None and b is not None and torch.equal(a,b)) for a,b in zip(reference_grads,grads['native'])))
                if not all(equal.values()):raise RuntimeError('Native loss/gradient parity failed')
                del reference_grads,reference_value
            for key in ['G','T']:
                if metrics[key]['finite_nonzero']:ratios[key].append(metrics['native']['gradient_norm']/metrics[key]['gradient_norm'])
            colors,_,_=palette();D=bundle['reconstructed_unquantized'];p=q['p'];u=q['u'];palette_rgb=colors[0][None,:,None,None]+p[:,0:1]*(colors[1]-colors[0])[None,:,None,None]+p[:,1:2]*(colors[2]-colors[0])[None,:,None,None]
            rows.append(dict(microbatch=j,indices=indices[j*4:j*4+4],noise_branch=noise.branch,conditioning_rng_hashes=rngs.hashes(),objectives=metrics,native_parity=equal,
                parameter_gradient_cosines=dict(native_G=cosine(grads['native'],grads['G']),native_T=cosine(grads['native'],grads['T']),G_T=cosine(grads['G'],grads['T'])),
                u_saturated_fraction=float(((u<=0)|(u>=1)).float().mean()),u_strict_outside_fraction=float(((u<0)|(u>1)).float().mean()),u_exact_boundary_fraction=float(((u==0)|(u==1)).float().mean()),mask_areas=p.sum((-2,-1)).detach().cpu().tolist(),RGB_palette_residual_mean=float((D-palette_rgb).square().sum(1).sqrt().mean())))
            write(ROOT/'preflight-progress.json',dict(rows=rows,model_image_forward_count=sum(v['images_completed'] for k,v in ledger.counts.items() if k.startswith('preflight/'))))
            del grads,bundle,q,gg,D,p,u,palette_rgb
    coefficients=dict(cG=statistics.median(ratios['G']) if len(ratios['G'])==4 else None,cT=statistics.median(ratios['T']) if len(ratios['T'])==4 else None,
        ratios=ratios,method='Ordinary even-sample median(mean of middle2); first4 B4 training microbatches, same native forward, full model FP32 parameter-gradient norms before clipping. No optimizer step or val selection.',targets=indices)
    write(ROOT/'coefficients.json',coefficients)
    from objectives import loss_for
    arm_checks=[]
    for arm in ['G_ONLY','NATIVE_G','NATIVE_T']:
        if coefficients['cT' if arm=='NATIVE_T' else 'cG'] is None:
            arm_checks.append(dict(arm=arm,usable=False,reason='Calibration zero/nonfinite gradient'));continue
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            with ledger.phase('preflight/PARITY'),torch.autocast('cuda',dtype=torch.float16):bundle=forward_bundle(permit,model,images[:4,:4],images[:4,4],actions[:4],first_noise)
            value,_=loss_for(arm,bundle,coefficients);ledger.grad_begin('FINAL_'+arm,True);gg=torch.autograd.grad(value,params,allow_unused=True);ledger.grad_end('FINAL_'+arm,True)
            nn=norm(gg);usable=bool(torch.isfinite(nn)) and float(nn)>0 and bool(torch.isfinite(value)) and all(bool(torch.isfinite(v).all()) for v in gg if v is not None)
            arm_checks.append(dict(arm=arm,usable=usable,total_loss=float(value) if bool(torch.isfinite(value)) else None,gradient_norm=float(nn) if bool(torch.isfinite(nn)) else None))
            del bundle,value,gg,nn
    result=dict(complete=True,native_parity_passed=True,G_usable=coefficients['cG'] is not None,T_usable=coefficients['cT'] is not None,usable_arms=[x['arm'] for x in arm_checks if x['usable']],arm_checks=arm_checks,coefficients=coefficients,coefficients_sha256=sha(ROOT/'coefficients.json'),rows=rows,
        model_image_forwards=sum(v['images_completed'] for k,v in ledger.counts.items() if k.startswith('preflight/')),model_gradient_calls=sum(v for k,v in ledger.backward.items() if k.startswith('preflight/') and k.endswith('/completed')),optimizer_updates=0,wall_seconds=time.perf_counter()-begin)
    write(ROOT/'preflight.json',result);model.zero_grad(set_to_none=True);return result
