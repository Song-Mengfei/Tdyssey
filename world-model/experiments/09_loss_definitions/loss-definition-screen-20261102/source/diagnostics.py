"""Bounded GT-only definition responses. Constructed errors are never training data."""
import math,time,io
from common import ROOT,read,write,write_bytes
GRADIENT_CALLS=0
def grad(value,x):
    global GRADIENT_CALLS
    import torch
    if GRADIENT_CALLS>=64:raise RuntimeError('Tensor-only gradient cap')
    GRADIENT_CALLS+=1
    return torch.autograd.grad(value,x,retain_graph=True)[0]
def compose(m):
    from objectives import palette
    colors,_,_=palette(str(m.device))
    # Diagnostic construction only. Linear RGB mixing is not an output repair.
    return colors[0][None,:,None,None]+m[:,0:1]*(colors[1]-colors[0])[None,:,None,None]+m[:,1:2]*(colors[2]-colors[0])[None,:,None,None]
def shift(m,dx=0.,dy=0.,angle=0.,center=None):
    import torch
    import torch.nn.functional as F
    axis=torch.arange(96,device=m.device,dtype=torch.float32);y,x=torch.meshgrid(axis,axis,indexing='ij')
    cx,cy=(47.5,47.5) if center is None else center
    co=math.cos(angle);si=math.sin(angle)
    # Pullback sampling: positive dx moves the foreground right.
    xx=x-dx-cx;yy=y-dy-cy
    sx=co*xx+si*yy+cx;sy=-si*xx+co*yy+cy
    grid=torch.stack([sx/47.5-1,sy/47.5-1],-1)[None]
    return F.grid_sample(m,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
def distances(g):
    import numpy as np,torch,cv2
    values=[]
    for mask in (g[0].detach().cpu().numpy()>.5).astype(np.uint8):
        # GT RGB-derived binary mask only; no simulator geometry/state labels.
        outside=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        inside=cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        values.append((outside-inside)/(95*math.sqrt(2)))
    return torch.tensor(np.stack(values)[None],device=g.device)
def values(image,target,g,phi):
    import torch
    from objectives import masks,geometry,transport,soft_jaccard,cldice,moments
    p=masks(image);T,w,area=transport(p,g,True);cen,cov=moments(p,g)
    return dict(image_MSE=(image-target).square().mean(),G=geometry(p,g),soft_Jaccard=soft_jaccard(p,g),
        signed_boundary=(p*phi).mean(),W1=w,relative_area=area,T=T,clDice_foreground=cldice(p,g),
        clDice_background=cldice(1-p,1-g),centroid_squared=cen,covariance_squared=cov)
def run():
    import torch,numpy as np,cv2
    import torch.nn.functional as F
    from PIL import Image,ImageDraw
    from objectives import masks,geometry,transport,palette
    started=time.perf_counter();spec=read(ROOT/'evaluation-spec.json');cases=[r for b in spec['batches'] for r in b['rows'] if r['selection_role']=='train']
    assert len(cases)==12 and all(sum(r['letter']==l for r in cases)==4 for l in ['O','C','B'])
    colors,P,A=palette();pure=colors[:,:,None,None];pure_masks,u=masks(pure,True)
    expected=torch.tensor([[0.,0.],[1.,0.],[0.,1.]],device='cuda')[:,:,None,None]
    zero=torch.zeros(1,2,96,96,device='cuda');full=zero.clone();full[:,:,20:30,20:30]=1
    clamp_probe=torch.tensor([-.1,0.,.5,1.,1.1],device='cuda',requires_grad=True)
    clamp_gradient=grad(clamp_probe.clamp(0,1).sum(),clamp_probe)
    equation_checks=dict(pure_colors_close=bool(torch.allclose(pure_masks,expected,atol=2e-6,rtol=0)),equal_G_zero=float(geometry(full,full))==0,equal_T_zero=float(transport(full,full))==0,
        both_empty_G_zero=float(geometry(zero,zero))==0,both_empty_T_zero=float(transport(zero,zero))==0,clamp_gradients=clamp_gradient.detach().cpu().tolist(),P=P.cpu().tolist(),A=A.cpu().tolist(),condition_number=float(torch.linalg.cond(A.double().T@A.double())))
    if not all(equation_checks[k] for k in ['pure_colors_close','equal_G_zero','equal_T_zero','both_empty_G_zero','both_empty_T_zero']):raise RuntimeError('Specified mask/G/T equation check failed')
    records=[];palette_rows=[];panels=[];responses=[]
    for ci,row in enumerate(cases):
        h=next((int(t['step']) for t in row['target_motion'] if t['strong']),1);index=int(row['anchor'])+h
        with np.load(row['raw_path'],allow_pickle=False) as z:
            target=torch.tensor(z['rgb'][index][None],device='cuda',dtype=torch.float32)/255
            previous=torch.tensor(z['rgb'][index-1][None],device='cuda',dtype=torch.float32)/255
        g,u=masks(target,True);phi=distances(g)
        reconstructed=compose(g);residual=(target-reconstructed).square().sum(1).sqrt()
        # Original hard gray palette projection is measured, not replaced.
        proto=torch.tensor(read(ROOT.parent/'native-rgb-20261028/frozen-renderer-calibration.json')['object_rgb'],device='cuda',dtype=torch.float64)
        direction=1-proto;alpha=((1-target.double())*direction[None,:,None,None]).sum(1)/direction.square().sum()
        rr=(target.double()-(1-alpha[:,None]*direction[None,:,None,None])).square().sum(1).sqrt()
        hard=(alpha>=.4)&(alpha<=1.25)&(rr<=.12);newhard=g[:,0]>.4
        palette_rows.append(dict(sample_id=row['sample_id'],letter=row['letter'],horizon=h,frame=index,
            source='existing GT RGB only',max_RGB_reconstruction_residual=float(residual.max()),mean_RGB_reconstruction_residual=float(residual.mean()),
            GT_u_outside_01_fraction=float(((u<0)|(u>1)).float().mean()),GT_u_min=float(u.min()),GT_u_max=float(u.max()),
            mask_areas=g.sum((-2,-1)).cpu().tolist(),hard_gray_disagreement_pixels=int((hard!=newhard).sum())))
        # All controlled errors use this same GT. Their eligibility is explicit.
        gray=g[:,0:1];blue=g[:,1:2];occupied=(gray[0,0]>.01).nonzero();cx=float((gray[0,0]*torch.arange(96,device='cuda')[None]).sum()/(gray.sum()+1e-6));cy=float((gray[0,0]*torch.arange(96,device='cuda')[:,None]).sum()/(gray.sum()+1e-6))
        variants=[('correct_RGB',target,'eligible')]
        for axis in ['x','y']:
            for n in [2,4,8]:
                moved=shift(gray,dx=n if axis=='x' else 0,dy=n if axis=='y' else 0)
                eligible=bool(len(occupied) and (occupied[:,1 if axis=='x' else 0].max()+n<96))
                variants.append((f'shift_{axis}_{n}',compose(torch.cat([moved,blue],1)),'eligible' if eligible else 'unknown_cropped'))
        for deg in [5,-5,15,-15]:
            ang=math.radians(deg);moved=shift(gray,angle=ang,center=(cx,cy))
            dx=occupied[:,1]-cx;dy=occupied[:,0]-cy;xx=dx*math.cos(ang)-dy*math.sin(ang)+cx;yy=dx*math.sin(ang)+dy*math.cos(ang)+cy
            ok=bool(((xx>=0)&(xx<=95)&(yy>=0)&(yy<=95)).all())
            variants.append((f'rotate_{deg}',compose(torch.cat([moved,blue],1)),'eligible_symmetry_not_resolved' if ok else 'unknown_cropped'))
        variants.extend([('dilate_gray',compose(torch.cat([F.max_pool2d(gray,3,1,1),blue],1)),'eligible'),('erode_gray',compose(torch.cat([-F.max_pool2d(-gray,3,1,1),blue],1)),'eligible')])
        binary=(gray[0,0].cpu().numpy()>.5).astype(np.uint8);nb,labels,stats,_=cv2.connectedComponentsWithStats(1-binary,connectivity=8)
        border=set(np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]).tolist());holes=[i for i in range(1,nb) if i not in border]
        filled=gray.clone()
        blue_binary=(blue[0,0].cpu().numpy()>.4).astype(np.uint8)
        occluded_holes=[i for i in holes if np.any(cv2.dilate((labels==i).astype(np.uint8),np.ones((3,3),np.uint8))&blue_binary)]
        if holes and not occluded_holes:filled[0,0][torch.tensor(np.isin(labels,holes),device='cuda')]=1
        variants.append(('fill_visible_holes',compose(torch.cat([filled,blue],1)),'unknown_occluded_blue_in_or_touching_hole' if occluded_holes else 'eligible' if holes else 'unknown_no_visible_hole'))
        cut=gray.clone();cut[:,:,:,max(0,round(cx)-1):min(96,round(cx)+1)]=0
        cc0=cv2.connectedComponents(binary,connectivity=8)[0]-1;cc1=cv2.connectedComponents((cut[0,0].cpu().numpy()>.5).astype(np.uint8),connectivity=8)[0]-1
        variants.append(('cut_connection',compose(torch.cat([cut,blue],1)),'eligible' if cc1>cc0 else 'unknown_not_disconnected'))
        duplicate=torch.maximum(gray,shift(gray,dx=16));duplicate_cc=cv2.connectedComponents((duplicate[0,0].cpu().numpy()>.5).astype(np.uint8),connectivity=8)[0]-1
        duplicate_qualification='unknown_cropped' if not bool(occupied[:,1].max()+16<96) else 'unknown_not_extra_component' if duplicate_cc<=cc0 else 'eligible'
        variants.append(('duplicate_gray_block',compose(torch.cat([duplicate,blue],1)),duplicate_qualification))
        variants.extend([('remove_gray',compose(torch.cat([torch.zeros_like(gray),blue],1)),'eligible'),('remove_blue',compose(torch.cat([gray,torch.zeros_like(blue)],1)),'eligible'),('all_white',torch.ones_like(target),'eligible'),
            ('blue_shift_x4',compose(torch.cat([gray,shift(blue,dx=4)],1)),'eligible' if float(shift(blue,dx=4).sum())>=float(blue.sum())-.01 else 'unknown_cropped'),
            ('background_minus_001',target-.01*(g.sum(1,keepdim=True)<.001),'eligible'),('gray_brightness_plus_002',target+.02*gray,'eligible'),('previous_real_frame',previous,'eligible_recorded_strong_motion')])
        assert len(variants)<=24
        tile=Image.new('RGB',(6*104,4*122),'white');draw=ImageDraw.Draw(tile)
        for vi,(name,image,qualification) in enumerate(variants):
            image=image.detach();p,pu=masks(image,True)
            result=values(image,target,g,phi)
            entry=dict(sample_id=row['sample_id'],letter=row['letter'],frame=index,horizon=h,variant=name,qualification=qualification,
                losses={k:float(v) for k,v in result.items()},finite=all(bool(torch.isfinite(v)) for v in result.values()),
                u_saturated_fraction=float(((pu<=0)|(pu>=1)).float().mean()),u_strict_outside_fraction=float(((pu<0)|(pu>1)).float().mean()),u_exact_boundary_fraction=float(((pu==0)|(pu==1)).float().mean()),predicted_mask_areas=p.sum((-2,-1)).cpu().tolist(),
                GT_gray_components=cc0,constructed_gray_components=cv2.connectedComponents((p[0,0].cpu().numpy()>.5).astype(np.uint8),connectivity=8)[0]-1,
                GT_gray_binary_area=int(binary.sum()),constructed_gray_binary_area=int((p[0,0]>.5).sum()),GT_holes=len(holes),blue_occluded_holes=occluded_holes)
            if ci==0 and name in ['correct_RGB','shift_x_4','all_white','background_minus_001']:
                q=image.clone().requires_grad_();losses=values(q,target,g,phi);entry['RGB_gradients']={}
                for key,value in losses.items():
                    gg=grad(value,q);entry['RGB_gradients'][key]=dict(finite=bool(torch.isfinite(gg).all()),norm=float(gg.norm()),nonzero_pixels=int((gg.abs().sum(1)>0).sum()))
            records.append(entry)
            a=(image[0].clamp(0,1).permute(1,2,0)*255).byte().cpu().numpy();xx=(vi%6)*104;yy=(vi//6)*122
            draw.text((xx,yy),name[:18],fill='black');tile.paste(Image.fromarray(a),(xx,yy+20))
        if ci==0:
            for axis in ['x','y']:
                measured=[]
                for n in [2.,4.,8.]:
                    displacement=torch.tensor(n,device='cuda',requires_grad=True)
                    moved=shift(gray,dx=displacement if axis=='x' else 0,dy=displacement if axis=='y' else 0)
                    value=transport(torch.cat([moved,blue],1),g);derivative=grad(value,displacement)
                    measured.append(dict(displacement=n,T=float(value),derivative=float(derivative),descent_toward_GT=bool(derivative>0)))
                responses.append(dict(axis=axis,values=measured,monotone=all(a['T']<b['T'] for a,b in zip(measured,measured[1:]))))
        buf=io.BytesIO();tile.save(buf,format='PNG');path=ROOT/f'diagnosis/train-{ci:02d}.png';write_bytes(path,buf.getvalue());panels.append(dict(path=str(path),sample_id=row['sample_id'],letter=row['letter'],frame=index))
    # Same centroid, different internal shape and equal-overlap/different-distance
    # counterexamples in mask space, included without treating them as model GT.
    square=torch.zeros(1,2,96,96,device='cuda');square[:,:,42:54,42:54]=1
    near=torch.zeros_like(square);near[:,:,42:54,60:72]=1
    far=torch.zeros_like(square);far[:,:,42:54,78:90]=1
    ring=square.clone();ring[:,:,46:50,46:50]=0
    counterexamples=dict(disjoint_same_Dice_near=float(geometry(near,square)),disjoint_same_Dice_far=float(geometry(far,square)),near_T=float(transport(near,square)),far_T=float(transport(far,square)),
        same_centroid_different_shape_moments=[float(x) for x in __import__('objectives').moments(ring,square)],ring_G=float(geometry(ring,square)),ring_T=float(transport(ring,square)))
    definitions=dict(image_MSE='mean_BCHW (I-y)^2',G='squared-denominator Dice, eps1e-6; mean sample/foreground classes',soft_Jaccard='1-(sum p*g+eps)/(sum(p+g-p*g)+eps)',signed_boundary='mean_BCHW p*phi(hard RGB GT), phi=(distanceOutside-distanceInside)/(95sqrt2); negative constant offset at GT preserved',W1='mean16 fixed spatial projections, exact weightedCDF integration',relative_area='abs(sum p-sum g)/(sum g+1e-6)',T='W1+relative_area',clDice_foreground='soft skeleton10 iterations, precision/sensitivity harmonic mean',clDice_background='same on1-p/1-g; no Betti guarantee',centroid_squared='normalized RGB mask centroid squared distance; empty class not meaningful',covariance_squared='RGB mask2x2 central moment Frobenius squared; not rigid pose')
    summary=dict(complete=True,model_calls=0,optimizer_updates=0,constructed_images=len(records),GT_cases=12,definitions=definitions,tensor_only_gradient_calls=GRADIENT_CALLS,
        equation_checks=equation_checks,controlled_translation_responses=responses,mask_counterexamples=counterexamples,palette_measurements=palette_rows,panels=panels,
        unknowns_preserved=True,not_training_augmentation=True,wall_seconds=time.perf_counter()-started,
        limitations=['RGB color-unmix is piecewise differentiable and has zero gradients strictly outside clamp range; colors can be exploited.','u_saturated_fraction includes exact0/1 boundaries; Torch clamp derivative at those exact boundaries is1. Strict-outside and exact-boundary fractions are distinct; total saturation is not a zero-gradient fraction.','G does not distinguish equal overlap at different distances or ensure holes/rigidity.','T is finite16-projection slicedW1 plus area, not2D OT; finite directions do not identify every shape.','SDF target is binary RGB-derived, unlike soft antialiased GT masks; its raw loss need not be zero/minimal at soft GT.','clDice soft skeleton does not certify Betti numbers or complete unseen contours.','Centroid/covariance are not rigid pose; symmetric rotation and occlusion remain.','Diagnostic image transformations do not establish physically possible futures.'])
    assert len(records)<=288 and GRADIENT_CALLS<=64 and len(definitions)<=12
    write(ROOT/'diagnosis/measurements.json',records);write(ROOT/'diagnosis/summary.json',summary)
    return summary
