"""Specified RGB-derived error definitions; no output modification or new model."""
import math
from functools import lru_cache
EPS=1e-6
@lru_cache(maxsize=4)
def palette(device='cuda'):
    import torch
    colors=torch.tensor([[255,255,255],[119,136,153],[65,105,225]],dtype=torch.float64,device=device)/255
    A=(colors[1:]-colors[0]).T
    P=torch.linalg.solve(A.T@A,A.T)
    return colors.float(),P.float(),A.float()
def masks(image,return_u=False):
    import torch
    colors,P,A=palette(str(image.device))
    u=torch.einsum('kc,bchw->bkhw',P,image.float()-colors[0][None,:,None,None])
    return (u.clamp(0,1),u) if return_u else u.clamp(0,1)
def geometry(p,g):
    return (1-(2*(p*g).sum((-2,-1))+EPS)/(p.square().sum((-2,-1))+g.square().sum((-2,-1))+EPS)).mean()
@lru_cache(maxsize=4)
def projections(device='cuda'):
    import torch
    xy=torch.arange(96,dtype=torch.float64,device=device)/(95*math.sqrt(2))
    y,x=torch.meshgrid(xy,xy,indexing='ij')
    theta=torch.arange(16,dtype=torch.float64,device=device)*math.pi/16
    v=x.flatten()[None]*theta.cos()[:,None]+y.flatten()[None]*theta.sin()[:,None]
    values,indices=v.sort(dim=-1,stable=True)
    return indices,(values[:,1:]-values[:,:-1]).float()
def transport(p,g,return_parts=False):
    import torch
    indices,spacing=projections(str(p.device));pp=p.flatten(-2);gg=g.flatten(-2)
    pm=pp.sum(-1,keepdim=True);gm=gg.sum(-1,keepdim=True)
    a=(pp+EPS/9216)/(pm+EPS);b=(gg+EPS/9216)/(gm+EPS)
    delta=a-b
    ordered=delta[:,:,indices]
    w=(ordered.cumsum(-1)[...,:-1].abs()*spacing[None,None]).sum(-1).mean(-1)
    area=((pm-gm).abs()/(gm+EPS)).squeeze(-1)
    result=(w+area).mean()
    return (result,w.mean(),area.mean()) if return_parts else result
def moments(p,g):
    import torch
    xy=torch.arange(96,dtype=torch.float32,device=p.device)/(95*math.sqrt(2));y,x=torch.meshgrid(xy,xy,indexing='ij');coords=torch.stack([x.flatten(),y.flatten()],-1)
    def one(v):
        q=v.flatten(-2);a=q/(q.sum(-1,keepdim=True)+EPS);mean=a@coords
        d=coords[None,None]-mean[:,:,None];cov=torch.einsum('bcn,bcni,bcnj->bcij',a,d,d)
        return mean,cov
    a,A=one(p);b,B=one(g)
    return (a-b).square().sum(-1).mean(),(A-B).square().sum((-2,-1)).mean()
def soft_jaccard(p,g):
    inter=(p*g).sum((-2,-1));union=(p+g-p*g).sum((-2,-1));return (1-(inter+EPS)/(union+EPS)).mean()
def soft_erode(x):
    import torch
    import torch.nn.functional as F
    return torch.minimum(-F.max_pool2d(-x,(3,1),(1,1),(1,0)),-F.max_pool2d(-x,(1,3),(1,1),(0,1)))
def soft_skeleton(x):
    import torch.nn.functional as F
    eroded=soft_erode(x);opened=F.max_pool2d(eroded,3,1,1);sk=(x-opened).relu()
    for _ in range(10):
        x=soft_erode(x);opened=F.max_pool2d(soft_erode(x),3,1,1);delta=(x-opened).relu();sk=sk+(delta-sk*delta).relu()
    return sk
def cldice(p,g):
    sp=soft_skeleton(p);sg=soft_skeleton(g)
    precision=((sp*g).sum((-2,-1))+EPS)/(sp.sum((-2,-1))+EPS)
    sensitivity=((sg*p).sum((-2,-1))+EPS)/(sg.sum((-2,-1))+EPS)
    return (1-(2*precision*sensitivity+EPS)/(precision+sensitivity+EPS)).mean()
def parts(bundle):
    import torch
    with torch.autocast('cuda',enabled=False):
        D=bundle['reconstructed_unquantized'].float();y=bundle['target'].float();p,u=masks(D,True);g=masks(y).detach()
        G=geometry(p,g);T,w,area=transport(p,g,True)
        return dict(native=bundle['native_loss'],G=G,T=T,W1=w,area=area,p=p,g=g,u=u)
def loss_for(arm,bundle,coefficients):
    values=parts(bundle)
    if arm=='G_ONLY':loss=coefficients['cG']*values['G']
    elif arm=='NATIVE_G':loss=values['native']+coefficients['cG']*values['G']
    elif arm=='NATIVE_T':loss=values['native']+coefficients['cT']*values['T']
    else:raise ValueError('Unapproved arm')
    return loss,values
