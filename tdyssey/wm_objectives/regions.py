"""Approved target-side region/detail residual weights, no model inputs.

Mask construction sees saved uint8 truth only. No geometry, pose or letter IDs.
Pixel radii use discrete Euclidean disks; all choices are frozen in loss-spec.
"""
from pathlib import Path
from functools import lru_cache
import json
import os
import numpy as np
from scipy import ndimage

REPO=Path(__file__).resolve().parents[2]
CALIBRATION=Path(os.environ.get('TDYSSEY_REGION_CALIBRATION',REPO/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json'))
COEFFICIENTS=(.50,.25,.25)

@lru_cache(maxsize=1)
def calibration():return json.loads(CALIBRATION.read_text(encoding='utf-8-sig'))

def disk(radius):
    y,x=np.mgrid[-radius:radius+1,-radius:radius+1]
    return x*x+y*y<=radius*radius

def components(image):
    from tdyssey.native_rgb_evaluation import palette_masks
    if image.dtype!=np.uint8 or image.shape!=(3,96,96):raise ValueError('Mask source must be original uint8 truth RGB96')
    grey,blue=palette_masks(image,calibration())
    # Same dual topology convention as the frozen score: 8-connected foreground,
    # 4-connected background. Hole support includes every enclosed background
    # pixel, with no new white threshold, small-hole removal or label geometry.
    holes=ndimage.binary_fill_holes(grey,structure=ndimage.generate_binary_structure(2,1))&~grey
    support=grey|blue|holes
    # A symmetric one-pixel morphological gradient retains both sides of all
    # object/pusher outer and inner boundary loops before approved E dilation.
    edge=np.zeros((96,96),bool)
    for mask in (grey,blue):
        edge|=ndimage.binary_dilation(mask,structure=disk(1))^ndimage.binary_erosion(mask,structure=disk(1),border_value=0)
    return support,edge,holes

def _pair(previous,target,pc,tc):
    region=ndimage.binary_dilation(pc[0]|tc[0],structure=disk(2))
    # For uint8 RGB, >1/255 in the original float32 image convention is a
    # per-channel absolute integer difference >1. Use integer arithmetic to
    # avoid threshold roundoff and retain both vacated and newly occupied pixels.
    changed=np.max(np.abs(target.astype(np.int16)-previous.astype(np.int16)),axis=0)>1
    detail=ndimage.binary_dilation(pc[1]|tc[1]|changed,structure=disk(1))
    return region,detail

def _stats(R,E,hole_counts):
    r=R.reshape(-1,96,96).sum((1,2));e=E.reshape(-1,96,96).sum((1,2))
    return dict(targets=len(r),R_empty=int((r==0).sum()),E_empty=int((e==0).sum()),
        R_pixels=int(r.sum()),E_pixels=int(e.sum()),R_min=int(r.min()),R_max=int(r.max()),E_min=int(e.min()),E_max=int(e.max()),
        enclosed_hole_truth_pixels=int(hole_counts))

def compute_masks(previous,target):
    if previous.shape!=target.shape or previous.ndim!=4 or previous.shape[1:]!=(3,96,96) or previous.dtype!=np.uint8 or target.dtype!=np.uint8:raise ValueError('Paired original uint8 truth batches required')
    rs=[];es=[];holes=0
    for p,t in zip(previous,target):
        pc,tc=components(p),components(t);r,e=_pair(p,t,pc,tc);rs.append(r);es.append(e);holes+=int(tc[2].sum())
    R=np.stack(rs);E=np.stack(es)
    return dict(R=R,E=E,stats=_stats(R,E,holes))

def batch(images):
    if images.dtype!=np.uint8 or images.ndim!=5 or images.shape[1:]!=(8,3,96,96):raise ValueError('Original four-target truth segment batch required')
    R=np.empty((len(images),4,96,96),bool);E=np.empty_like(R);holes=0
    for i,seq in enumerate(images):
        parts=[components(im) for im in seq[3:8]]
        for h in range(4):
            R[i,h],E[i,h]=_pair(seq[h+3],seq[h+4],parts[h],parts[h+1]);holes+=int(parts[h+1][2].sum())
    return dict(R=R,E=E,stats=_stats(R,E,holes))

def weighted_residual_loss(output,residual,R,E,coeffs=COEFFICIENTS):
    import torch
    import torch.nn.functional as F
    if output.shape!=residual.shape or output.ndim!=4 or output.shape[1:]!=(3,96,96):raise ValueError('Native residual RGB shape required')
    b=len(output)
    if tuple(R.shape)!=(b,96,96) or E.shape!=R.shape or R.dtype!=torch.bool or E.dtype!=torch.bool or R.device!=output.device or E.device!=output.device or R.requires_grad or E.requires_grad:raise ValueError('Detached target-side boolean masks required')
    if len(coeffs)!=3 or any(not np.isfinite(x) or x<0 for x in coeffs):raise ValueError('Three finite nonnegative coefficients required')
    all_loss=F.mse_loss(output.float(),residual.float())
    squared=(output.float()-residual.float()).square().mean(dim=1)
    per_sample=squared.mean((1,2))
    def regional(mask):
        count=mask.sum((1,2));value=(squared*mask).sum((1,2))/count.clamp_min(1)
        return torch.where(count>0,value,per_sample).mean(),count
    region,rn=regional(R);detail,en=regional(E)
    # Exact original native MSE path for the required (1,0,0) equivalence check.
    loss=all_loss if tuple(coeffs)==(1,0,0) else coeffs[0]*all_loss+coeffs[1]*region+coeffs[2]*detail
    stats=dict(all=all_loss.detach(),R=region.detach(),E=detail.detach(),R_empty=(rn==0).sum().detach(),E_empty=(en==0).sum().detach())
    finite=torch.stack([torch.isfinite(x).all() for x in (output,residual,loss)]).all()
    return loss,stats,finite

def specification():
    return dict(kind='native-residual-target-regions.v1',coefficients=list(COEFFICIENTS),
        error='FP32 squared native residual error, RGB-channel mean; per-sample R/E normalization, batch mean; native full term F.mse_loss',
        R='Union grey object, blue pusher, enclosed background hole pixels in previous/target truth, Euclidean disk dilation radius2',
        E='Union previous/target grey+blue symmetric cross morphological gradients and max uint8 RGB absolute difference>1, Euclidean disk dilation radius1',
        topology='Grey foreground8/background4, every enclosed background pixel retained, frozen palette thresholds; no new hole-size threshold',
        border='Binary dilation outsideFalse; erosion outsideFalse',empty_region='Fall back to that sample full image mean and count each empty region',
        masks_inputs='Training saved uint8 truth RGB at t and t+1 only, no prediction/physicalstate/letter/validation data',
        model_input_masks=False,prediction_repair=False,true_RGB_modified=False,original_MSE_coefficients=[1,0,0])
