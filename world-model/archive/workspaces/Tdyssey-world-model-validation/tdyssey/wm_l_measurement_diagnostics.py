"""Localize frozen-fit residuals without refitting or changing acceptance.

Pusher discs are derived from RGB-only pusher fits. Excluding such a disc is a
counterfactual diagnostic, never a new shape score or an accepted hidden pose.
No physical truth, action or goal enters the residual partition.
"""

import json
from pathlib import Path

import numpy as np


def partition_residual(alpha, template, pusher_position, *, radius=15.):
    alpha, template = np.asarray(alpha), np.asarray(template)
    if (alpha.shape != (96,96) or template.shape != alpha.shape
            or not np.isfinite(alpha).all() or not np.isfinite(template).all()):
        raise ValueError("Expected finite full-frame alpha and rigid template")
    residual=np.abs(alpha-template)
    result=dict(total_absolute_alpha_residual=float(residual.sum()),
        eligibility_changed=False,truth_used=False,discs=[],
        interpretation="Outside-disc IoU is an occlusion/color sensitivity diagnostic, not a replacement gate")
    if pusher_position is None:
        result["status"]="pusher_unavailable"
        return result
    position=np.asarray(pusher_position,dtype=float)
    if position.shape!=(2,) or not np.isfinite(position).all() or not 0<radius<512:
        raise ValueError("Invalid RGB-derived pusher geometry")
    y,x=np.mgrid[:96,:96]
    world_x,world_y=(x+.5)*512/96-.5,(y+.5)*512/96-.5
    distance=np.hypot(world_x-position[0],world_y-position[1])
    for margin in (0,1):
        disc=distance<=radius+margin*512/96
        outside=~disc
        denominator=float(np.maximum(alpha[outside],template[outside]).sum())
        result["discs"].append(dict(margin_image_pixels=margin,image_pixels=int(disc.sum()),
            residual_inside_fraction=float(residual[disc].sum()/residual.sum()) if residual.sum()>0 else None,
            outside_soft_iou=float(np.minimum(alpha[outside],template[outside]).sum()/denominator) if denominator else None))
    result["status"]="diagnostic_only"
    return result


def audit_selected_residuals(run_dir,selection_path):
    from .wm_l_diagnostic_figures import _saved_raw,frame_diagnostics
    root=Path(run_dir).resolve()
    results=root/"results" if (root/"results").is_dir() else root
    selection=json.loads(Path(selection_path).read_text())
    rows=[]
    for case in selection["cases"]:
        for mode,horizons in (("AR",(1,2,3,4)),("TF",(2,4))):
            for h in horizons:
                key=f"{case['anchor']}/{case['candidate']}/{case['seed']}/{mode}/B/H{h}"
                record=json.loads((results/"records"/(key.replace("/","-")+".json")).read_text())
                raw,_=_saved_raw(record,results)
                diagnostic=frame_diagnostics(raw,record["measurement"])
                full=diagnostic["summary"]["offline_frozen_pusher"]
                position=full.get("position") if full.get("valid") else None
                value=(partition_residual(diagnostic["alpha"],diagnostic["template"],position)
                       if diagnostic["template"] is not None else dict(status="no_original_candidate",eligibility_changed=False))
                rows.append(dict(key=key,categories=case["categories"],original_measurement=record["measurement"],
                    pusher_fit_iou=full.get("fit_iou"),**value))
    return dict(kind="frozen_fit_residual_localization.v1",selection_file=str(Path(selection_path).resolve()),
        records=len(rows),rows=rows,full_object_refits=0,offline_pusher_measurements=len(rows),
        new_model_predictions=0,new_model_loads=0,new_physical_operations=0,threshold_changes=0,
        limitation="Localized mismatch can reflect occlusion, mixed color or nearby deformation; it does not identify hidden true geometry")
