"""Read-only frame-by-frame diagnostics of saved, frozen R3B predictions.

No model, simulator, fitted image, repair or future-RGB predictor input is made
here. Thresholded visible-image measurements and manual observations are separate
from occlusion-aware availability; neither is called a rigidity certificate.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DIAGNOSTICS_ROOT=ROOT/'artifacts/r3b-playground-diagnostics-20261030'
CALIBRATION=ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json'
TRAINING_PHYSICS_SHA='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6'
DISPLAY_SCALE=3

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def safe(value):
    if isinstance(value,dict):return{str(k):safe(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return[safe(v)for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float)and not math.isfinite(value):return None
    if isinstance(value,Path):return str(value)
    return value
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or['no_rows']);writer.writeheader()
        writer.writerows({k:json.dumps(safe(v),ensure_ascii=False)if isinstance(v,(list,dict))else safe(v)for k,v in row.items()}for row in rows)
def mean(values):
    values=[float(v)for v in values if v is not None and math.isfinite(float(v))]
    return sum(values)/len(values)if values else None
def fmt(value,digits=4):return 'N/A'if value is None else f'{value:.{digits}f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows])


def sampler_timing(records,expected_count):
    """Read runner seconds (or legacy sampler_s); never invent missing timing."""
    values=[];keys=Counter();invalid=0
    for record in records:
        key='seconds'if record.get('seconds')is not None else'sampler_s'
        value=record.get(key)
        try:value=float(value)
        except(TypeError,ValueError):invalid+=1;continue
        if not math.isfinite(value)or value<0:invalid+=1;continue
        values.append(value);keys[key]+=1
    missing=max(0,expected_count-len(records))+invalid
    extra=max(0,len(records)-expected_count)
    recorded=sum(values)if values else None
    return dict(total_seconds=recorded if not missing and not extra else None,
        recorded_seconds=recorded,expected_records=expected_count,provided_records=len(records),
        valid_records=len(values),missing_or_invalid_records=missing,unexpected_records=extra,source_keys=dict(keys))


def observed_defects(row):
    """Keep manual visual evidence explicit without changing palette availability."""
    notes=row.get('visual_observation',{})
    manual=['manual_'+name for name in('rounding','deformation','hole_merge','gray_fragment')if notes.get('visible_'+name)is True]
    return list(dict.fromkeys(row.get('visible_defects',[])+manual))


def raw_masks(rgb,calibration):
    """Same palette projection as frozen metrics, with finite raw OOR retained."""
    image=np.asarray(rgb,dtype=np.float32)
    if image.shape!=(3,96,96)or not np.isfinite(image).all():raise ValueError('Finite raw float32 CHW96 required')
    prototype=np.asarray(calibration['object_rgb'],dtype=np.float64);direction=1-prototype
    pixels=image.transpose(1,2,0);alpha=((1-pixels)*direction).sum(-1)/float(direction@direction)
    residual=np.linalg.norm(pixels-(1-alpha[...,None]*direction),axis=-1)
    policy=calibration['palette']
    gray=(alpha>=policy['minimum_alpha'])&(alpha<=policy['maximum_alpha'])&(residual<=policy['colour_residual'])
    r,g,b=image;blue=(b>.60)&(r<.45)&(g<.65)&(b-np.maximum(r,g)>.25)
    return gray,blue


def centroid(mask):
    ys,xs=np.nonzero(mask)
    return np.asarray([xs.mean(),ys.mean()])if len(xs)else None
def distance(a,b):return None if a is None or b is None else float(np.linalg.norm(a-b))
def iou(a,b):
    union=(a|b).sum();return float((a&b).sum()/union)if union else None
def border(mask):return bool(mask[0].any()or mask[-1].any()or mask[:,0].any()or mask[:,-1].any())
def topology(mask):
    from scipy import ndimage
    labels,_=ndimage.label(mask,structure=np.ones((3,3)))
    areas=np.bincount(labels.ravel());components=[]
    for label in range(1,len(areas)):
        ys,xs=np.nonzero(labels==label)
        components.append(dict(label=label,pixels=int(areas[label]),bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]))
    bg,count=ndimage.label(~np.pad(mask,1),structure=ndimage.generate_binary_structure(2,1));outside=int(bg[0,0])
    holes=[]
    for label in range(1,count+1):
        selected=bg==label;area=int(selected.sum())
        if label!=outside and area>=2:holes.append(dict(pixels=area,mask=selected[1:-1,1:-1]))
    return dict(components=sorted(components,key=lambda c:c['pixels'],reverse=True),labels=labels,
        large_components=sum(c['pixels']>=2 for c in components),tiny_pixels=sum(c['pixels']for c in components if c['pixels']<2),holes=holes)
def contours(mask):
    import cv2
    found,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    return [len(cv2.approxPolyDP(c,.75,True))for c in found if cv2.contourArea(c)>=2]
def contour_error(a,b):
    from scipy import ndimage
    aa=a&~ndimage.binary_erosion(a,structure=np.ones((3,3)));bb=b&~ndimage.binary_erosion(b,structure=np.ones((3,3)))
    if not aa.any()or not bb.any():return None
    da=ndimage.distance_transform_edt(~aa);db=ndimage.distance_transform_edt(~bb)
    return float((da[bb].mean()+db[aa].mean())/2)


def measure_frame(prediction,truth,last,calibration):
    """Unregistered visible metrics; visible defects survive global occlusion."""
    from scipy import ndimage
    prediction=np.asarray(prediction);truth=np.asarray(truth);last=np.asarray(last)
    if prediction.shape!=(3,96,96):return dict(status='wrong_shape',visible_defects=['wrong_shape'],shape_assessment='unknown')
    if not np.isfinite(prediction).all():return dict(status='nonfinite',visible_defects=['nonfinite'],shape_assessment='unknown',nonfinite_values=int((~np.isfinite(prediction)).sum()))
    gray,blue=raw_masks(prediction,calibration);target,target_blue=raw_masks(truth,calibration);prior,_=raw_masks(last,calibration)
    p,t=topology(gray),topology(target);dilation=ndimage.binary_dilation
    occluded=bool((dilation(gray)&blue).any()or(dilation(target)&target_blue).any());clipped=border(gray)or border(target)
    object_roi=dilation(gray|target,iterations=1);blue_roi=dilation(blue|target_blue,iterations=1)
    gray_center,target_center=centroid(gray),centroid(target);blue_center,target_blue_center=centroid(blue),centroid(target_blue)
    expected_holes=len(t['holes']);predicted_holes=len(p['holes'])
    near_blue_hole=any((dilation(h['mask'])&blue).any()for h in p['holes'])or any((dilation(h['mask'])&target_blue).any()for h in t['holes'])
    small_hole=any(h['pixels']<4 for h in p['holes']+t['holes'])
    hole_status='unknown_small_or_blue_occluded_hole'if near_blue_hole or small_hole or clipped else'visible_count_assessable'
    extras=[]
    for component in p['components'][1:]:
        if component['pixels']<2:continue
        mask=p['labels']==component['label'];outside_truth=not bool((mask&dilation(target,iterations=2)).any())
        extras.append(dict(**component,outside_truth_neighborhood=outside_truth,near_blue=bool((dilation(mask)&blue).any()),
            anti_alias_review_required=component['pixels']<4))
    defects=[]
    if target_blue.sum()>=2 and not blue.any():defects.append('pusher_disappeared')
    if target.any()and not gray.any():defects.append('gray_object_disappeared')
    open_extras=[c for c in extras if c['outside_truth_neighborhood']and c['pixels']>=4]
    if open_extras:defects.append('open_field_gray_fragment')
    if predicted_holes!=expected_holes and hole_status=='visible_count_assessable':defects.append('visible_hole_count_change')
    area_error=float(abs(int(gray.sum())-int(target.sum()))/max(1,int(target.sum())))
    if area_error>.20:defects.append('large_visible_area_difference')
    relative=None
    if all(v is not None for v in(gray_center,target_center,blue_center,target_blue_center)):
        relative=float(np.linalg.norm((gray_center-blue_center)-(target_center-target_blue_center)))
    status='visible_defect'if defects else'unknown_occlusion_or_clip'if occluded or clipped else'no_listed_visible_defect'
    return dict(status='measured',raw_min=float(prediction.min()),raw_max=float(prediction.max()),
        raw_finite_out_of_range=bool(prediction.min()<0 or prediction.max()>1),gray_pixels=int(gray.sum()),truth_gray_pixels=int(target.sum()),
        blue_pixels=int(blue.sum()),truth_blue_pixels=int(target_blue.sum()),object_roi_mae=float(np.abs(prediction-truth)[:,object_roi].mean())if object_roi.any()else None,
        blue_roi_mae=float(np.abs(prediction-truth)[:,blue_roi].mean())if blue_roi.any()else None,gray_iou=iou(gray,target),blue_iou=iou(blue,target_blue),
        gray_visible_centroid_error_px=distance(gray_center,target_center),blue_centroid_error_px=distance(blue_center,target_blue_center),
        gray_visible_drift_px=distance(gray_center,centroid(prior)),relative_position_error_px=relative,gray_contour_error_px=contour_error(gray,target),
        predicted_holes=predicted_holes,truth_holes=expected_holes,hole_measurement_status=hole_status,
        predicted_components=p['large_components'],truth_components=t['large_components'],extra_components=extras,
        tiny_fragment_pixels=p['tiny_pixels'],area_fraction_error=area_error,raw_corner_counts=contours(gray),truth_raw_corner_counts=contours(target),
        occlusion_ambiguous=occluded,viewport_clipped=clipped,visible_defects=defects,shape_assessment=status,
        shape_assessment_meaning='Visible defects are reported even if another region is occluded; no listed defect is not a rigidity pass. Corner counts are raster diagnostics only.')


def font(size=16):
    from PIL import ImageFont
    for candidate in('C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/segoeui.ttf'):
        if Path(candidate).exists():return ImageFont.truetype(candidate,size)
    return ImageFont.load_default()
def display_image(raw,scale):
    from PIL import Image
    value=np.asarray(raw)
    if value.shape!=(3,96,96)or not np.isfinite(value).all():return None
    pixels=np.rint(np.clip(value,0,1)*255).astype(np.uint8).transpose(1,2,0)
    return Image.fromarray(pixels).resize((96*scale,96*scale),Image.Resampling.NEAREST)
def draw_grid(arrays,metadata,path,*,crop=None,notes=None):
    from PIL import Image,ImageDraw
    truth=arrays['truth_rgb'];a=arrays['predictions_A'];b=arrays['predictions_B']
    scale=DISPLAY_SCALE if crop is None else 7;tile=96*scale if crop is None else max(crop[2]-crop[0],crop[3]-crop[1])*scale
    margin=180;top=124;row_title=30;canvas=Image.new('RGB',(margin+5*tile,top+3*(tile+row_title)),(247,248,250));draw=ImageDraw.Draw(canvas)
    title=f"{metadata.get('case_id','case')} anchor {metadata.get('anchor','?')} | {metadata.get('letter','?')} | sampler {metadata.get('sampler',{}).get('num_steps_denoising','?')}"
    draw.text((12,8),title,font=font(20),fill='black');draw.text((12,36),'Exact same trajectory/actions/seeds. Display clamp only; raw float32 feedback retained.',font=font(15),fill='#333333')
    if crop is not None:draw.text((12,58),'GT-defined identical integer crop, nearest-neighbor magnification; no image alignment.',font=font(14),fill='#333333')
    for step in range(5):
        motion=(metadata.get('motion_contact')or[{}]*4)[step-1]if step else{}
        absolute=motion.get('absolute_control_step','?')
        draw.text((margin+step*tile+8,top-43),'t0'if step==0 else f'+{step/10:.1f}s | control {absolute}',font=font(15),fill='black')
        if step:
            target=motion.get('actual_target');target_text='N/A'if target is None else ','.join(f'{float(v):.1f}'for v in target)
            seed=int(arrays['seed_steps'][step-1]);draw.text((margin+step*tile+8,top-23),f'executed ({target_text}) seed {seed}',font=font(12),fill='#333333')
    for row,label in enumerate(('Ground truth','A: true-history H1','B: open-loop H4')):
        y=top+row*(tile+row_title);draw.text((10,y+10),label,font=font(17),fill='black')
        for step in range(5):
            raw=truth[step]if row==0 or step==0 else(a if row==1 else b)[step-1]
            image=display_image(raw,1)
            if image is None:draw.text((margin+step*tile+8,y+12),'INVALID RAW',font=font(18),fill='red');continue
            if crop is not None:
                image=image.crop(crop);image.thumbnail((tile,tile),Image.Resampling.NEAREST)
                image=image.resize((image.width*scale,image.height*scale),Image.Resampling.NEAREST)
            else:image=image.resize((tile,tile),Image.Resampling.NEAREST)
            canvas.paste(image,(margin+step*tile,y));draw.rectangle((margin+step*tile,y,margin+(step+1)*tile-1,y+tile-1),outline='#dddddd')
            if step and row==0:
                motion=(metadata.get('motion_contact')or[{}]*4)[step-1]
                label=f"GT move {fmt(motion.get('translation_image96_px'),2)}px / {fmt(motion.get('signed_rotation_deg'),1)}deg; contact {motion.get('contact_substeps','?')}/10"
                draw.text((margin+step*tile+3,y+tile+4),label,font=font(11),fill='#333333')
            elif step and notes:
                metric=notes.get(('A'if row==1 else'B',step),{})
                flags=','.join(observed_defects(metric)).replace('pusher_disappeared','blue lost').replace('visible_hole_count_change','holes').replace('open_field_gray_fragment','fragment').replace('large_visible_area_difference','area')
                manual_flags=','.join(flag.replace('manual_','')for flag in observed_defects(metric)if flag.startswith('manual_'))
                if manual_flags:flags='manual '+manual_flags+'; '+flags.replace('manual_','')
                label='ROI '+fmt(metric.get('object_roi_mae'))+' | '+(flags or metric.get('shape_assessment','unknown'))
                draw.text((margin+step*tile+3,y+tile+4),label[:max(25,tile//6)],font=font(11),fill='#b11d26'if flags else'#333333')
    canvas.save(path)
    return dict(width=canvas.width,height=canvas.height,crop=crop,scale=scale,nearest=True,registration=False)


def crop_from_truth(truth,calibration):
    masks=[raw_masks(v,calibration)[0]for v in truth];mask=np.logical_or.reduce(masks)
    ys,xs=np.nonzero(mask)
    if not len(xs):return(0,0,96,96)
    return(max(0,int(xs.min())-5),max(0,int(ys.min())-5),min(96,int(xs.max())+6),min(96,int(ys.max())+6))
def draw_video(arrays,metadata,path):
    import cv2
    import imageio_ffmpeg
    from PIL import Image,ImageDraw
    scale=3;w=96*scale;canvas_size=(3*w,96*scale+70)
    writer=imageio_ffmpeg.write_frames(str(path),canvas_size,fps=10,codec='libx264',quality=8,
        macro_block_size=1,ffmpeg_log_level='error',output_params=['-movflags','+faststart'])
    writer.send(None)
    try:
        for step in range(5):
            canvas=Image.new('RGB',canvas_size,'white');draw=ImageDraw.Draw(canvas)
            for column,label in enumerate(('GT','A true-history H1','B open-loop H4')):
                raw=arrays['truth_rgb'][step]if column==0 or step==0 else arrays['predictions_A'if column==1 else'predictions_B'][step-1]
                image=display_image(raw,scale)
                draw.text((column*w+6,5),label,font=font(17),fill='black')
                if image is not None:canvas.paste(image,(column*w,35))
                else:draw.text((column*w+20,100),'INVALID RAW',font=font(17),fill='red')
            motion=(metadata.get('motion_contact')or[{}]*4)[step-1]if step else{}
            label=f"t0+{step/10:.1f}s | 4x slow motion | control {motion.get('absolute_control_step','t0')} | executed {motion.get('actual_target','initial state')}"
            draw.text((8,canvas_size[1]-27),label,font=font(15),fill='black')
            frame=np.asarray(canvas)
            for _ in range(4 if step<4 else 8):writer.send(frame)
    finally:writer.close()
    capture=cv2.VideoCapture(str(path));decoded=0
    while capture.read()[0]:decoded+=1
    capture.release()
    if decoded!=24:raise RuntimeError(f'Video decode count {decoded}, expected24')
    return dict(file=path.name,decoded_frames=decoded,fps=10,slow_motion=4,codec='H264/libx264',preview_is_lossy=True,
        coordinates_resized_by_codec=False,original_times_seconds=[0,.1,.2,.3,.4])


def reference_problem_rows(groups,manual):
    """No original screenshot can be an exact replay: action/RNG history absent."""
    conditions={1:('B',('visible_hole_count_change','pusher_disappeared')),2:('X',('open_field_gray_fragment',)),
        3:('C',('rounding',)),4:('C',('open_field_gray_fragment',)),5:('O',('rounding',)),6:('O',('pusher_disappeared',))}
    result=[]
    for number,(letter,required)in conditions.items():
        hits={name:[]for name in required}
        for group in groups:
            if group['letter']!=letter or group['sampler_steps']!=3:continue
            for row in group['rows']:
                if row['branch']!='B':continue
                for name in required:
                    observed=name in row.get('visible_defects',[])
                    notes=manual.get(group['id'],{}).get(f"B{row['step']}",{})
                    if name=='rounding':observed=notes.get('visible_rounding')is True
                    if observed:hits[name].append(dict(group_id=group['id'],step=row['step'],grid=group['grid'],manual_note=notes.get('note'),physics_cohort=group['physics_cohort'],physics_protocol_sha256=group['physics_protocol_sha256']))
        all_found=all(hits.values())
        result.append(dict(screenshot=number,letter=letter,status='已复现同类'if all_found else'未复现',exact_original_screenshot_replay=False,
            reason='Original screenshot lacks input history/actions/RNG. These are same-class visual components under new original physics; components may come from different clips/steps and do not claim complete same-frame reproduction.',components=hits,
            compound_same_frame_not_claimed=True,cross_physics_hits_are_visual_class_only=True))
    return result


def paired_sampler_comparison(groups):
    matched=defaultdict(dict)
    for group in groups:
        key=(group['physics_cohort'],group['case_id'],group['anchor'],tuple(group['seed_steps']))
        matched[key][group['sampler_steps']]=group
    comparisons=[]
    for key,variants in sorted(matched.items()):
        if not all(s in variants for s in(3,8,16)):continue
        for branch in('A','B'):
            for steps in(3,8,16):
                group=variants[steps];rows=[r for r in group['rows']if r['branch']==branch]
                records=group['source_metadata'].get('step_records_'+branch,[])
                timing=sampler_timing(records,4)
                bad=[r for r in rows if observed_defects(r)]
                comparisons.append(dict(physics_cohort=key[0],case_id=key[1],anchor=key[2],seed_steps=list(key[3]),branch=branch,
                    sampler_steps=steps,group_id=group['id'],all4_ROI=mean(r.get('object_roi_mae')for r in rows),
                    H4_ROI=rows[-1].get('object_roi_mae'),H4_gray_iou=rows[-1].get('gray_iou'),
                    four_sampler_seconds=timing['total_seconds'],sampler_timing=timing,
                    first_visible_defect_step=bad[0]['step']if bad else None,
                    first_visible_defects=observed_defects(bad[0])if bad else[],grid=group['grid']))
    return comparisons


def build_report(stage,*,videos=True):
    stage=Path(stage).resolve();out=stage/'report';assets=out/'assets';assets.mkdir(parents=True,exist_ok=True)
    calibration=read(CALIBRATION);manual_path=stage/'review/manual-observations.json';manual=read(manual_path)if manual_path.exists()else{}
    groups=[];all_rows=[];failures=[]
    for index,metadata_path in enumerate(sorted((stage/'clips').rglob('metadata.json'))):
        metadata=read(metadata_path);source=metadata_path.with_name('clip.npz');identifier=f'clip{index:03d}'
        if not source.exists():failures.append(dict(metadata=str(metadata_path),reason='No saved raw archive',data=metadata));continue
        with np.load(source,allow_pickle=False)as raw:arrays={k:raw[k]for k in raw.files}
        for key,shape in(('truth_rgb',(5,3,96,96)),('predictions_A',(4,3,96,96)),('predictions_B',(4,3,96,96))):
            if arrays.get(key,np.empty(0)).shape!=shape:raise ValueError('Wrong saved clip array '+str(source)+': '+key)
        inferred_case=metadata.get('case_id',source.parents[1].name);letter=metadata.get('letter',inferred_case[0]);sampler=metadata.get('sampler',{})
        physics_sha=metadata.get('physics_protocol_sha256',metadata.get('physics_sha256'))
        physics_cohort=metadata.get('physics_cohort','historical_dynamic_half_life_0.1s'if physics_sha==TRAINING_PHYSICS_SHA else'original_kinematic_cross_physics_preview')
        steps=int(sampler.get('num_steps_denoising',metadata.get('sampler_steps',source.parent.name.split('-')[0])))
        title=dict(metadata,case_id=inferred_case,letter=letter,sampler=sampler)
        grid=assets/f'{identifier}-frames.png';closeup=assets/f'{identifier}-closeup.png'if letter in'OCB'else None
        video=assets/f'{identifier}.mp4'
        rows=[]
        for branch in('A','B'):
            for step in range(1,5):
                result=measure_frame(arrays[f'predictions_{branch}'][step-1],arrays['truth_rgb'][step],arrays['truth_rgb'][0],calibration)
                motion=(metadata.get('motion_contact')or[{}]*4)[step-1]
                row=dict(group_id=identifier,case_id=inferred_case,letter=letter,anchor=metadata.get('anchor'),sampler_steps=steps,physics_cohort=physics_cohort,physics_protocol_sha256=physics_sha,
                    branch=branch,step=step,time_seconds=step/10,seed=int(arrays['seed_steps'][step-1]),**result,
                    truth_translation_image96_px=motion.get('translation_image96_px'),truth_signed_rotation_deg=motion.get('signed_rotation_deg'),
                    contact_substeps=motion.get('contact_substeps'),completed_substeps=motion.get('completed_substeps'),
                    actual_target=motion.get('actual_target'),mouse_preview_target=motion.get('mouse_preview_target'),
                    visual_observation=manual.get(identifier,{}).get(f'{branch}{step}',{}))
                rows.append(row);all_rows.append(row)
        notes={(r['branch'],r['step']):r for r in rows}
        grid_info=draw_grid(arrays,title,grid,notes=notes)
        if closeup is not None:draw_grid(arrays,title,closeup,crop=crop_from_truth(arrays['truth_rgb'],calibration),notes=notes)
        video_info=draw_video(arrays,title,video)if videos else None
        first_equal=bool(np.array_equal(arrays['predictions_A'][0],arrays['predictions_B'][0]))
        if metadata.get('first_output_equal')is not True or not first_equal:raise RuntimeError('Saved H1/H4 first prediction mismatch')
        timing=sampler_timing([r for b in('A','B')for r in metadata.get(f'step_records_{b}',[])],8)
        group=dict(id=identifier,case_id=inferred_case,letter=letter,anchor=metadata.get('anchor'),sampler_steps=steps,physics_cohort=physics_cohort,physics_protocol_sha256=physics_sha,
            seed_steps=arrays['seed_steps'].tolist(),metadata_path=str(metadata_path),archive_path=str(source),archive_sha256=sha(source),
            first_frame_check_mode=metadata.get('first_frame_check_mode'),first_output_equal=first_equal,
            rows=rows,grid='assets/'+grid.name,grid_info=grid_info,closeup='assets/'+closeup.name if closeup else None,
            video='assets/'+video.name if videos else None,video_info=video_info,
            model_seconds=timing['total_seconds'],model_timing=timing,
            source_metadata=metadata)
        groups.append(group)
    references=reference_problem_rows(groups,manual);summary=[]
    for cohort,steps in sorted({(g['physics_cohort'],g['sampler_steps'])for g in groups}):
        chosen=[g for g in groups if g['sampler_steps']==steps and g['physics_cohort']==cohort]
        for branch in('A','B'):
            for step in range(1,5):
                rows=[r for g in chosen for r in g['rows']if r['branch']==branch and r['step']==step]
                summary.append(dict(physics_cohort=cohort,sampler_steps=steps,branch=branch,step=step,planned_saved_groups=len(rows),
                    ROI_MAE=mean(r.get('object_roi_mae')for r in rows),gray_iou=mean(r.get('gray_iou')for r in rows),
                    gray_centroid_error_px=mean(r.get('gray_visible_centroid_error_px')for r in rows),
                    blue_centroid_error_px=mean(r.get('blue_centroid_error_px')for r in rows),
                    pusher_disappeared=sum('pusher_disappeared'in r.get('visible_defects',[])for r in rows),
                    fragment_rows=sum('open_field_gray_fragment'in r.get('visible_defects',[])for r in rows),
                    hole_change_rows=sum('visible_hole_count_change'in r.get('visible_defects',[])for r in rows),
                    unknown=sum(r.get('shape_assessment','unknown').startswith('unknown')for r in rows),
                    nonfinite_or_missing=sum(r['status']!='measured'for r in rows)))
    sampler_pairs=paired_sampler_comparison(groups)
    result=dict(kind='R3B-playground-saved-frame-diagnostic-report.v1',groups=groups,summary=summary,user_reference_correspondence=references,paired_sampler_comparisons=sampler_pairs,
        failed_clip_records=failures,manual_observation_file=str(manual_path)if manual_path.exists()else None,
        no_population_failure_rate_claim=True,optimizer_updates=0,model_calls_by_report=0,physics_steps_by_report=0,
        report_source_sha256=sha(Path(__file__)),
        sampler_timing=sampler_timing([r for g in groups for b in('A','B')for r in g['source_metadata'].get(f'step_records_{b}',[])],len(groups)*8),
        model_training_physics_sha256=TRAINING_PHYSICS_SHA,physics_cohorts=sorted({g['physics_cohort']for g in groups}),
        cross_physics_scope='The frozen7499 model was trained on historical dynamic pusher physics; original kinematic preview results cannot establish failures under the old dynamics or a training/autoregression mismatch by themselves.',
        display_clamp_only=True,raw_files_unchanged=True,calibration_path=str(CALIBRATION),calibration_sha256=sha(CALIBRATION),
        motion_fit_note='Gray/blue centroids are visible-image measurements, not privileged predicted simulator pose; true motion/contact labels are recorded separately.')
    timing_note=''
    proof_path=stage/'model-proof.json'
    if proof_path.exists():
        proof=read(proof_path);inner=proof.get('sampler_seconds');outer=result['sampler_timing']['total_seconds']
        result['sampler_timer_comparison']=dict(metadata_outer_seconds=outer,model_proof_inner_seconds=inner,
            difference_seconds=outer-inner if outer is not None and inner is not None else None,
            timers_expected_equal=False,proof_path=str(proof_path),proof_sha256=sha(proof_path),
            outer_scope='Runner paired_predictions: input copies, sampler return/index and finite-value scan before seconds is recorded.',
            inner_scope='FrozenPredictor.sampler: only the sampler wrapper invocation.',
            runner_source_sha256=sha(ROOT/'tdyssey/native_rgb_playground_diagnostic_runner.py'))
        timing_note=f" 外层合计{fmt(outer,6)}秒与model-proof内层sampler计时{fmt(inner,6)}秒边界不同，差{fmt(result['sampler_timer_comparison']['difference_seconds'],6)}秒；外层另含输入copy、返回索引和finite扫描，不能强行对齐或当纯GPU费用。"
    physics_path=DIAGNOSTICS_ROOT/'physics-alignment/validated-v2/summary.json'
    physical=None;physical_html='';physical_report=''
    if physics_path.exists():
        physical=read(physics_path)
        result['original_physics_validation']=dict(summary_path=str(physics_path),summary_sha256=sha(physics_path),passed=physical['passed'],
            protocol_sha256=physical['protocol_sha256'],cost=physical['cost'],parity=[{k:r[k]for k in('letter','passed','max_pre_error','max_post_error')}for r in physical['parity']])
        prefix='/diagnostics-root/physics-alignment/validated-v2/'
        physical_html='<section><h2>Original-source physics alignment and real continuing controls</h2><p>Six-letter material/body/shape identity and direct-source T/C substep parity. O hold2/push6/withdraw6/continue-hold10; no velocity clearing or pose rewrite.</p><img src="'+prefix+'separation-curves.png"><p><a href="'+prefix+'summary.json">Exact physical summary</a> · <a href="/diagnostics-root/physics-alignment/INDEX.md">Failure and cost record</a></p>'
        for case in('O-center','O-near-wall'):
            video_name=case+'-browser.mp4'if(physics_path.parent/(case+'-browser.mp4')).exists()else case+'.mp4'
            physical_html+='<p>'+case+'</p><video controls preload="none" src="'+prefix+video_name+'"></video><p><a href="'+prefix+case+'-controls.csv">Control CSV</a> · <a href="'+prefix+case+'-substeps.csv">Substep/contact/positive-impulse CSV</a> · <a href="'+prefix+case+'.npz">Saved raw states and RGB</a></p>'
        physical_html+='</section>'
        physical_report='\n## 原始物理与持续推进验收\n\n6个已训练字母的实际body/material/geometry与原source参考一致；T、C逐前后state15最大误差均0。O中心和近墙执行hold2/push6/withdraw6/继续hold10，继续真实推进，不清速度、不回写位姿。中心首次整控无AO为9，近墙为10；末速度0不等于严格零位移，首脱离控仍有.130603/.036994世界像素残余位移/接触位置修正。近墙有真实OW接触，最大墙穿透.5031px；中心原运动学推时AO穿透达2.291px，未用旧finiteforce保护隐藏。源物理damping0与旧0.1秒半衰期训练协议不同。\n\n[参数/子步一致性/耗费完整记录]('+prefix+'summary.json) · [速度/角速度/接触曲线]('+prefix+'separation-curves.png) · [中心短片]('+prefix+'O-center-browser.mp4) · [近墙短片]('+prefix+'O-near-wall-browser.mp4)。\n'
    write(out/'summary.json',result);write(out/'reference-correspondence.json',references);csv_write(out/'frame-metrics.csv',all_rows);csv_write(out/'step-summary.csv',summary);csv_write(out/'paired-sampler-comparison.csv',sampler_pairs)
    cards=[]
    for group in groups:
        info=html.escape(f"{group['id']} | {group['case_id']} anchor{group['anchor']} | sampler{group['sampler_steps']} | A true-history correction vs B open loop")
        body=f'<section id="{group["id"]}" data-letter="{html.escape(group["letter"])}"><h2>{info}</h2><p>Physics cohort: {html.escape(group["physics_cohort"])}. Model training physics remains historical dynamic0.1s.</p><img src="{group["grid"]}" alt="Same-time GT/A/B frame grid">'
        if group['closeup']:body+=f'<details><summary>Identical integer closeup</summary><img src="{group["closeup"]}"></details>'
        if group['video']:body+=f'<video controls preload="none" src="{group["video"]}"></video>'
        uri_root=DIAGNOSTICS_ROOT if stage.is_relative_to(DIAGNOSTICS_ROOT)else stage
        archive_url='/diagnostics-root/'+str(Path(group['archive_path']).relative_to(uri_root)).replace('\\','/')
        metadata_url='/diagnostics-root/'+str(Path(group['metadata_path']).relative_to(uri_root)).replace('\\','/')
        body+='<p>Independent first-frame check: '+html.escape(str(group['first_frame_check_mode']))+'; equality '+str(group['first_output_equal'])+'.</p>'
        body+=f'<p><a href="{html.escape(archive_url)}">Saved raw float32 / truth / actions / states</a> · <a href="{html.escape(metadata_url)}">Replay parameters and identities</a></p></section>';cards.append(body)
    reference_cards=''.join(f'<li>Screenshot{r["screenshot"]:02d} ({r["letter"]}): {r["status"]}; visual components may span different clips/steps; complete same-frame reproduction is not claimed. New original physics is not an exact old-protocol replay. <a href="/diagnostics-root/user-references/screenshot-{r["screenshot"]:02d}.png">original</a> · <a href="reference-correspondence.json">evidence steps and physics cohort</a></li>'for r in references)
    page='''<!doctype html><meta charset="utf-8"><title>R3B playground frame diagnostics</title><style>body{font:16px system-ui;margin:24px;background:#f1f3f5;color:#202735}section{background:white;padding:18px;margin:20px 0;border-radius:10px}img{max-width:100%;height:auto;image-rendering:pixelated}video{display:block;max-width:100%;margin-top:12px}a{color:#1554a0}summary{cursor:pointer}select{font:inherit;padding:5px}</style><h1>R3B 7499: frozen-model frame diagnostics</h1><p>GT / A: repeated H1 with true-history correction / B: H4 raw open-loop. t0,+0.1,+0.2,+0.3,+0.4s. Same actions and paired seeds. Prediction display is clipped/rounded only for PNG; raw float32 feedback remains unchanged. Deliberately chosen diagnostic scenes are not a population failure-rate estimate. No training.</p><p><a href="frame-metrics.csv">Each frame metrics</a> · <a href="step-summary.csv">Step summary</a> · <a href="paired-sampler-comparison.csv">Matched3/8/16</a> · <a href="summary.json">Exact data/source links</a></p><label>Letter <select id="letter"><option value="all">All</option>'''+''.join(f'<option>{v}</option>'for v in'B C O X T L'.split())+'</select></label><h2>User screenshots: same-class status</h2><ul>'+reference_cards+'</ul>'+physical_html+''.join(cards)+'''<script>document.querySelector('#letter').addEventListener('change',e=>document.querySelectorAll('section[data-letter]').forEach(s=>s.hidden=e.target.value!=='all'&&s.dataset.letter!==e.target.value));</script>'''
    if any(g['physics_protocol_sha256']!=TRAINING_PHYSICS_SHA for g in groups):
        page=page.replace('<h1>R3B 7499: frozen-model frame diagnostics</h1>','<h1>R3B 7499: frozen-model frame diagnostics</h1><p><strong>Cross-physics preview:</strong> original kinematic-pusher cases use different physics from this checkpoint\'s historical dynamic-pusher0.1s training. Cohorts are separate. This is no proof of an autoregressive training-distribution cause.</p>')
    (out/'index.html').write_text(page,encoding='utf-8')
    first_bad=[]
    for group in groups:
        for branch in('A','B'):
            chosen=[r for r in group['rows']if r['branch']==branch and observed_defects(r)]
            first_bad.append([group['id'],group['letter'],group['sampler_steps'],branch,chosen[0]['step']if chosen else'none listed',','.join(observed_defects(chosen[0]))if chosen else'No rigidity pass inferred'])
    report=f'''# R3B试玩第一步诊断：逐帧对照

本报告只读取本次保存的实际轨迹和原始float32预测，不训练、不修改权重，不隐藏碎块、补孔、修边或贴回模板。
当前包含{len(groups)}组保存片段，{len(all_rows)}个A/B逐帧记录；有{len(failures)}个缺少raw的失败片段记录保留。选择是定向诊断，不是总体失败率。
**物理分组必须分开读：7499模型仍是旧动态推杆/0.1秒半衰期训练；新默认原始运动学推杆结果是跨物理预览，不是旧协议故障的动力学精确重放，也不能单独证明训练与递推分布不一致。**

A每控制重新使用真实四帧历史，有真实历史校正；B从相同真实起点连续预测四步，后续只喂自己raw预测。
每步动作/seed配对，首帧独立调用并检查一致。现有UI的H4每批重新取真实历史，不是从最初场景无限递推。
GT未来只作图像/误差标签。图为同坐标整数nearest放大；原始raw数值不被显示clip影响。

[逐帧图与同步慢放视频](report/index.html) · [全部逐帧指标](report/frame-metrics.csv) · [结构化原始引用](report/summary.json)

{physical_report}

## 固定片段去噪3/8/16对照

下表只比较同物理、同case/anchor、同动作及逐步seed都有3/8/16结果的固定代表片段，不能拿3步全部片段平均与8/16的子集平均直接比较。

{table(['case','anchor','分支','去噪步','四帧平均ROI','H4 ROI','4次调用外层秒','最早可见异常'],[[r['case_id'],r['anchor'],r['branch'],r['sampler_steps'],fmt(r['all4_ROI']),fmt(r['H4_ROI']),fmt(r['four_sampler_seconds'],3),r['first_visible_defect_step']if r['first_visible_defect_step']is not None else'none listed']for r in sampler_pairs])}

配对原始明细：[paired-sampler-comparison.csv](report/paired-sampler-comparison.csv)。耗时读取实际runner逐调用`seconds`字段，兼容旧`sampler_s`；缺失或不完整显示N/A，不能填0。全片段实际调用外层耗时合计{fmt(result['sampler_timing']['total_seconds'],6)}秒，{result['sampler_timing']['valid_records']}/{result['sampler_timing']['expected_records']}条有效调用记录。{timing_note} 不含轨迹、图表或视频生成时间，不给模型时间硬上限，不删除更多步数反而更差的结果。

## 六张截图对应

{table(['截图','字母','状态','说明'],[[r['screenshot'],r['letter'],r['status'],'新原始物理下同类组成现象；可能跨片段/步，不称整张同帧复现或旧协议精确重放；证据见JSON']for r in references])}

截图的多个组成现象若在不同片段/不同步出现，会逐项链接，不能冒称同一帧精确还原。
O/C圆化以人工逐帧观察确认并记录，没有用不可靠raw角点数量直接认定“棱角恢复率”。

[实际逐图人工观察](review/VISUAL-REVIEW.md) · [只标注实际看过的逐帧记录](review/manual-observations.json)。人工观察保留首步已有圆化/形变、后续累积、真实历史校正也出现碎块等例外；没有异常标注不等于刚体通过。

## 可见异常最早发生步

manual_*为实际看过同坐标保存图的人工形状判断，与自动palette可测性分开保留；遮挡unknown不会盖过这些可见形变。

{table(['片段','字母','去噪步','分支','最早异常步','可见异常'],first_bad)}

## 解释边界

- 第1步A/B相同且已错，说明该误差不需要多步递推才能出现；仍不能仅凭截图归因网络结构或训练分布。
- 同一轨迹/seed下B后续恶化而A较好，支持预测历史递推对本片段的影响；不能直接证明具体训练机制或所有场景。
- 形状计数区分可见孔洞、独立灰块、近蓝色边缘的小区域/抗锯齿和遮挡。全局occlusion不会抹掉清楚的推杆消失、可见孔变化/碎块；unknown不算pass。
- 原始角点数受96像素栅格、位置朝向及遮挡影响，只作诊断。明显圆化需结合原始同坐标图与人工记录。
- 质心/相对位置是可见图像测量，不冒充真实物理位姿；实际平移、转角与接触由保存标签单列。推杆消失时相关位置误差不可测，失效计数仍保留。
- 3/8/16只在固定代表片段使用同checkpoint/动作/seed比较；所有慢/坏结果保留，不保证去噪更多更好。
- 确认的UI/输入/时序问题及修复证据由本阶段接口审计另附；报告不会把鼠标预览十字当作实际四步执行目标。

采样速度、per-step异常和全部分母见step-summary.csv/summary.json；不得以缺失值过滤失败后宣称模型可靠。
完成本诊断后STOP waiting_user_approval，不开始新训练、采集或模型方案。
'''
    (stage/'REPORT.md').write_text(report,encoding='utf-8')
    write(out/'inventory.json',dict(kind='R3B-playground-report-assets.v1',files=[dict(file=str(p.relative_to(out)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(out.rglob('*'))if p.is_file()and p.name!='inventory.json'],model_calls=0,physics_steps=0,source_clip_files_modified=False))
    return dict(report=str(stage/'REPORT.md'),index=str(out/'index.html'),groups=len(groups),rows=len(all_rows),reference_statuses=[dict(screenshot=r['screenshot'],status=r['status'])for r in references],model_calls=0,physics_steps=0)
