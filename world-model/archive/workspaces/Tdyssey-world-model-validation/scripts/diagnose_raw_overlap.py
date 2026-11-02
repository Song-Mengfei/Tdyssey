"""Two bounded offline stages; saved RGB/JSON only, never neural/physics work."""
import argparse
from collections import Counter
import hashlib
import importlib.abc
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','pygame','pymunk','gym','cv2','scipy'}:
            raise ImportError('Offline diagnosis forbids runtime import: '+fullname)


sys.meta_path.insert(0,NoRuntime())
import numpy as np
from PIL import Image,ImageDraw
from tdyssey.envs.official import official_shape
from tdyssey.geometry import Pose
from tdyssey.evaluation import goal_metrics

OUTPUT = ROOT/'artifacts/raw-tl-overlap-diagnosis-20261027'
MANUAL = ROOT/'artifacts/manual-official-world-model/20261026-235943-407588'
CASES = ('view-001009-813929-case','view-001013-708426-case')
MODEL_SHA = '79c1fe8b9eaa44b80f8c90ae27f3280b84a01456aa3efd6ebc75dda253558c05'
SCHEDULE = ROOT/'artifacts/gpc-raw-tl-fulltrain-preparation-20261026/training-schedule.json'
SCHEDULE_SHA = '45770d001c7fb3f7b20743736ba44b3bdd4482cba19aa5752634ee5b9b217b0a'
BIN_NAMES = ('zero','positive_lt_0.5','0.5_to_lt_0.8','gte_0.8')
PALETTE = np.array([[119,136,153],[143,163,184],[144,238,144],
                    [65,105,225],[78,126,255],[255,255,255]],dtype=float)
CLASS_NAMES = ('gray_outline','gray_fill','green','blue_outline','blue_fill','white')
_Y,_X = np.mgrid[:96,:96]
WORLD_X,WORLD_Y = (_X+.5)*512/96-.5,(_Y+.5)*512/96-.5


def encoded(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(value).hexdigest()
def file_hash(path):
    if Path(path).suffix.lower() in ('.pt','.pth'):raise PermissionError('No weight content reads')
    return digest(Path(path).read_bytes())
def output(name,payload):
    path=OUTPUT/name
    if path.exists() or not path.resolve().is_relative_to(OUTPUT.resolve()):raise FileExistsError(name)
    # Simple finite output admission only; no guardian/framework or old scans.
    existing=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
    if existing+len(payload)>15*1024**2:raise OSError('15MiB payload reserve exceeded')
    with path.open('xb') as stream:stream.write(payload)
    return dict(path=str(path),bytes=len(payload),sha256=digest(payload))
def write_json(name,value):return output(name,encoded(value))
def pose(value):return Pose(tuple(value['position']),value['angle'])
def coverage(state):
    return goal_metrics(official_shape(state['shape']),pose(state['object_body']['pose']),pose(state['goal'])).coverage
def bin_name(value):return BIN_NAMES[0 if value==0 else 1 if value<.5 else 2 if value<.8 else 3]


def geometry_mask(state,goal=False):
    body=pose(state['goal'] if goal else state['object_body']['pose'])
    c,s=math.cos(body.angle),math.sin(body.angle)
    dx,dy=WORLD_X-body.position[0],WORLD_Y-body.position[1]
    x,y=c*dx+s*dy,-s*dx+c*dy
    mask=np.zeros((96,96),dtype=bool)
    for rect in official_shape(state['shape']).components:
        mask |= (x>=rect.x0)&(x<=rect.x1)&(y>=rect.y0)&(y<=rect.y1)
    return mask


def margin(mask,erode):
    pad=np.pad(mask,1,constant_values=False)
    neighbours=[pad[dy:dy+96,dx:dx+96] for dy in range(3) for dx in range(3)]
    return (np.logical_and if erode else np.logical_or).reduce(neighbours)


def palette_maps(image):
    rgb=image.transpose(1,2,0).astype(float)*255
    distance=np.linalg.norm(rgb[:,:,None,:]-PALETTE[None,None,:,:],axis=-1)/(255*math.sqrt(3))
    nearest=distance.argmin(-1)
    ties=(distance==distance.min(-1,keepdims=True)).sum(-1)>1
    nearest[ties]=-1
    return dict(nearest=nearest,gray=distance[:,:,:2].min(-1),green=distance[:,:,2])


def region_metrics(mask,truth,prediction):
    n=int(mask.sum()); valid=mask&np.isin(truth['nearest'],(0,1));count=int(valid.sum())
    green=valid&(prediction['nearest']==2);gray=valid&np.isin(prediction['nearest'],(0,1))
    points=np.argwhere(green)
    return dict(geometric_pixels=n,actual_gray_pixels=count,
        actual_gray_fraction=None if not n else count/n,
        prediction_gray_retained=int(gray.sum()),prediction_green_intrusion=int(green.sum()),
        gray_retention_fraction=None if not count else int(gray.sum())/count,
        green_intrusion_fraction=None if not count else int(green.sum())/count,
        prediction_class_counts={name:int((valid&(prediction['nearest']==i)).sum()) for i,name in enumerate(CLASS_NAMES)},
        prediction_ties=int((valid&(prediction['nearest']==-1)).sum()),
        mean_gray_distance_actual=None if not count else float(truth['gray'][valid].mean()),
        mean_gray_distance_prediction=None if not count else float(prediction['gray'][valid].mean()),
        mean_green_distance_prediction=None if not count else float(prediction['green'][valid].mean()),
        green_pixels_xy=[[int(x),int(y)] for y,x in points])


def pil_frame(frame):
    pixels=np.rint(np.clip(frame,0,1)*255).astype(np.uint8).transpose(1,2,0)
    return Image.fromarray(pixels)
def save_image(name,image):
    buf=io.BytesIO();image.save(buf,format='PNG');return output(name,buf.getvalue())
def sheet(name,frames,labels,size=192):
    image=Image.new('RGB',(size*len(frames),size+44),'white');draw=ImageDraw.Draw(image)
    for i,(frame,label) in enumerate(zip(frames,labels)):
        tile=pil_frame(frame) if isinstance(frame,np.ndarray) else frame
        image.paste(tile.resize((size,size),Image.Resampling.NEAREST),(i*size,0))
        draw.text((i*size+4,size+4),label,fill='black')
    return save_image(name,image)


def case_analysis():
    from tdyssey.manual_world_model import load_case_bundle,request_from_case
    records=[];unique={};panels=[]
    for name in CASES:
        folder=MANUAL/name; bundle=load_case_bundle(folder)
        meta,arrays=bundle['metadata'],bundle['arrays'];request=request_from_case(bundle)
        accepted=meta['predictions'][meta['execution']['prediction_index']]
        assert request['horizon']==1 and meta['image_domain']=='official-original-color'
        assert accepted['checkpoint_identity']['checkpoint']['sha256']==MODEL_SHA
        assert accepted['checkpoint_identity']['sidecar']['metadata']['updates']==2328
        assert meta['t1_sha256'] is None
        assert np.array_equal(arrays['history_raw'],arrays['history_model'])
        assert np.array_equal(arrays['actual_raw'],arrays['actual_model'])
        pred=arrays[accepted['array']];actual=arrays['actual_raw'];history=arrays['history_raw']
        stem=name.removesuffix('-case');npypath=MANUAL/(stem+'-prediction.npy')
        saved=np.load(npypath,allow_pickle=False)
        assert np.array_equal(saved,pred)
        steps=meta['execution']['steps'];assert len(steps)==1 and steps[0]['action']==meta['requested_actions'][0]
        assert steps[0]['before_state']==meta['request']['history_states'][-1]
        assert [s['control_step'] for s in meta['request']['history_states']]==[84,85,86,87]
        assert steps[0]['after_state']['control_step']==88
        trace=accepted['prediction_trace'][0]
        assert trace['input_sha256']==digest(history.tobytes()) and trace['raw_sha256']==digest(pred.tobytes())
        assert trace['t1_applied'] is False and trace['sigma_cond'] is None
        assert np.array_equal(np.asarray(trace['actions_absolute'][:3],np.float32),arrays['past_actions'])
        assert trace['actions_absolute'][-1]==meta['requested_actions'][0]
        key=(meta['request']['request_key'],digest(history.tobytes()),digest(pred.tobytes()),digest(actual.tobytes()),MODEL_SHA)
        ui=np.asarray(Image.open(MANUAL/(stem+'.png')).convert('RGB'))
        assert ui.shape==(538,1000,3)
        pixels=np.asarray(pil_frame(pred[0]));pane=ui[50:522,512:984]
        floor=(np.arange(472)*96//472).astype(int)
        centre=np.minimum(95,((np.arange(472)+.5)*96/472).astype(int))
        ui_errors={label:dict(unequal_pixels=int(np.any(pane!=pixels[index[:,None],index[None,:]],axis=-1).sum()),
            mean_channel_error=float(np.abs(pane.astype(float)-pixels[index[:,None],index[None,:]]).mean()))
            for label,index in (('floor',floor),('pixel_centre',centre))}
        records.append(dict(path=str(folder),case_json_sha256=file_hash(folder/'case.json'),
            arrays_sha256=file_hash(folder/'arrays.npz'),sibling_npy_sha256=file_hash(npypath),
            request_key=key[0],input_sha256=key[1],prediction_sha256=key[2],actual_sha256=key[3],
            saved_npy_equals_case=True,raw_equals_model=True,ui_scale_comparisons=ui_errors,
            history_control_steps=[84,85,86,87],executed_steps=[87,88],model_declared_sha256=MODEL_SHA,
            model_weight_bytes_read=0))
        if key in unique:continue
        unique[key]=True; state=steps[0]['after_state']; O,G=geometry_mask(state),geometry_mask(state,True)
        p=state['pusher_body']['pose']['position']; P=(WORLD_X-p[0])**2+(WORLD_Y-p[1])**2<=15**2
        interior=margin(O,True)&~margin(P,False)
        overlap=interior&margin(G,True); nonoverlap=interior&~margin(G,False)
        truth,predmap=palette_maps(actual[0]),palette_maps(pred[0])
        regions={label:region_metrics(mask,truth,predmap) for label,mask in (('overlap',overlap),('nonoverlap',nonoverlap))}
        region_image=np.full((96,96,3),255,dtype=np.uint8)
        region_image[O]=(200,200,200);region_image[overlap]=(150,90,210);region_image[nonoverlap]=(240,150,30)
        intrusion=np.full((96,96,3),255,dtype=np.uint8)
        valid=interior&np.isin(truth['nearest'],(0,1));intrusion[valid]=(170,170,170)
        intrusion[valid&(predmap['nearest']==2)]=(0,220,0)
        coords=np.argwhere(O|G|P);lo=np.maximum(0,coords.min(0)-4);hi=np.minimum(96,coords.max(0)+5)
        box=(int(lo[1]),int(lo[0]),int(hi[1]),int(hi[0]))
        panels.append(sheet('case-input-truth-prediction.png',list(history)+[actual[0],pred[0]],
            ['history84','history85','history86','history87','actual88','raw2328 prediction88']))
        panels.append(sheet('case-overlap-crops.png',[pil_frame(actual[0]).crop(box),pil_frame(pred[0]).crop(box),
            Image.fromarray(region_image).crop(box),Image.fromarray(intrusion).crop(box)],
            ['actual crop','prediction crop','TRUTH: overlap purple / other orange','TRUTH gray -> predicted green'],size=288))
        panels.append(save_image('actual-96.png',pil_frame(actual[0])))
        panels.append(save_image('prediction-96.png',pil_frame(pred[0])))
        metrics=dict(shape=state['shape'],coverage_truth=coverage(state),reported_coverage=meta['execution']['result']['truth_coverage'],
            history_coverage_truth=[coverage(s) for s in meta['request']['history_states']],
            original_world_pusher=state['pusher_body']['pose']['position'],
            region_pixels=dict(object=int(O.sum()),object_interior_without_pusher=int(interior.sum()),
                overlap=int(overlap.sum()),nonoverlap=int(nonoverlap.sum()),
                excluded_goal_boundary=int((interior&~(overlap|nonoverlap)).sum())),
            regions=regions,truth_only_labels=True,pose_and_predicted_coverage='not_measured',crop_box_xyxy=list(box),
            color_rule='strict nearest of fixed source swatches; ties separate; no tuned threshold',
            caveat='Loss of gray can reflect shifted predicted geometry as well as colour change; no causal palette-leak claim')
    return dict(saved_directories=2,unique_cases=len(unique),records=records,case=metrics,panels=panels,
        training_or_inference_operations=0,source_case_pixels_changed=False)


def coverage_analysis():
    from tdyssey.gpc_adaptation_data import read_adaptation_manifest
    from tdyssey.gpc_raw_fulltrain_data import validate_full_schedule
    source=read_adaptation_manifest(ROOT/'artifacts/gpc-data-stage-a-20261026/adaptation-data.json')
    assert file_hash(SCHEDULE)==SCHEDULE_SHA
    schedule=json.loads(SCHEDULE.read_bytes());validate_full_schedule(schedule,data=source)
    episodes={row['trajectory_id']:row for row in source['episodes'] if row['role']=='train'}
    assert len(episodes)==24
    records={};identities=[]
    for identifier,row in episodes.items():
        assert row['role']=='train' and row['split']=='train'
        item=next(item for item in row['entry']['files'] if item['name'].endswith('.json'))
        path=Path(row['root'])/item['name'];payload=path.read_bytes()
        assert len(payload)==item['bytes'] and digest(payload)==item['sha256']
        records[identifier]=json.loads(payload)
        identities.append(dict(trajectory_id=identifier,record_sha256=item['sha256'],bytes=len(payload)))
    rows=[];diagnostic_deltas=[]
    for entry in schedule['entries']:
        episode=episodes[entry['trajectory_id']];record=records[entry['trajectory_id']]
        anchor=entry['anchor'];transition=record['trajectory']['transitions'][anchor]
        state=transition['after'];assert state['shape']==entry['shape']
        value=coverage(state);diagnostic=record['step_diagnostics'][anchor]
        for field in ('coverage','truth_coverage','raw_coverage'):
            if field in diagnostic:diagnostic_deltas.append(abs(value-diagnostic[field]))
        row=dict(trajectory_id=entry['trajectory_id'],shape=entry['shape'],anchor=anchor,
            scene_sha256=entry['initial_state_sha256'],schedule_step=entry['step'],
            target_coverage=value,coverage_bin=bin_name(value),contact=entry['contact'],
            rotation_degrees=entry['rotation_degrees'],target_state=state,
            source_role='train',source_split='train',source_root=episode['root'],
            image_file=record['image_file'],image_sha256=record['image_sha256'])
        rows.append(row)
    assert len(rows)==2328 and len({(r['trajectory_id'],r['anchor']) for r in rows})==2328
    summary={};selected=[]
    for shape in ('T','L'):
        subset=[row for row in rows if row['shape']==shape]
        bins={};thresholds={}
        for label in BIN_NAMES:
            group=sorted((r for r in subset if r['coverage_bin']==label),key=lambda r:(r['trajectory_id'],r['anchor']))
            bins[label]=dict(windows=len(group),scenes=len({r['scene_sha256'] for r in group}),
                trajectories=len({r['trajectory_id'] for r in group}),contacts=sum(r['contact'] for r in group))
            if group:selected.append(group[0])
        for threshold in (.1,.5,.8,.9,.95):
            group=[r for r in subset if r['target_coverage']>=threshold]
            thresholds[str(threshold)]=dict(windows=len(group),scenes=len({r['scene_sha256'] for r in group}))
        success=[r for r in subset if r['target_coverage']>.95]
        summary[shape]=dict(windows=len(subset),scenes=len({r['scene_sha256'] for r in subset}),
            minimum=min(r['target_coverage'] for r in subset),maximum=max(r['target_coverage'] for r in subset),
            mean=sum(r['target_coverage'] for r in subset)/len(subset),bins=bins,at_least=thresholds,
            strict_success_gt_0_95=dict(windows=len(success),scenes=len({r['scene_sha256'] for r in success})))
    write_json('training-window-coverage.json',dict(rows=rows,summary=summary,train_record_identities=identities,
        schedule_sha256=SCHEDULE_SHA,source_manifest_sha256=source['sha256'],
        train_json_reads=24,train_image_frames_read=0,development_payload_reads=0,
        independent_validation_payload_reads=0,held_payload_reads=0,
        existing_diagnostic_comparisons=len(diagnostic_deltas),
        existing_diagnostic_max_delta=max(diagnostic_deltas) if diagnostic_deltas else None))
    write_json('frozen-training-frame-selection.json',dict(rule='First trajectory_id/anchor per shape and fixed coverage bin; empty bins not filled',
        bins=list(BIN_NAMES),maximum_windows=8,maximum_frames=40,entries=selected))
    return summary


def frame_checks():
    plan=json.loads((OUTPUT/'frozen-training-frame-selection.json').read_bytes())
    assert len(plan['entries'])<=8
    results=[];decoded=0;hashes={}
    for ordinal,row in enumerate(plan['entries']):
        assert row['source_role']=='train' and row['source_split']=='train'
        path=Path(row['source_root'])/row['image_file']
        if str(path) not in hashes:
            hashes[str(path)]=file_hash(path)
            assert hashes[str(path)]==row['image_sha256']
        if decoded+5>40:raise ValueError('Forty selected-frame limit exceeded')
        # Only five selected frames become arrays. ZipExtFile may inflate and
        # discard compressed prefix bytes while seeking; disclose that IO below.
        with zipfile.ZipFile(path) as archive,archive.open('images.npy') as member:
            version=np.lib.format.read_magic(member)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(member)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(member)
            else:raise ValueError('Unexpected source NPY header version')
            assert shape[1:]==(3,96,96) and dtype==np.dtype('uint8') and not fortran
            header=member.tell();start=row['anchor']-3;frame_bytes=3*96*96
            assert 0<=start and start+5<=shape[0]
            member.seek(header+start*frame_bytes)
            payload=member.read(5*frame_bytes);assert len(payload)==5*frame_bytes
            frames=np.frombuffer(payload,dtype=np.uint8).reshape(5,3,96,96).astype(np.float32)/np.float32(255)
        decoded+=5;state=row['target_state'];O,G=geometry_mask(state),geometry_mask(state,True)
        p=state['pusher_body']['pose']['position'];P=(WORLD_X-p[0])**2+(WORLD_Y-p[1])**2<=15**2
        interior=margin(O,True)&~margin(P,False)
        maps=palette_maps(frames[4]);target_metrics={}
        for label,mask in (('overlap',interior&margin(G,True)),('nonoverlap',interior&~margin(G,False))):
            n=int(mask.sum());target_metrics[label]=dict(pixels=n,
                nearest_gray=int((mask&np.isin(maps['nearest'],(0,1))).sum()),
                nearest_green=int((mask&(maps['nearest']==2)).sum()))
        panel=sheet(f'train-{ordinal:02d}-{row["shape"]}-{row["coverage_bin"]}.png',list(frames),
            [f'real frame{start+i}' for i in range(5)],size=144)
        results.append(dict(trajectory_id=row['trajectory_id'],shape=row['shape'],anchor=row['anchor'],
            coverage=row['target_coverage'],coverage_bin=row['coverage_bin'],source_npz_sha256=hashes[str(path)],
            decoded_indices=list(range(start,start+5)),prefix_bytes_inflated_or_skipped=start*frame_bytes,
            target_region_color=target_metrics,panel=panel))
    return dict(selected_windows=len(results),decoded_train_frames=decoded,results=results,
        selection_uses_model_predictions=False,full_episode_arrays_materialized=0,
        note='Source NPZ compressed hashes verified; prefix IO is not hidden. Only selected5frames/window are numerically decoded.')


def source_checks():
    # No subprocess, native resource query, SDK or hidden extra command.
    for name in ('check_repo','check_messages'):
        spec=importlib.util.spec_from_file_location('_overlap_'+name,ROOT/'scripts'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if name=='check_messages':
            errors=module.check_commit((OUTPUT/'commit-message.txt').read_text(encoding='utf-8'))
            if errors:raise AssertionError(errors)
        else:
            old=sys.argv;sys.argv=[str(ROOT/'scripts/check_repo.py')]
            try:
                if module.main():raise AssertionError('Source check failed')
            finally:sys.argv=old


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--stage',required=True,choices=('case-and-coverage','train-frames-and-checks'))
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    claim=OUTPUT/(args.stage+'.started.json')
    with claim.open('xb') as stream:stream.write(encoded(dict(stage=args.stage,neural_operations=0)))
    if args.stage=='case-and-coverage':
        # Exactly one bounded snapshot of the mutable GUI log, never polled.
        log=MANUAL/'events.jsonl'
        with log.open('rb') as stream:
            stream.seek(0,2);length=stream.tell();start=max(0,length-65536);stream.seek(start)
            snapshot=stream.read(length-start)
        if start and b'\n' in snapshot:snapshot=snapshot.split(b'\n',1)[1]
        event_info=output('events-bounded-snapshot.jsonl',snapshot)
        cases=case_analysis();write_json('case-analysis.json',cases)
        coverage_summary=coverage_analysis()
        result=dict(stage=args.stage,status='complete',cases=cases['unique_cases'],regions=cases['case']['regions'],
            coverage_truth=cases['case']['coverage_truth'],training_coverage=coverage_summary,
            event_snapshot=dict(original_bytes_at_snapshot=length,read_bytes=len(snapshot),identity=event_info),
            neural_operations=0,physics=0,training_frames_decoded=0)
    else:
        result=frame_checks();source_checks();result.update(stage=args.stage,status='complete',neural_operations=0,physics=0)
    assert not any(name.split('.')[0] in {'torch','pygame','pymunk','cv2','scipy'} for name in sys.modules)
    write_json(args.stage+'-result.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
