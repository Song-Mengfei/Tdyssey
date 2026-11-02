"""One bounded CPU-only original-PushT parity and separation observation pass."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from tdyssey.native_rgb_original_physics import make_original_env,PROTOCOL,PROTOCOL_SHA256
from tdyssey.native_rgb_env import load_physics_source,STATE_COLUMNS
from tdyssey.envs.official import make_official_shape_env_class
from tdyssey.native_rgb_r2a_env import _body_identity,_shape_geometry,_xy,_scalar

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(value):
    if isinstance(value,dict):return{str(k):safe(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return[safe(v)for v in value]
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float)and not math.isfinite(value):return str(value)
    return value
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(safe(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',encoding='utf-8',newline='')as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(safe(v))if isinstance(v,(list,tuple,dict,np.ndarray))else safe(v)for k,v in r.items()}for r in rows)
def state15(env):return np.asarray((*env.agent.position,*env.block.position,env.block.angle,*env.agent.velocity,*env.block.velocity,env.block.angular_velocity,*env.agent.force,*env.block.force,env.block.torque),np.float64)
def fullbody(body):
    return dict(position=_xy(body.position),angle=float(body.angle),velocity=_xy(body.velocity),angular_velocity=float(body.angular_velocity),force=_xy(body.force),torque=float(body.torque),**_body_identity(body))
def fullstate(env):
    return dict(agent=fullbody(env.agent),object=fullbody(env.block),latest_action=None if env.latest_action is None else list(env.latest_action),control_step=int(getattr(env,'control_step',0)))
def copy_initial_body(source,target):
    # Nonzero center-of-gravity bodies require angle before position, matching
    # the project's canonical _write_body and upstream nonlegacy assignment.
    for name in('angle','position','velocity','angular_velocity','force','torque'):
        value=getattr(source,name);setattr(target,name,tuple(value)if name in('position','velocity','force')else value)
def physical(env):
    roles=lambda body:'agent'if body is env.agent else'object'if body is env.block else'wall'
    shapes=[]
    for shape in env.space.shapes:
        f=shape.filter
        shapes.append(dict(role=roles(shape.body),body=_body_identity(shape.body),geometry=_shape_geometry(shape),friction=float(shape.friction),elasticity=float(shape.elasticity),sensor=bool(shape.sensor),collision_type=int(shape.collision_type),filter=dict(group=int(f.group),categories=int(f.categories),mask=int(f.mask)),surface_velocity=_xy(shape.surface_velocity)))
    shapes.sort(key=lambda r:json.dumps(r,sort_keys=True))
    return dict(kp=float(env.k_p),kv=float(env.k_v),sim_hz=int(env.sim_hz),control_hz=int(env.control_hz),space={k:_scalar(getattr(env.space,k))for k in('damping','iterations','collision_slop','collision_bias','collision_persistence','sleep_time_threshold','idle_speed_threshold')},agent=_body_identity(env.agent),object=_body_identity(env.block),shapes=shapes)


def make_reference(source,letter,wrapper):
    """No reset/_set_state hidden step: direct official setup, initial body copy."""
    pinned=load_physics_source(source);Ref=make_official_shape_env_class(pinned)
    reference=Ref(shape=letter,render_size=96,render_action=False);reference._setup()
    copy_initial_body(wrapper.agent,reference.agent);copy_initial_body(wrapper.block,reference.block)
    reference.space.reindex_shapes_for_body(reference.agent);reference.space.reindex_shapes_for_body(reference.block)
    reference.control_step=0;reference.latest_action=None
    return reference,pinned


def parity(source,letter,out,cost):
    env=make_original_env(source,letter);reference=None;phase=[];initial=None;done=[]
    try:
        rgb=env.reset_scene(center=(256,256),angle=.23 if letter=='T'else.41,pusher=(155,254))
        reference,pinned=make_reference(source,letter,env)
        initial=dict(fullstate_equal=fullstate(env._base)==fullstate(reference),physics_equal=physical(env._base)==physical(reference))
        # Native clean renderer is reused only as an observation on the direct
        # reference world; reference physics remains pinned PushTEnv.step.
        ref_rgb=type(env._base).render_layers(reference)['rgb']
        initial['clean_RGB_exact']=bool(np.array_equal(rgb,ref_rgb));initial['RGB_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
        if not all(initial[k]for k in('fullstate_equal','physics_equal','clean_RGB_exact')):raise AssertionError('Initial direct-source reference differs')
        raw_step=reference.space.step;reference_rows=[]
        def observe(dt):
            before=state15(reference);raw_step(dt);after=state15(reference)
            reference_rows.append(dict(dt=float(dt),pre=before.copy(),post=after.copy()))
            cost['known_substeps']+=1
        reference.space.step=observe
        actions=np.asarray([(155,254)]*2+[(208,242),(232,228),(258,226),(260,255),(208,285),(160,290)],np.float64)
        saved=[]
        for control,action in enumerate(actions):
            cost['attempted_controls']+=1
            try:actual,info=env.step(action)
            finally:cost['known_substeps']+=sum(bool(r.get('executed'))for r in env.last_substeps)
            cost['completed_controls']+=1;cost['attempted_controls']+=1;before=len(reference_rows)
            pinned.step(reference,action);reference.control_step+=1;cost['completed_controls']+=1
            current=reference_rows[before:]
            if len(current)!=10 or len(env.last_substeps)!=10:raise AssertionError('Ten original substeps required')
            for sub,(a,b)in enumerate(zip(env.last_substeps,current)):
                pre=np.asarray(a['pre_state']);post=np.asarray(a['post_state'])
                row=dict(letter=letter,control=control+1,substep=sub+1,pre_max_abs_error=float(np.max(np.abs(pre-b['pre']))),post_max_abs_error=float(np.max(np.abs(post-b['post']))))
                phase.append(row);saved.append((pre,post,b['pre'],b['post']))
            same_rgb=bool(np.array_equal(actual,type(env._base).render_layers(reference)['rgb']))
            done.append(dict(control=control+1,action=action.tolist(),RGB_exact=same_rgb,actual_state=state15(env).tolist(),reference_state=state15(reference).tolist()))
            if not same_rgb or max(r['post_max_abs_error']for r in phase)>1e-9:raise AssertionError('Original control/substep parity failed')
        np.savez_compressed(out/f'parity-{letter}.npz',actions=actions,actual_pre=np.stack([r[0]for r in saved]),actual_post=np.stack([r[1]for r in saved]),reference_pre=np.stack([r[2]for r in saved]),reference_post=np.stack([r[3]for r in saved]))
        return dict(letter=letter,passed=True,controls_per_world=8,substeps_per_world=80,initial=initial,max_pre_error=max(r['pre_max_abs_error']for r in phase),max_post_error=max(r['post_max_abs_error']for r in phase),controls=done,substeps=phase)
    except BaseException as exc:
        write(out/f'parity-{letter}-failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),initial=initial,complete_controls=done,paired_substeps=phase,actual_partial=env.last_substeps));raise
    finally:
        if reference is not None:reference.close()
        env.close()


def contact_counts(substeps):
    result={kind:0 for kind in('agent_object','object_wall','agent_wall')};positive=dict(result)
    for row in substeps:
        for kind in result:
            events=[e for e in row.get('contacts',[])if e['kind']==kind and e['phase']=='post_solve']
            result[kind]+=int(bool(events));positive[kind]+=int(any(np.linalg.norm(e.get('total_impulse',[0,0]))>0 for e in events))
    return result,positive


def video(path,frames,rows,name):
    import cv2
    from PIL import Image,ImageDraw,ImageFont
    f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    w,h=480,560;writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(w,h))
    if not writer.isOpened():raise RuntimeError('Existing MP4 writer unavailable')
    try:
        for i,raw in enumerate(frames):
            canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
            im=Image.fromarray(np.asarray(raw).transpose(1,2,0)).resize((480,480),Image.Resampling.NEAREST);canvas.paste(im,(0,48))
            row=rows[i-1]if i else None
            draw.text((6,6),f'{name} | t={i/10:.1f}s | original kinematic',font=f,fill='black')
            label='initial'if row is None else f"{row['phase']} | AO {row['agent_object_contact_substeps']}/10 | speed {row['object_speed']:.5f} | omega {row['omega']:.5f}"
            draw.text((6,29),label,font=f,fill='black');draw.text((6,534),'Continuous controls after withdrawal; 4x slow motion',font=f,fill='black')
            for _ in range(4):writer.write(np.asarray(canvas)[...,::-1])
    finally:writer.release()
    capture=cv2.VideoCapture(str(path));count=0
    while capture.read()[0]:count+=1
    capture.release()
    if count!=len(frames)*4:raise AssertionError('MP4 decode frame count differs')
    return dict(file=path.name,decoded_frames=count,fps=10,slow_motion=4)


def separation(source,near_wall,out,cost):
    env=make_original_env(source,'O');name='O-near-wall'if near_wall else'O-center';rows=[];subs=[];states=[];frames=[];actions=[];failure=None
    try:
        center=(430,256)if near_wall else(256,256)
        # Geometry comes from the same declared O polygon, not future simulation.
        from tdyssey.native_rgb_env import shape_geometry
        minx,miny,maxx,maxy=shape_geometry('O').bounds
        x0=center[0]+minx;right=center[0]+maxx;start=np.asarray([x0-15,(center[1]+miny+center[1]+maxy)/2],float)
        push=start+np.asarray([max(1.,504-right+1)if near_wall else 80.,0.])
        rgb=env.reset_scene(center=center,angle=0.,pusher=start)
        identity=env.initial_identity();states=[env.state().copy()];frames=[rgb.copy()]
        targets=[start.copy()]*2+[push.copy()]*6+[start.copy()]*6+[start.copy()]*10
        phases=['hold_initial']*2+['push']*6+['withdraw']*6+['continue_hold']*10
        for control,(target,phase)in enumerate(zip(targets,phases)):
            cost['attempted_controls']+=1;before=env.state().copy()
            try:rgb,info=env.step(target)
            finally:cost['known_substeps']+=sum(bool(r.get('executed'))for r in env.last_substeps)
            cost['completed_controls']+=1;after=env.state().copy();contacts,positive=contact_counts(env.last_substeps)
            row=dict(case=name,control=control+1,time_seconds=(control+1)/10,phase=phase,actual_target=target.tolist(),
                object_x=float(after[2]),object_y=float(after[3]),angle_rad=float(after[4]),object_speed=float(np.linalg.norm(after[7:9])),omega=float(after[9]),
                translation_world_px=float(np.linalg.norm(after[2:4]-before[2:4])),rotation_deg=float(np.degrees(after[4]-before[4])),
                max_penetration=np.asarray(info['penetration']).tolist(),**{k+'_contact_substeps':v for k,v in contacts.items()},**{k+'_positive_impulse_substeps':v for k,v in positive.items()})
            rows.append(row);states.append(after);frames.append(rgb.copy());actions.append(target.copy())
            for sub in env.last_substeps:
                sc,sp=contact_counts([sub]);ss=np.asarray(sub['post_state']);subs.append(dict(case=name,control=control+1,phase=phase,substep=sub['index']+1,time_seconds=control/10+(sub['index']+1)/100,
                    actual_target=target.tolist(),pre_state=sub['pre_state'],post_state=sub['post_state'],object_speed=float(np.linalg.norm(ss[7:9])),omega=float(ss[9]),
                    signed_wall_gaps=sub['signed_wall_gaps'],penetrations=sub['penetrations'],contacts=sub['contacts'],**{k+'_contact':v for k,v in sc.items()},**{k+'_positive_impulse':v for k,v in sp.items()}))
        np.savez_compressed(out/f'{name}.npz',state15=np.asarray(states),frame96=np.asarray(frames),actions_absolute=np.asarray(actions),substep_pre=np.asarray([r['pre_state']for r in subs]),substep_post=np.asarray([r['post_state']for r in subs]),substep_time=np.asarray([r['time_seconds']for r in subs]))
        csvwrite(out/f'{name}-controls.csv',rows);csvwrite(out/f'{name}-substeps.csv',subs);write(out/f'{name}-contacts.json',subs)
        detached=[r for r in rows if r['control']>=9 and r['agent_object_contact_substeps']==0]
        continued=[r for r in rows if r['phase']=='continue_hold']
        summary=dict(case=name,passed=True,identity=identity,controls=24,known_substeps=240,
            action_schedule=dict(hold_initial=2,push=6,withdraw=6,continued_real_controls=10),start_target=start.tolist(),push_target=push.tolist(),
            wall_initial_gap=504-right,first_full_no_AO_control=detached[0]['control']if detached else None,
            continued_controls_advanced=len(continued),maximum_post_separation_speed=max((r['object_speed']for r in detached),default=None),
            maximum_post_separation_translation_world_px=max((r['translation_world_px']for r in detached),default=None),
            maximum_post_separation_abs_omega=max((abs(r['omega'])for r in detached),default=None),
            total_AO_contact_substeps=sum(r['agent_object_contact_substeps']for r in rows),total_OW_contact_substeps=sum(r['object_wall_contact_substeps']for r in rows),total_AW_contact_substeps=sum(r['agent_wall_contact_substeps']for r in rows),
            max_penetration=np.max(np.asarray([r['max_penetration']for r in rows]),axis=0).tolist(),no_manual_velocity_or_pose_reset=True,
            video=video(out/f'{name}.mp4',frames,rows,name),rows=rows)
        return summary,subs
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),rows=rows,substeps=subs,partial=env.last_substeps)
        write(out/f'{name}-failure.json',failure);raise
    finally:env.close()


def curves(out,separations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,8),sharex=True,constrained_layout=True)
    for column,result in enumerate(separations):
        rows=result['rows'];x=[r['time_seconds']for r in rows]
        axes[0,column].plot(x,[r['object_speed']for r in rows],label='object linear speed');axes[0,column].set_title(result['case'])
        axes[1,column].plot(x,[r['omega']for r in rows],label='object angular speed')
        for kind in('agent_object','object_wall','agent_wall'):axes[2,column].step(x,[r[kind+'_contact_substeps']for r in rows],where='post',label=kind)
        for row,axis in enumerate(axes[:,column]):
            for boundary in(.2,.8,1.4):axis.axvline(boundary,color='gray',linestyle='--',linewidth=.7)
            axis.grid(alpha=.2);axis.legend(fontsize=8)
        axes[2,column].set_xlabel('Actual simulated seconds (continuous controls)')
    axes[0,0].set_ylabel('world pixels/s');axes[1,0].set_ylabel('radians/s');axes[2,0].set_ylabel('contact substeps /10')
    fig.suptitle('Original source damping0 / kinematic pusher: hold2, push6, withdraw6, continue hold10')
    fig.savefig(out/'separation-curves.png',dpi=140);plt.close(fig)


def run(output,source):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    with(output/'once-claim.json').open('x',encoding='utf-8')as f:json.dump(dict(pid=os.getpid(),at_utc=utc()),f)
    began=time.perf_counter();cost=dict(attempted_controls=0,completed_controls=0,known_substeps=0,unknown_substeps=0,optimizer_updates=0,GPU_calls=0,model_calls=0)
    result=dict(kind='original-playground-physical-alignment.v1',started_at_utc=utc(),passed=False,protocol=PROTOCOL,protocol_sha256=PROTOCOL_SHA256,
        original_source_hashes=PROTOCOL['source_sha256'],validation_source_sha256=sha(__file__),wrapper_source_sha256=sha(ROOT/'tdyssey/native_rgb_original_physics.py'),identities=[],parity=[],separation=[],cost=cost)
    try:
        for letter in'TLOCXB':
            env=make_original_env(source,letter);reference=None
            try:
                env.reset_scene(center=(256,256),angle=0,pusher=(120,256));reference,_=make_reference(source,letter,env)
                actual=physical(env._base);expected=physical(reference);matched=actual==expected
                if not matched or actual['agent']['body_type']!=1 or actual['object']['body_type']!=0 or actual['space']['damping']!=0 or actual['space']['iterations']!=10:raise AssertionError('Actual source parameter/material identity differs')
                result['identities'].append(dict(letter=letter,passed=True,parameters_materials_geometry_exact=matched,actual=actual,expected=expected))
            finally:
                if reference is not None:reference.close()
                env.close()
        for letter in('T','C'):result['parity'].append(parity(source,letter,output,cost))
        for near_wall in(False,True):
            summary,_=separation(source,near_wall,output,cost);result['separation'].append(summary)
        curves(output,result['separation'])
        result['passed']=all(v['passed']for key in('identities','parity','separation')for v in result[key])and cost['completed_controls']==80 and cost['known_substeps']==800
        result['interpretation']='Wrapper matches actual pinned source step and official shape construction. Post-separation behavior is observed under original damping0 with real continuing controls, not hand-cleared velocities; this is distinct from historical0.1s half-life training physics.'
    except BaseException as exc:result['failure']=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        cost['wall_seconds']=time.perf_counter()-began;cost['Torch_imported']='torch'in sys.modules;result['finished_at_utc']=utc();write(output/'summary.json',result)
        write(output/'inventory.json',dict(files=[dict(file=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(output.rglob('*'))if p.is_file()and p.name!='inventory.json']))
    print(json.dumps(dict(passed=result['passed'],cost=cost,summary=str(output/'summary.json'),failure=result.get('failure')),ensure_ascii=False));return 0 if result['passed']else 2

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=ROOT.parent/'Tdyssey/third_party/diffusion_policy');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required for the single bounded CPU physics pass')
    raise SystemExit(run(a.output,a.source))
