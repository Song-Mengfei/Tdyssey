"""Render recorded states only. No stepping, model, GPU or pixel repair."""
from pathlib import Path
import csv,gzip,hashlib,json,math,sys,time
import numpy as np
import cv2
from shapely.geometry import Point,Polygon
repo=Path(__file__).resolve().parents[4];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_env import make_env,shape_geometry
from tdyssey.envs.official import official_shape
d=Path(__file__).resolve().parent;stage=d.parents[1]
newpath=stage/'physics-regression/v4/B-corner_tl-candidate.npz'
oldpath=d.parent/'B-corner-paired-final/B-corner_tl-original.npz'
oldraw=d.parent/'B-corner-paired-final/B-corner_tl-original-substeps.json.gz'
with np.load(newpath)as z:
 newrows=json.loads(z['substep_metadata_json'].item());newpost=z['substep_post'].copy();spec=json.loads(z['spec_json'].item());newpre=z['substep_pre'].copy()
oldrecords=json.load(gzip.open(oldraw,'rt'));oldrows=[dict(control=i+1,**r)for i,c in enumerate(oldrecords)for r in c['substeps']]
oldpost=np.asarray([r['post_state']for r in oldrows])
geom=shape_geometry('B');holes=[Polygon(r)for r in geom.interiors]
renderer=make_env(repo.parent/'Tdyssey/third_party/diffusion_policy',letter='B',controller='upstream-kinematic',render_size=96)
renderer.reset_scene(center=spec['center'],angle=spec['angle'],pusher=spec['pusher'])
def no_step(*args,**kwargs):raise RuntimeError('Rendering saved state only: physics step disabled')
renderer.space.step=no_step
renders=0
def render(state):
 global renders
 renderer.set_render_state(np.asarray(state));renders+=1
 return np.moveaxis(renderer.render_clean(),0,-1)
def events(row,kind):
 ev=[e for e in row['contacts']if e['phase']=='post_solve'and e['kind']==kind]
 return len(ev),sum(float(np.linalg.norm(e['total_impulse']))for e in ev)
def signed_geometry(s):
 a=s[4];delta=s[:2]-s[2:4];p=Point(math.cos(a)*delta[0]+math.sin(a)*delta[1],-math.sin(a)*delta[0]+math.cos(a)*delta[1])
 inside=geom.contains(p);distance=-geom.boundary.distance(p)if inside else geom.distance(p)
 region='solid'if inside else'hole'if any(h.contains(p)for h in holes)else'outside'
 return float(distance-15),region
def text(canvas,value,x,y,color=(25,25,25),scale=.48):cv2.putText(canvas,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
metrics=[];main=np.full((960,1400,3),248,np.uint8)
text(main,'ACTUAL SAVED CONTACT FRAMES: B top-left corner',16,30,scale=.8)
text(main,'Same initial state/requested prefix; different trajectories. Recorded-state renderer only. No new physics/model.',16,55,scale=.55)
for j,t in enumerate([.32,.33]):
 y=80+j*435
 for branch,(name,rows,post,dt,path)in enumerate([('BEFORE ORIGINAL',oldrows,oldpost,.01,oldpath),('AFTER V6',newrows,newpost,.0005,newpath)]):
  index=round(t/dt)-1;row=rows[index];state=post[index];frame=render(state)
  a,aj=events(row,'agent_object');w,wj=events(row,'object_wall');gap,region=signed_geometry(state)
  previous=post[round((t-.1)/dt)-1];move=float(np.linalg.norm(state[2:4]-previous[2:4]));rot=abs(float(state[4]-previous[4]))*180/np.pi
  metrics.append(dict(case='B-corner_tl',branch=name,physical_time_seconds=t,flat_substep_index=index,control=row['control'],substep_index=row['index'],substep_dt=dt,
   AO_post_events=a,AO_impulse_norm_sum=aj,OW_post_events=w,OW_impulse_norm_sum=wj,signed_agent_object_gap_world=gap,agent_object_overlap_world=max(0,-gap),object_wall_overlap_world=row['penetrations'][0],
   pusher_center_region=region,H1_centroid_displacement_world=move,H1_centroid_displacement_image=move*96/512,H1_angle_change_deg=rot,source=str(path)))
  if branch:assert a and aj>0 and w and wj>0
  x=15+branch*700
  text(main,f'{name}: t={t:.3f}s, control{row["control"]}, micro{row["index"]+1}, flat row{index}',x,y+20,scale=.53)
  text(main,'whole96 x3',x,y+43);text(main,'same crop x[0:36],y[0:36] x8',x+310,y+43)
  main[y+55:y+343,x:x+288]=cv2.resize(frame,(288,288),interpolation=cv2.INTER_NEAREST)
  main[y+55:y+343,x+310:x+598]=cv2.resize(frame[0:36,0:36],(288,288),interpolation=cv2.INTER_NEAREST)
  c=(20,110,35)if branch else(170,35,35)
  text(main,f'AO post={a}, J={aj:.3f}; OW post={w}, J={wj:.3f}',x,y+368,c,.52)
  text(main,f'AO signed gap={gap:.6f}w; OW depth={row["penetrations"][0]:.6f}w; center={region}',x,y+390,c,.46)
  text(main,f'H1(0.1s) CoG move {move:.3f}w / {move*96/512:.3f}imagepx; angle {rot:.2f}deg',x,y+412,scale=.46)
text(main,'Contact is proved by THIS post-solve event/positive impulse, not by a whole-control contact total.',16,947,scale=.58)
cv2.imwrite(str(d/'B-TL-exact-contact-proof.png'),cv2.cvtColor(main,cv2.COLOR_RGB2BGR))

# All0.01-grid frames through control4, including actual loss at .34; do not hide it.
grid=[]
for t in np.arange(.31,.401,.01):
 i=round(float(t)/.0005)-1;r=newrows[i];a,aj=events(r,'agent_object');w,wj=events(r,'object_wall');gap,region=signed_geometry(newpost[i])
 grid.append(dict(time_seconds=round(float(t),3),flat_index=i,control=r['control'],micro_index=r['index'],AO_post_events=a,AO_J=aj,OW_post_events=w,OW_J=wj,AO_signed_gap_world=gap,OW_depth_world=r['penetrations'][0],center_region=region))
with(d/'TL-control4-all-0_01-grid.csv').open('w',newline='',encoding='utf-8')as f:
 writer=csv.DictWriter(f,fieldnames=list(grid[0]));writer.writeheader();writer.writerows(grid)

freepath=stage/'physics-regression/v4/B-free_left-candidate.npz'
with np.load(freepath)as z:
 freerows=json.loads(z['substep_metadata_json'].item());freepost=z['substep_post'].copy();freergb=z['rgb'].copy()
free=np.full((760,1220,3),248,np.uint8)
text(free,'ACTUAL B FREE-CONTACT MOTION: fixed coordinates, saved v6 states',15,30,scale=.72)
text(free,'Whole96 x3 above; SAME crop x[30:76],y[30:66] x6 below. No alignment, interpolation or simulation.',15,55,scale=.52)
free_summary=[]
for j,t in enumerate([.30,.40,.50,.60]):
 i=round(t/.0005)-1;r=freerows[i];s=freepost[i];frame=render(s);a,aj=events(r,'agent_object');w,wj=events(r,'object_wall');gap,region=signed_geometry(s)
 assert a and aj>0 and not w
 assert np.array_equal(frame,np.moveaxis(freergb[r['control']],0,-1)), 'Saved end-control RGB differs from recorded-state renderer'
 prev=freepost[round((t-.1)/.0005)-1];move=float(np.linalg.norm(s[2:4]-prev[2:4]));rot=abs(float(s[4]-prev[4]))*180/np.pi
 x=10+j*305;text(free,f't={t:.2f}s c{r["control"]} row{i}',x,85,scale=.54)
 free[100:388,x:x+288]=cv2.resize(frame,(288,288),interpolation=cv2.INTER_NEAREST)
 free[420:636,x:x+276]=cv2.resize(frame[30:66,30:76],(276,216),interpolation=cv2.INTER_NEAREST)
 text(free,f'AO J={aj:.2f}, OW=none',x,665,(20,110,35),.49)
 text(free,f'gap={gap:.5f}w, {region}',x,690,scale=.48)
 text(free,f'H1 move {move:.2f}w/{move*96/512:.2f}px',x,715,scale=.46)
 free_summary.append(dict(time_seconds=t,flat_substep_index=i,control=r['control'],micro_index=r['index'],AO_impulse=aj,OW_events=w,signed_gap_world=gap,H1_centroid_world=move,H1_centroid_image=move*96/512,H1_rotation_deg=rot))
span=[r for i,r in enumerate(freerows)if .30<(i+1)*.0005<=.60+1e-12]
span_ao=sum(r['dt']for r in span if events(r,'agent_object')[0]and events(r,'agent_object')[1]>0)
text(free,f'Actual (.30,.60] span: positive-AO {span_ao:.4f}s / .3000s; all attempted microsteps retained.',15,750,scale=.56)
cv2.imwrite(str(d/'B-free-sustained-contact-motion.png'),cv2.cvtColor(free,cv2.COLOR_RGB2BGR))

result=dict(new_physics_controls=0,new_physics_substeps=0,new_model_calls=0,optimizer_updates=0,rendered_saved_states=renders,
 source_geometry_hash=official_shape('B').geometry_hash,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[newpath,oldpath,oldraw,freepath]},
 selected_exact_contact_frames=metrics,control4_grid=grid,free_contact_frames=free_summary,free_positive_AO_span_seconds=span_ao,free_span_seconds=.3,
 limitation='Selected exact contact examples, not a rate estimate. TL .34s has no AO although control4 aggregate has81.5ms. TR control4 and old prefix31/32 do not establish continued AO contact.',
 old_prefix_figures_status='Historical trajectory-separation images; B-v6 control31/32 have no contact and cannot be contact proof.')
(d/'exact-frame-proof.json').write_text(json.dumps(result,indent=2)+'\n')
with(d/'selected-frame-metrics.csv').open('w',newline='',encoding='utf-8')as f:
 writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
prefix=json.load(gzip.open(d.parent/'B-v6-replay-final-substeps.json.gz','rt'))['records']
warning=[]
for control in[31,32]:
 row=prefix[control-1]['substeps'][-1];gap,region=signed_geometry(np.asarray(row['post_state']))
 warning.append(dict(control=control,physical_time_seconds=control/10,whole_control_AO_seconds=prefix[control-1]['agent_object_contact_seconds'],end_frame_AO_events=events(row,'agent_object')[0],end_frame_AO_impulse=events(row,'agent_object')[1],end_frame_AO_signed_gap_world=gap,center_region=region))
result['historical_prefix_warning']=warning
result['free_saved_end_RGB_bitwise_matches_recorded_renderer']=True
result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),repo/'tdyssey/native_rgb_env.py',repo/'tdyssey/native_rgb_contact_safe_physics.py']}
result['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in[d/'B-TL-exact-contact-proof.png',d/'B-free-sustained-contact-motion.png',d/'selected-frame-metrics.csv',d/'TL-control4-all-0_01-grid.csv']}
(d/'exact-frame-proof.json').write_text(json.dumps(result,indent=2)+'\n')
(d/'historical-prefix-NOT-CONTACT-PROOF.json').write_text(json.dumps(dict(warning='Trajectory separation only; original prefix control31/32 are not contact evidence.',frames=warning,new_physics=0),indent=2)+'\n')
old_image=d.parent/'B-v6-before-after-control32.png'
old=cv2.imread(str(old_image));annotated=np.full((old.shape[0]+90,max(old.shape[1],1000),3),248,np.uint8)
annotated[90:90+old.shape[0],0:old.shape[1]]=old
cv2.putText(annotated,'HISTORICAL TRAJECTORY SEPARATION ONLY',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,190),2,cv2.LINE_AA)
cv2.putText(annotated,'V6 control31/32: NO AO CONTACT. This image cannot prove contact repair.',(15,63),cv2.FONT_HERSHEY_SIMPLEX,.57,(20,20,190),1,cv2.LINE_AA)
cv2.imwrite(str(d/'historical-prefix31-32-NO-CONTACT.png'),annotated)
print(json.dumps({'rendered_saved_states':renders,'new_physics':0,'selected_after':metrics[1::2],'free_AO_span':span_ao}))
