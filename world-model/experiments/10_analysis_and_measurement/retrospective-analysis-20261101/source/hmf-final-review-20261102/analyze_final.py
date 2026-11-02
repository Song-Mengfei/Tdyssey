"""Saved-only HMF analysis. No project imports, torch, simulator, model or network.
All mutations are confined to this independent report directory.
"""
from pathlib import Path
from collections import Counter
import json,hashlib,datetime,os
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage

O=Path(__file__).resolve().parent;P=O.parent;R=P.parents[1];H=R/'artifacts/high-motion-fit-20261101'
sources={}
def load(p):
 p=Path(p);b=p.read_bytes();sources[str(p.resolve())]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 return json.loads(b.decode('utf-8-sig'))
def save(n,j):(O/n).write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
selection=load(H/'selection/index.json');panel={x['sample_id']for x in selection['fixed_train_panel12']}
labels=['0','250','500','1000','2000','final'];steps=[0,250,500,1000,2000,3000]
payload={};curves=[];full=[];casewise=[];trajectories=[]
keys=['raw_mask_iou','object_position_error_image_px','raw_contour_error_image_px','pusher_position_error_image_px']
def stats(rows):
 out={}
 for key in keys:
  v=np.array([x['metrics'][key]for x in rows if x['metrics'].get(key)is not None])
  out[key]=dict(n=len(rows),available=len(v),mean=float(v.mean())if len(v)else None,median=float(np.median(v))if len(v)else None,P10=float(np.quantile(v,.1))if len(v)else None,P90=float(np.quantile(v,.9))if len(v)else None,min=float(v.min())if len(v)else None,max=float(v.max())if len(v)else None)
 return out
for label,step in zip(labels,steps):
 j=load(H/f'evaluation/{label}/primary/metrics.json')
 if label in ['0','final']:payload[label]=j
 rows=j['rows'];pr=[x for x in rows if x['sample_id']in panel]
 for letter in 'OCB':
  for mode in ['real_history','autoregressive']:
   lr=[x for x in pr if x['letter']==letter and x['input_mode']==mode]
   for horizon in [1,4,8,16]:
    sr=[x for x in lr if x['horizon']==horizon]
    curves.append(dict(label=label,updates=step,letter=letter,mode=mode,subset='fixed12_horizon',horizon=horizon,metrics=stats(sr)))
   if mode=='real_history':
    sr=[x for x in lr if x['strong_H1']]
    curves.append(dict(label=label,updates=step,letter=letter,mode=mode,subset='fixed12_strong_H1',horizon=None,metrics=stats(sr)))
 for row in pr:
  trajectories.append({k:row[k]for k in ['sample_id','letter','horizon','input_mode','strong_H1']}|dict(label=label,updates=step,metrics={k:row['metrics'].get(k)for k in keys}))
 if label in ['0','final']:
  for role in sorted({x['selection_role']for x in rows}):
   for letter in 'OCB':
    for mode in ['real_history','autoregressive']:
     rr=[x for x in rows if x['selection_role']==role and x['letter']==letter and x['input_mode']==mode]
     for h in [1,4,8,16]:
      sub=[x for x in rr if x['horizon']==h]
      full.append(dict(label=label,role=role,letter=letter,mode=mode,horizon=h,subset='endpoint',metrics=stats(sub)))
     if mode=='real_history':
      full.append(dict(label=label,role=role,letter=letter,mode=mode,horizon=None,subset='strong_H1',metrics=stats([x for x in rr if x['strong_H1']])))
  for x in rows:
   if x['horizon']in [1,4,8,16]:casewise.append({k:x[k]for k in ['sample_id','family_id','letter','selection_role','input_mode','horizon','strong_H1']}|dict(label=label,metrics={k:x['metrics'].get(k)for k in keys}))

# Paired changes rather than comparing full48 to fixed12.
changes=[]
lookup={(x['label'],x['sample_id'],x['input_mode'],x['horizon']):x for x in trajectories}
for before,after in zip(labels,labels[1:]):
 for letter in 'OCB':
  for mode in ['real_history','autoregressive']:
   for horizon in [4,8,16]:
    pairs=[(x,lookup[(after,x['sample_id'],mode,horizon)])for x in trajectories if x['label']==before and x['letter']==letter and x['input_mode']==mode and x['horizon']==horizon]
    for metric in keys[:2]:
     delta=np.array([b['metrics'][metric]-a['metrics'][metric]for a,b in pairs]);good=delta>0 if metric=='raw_mask_iou'else delta<0
     changes.append(dict(before=before,after=after,letter=letter,mode=mode,horizon=horizon,metric=metric,mean_delta=float(delta.mean()),improved=int(good.sum()),ties=int((delta==0).sum()),regressed=int(((~good)&(delta!=0)).sum()),n=len(delta)))
losses=[json.loads(s)for s in(H/'training/losses.jsonl').read_text().splitlines()if s.strip()]
lossbins=[]
for start in range(0,3000,250):
 rr=losses[start:start+250];v=np.array([x['loss']for x in rr]);lossbins.append(dict(attempts=[start,start+len(rr)-1],mean=float(v.mean()),median=float(np.median(v)),P90=float(np.quantile(v,.9)),gradient_norm_mean=float(np.mean([x['gradient_norm']for x in rr]))))
summary=load(H/'report/summary.json');audit=load(H/'postrun-fit-diagnostic-audit.json');result=load(H/'training/result.json')
save('CURVES.json',dict(fixed_panel_ids=sorted(panel),curves=curves,paired_changes=changes,loss_bins=lossbins,loss_caveat='Stochastic sampled training losses; different target/noise draws. Not fixed-noise validation loss.',midpoint_validation_available=False,midpoint_full48_available=False))
save('FULL_ENDPOINTS.json',full);save('CASE_ENDPOINTS.json',casewise);save('PANEL_TRACES.json',trajectories)
save('FINAL_STATE.json',dict(read_at_singapore=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),run_id=summary['run_id'],progress=result['progress'],stop_reason=result['stop_reason'],final=result['final'],fit_support=summary['fit_support'],timings=result['timings'],new_analysis_model_calls=0,new_analysis_physics_steps=0))

# Reuse raw RGB for end-frame sheets of every one of the 75 cases. Clip only for
# this display. Original arrays are neither changed nor fed to any model.
maps={label:{(x['sample_id'],x['input_mode']):x for x in d['raw_files']}for label,d in payload.items()}
font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14);small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',12)
def raw(rec):
 p=Path(rec['path']);b=p.read_bytes();digest=hashlib.sha256(b).hexdigest();assert digest==rec['sha256'];sources[str(p)]=dict(bytes=len(b),sha256=digest)
 with np.load(p,allow_pickle=False)as z:return z['raw']
def frame(a):return Image.fromarray(np.rint(np.clip(a,0,1).transpose(1,2,0)*255).astype('uint8')).resize((144,144),Image.Resampling.NEAREST)
sheets=[];rawcheck=[]
for role in ['train','seen_val','original_failure']:
 for letter in 'OCB':
  records=[x for x in payload['final']['raw_files']if x['input_mode']=='autoregressive'and x['selection_role']==role and x['letter']==letter]
  if not records:continue
  sheet=Image.new('RGB',(1160,48+int(np.ceil(len(records)/2))*180),'white');draw=ImageDraw.Draw(sheet)
  draw.text((8,5),f'{role}/{letter} | H16=1.6s | each block: GT | E2 AR | final3000 AR | final3000 RH',font=font,fill='black')
  for i,rec in enumerate(records):
   sid=rec['sample_id'];x=(i%2)*580;y=36+(i//2)*180
   base=raw(maps['0'][(sid,'autoregressive')]);ar=raw(rec);rh=raw(maps['final'][(sid,'real_history')])
   source=Path(rec['raw_path'])
   with np.load(source,allow_pickle=False)as z:gt=z['rgb'][rec['anchor']+16].astype('float32')/255
   draw.text((x+4,y),sid.replace('HMF-',''),font=small,fill='black')
   for c,im in enumerate([gt,base[-1],ar[-1],rh[-1]]):sheet.paste(frame(im),(x+c*144,y+22))
   m=next(x for x in payload['final']['rows']if x['sample_id']==sid and x['input_mode']=='autoregressive'and x['horizon']==16)['metrics']
   rawcheck.append(dict(sample_id=sid,role=role,letter=letter,final_AR16_IoU=m['raw_mask_iou'],final_AR16_position=m['object_position_error_image_px'],raw=np.asarray(ar[-1]).shape))
  p=O/f'all-{role}-{letter}-H16.png';sheet.save(p);sheets.append(dict(role=role,letter=letter,cases=len(records),path=str(p)))

# A separate numerical illustration from actual saved rows.
os.environ['MPLCONFIGDIR']=str(P/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for col,letter in enumerate('OCB'):
 for mode,style in [('real_history','--'),('autoregressive','-')]:
  for h,color in [(4,'#a6b6bf'),(8,'#4f8b99'),(16,'#c25336')]:
   rr=[x for x in curves if x['letter']==letter and x['mode']==mode and x['subset']=='fixed12_horizon'and x['horizon']==h]
   axes[0,col].plot(steps,[x['metrics']['raw_mask_iou']['mean']for x in rr],style,color=color,marker='o',label=f'{mode[:2]} H{h}')
  rr=[x for x in curves if x['letter']==letter and x['mode']==mode and x['subset']=='fixed12_horizon'and x['horizon']==16]
  axes[1,col].plot(steps,[x['metrics']['object_position_error_image_px']['mean']for x in rr],style,marker='o',label=f'{mode} H16')
 axes[0,col].set(title=f'{letter}: same four trained cases',ylabel='Gray IoU',xlabel='Successful updates',ylim=(0,1.02));axes[0,col].legend(fontsize=7,ncols=2)
 axes[1,col].set(xlabel='Successful updates',ylabel='Visible centroid error (image96 px)');axes[1,col].legend(fontsize=7)
 for a in axes[:,col]:a.grid(alpha=.2)
fig.suptitle('Matched fixed12 checkpoint curves (not full48 / not validation)');fig.savefig(O/'independent-checkpoint-curves.png',dpi=140);plt.close(fig)
save('SAVED_ONLY_AUDIT.json',dict(sources=sources,sheets=sheets,cases_reviewed=rawcheck,method='Reaggregate every primary saved metric row, fixed12 paired checkpoint curves, display all75 H16 cases from raw. No checkpoint deserialization/model/physics.',actions=dict(training=0,inference=0,physics=0,packages=0)))
print('ROLES',dict(Counter(x['selection_role']for x in payload['final']['rows'])))
for letter in 'OCB':
 print('LETTER',letter)
 for mode in ['real_history','autoregressive']:
  print(mode,'H16 IoU',[round(x['metrics']['raw_mask_iou']['mean'],6)for x in curves if x['letter']==letter and x['mode']==mode and x['subset']=='fixed12_horizon'and x['horizon']==16])
  print(mode,'H16 pos',[round(x['metrics']['object_position_error_image_px']['mean'],6)for x in curves if x['letter']==letter and x['mode']==mode and x['subset']=='fixed12_horizon'and x['horizon']==16])
 print('strong RH IoU',[round(x['metrics']['raw_mask_iou']['mean'],6)for x in curves if x['letter']==letter and x['subset']=='fixed12_strong_H1'])
print('LOSS_BINS',lossbins)
print('SHEETS',sheets)
