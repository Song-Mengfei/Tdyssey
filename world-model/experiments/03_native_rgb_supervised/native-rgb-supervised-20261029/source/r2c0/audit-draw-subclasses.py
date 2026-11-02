"""Supplement frozen draw exposure audit with letter-scoped actual arcs/rest tags."""
import sys,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r2b_run import EXPERIMENT,read,sha,atomic_json,utc
OUT=EXPERIMENT/'r2c0';D=OUT/'data';rows=read(D/'pools.json')['windows']
meta={e['trajectory_id']:read(EXPERIMENT/'r2b'/e['metadata_file']) for e in read(D/'manifest.json')['entries'] if e['split']=='train'}
classes=['exact_stationary','near_static_with_AO','near_static_without_AO','small_motion','release','actual_AO_loss','wall_contact_stall','recorded_miss_or_recovery']
def tags(r):
 if r['moving']:return []
 x,a=r['translation_image96_px'],r['abs_rotation_deg'];near=x<.1 and a<.5
 rec=meta[r['trajectory_id']]['control_records'];t=r['anchor']
 return [k for k,flag in [('exact_stationary',x==0 and a==0),('near_static_with_AO',near and r['agent_object_contact']),('near_static_without_AO',near and not r['agent_object_contact']),('small_motion',not near),('release','withdraw' in r['planned_phases']),('actual_AO_loss',not r['agent_object_contact'] and rec[t-1]['contact']>0),('wall_contact_stall',r['boundary_process']=='observed_wall_contact_stall'),('recorded_miss_or_recovery',any(q in r['phase'] for q in ['miss','recovery','unconfirmed']))] if flag]
def summarize(indices):
 selected=[rows[i] for i in indices];by={}
 for letter in 'TLOCXB':
  s=[r for r in selected if r['letter']==letter];arcs=sorted({a for r in s for a in r['actual_contact_arc_ids']},key=int)
  by[letter]=dict(draws=len(s),independent_families=len({r['family_id'] for r in s}),actual_arc_support={a:dict(exposures=sum(a in r['actual_contact_arc_ids'] for r in s),unique_windows=len({r['window_id'] for r in s if a in r['actual_contact_arc_ids']}),families=sorted({r['family_id'] for r in s if a in r['actual_contact_arc_ids']})) for a in arcs},signed_lever_support={q:dict(exposures=sum(q in r['signed_lever_signs'] for r in s),unique_windows=len({r['window_id'] for r in s if q in r['signed_lever_signs']}),independent_families=len({r['family_id'] for r in s if q in r['signed_lever_signs']})) for q in ['positive','negative']},other_subclasses={k:dict(exposures=sum(k in tags(r) for r in s),unique_windows=len({r['window_id'] for r in s if k in tags(r)}),families=len({r['family_id'] for r in s if k in tags(r)})) for k in classes})
 return dict(by_letter=by,other_subclass_exposures=dict(Counter(k for r in selected for k in tags(r))),missing_other_subclasses=[k for k in classes if not any(k in tags(r) for r in selected)])
result=dict(kind='R2C0-supplemental-actual-support.v1',at_utc=utc(),pool_sha256=sha(D/'pools.json'),draw480_sha256=sha(D/'draws480.json'),draw4800_sha256=sha(D/'draws4800.json'),
 rules=dict(moving='unchanged H1 translation>=2image96px OR absangle>=5deg',exact_stationary='exact zero translation AND exact zero angle; no rounding to manufacture static',near_static='translation<0.1 image96px AND absangle<0.5deg (audit only, no sampling-policy change)',tags_overlap=True,actual_AO_loss='current control0AO substeps and previouscontrol>0AO substeps',actual_arc='native measured bins scoped by letter; not planned arc or rigid coverage claim'),source=summarize(range(len(rows))),first480=summarize(read(D/'draws480.json')['indices']),proposed4800=summarize(read(D/'draws4800.json')['indices']),new_physics_or_models=0)
atomic_json(OUT/'supplemental-draw-support.json',result)
print(json.dumps({k:{'rest_tags':result[k]['other_subclass_exposures'],'missing':result[k]['missing_other_subclasses']} for k in ['source','first480','proposed4800']}))
