from pathlib import Path
import gzip,json
d=Path(__file__).resolve().parent
rows=[]
def add(name,data,warmup=True):
 wc=data.get('warmup_controls',3 if warmup else 0)
 sub=data.get('known_microsteps',data.get('known_substeps',0))+data.get('warmup_microsteps',0)
 rows.append(dict(name=name,attempted=data['attempted']+wc,completed=data['completed']+wc,warmup_controls=wc,known_substeps=sub,wall_seconds=data['wall_seconds'],wall_scope='factory/reset/warmup/physics/RGB; excludes evidence compression and video encoding'))
rows.append(dict(name='original-first-writer-failure',attempted=40,completed=40,warmup_controls=3,known_substeps=400,wall_seconds=4.0499113,wall_scope='whole tool including import/writer; not comparable with internal run timer'))
add('candidate1-adaptive-failure',json.load(gzip.open(d/'B-candidate-v1-failed.json.gz','rt')))
add('candidate2-B',json.loads((d/'B-v2-summary.json').read_text())[0])
add('candidate3-B',json.loads((d/'B-probe-summary.json').read_text())[0])
for row in json.loads((d/'v4-fourcase/summary.json').read_text()):add('candidate4-'+row['name'],row,False)
for row in json.loads((d/'final-B-paired-summary.json').read_text()):add(row['name'],row)
add('candidate5-C-corner_tr',json.loads((d/'C-corner_tr-v5-probe.json').read_text()),False)
add('candidate5-B-final',json.loads((d/'B-v5-replay-final.json').read_text()))
if(d/'v6-sevencase/summary.json').exists():
 for row in json.loads((d/'v6-sevencase/summary.json').read_text()):add('candidate6-'+row['name'],row,False)
if(d/'B-v6-replay-final.json').exists():add('candidate6-B-final',json.loads((d/'B-v6-replay-final.json').read_text()))
if(d/'B-corner-paired-final/summary.json').exists():
 for row in json.loads((d/'B-corner-paired-final/summary.json').read_text()):
  rows.append(dict(name=row['name']+'-original-final',attempted=row['original_attempted'],completed=row['original_completed'],warmup_controls=0,known_substeps=row['original_native_substeps'],wall_seconds=row['wall_seconds'],wall_scope='original factory/reset/physics/RGB; v6 counterpart reused without physics'))
result=dict(owner='contact_physics_review; excludes root regression/collector/UI',attempts=rows,
 attempted_control_slots_including_warmup=sum(r['attempted']for r in rows),completed_controls_including_warmup=sum(r['completed']for r in rows),warmup_controls=sum(r['warmup_controls']for r in rows),known_substeps=sum(r['known_substeps']for r in rows),
 model_calls=0,optimizer_updates=0,
 measured_environment_run_wall_seconds=sum(r['wall_seconds']for r in rows if r['name']!='original-first-writer-failure'),
 failed_original_writer_tool_wall_seconds=4.0499113,
 pure_cpu_checks=[json.loads((d/f).read_text())for f in ['focused-CPU-v4-checks.json','focused-CPU-v5-checks.json','focused-CPU-checks.json']],
 final_protocol_sha256=json.loads((d/'focused-CPU-checks.json').read_text())['protocol_sha256'],
 final_source_sha256=json.loads((d/'focused-CPU-checks.json').read_text())['source_sha256'],
 old_baseline_reused_for_v5_video=True,
 root_broad_gate_required_before_collection=True)
gate_path=d.parent/'physics-regression/v4/summary.json'
if gate_path.exists():
 gate=json.loads(gate_path.read_text())
 result['root_broad_gate_completed']=bool(gate['completed']and gate['physics_gate_passed'])
 result['root_broad_gate_summary']=str(gate_path)
 result['root_broad_gate_cost_excluded_from_child_totals']=True
(d/'COST-SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
records=json.load(gzip.open(d/'B-v5-replay-final-substeps.json.gz','rt'))
controller=dict(initial_last_micro_ticks=8,control_boundary_last_micro_ticks=[round(c['substeps'][-1]['dt']/.000125)for c in records['records']],solver_cache_serialized=False)
(d/'B-v5-controller-state.json').write_text(json.dumps(controller,indent=2)+'\n')
if(d/'B-v6-replay-final-substeps.json.gz').exists():
 records=json.load(gzip.open(d/'B-v6-replay-final-substeps.json.gz','rt'))
 controller=dict(initial_last_micro_ticks=1,control_boundary_last_micro_ticks=[round(c['substeps'][-1]['dt']/.0005)for c in records['records']],solver_cache_serialized=False)
 (d/'B-v6-controller-state.json').write_text(json.dumps(controller,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in['attempts','pure_cpu_checks']}))
