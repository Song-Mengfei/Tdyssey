from pathlib import Path
import json,math,gzip
import numpy as np
import cv2
d=Path(__file__).resolve().parent
v1=json.load(gzip.open(d/'B-candidate-v1-failed.json.gz','rt'))
v2=json.loads((d/'B-v2-summary.json').read_text())[0]
v3=json.loads((d/'B-probe-summary.json').read_text())[0]
v4=json.loads((d/'v4-fourcase/summary.json').read_text())
final=json.loads((d/'final-B-paired-summary.json').read_text())
cost={
 'owner':'contact_physics_review; excludes root regression/collector/UI',
 'attempted_control_slots_including_warmup':328,'completed_controls_including_warmup':327,
 'partial_controls':1,'warmup_controls':18,'known_substeps':30701,
 'measured_environment_run_wall_seconds':v1['wall_seconds']+v2['wall_seconds']+v3['wall_seconds']+sum(r['wall_seconds']for r in v4+final),
 'failed_original_writer_tool_wall_seconds':4.0499113,
 'failed_original_writer_scope':'40complete controls400substeps; full tool wall includes import and writer, no complete original-state proof',
 'pure_cpu_projection_metadata_wall_seconds':json.loads((d/'focused-CPU-checks.json').read_text())['wall_seconds'],
 'measured_wall_scope':'factory/reset/physics/render; excludes output compression, movie encoding and most process imports',
 'model_calls':0,'optimizer_updates':0,
 'v1_failure':'adaptive tiny remainder dt warm-start numerical overflow in control29',
 'v3_broad_gate':'root-owned96case failed28, separately accounted',
 'final_v4_protocol_sha256':final[1]['protocol_sha256'],
 'root_must_finish_broad_gate_before_collection':True,
}
(d/'COST-SUMMARY.json').write_text(json.dumps(cost,indent=2)+'\n')
canvas=np.full((620,800,3),248,np.uint8)
for branch,name in enumerate(['B-original-replay-final','B-v4-replay-final']):
 with np.load(d/f'{name}.npz') as z:
  frames=z['rgb'];states=z['states']
  for j,index in enumerate([31,32]):
   img=cv2.resize(np.moveaxis(frames[index],0,-1),(280,280),interpolation=cv2.INTER_NEAREST)
   x=20+branch*400;y=40+j*285
   canvas[y:y+280,x:x+280]=img
   title=f'{"BEFORE"if branch==0 else"AFTER v4"} | B control{index} t={index/10:.1f}s'
   cv2.putText(canvas,title,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,.52,(25,25,25),1,cv2.LINE_AA)
cv2.imwrite(str(d/'B-before-after-control32.png'),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
print(json.dumps(cost))
