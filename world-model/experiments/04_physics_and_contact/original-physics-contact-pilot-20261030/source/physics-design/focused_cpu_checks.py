from pathlib import Path
import hashlib,json,sys,time
import numpy as np
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo))
from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,project_halfplanes,PROTOCOL,PROTOCOL_SHA256
started=time.perf_counter();checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
v,ok=project_halfplanes([2,3],[([1,0],0),([-1,0],0)])
check('wall clips normal while preserving tangent',ok and np.allclose(v,[0,3]))
v,ok=project_halfplanes([0,0],[([1,0],1)])
check('zero PD speed cannot be amplified to satisfy infeasible contact',not ok and np.allclose(v,[0,0]))
v,ok=project_halfplanes([1,0],[([0,1],.8)])
check('active contact and speed circle meet without amplification',ok and np.allclose(v,[.6,.8]))
v,ok=project_halfplanes([-1,-1],[([1,0],0),([0,1],0)])
check('two wall normals jointly stop inward command',ok and np.allclose(v,[0,0]))
v,ok=project_halfplanes([2,0],[([1,0],1),([-1,0],-3)])
check('feasible request remains unchanged',ok and np.allclose(v,[2,0]))
env=make_contact_safe_env(repo.parent/'Tdyssey/third_party/diffusion_policy','B')
env.reset_scene(center=(256,256),angle=0,pusher=(184.66666666666666,256))
identity=env.initial_identity();snapshot=env.capture_state()
check('snapshot execution identity differs from base-body-only identity',snapshot.physics_hash==env._physics_hash() and snapshot.physics_hash!=env._base._physics_hash())
check('fixed2000Hz integration identity is truthful',identity['protocol_sha256']==PROTOCOL_SHA256 and identity['protocol']['integration_tick_seconds']==.0005 and identity['physics']['substep_dt']==.0005 and identity['physics']['substeps_per_control']==200 and env.space.iterations==50)
check('immutable boundaries remain kinematic and zero damping',env.agent.body_type==1 and env.block.body_type==0 and env.space.damping==0 and env.k_p==100 and env.k_v==20)
check('new snapshot has unchanged actual body state',np.allclose(snapshot.object_body.pose.position,env.block.position) and np.allclose(snapshot.pusher_body.pose.position,env.agent.position))
import gzip
d=Path(__file__).resolve().parent
rows=json.load(gzip.open(d/'v6-sevencase/C-corner_tr-substeps.json.gz','rt'))
check('saved actual dt are fixed2000Hz with true control duration',all(len(c['substeps'])==200 and all(r['dt']==.0005 for r in c['substeps'])and abs(sum(r['dt']for r in c['substeps'])-.1)<1e-10 for c in rows))
result=dict(passed=True,checks=checks,source_sha256=hashlib.sha256((repo/'tdyssey/native_rgb_contact_safe_physics.py').read_bytes()).hexdigest(),protocol_sha256=PROTOCOL_SHA256,physical_controls=0,physical_substeps=0,model_calls=0,optimizer_updates=0,wall_seconds=time.perf_counter()-started)
old=d/'focused-CPU-checks.json'
if old.exists()and not(d/'focused-CPU-v4-checks.json').exists():(d/'focused-CPU-v4-checks.json').write_bytes(old.read_bytes())
old.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
