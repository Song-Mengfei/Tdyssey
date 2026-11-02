"""Stage UI integration without changing the running service/static page."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
repo=p.parent.parent
app=repo/'apps/r3b-playground'
old=p/'ui-source-before';new=p/'ui-next'
old.mkdir(exist_ok=True);new.mkdir(exist_ok=True)
def edit(name,replacements):
 original=(app/name).read_bytes()
 if not(old/name).exists():(old/name).write_bytes(original)
 text=original.decode('utf-8').replace('\r\n','\n')
 for before,after,count in replacements:
  actual=text.count(before)
  if actual!=count:raise RuntimeError((name,before[:100],actual,count))
  text=text.replace(before,after)
 (new/name).write_text(text,encoding='utf-8',newline='')
edit('server.py',[
 ("kind='R3B-playground-complete-session.v1'","kind='R3B-playground-public-session.v2'",1),
 ('from tdyssey.native_rgb_original_physics import make_original_env','from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL',1),
 ('newenv=make_original_env(PHYSICS,letter)','newenv=make_contact_safe_env(PHYSICS,letter)',1),
 ("if metadata['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Saved model differs')", "if metadata['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Saved model differs')\n        if metadata['environment_physics_sha256']!=self.physics_sha:raise ValueError('Saved environment differs; current session was preserved')\n        saved_ticks=metadata.get('physical_state',{}).get('controller',{}).get('integration_microstep_ticks')\n        allowed_ticks=self.env.initial_identity()['protocol']['integration_step_tick_choices']\n        if type(saved_ticks)is not int or saved_ticks not in allowed_ticks:raise ValueError('Saved integration-controller state is incomplete; current session was preserved')",1),
 ("identity['physics']['space']['iterations']!=10","identity['physics']['space']['iterations']!=PROTOCOL['solver_iterations']",1),
 ("raise RuntimeError('Expected original kinematic pusher/damping/iterations')","raise RuntimeError('Expected versioned contact-safe kinematic pusher/damping/iterations')",1),
 ("contact_substeps=0,latency_ms=0.","contact_substeps=0,physics_substeps=0,contact_seconds=0.,intervention_ratio=0.,latency_ms=0.",1),
 ("self.message='原始物理环境已就绪；当前模型是在旧物理上训练的跨物理预览。'","self.message='接触约束修复版已就绪：运动学推杆、阻尼0；当前模型仍是旧物理训练的跨物理预览。'",1),
 ('executed_target=a.tolist()','requested_target=a.tolist()',2),
 ("contact_substeps=int(self.info['agent_object_contact_substeps']),latency_ms=inference_ms,","contact_substeps=int(self.info['agent_object_contact_substeps']),physics_substeps=int(self.info['physics_substeps']),contact_seconds=float(self.info['agent_object_contact_seconds']),intervention_ratio=float(self.info['intervention_time_ratio']),max_penetration=max(self.info['penetration']),latency_ms=inference_ms,",1),
 ("environment_physics_sha256=self.physics_sha,sampler=SAMPLER,\n            normalized_actions=","environment_physics_sha256=self.physics_sha,sampler=SAMPLER,\n            action_contract='requested absolute controller target retained; actual endpoints/velocities and unilateral contact interventions in control_infos',\n            normalized_actions=",1),
 ("physics='环境：原始运动学推杆 / 原始阻尼；跨物理预览'","physics='环境：运动学推杆 / 阻尼0 / 接触约束修复版；跨物理预览'",1),
 ("actual_batch_target=self.target,report_available=","actual_batch_target=self.target,command_target=self.target,action_contract='requested target; actual pusher may be constrained by contact',report_available=",1),
 ("for prefix,root in (('/captures/',","for prefix,root in (('/contact-pilot/',REPO/'artifacts/original-physics-contact-pilot-20261030'),('/captures/',",1),
 ("(DIAGNOSTICS/'runs/original-20261030-v1/report/index.html').is_file()", "(REPO/'artifacts/original-physics-contact-pilot-20261030/index.html').is_file()",2),
 ("url='/diagnostics-root/runs/original-20261030-v1/report/index.html'", "url='/contact-pilot/index.html'",1),
 ("('.html','.png','.mp4','.webm','.json','.csv','.npz','.md')", "('.html','.png','.mp4','.webm','.json','.csv','.npz','.md','.gz','.txt')",1),
 ('已恢复保存的身体/控制器和模型历史；接触求解缓存未序列化，后续接触物理并非已验证的精确重放。','已恢复公开身体、控制器和模型历史；隐藏位置偏置速度与碰撞缓存未序列化，不承诺后续接触物理精确重放。',1),
])
edit('capture.py',[
 ('Both complete bodies, controller target/count and recorder boundary state.','Public body properties, controller target/count and recorder boundary state.',1),
 ("schema='r3b-playground-full-control-boundary.v1'", "schema='r3b-playground-public-control-boundary.v2'",1),
 ('Body/controller state is complete; native contact-solver cache is not serialized.', 'Public body/controller state is retained; native contact-solver cache and hidden position-bias velocities are not serialized.',1),
 ('physics_state_complete=True,solver_cache_serialized=False', 'physics_state_complete=False,public_physics_state_complete=True,native_solver_and_bias_state_complete=False,solver_cache_serialized=False',1),
 ("control_step=int(env.control_step),pending_action=", "control_step=int(env.control_step),integration_microstep_ticks=getattr(env,'_last_micro_ticks',None),pending_action=",1),
 ("if snapshot.shape!=env.letter or snapshot.physics_hash!=env._physics_hash():raise ValueError('Saved geometry/physics differs')", "if snapshot.shape!=env.letter or snapshot.physics_hash!=env._physics_hash():raise ValueError('Saved geometry/physics differs')\n    micro_ticks=value['controller'].get('integration_microstep_ticks')\n    allowed_ticks=env.initial_identity()['protocol']['integration_step_tick_choices']\n    if type(micro_ticks)is not int or micro_ticks not in allowed_ticks:raise ValueError('Saved integration-controller state is incomplete')",1),
 ("env.control_step=snapshot.control_step", "env.control_step=snapshot.control_step\n    env._last_micro_ticks=micro_ticks",1),
 ("repo/'apps/r3b-playground']","repo/'apps/r3b-playground',repo/'artifacts/original-physics-contact-pilot-20261030',repo/'tdyssey/native_rgb_contact_safe_physics.py',repo/'tdyssey/original_contact_pilot.py',repo/'tdyssey/original_contact_pilot_report.py']",1),
 ("upper=int(horizon)*(4*3*96*96*4+3*96*96*4+3*96*96)+1024**2","upper=int(horizon)*(4*3*96*96*4+3*96*96*4+3*96*96+4*1024**2)+1024**2",1),
])
# Readable user labels distinguish a requested target from constrained execution.
text=(app/'app.js').read_text(encoding='utf-8')
if not(old/'app.js').exists():(old/'app.js').write_bytes((app/'app.js').read_bytes())
text=text.replace('本批实际执行','本批请求目标').replace('固定执行目标','固定请求目标')
text=text.replace("executed_target:fixed","requested_target:fixed")
text=text.replace('f.executed_target','(f.requested_target||f.executed_target)')
text=text.replace("s.metrics.contact_substeps+' <small>/ 10</small>'","Math.round((s.metrics.contact_seconds||0)*1000)+' <small>/ 100 ms</small>'")
text=text.replace("metric('contact',d.metrics.contact_substeps,0,'/ 10')","metric('contact',1000*(d.metrics.contact_seconds||0),0,'/ 100 ms')")
text=text.replace('原始物理 · 模型就绪','接触约束版 · 模型就绪')
(new/'app.js').write_text(text,encoding='utf-8')
text=(app/'index.html').read_text(encoding='utf-8')
if not(old/'index.html').exists():(old/'index.html').write_bytes((app/'index.html').read_bytes())
text=text.replace('在原始物理环境中的跨物理预览','在运动学推杆、阻尼0接触约束版中的跨物理预览')
text=text.replace('环境：原始运动学推杆 / 原始阻尼；','环境：运动学推杆 / 阻尼0 / 接触约束修复版；')
text=text.replace('本批固定执行目标','本批固定请求目标；接触受阻时实际移动会受限')
text=text.replace('0 <small>/ 10</small>','0 <small>/ 100 ms</small>')
text=text.replace('当前默认环境已切换为原始运动学推杆、原始阻尼、原始控制循环。','当前环境保留运动学推杆、阻尼0及原字母物理属性，使用接触约束与更细积分修复穿透；与原始实现的数值求解设置不同。')
text=text.replace('执行/预览目标','请求/预览目标')
text=text.replace('本批目标不会随之改变。','本批请求目标不会随之改变；实际推杆运动在受阻时受到约束，原始记录保留请求与实际端点/速度。')
text=text.replace('/diagnostics-root/runs/original-20261030-v1/report/index.html','/contact-pilot/index.html')
text=text.replace('接触求解器内部缓存未序列化','接触求解器缓存和隐藏位置偏置速度未序列化')
(new/'index.html').write_text(text,encoding='utf-8')
# Test source is staged too; it remains a meaningful causality/recording check.
edit('test_playground.py',[
 ('from tdyssey.native_rgb_original_physics import PROTOCOL_SHA256 as PHYSICS_SHA','from tdyssey.native_rgb_contact_safe_physics import PROTOCOL_SHA256 as PHYSICS_SHA,PROTOCOL',1),
 ('test_exact_historical_physics_and_real_history','test_versioned_contact_safe_physics_and_real_history',1),
 ("self.assertEqual(i['physics']['space']['iterations'],10)","self.assertEqual(i['physics']['space']['iterations'],PROTOCOL['solver_iterations'])\n        self.assertEqual(i['protocol']['integration_tick_seconds'],.0005)\n        self.assertEqual(i['physics']['substeps_per_control'],200)\n        self.assertEqual(self.app.env.capture_state().physics_hash,self.app.env._physics_hash())",1),
 ("exported=self.app.export_session();saved=self.app.env.state().copy();history=np.stack(self.app.history).copy()", "exported=self.app.export_session();saved=self.app.env.state().copy();history=np.stack(self.app.history).copy();saved_ticks=self.app.env._last_micro_ticks",1),
 ("np.testing.assert_array_equal(self.app.env.state(),saved)", "np.testing.assert_array_equal(self.app.env.state(),saved)\n        self.assertEqual(self.app.env._last_micro_ticks,saved_ticks)",1),
])
manifest={folder.name:{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in folder.iterdir()if f.is_file()}for folder in(old,new)}
(p/'ui-staging.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest['ui-next'],indent=2))
