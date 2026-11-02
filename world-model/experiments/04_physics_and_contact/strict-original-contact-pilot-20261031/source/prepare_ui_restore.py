"""Preserve deployed sources and prepare original-protocol UI without deploying."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
APP=ROOT/'apps/r3b-playground'
BEFORE=OUT/'ui-source-before'
STAGE=OUT/'ui-next'

def replace(text,old,new):
    if old not in text:raise ValueError('Expected deployed source missing: '+old[:100])
    return text.replace(old,new)

def main():
    BEFORE.mkdir(exist_ok=False);STAGE.mkdir(exist_ok=False)
    for source in APP.iterdir():
        if source.is_file():shutil.copyfile(source,BEFORE/source.name);shutil.copyfile(source,STAGE/source.name)
    p=STAGE/'server.py';s=p.read_text(encoding='utf-8')
    for a,b in [
      ('from tdyssey.native_rgb_contact_safe_physics import make_contact_safe_env,PROTOCOL','from tdyssey.native_rgb_original_physics import make_original_env,PROTOCOL'),
      ('newenv=make_contact_safe_env(PHYSICS,letter)','newenv=make_original_env(PHYSICS,letter)'),
      ('Expected versioned contact-safe kinematic pusher/damping/iterations','Expected strict original kinematic pusher/damping/iterations'),
      ("self.message='接触约束修复版已就绪：运动学推杆、阻尼0；当前模型仍是旧物理训练的跨物理预览。'","self.message='原始环境已就绪：100Hz、solver10、原PD；运动学推杆、阻尼0。模型仍为旧物理训练的跨物理预览。'"),
      ("saved_ticks=metadata.get('physical_state',{}).get('controller',{}).get('integration_microstep_ticks')\n        allowed_ticks=self.env.initial_identity()['protocol']['integration_step_tick_choices']\n        if type(saved_ticks)is not int or saved_ticks not in allowed_ticks:raise ValueError('Saved integration-controller state is incomplete; current session was preserved')","from capture import validate_original_controller\n        validate_original_controller(self.env,metadata.get('physical_state',{}).get('controller',{}))"),
      ("self.message='物理环境触发穿透保护，已停在最后一个完整对比步；请重置场景。'","self.message='原始物理环境未完成当前控制，已保留原始失败状态；请重置场景。'"),
      ("contact_seconds=float(self.info['agent_object_contact_seconds']),intervention_ratio=float(self.info['intervention_time_ratio'])","contact_seconds=float(self.info['agent_object_contact_substeps'])/100,intervention_ratio=0."),
      ("action_contract='requested absolute controller target retained; actual endpoints/velocities and unilateral contact interventions in control_infos'","action_contract='requested absolute target executes unchanged through pinned original 100Hz PD and10 solver iterations; observation-only contacts'"),
      ("physics='环境：运动学推杆 / 阻尼0 / 接触约束修复版；跨物理预览'","physics='环境：原始100Hz / solver10 / 原PD / 运动学推杆 / 阻尼0；跨物理预览'"),
      ("action_contract='requested target; actual pusher may be constrained by contact'","action_contract='requested target through unchanged original100Hz controller; no contact velocity projection'"),
    ]:s=replace(s,a,b)
    p.write_text(s,encoding='utf-8')
    p=STAGE/'capture.py';s=p.read_text(encoding='utf-8')
    s=replace(s,"control_step=int(env.control_step),integration_microstep_ticks=getattr(env,'_last_micro_ticks',None),pending_action=None if not hasattr(env,'_action') else np.asarray(env._action).tolist()),","control_step=int(env.control_step),physics_hz=int(env.sim_hz),control_hz=int(env.control_hz),solver_iterations=int(env.space.iterations),substep_dt=.01,substeps_per_control=10,pending_action=None if not hasattr(env,'_action') else np.asarray(env._action).tolist()),")
    s=replace(s,"    micro_ticks=value['controller'].get('integration_microstep_ticks')\n    allowed_ticks=env.initial_identity()['protocol']['integration_step_tick_choices']\n    if type(micro_ticks)is not int or micro_ticks not in allowed_ticks:raise ValueError('Saved integration-controller state is incomplete')","    validate_original_controller(env,value['controller'])")
    s=replace(s,'    env._last_micro_ticks=micro_ticks\n','')
    s=replace(s,"repo/'tdyssey/original_contact_pilot_report.py']","repo/'tdyssey/original_contact_pilot_report.py',repo/'artifacts/strict-original-contact-pilot-20261031',repo/'tdyssey/strict_original_contact_pilot.py',repo/'tdyssey/native_rgb_data_eligibility.py',repo/'scripts/strict_original_contact_pilot.py']")
    s=replace(s,'def restore_physics(env,value):', '''def validate_original_controller(env,controller):
    """Reject nonoriginal or incomplete controller state before body writes/reset."""
    expected=dict(name=env.controller,kp=100.,kv=20.,physics_hz=100,control_hz=10,solver_iterations=10,substep_dt=.01,substeps_per_control=10)
    if env.initial_identity()['protocol']['name']!='original-kinematic-pusht-goalfree-rgb.v1':raise ValueError('Only strict original protocol restores are allowed')
    if any(controller.get(key)!=value for key,value in expected.items()):raise ValueError('Saved original-controller state is incomplete or differs; current session was preserved')
    if controller.get('integration_microstep_ticks') is not None:raise ValueError('Modified integration-controller state cannot be restored')

def restore_physics(env,value):''')
    p.write_text(s,encoding='utf-8')
    p=STAGE/'test_playground.py';s=p.read_text(encoding='utf-8')
    s=replace(s,'from tdyssey.native_rgb_contact_safe_physics import PROTOCOL_SHA256 as PHYSICS_SHA,PROTOCOL','from tdyssey.native_rgb_original_physics import PROTOCOL_SHA256 as PHYSICS_SHA,PROTOCOL')
    s=replace(s,'test_versioned_contact_safe_physics_and_real_history','test_strict_original_physics_and_real_history')
    s=replace(s,"self.assertEqual(i['protocol']['integration_tick_seconds'],.0005)","self.assertEqual(i['physics']['substep_dt'],.01)")
    s=replace(s,"self.assertEqual(i['physics']['substeps_per_control'],200)","self.assertEqual(i['physics']['substeps_per_control'],10)")
    s=replace(s,'saved_ticks=self.app.env._last_micro_ticks','saved_controller=self.app.session_record()[\'physical_state\'][\'controller\']')
    s=replace(s,'self.assertEqual(self.app.env._last_micro_ticks,saved_ticks)',"self.assertEqual(self.app.session_record()['physical_state']['controller'],saved_controller)")
    s=replace(s,"dir=Path(__file__).resolve().parents[2]/'artifacts'","dir=__import__('server').REPO/'artifacts'")
    addition='''    def test_wrong_original_controller_rejected_before_reset(self):
        import base64
        from capture import unpack,session_export
        exported=self.app.export_session();metadata,arrays=unpack(base64.b64decode(exported['archive_base64']))
        metadata['physical_state']['controller']['physics_hz']=2000
        bad=session_export(metadata,arrays);before=self.app.env.state().copy();session=self.app.session
        with self.assertRaises(ValueError):self.app.restore_session(bad)
        self.assertEqual(self.app.session,session);np.testing.assert_array_equal(self.app.env.state(),before)
    def test_real_ten_substeps_and_contact_seconds(self):
        self.app.step(self.app.session,self.app.revision,np.asarray(self.app.target)+[8,0],1)
        info=self.app.info
        self.assertEqual(info['physics_substeps'],10)
        self.assertEqual([r['dt']for r in info['substeps']],[.01]*10)
        self.assertEqual(self.app.metrics['contact_seconds'],info['agent_object_contact_substeps']/100)
        self.assertNotIn('interventions',info)

'''
    s=replace(s,'class ConcurrentRequestTests(unittest.TestCase):',addition+'class ConcurrentRequestTests(unittest.TestCase):')
    p.write_text(s,encoding='utf-8')
    p=STAGE/'index.html';s=p.read_text(encoding='utf-8')
    for a,b in [
      ('运动学推杆、阻尼0接触约束版','原始100Hz、solver10、运动学推杆和阻尼0环境'),
      ('环境：运动学推杆 / 阻尼0 / 接触约束修复版','环境：原始100Hz / solver10 / 原PD / 运动学推杆 / 阻尼0'),
      ('；接触受阻时实际移动会受限','；目标经原始PD控制器执行'),
      ('当前环境保留运动学推杆、阻尼0及原字母物理属性，使用接触约束与更细积分修复穿透；与原始实现的数值求解设置不同。','当前环境严格执行原始100Hz、每控制10子步、solver10、原PD kp100/kv20，运动学推杆、阻尼0和原字母物理属性。原环境自身仍可能穿透，当前没有增加接触速度约束或物理修补。'),
      ('实际推杆运动在受阻时受到约束','实际推杆经原始PD控制器执行'),
    ]:s=replace(s,a,b)
    p.write_text(s,encoding='utf-8')
    p=STAGE/'app.js';s=p.read_text(encoding='utf-8');s=replace(s,'接触约束版 · 模型就绪','原始100Hz · 模型就绪');s=replace(s,'原始运动学推杆 / 原始阻尼；R3B旧物理训练检查点，跨物理预览','原始100Hz / solver10 / 运动学推杆 / 阻尼0；旧物理模型跨物理预览');p.write_text(s,encoding='utf-8')
    (STAGE/'README.txt').write_text('''Tdyssey R3B original-protocol playground

The active environment executes the pinned PushTEnv.step itself:100Hz,dt0.01,
10 solver iterations, original PD kp100/kv20 at100Hz,10Hz actions/images,
KINEMATIC pusher,DYNAMIC object,Space.damping0. Original geometry,mass,inertia,
friction,walls and collision parameters are unchanged. No extra microsteps,
velocity projection,speed disk,wall budget,position repair or velocity clearing.
The original environment can still penetrate; this restoration is not a fix.

R3B lineage7499 remains trained on historical dynamic-pusher/.1s half-life
physics. Predictions remain cross-physics preview; no model adaptation or update.
H1 uses4 real frames;H4 feeds raw predictions only within its4-control batch.
Requested absolute targets are the actual actions to the original controller.

Recording and clip behavior remains bounded and diagnostic-only:64MiBWebM,
128MiBstreaming raw,32MiB/16batch ring; shared3GiB with256MiBatomic reserve.
Snapshot restore preserves public body/controller state and model histories,
but does not contain native solver caches/hidden position-bias velocities and
does not claim arbitrary mid-contact exact physics replay. Nonoriginal protocol
and incomplete controller metadata are rejected before reset/body writes.

Old modified-physics data remain diagnostic,training_eligible=false. The new
strict-original collection has an independent output root and eligibility gate.
Training,architecture,loss,denoising and server experiments remain unapproved.
''',encoding='utf-8')
    manifest=dict(kind='strict-original-ui-staging.v1',deployed=False,model_calls=0,optimizer_updates=0,files=[])
    for source in sorted(BEFORE.iterdir()):
        target=STAGE/source.name
        manifest['files'].append(dict(file=source.name,before_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),staged_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changed=source.read_bytes()!=target.read_bytes()))
    (OUT/'ui-staging-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest))

if __name__=='__main__':main()
