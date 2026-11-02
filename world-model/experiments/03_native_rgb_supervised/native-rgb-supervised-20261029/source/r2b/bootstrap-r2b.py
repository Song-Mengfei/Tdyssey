"""Persist selected physics and delegated R2B limits; no simulator/model work."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
R=Path(__file__).resolve().parent.parent
O=R/'r2b'
ROOT=R.parent.parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=read(R/'stage-state.json')
assert s['current_stage']=='R2A2' and s['status']=='waiting_parent' and s['callback']['sent']
write(O/'prior-stage-state.json',s)
s['phase_r2a2_callback_history']=s['callback']
s.update(current_stage='R2B',status='planning_and_implementation',authorized_stages=['R2B'],
    authorized_scope='Only54 newly frozen6letterx9space contact-following slots<=24controls each/fresh cap1296. Selected0.1s half-life D.0009765625 only; no more damping comparison/validation/search.0models/GPU/optimizer/training/unseen/server.',
    r2b_execution_authorized=True,r2b_parent_release_sha256=sha(O/'PARENT_REVIEW.md'),
    r2a2_diagnostic_evidence_accepted=True,parent_review_received=True,
    formal_sampling_accepted=False,full_18letter_collection_authorized=False,
    future_stages_authorized=False,training_authorized=False,
    next_action='Freeze complete54-slot geometry plan and deterministic feedback rules; first6 role-specific slots within1296; implementation/timing/quality gate then frozen remaining48. All results+videos, actual callback, STOP.')
s['selected_physics']=dict(selection='direct human instruction via supervising parent',
    direct_user_text='那就全用0.1s半衰期的 不用验证了',half_life_seconds=.1,
    space_damping=.0009765625,protocol_sha256='58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6',
    display_name='0.1秒半衰期（已选）',applies_to='All new sampling and future TRAIN/validation/test data generation',
    historical_protocols_and_data_not_relabelled=True,further_damping_comparisons=False,
    further_damping_validation=False,further_damping_search=False)
s['budget'].update(r2b_new_controls_cap=1296,r2b_max_slots=54,r2b_controls_each_cap=24,
    r2b_max_substeps=12960,r2b_first_six_controls_cap=144,
    r2b_prior_supervised_controls=40704,r2b_prior_all_native_controls=75264,
    old_unused320_not_spent=True,r2b_replay_slots_authorized=0)
s['callback']=dict(required=True,sent=False,attempts=[],thread_id=s['parent_thread_id'],
    host_id='local',message_file='r2b/callback-r2b.txt',result_file='r2b/callback-result.json',
    standing_authorization_file='standing-callback-authorization.json',parent_review_received=False)
write(R/'stage-state.json',s)
shutil.copyfile(R/'native-rgb-cost-ledger.json',O/'prior-model-ledger.json')
shutil.copyfile(R/'r2a2/runtime-identity.json',O/'runtime-identity.json')
old=read(R/'r2a2/prior-resource-identities.json')
paths=[Path(x['path']) for x in old['files']]+[R/'r2a2/resource-ledger.json',R/'r2a2/r2a2-result.json',
    R/'r2a2/index.json',R/'r2a2/media/manifest.json']
write(O/'prior-resource-identities.json',dict(prior_supervised_controls=40704,
    prior_all_native_controls=75264,no_old_budget_reset=True,
    files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in paths]))
parentfiles=[p for p in (R/'r2a2/parent-review').rglob('*') if p.is_file()]
write(O/'parent-review-identities.json',dict(files=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in parentfiles]))
memo=('当前用户已选0.1秒半衰期，D精确.0009765625，既有协议58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6。'
    '之后所有新采样/未来TRAIN/val/test数据生成默认此配置；旧协议/旧数据身份不改，不把D0改标签。'
    '不再阻尼对照/验证/搜索，不再请用户选D；显示“0.1秒半衰期（已选）”，历史协议技术名candidate不重写SHA。\n'
    '当前仅R2B新54槽=6letter×9space/每槽<=24/总1296新控授权，先完整静态新计划和反馈规则冻结，再首6(Trot/Ctrans/Xtrans/Btrans/L反rot/Owall)最多144控作实现/时序/质量gate，之后既定剩余48。'
    '不是18字母正式集，不用R1旧B草案直跑，0模型/GPU/训练/未见/服务器/安装/上传包。'
    '控制规则当前/历史state反馈localcontact世界跟随，no未来仿真/不改物体状态/只初始reset；approach/withdraw/recontact合法free路径，恢复占24不追加。'
    '0..3approach/4..13push/14..19release-decay/20..23recontact，计划不是实际成功，正常小运动不删，gap/arc/lever/方向/壁角缺层如实。'
    '任何substep>2px/nonfinite/物理IO/记录规则差异立即停全phase不补采重放。'
    '累计40704/75264，跑满42000/76560；原余320不动，模型250/868.484字节不变。'
    '每阶段128MiB和共享3GiB含父审+256MiB原子预留逐槽核。实际幅度和全部单配置视频回父验收前不训练。'
    '持续人类callback授权有效，完成实际发送保存receipt后waiting_parent停止。完整r2b/PARENT_REVIEW.md。\n\n')
p=R/'SUPERVISION.md';p.write_text(memo+'以下旧候选/阶段记录仅为历史：\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
p=ROOT/'docs/STATUS.md';text=p.read_text(encoding='utf-8')
head=('## Selected0.1s physics; R2B contact-following resampling authorized (2026-10-29)\n\n'
    'The user selected0.1s free-velocity half-life, exact damping.0009765625 and\n'
    'existing protocol58563255...20f81f6. All new collection and future TRAIN/val/\n'
    'test generation must use this selected configuration; historical protocols\n'
    'and data keep their original identities. No further damping comparisons,\n'
    'validation or search, and no request to choose damping again.\n\n'
    'Only R2B54 new slots/1296 controls are authorized,6TRAIN representatives×9\n'
    'spaces,24-control cap. Freeze a new static plan and deterministic current-state\n'
    'contact-following rules before the first6role-specific implementation gate.\n'
    'Then continue frozen remaining48only without quality/rule errors. Preserve\n'
    'allsmallmotion/coasting/blocked cases and allsingle-configuration videos.\n'
    'No model/GPU/training,18letter formal collection,unseen access or servers.\n'
    'See r2b/PARENT_REVIEW.md, stage-state and SUPERVISION.md. Actively report\n'
    'actual amplitude/coverage/videos and stop for parent acceptance.\n\n')
p.write_text(text.replace('# Shared project status\n\n','# Shared project status\n\n'+head,1),encoding='utf-8')
print(json.dumps(dict(stage='R2B',selected_half_life=.1,selected_damping=.0009765625,
    new_controls_cap=1296,slots=54,parent_review_files=len(parentfiles)),ensure_ascii=False))
