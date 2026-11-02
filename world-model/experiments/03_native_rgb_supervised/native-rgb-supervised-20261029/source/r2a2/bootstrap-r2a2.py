"""Save delegated R2A2 authorization and exact inherited requests; no physics."""
from pathlib import Path
import copy
from datetime import datetime,timezone
import hashlib
import json
import shutil

R=Path(__file__).resolve().parent.parent
OUT=R/'r2a2'
ROOT=R.parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

s=read(R/'stage-state.json')
assert s['current_stage']=='R2A' and s['status']=='waiting_parent' and s['callback']['sent']
write(OUT/'prior-stage-state.json',s)
s['phase_r2a_callback_history']=s['callback']
s.update(current_stage='R2A2',status='preparing',authorized_stages=['R2A2'],
    authorized_scope='Only64 new finite-damping slots<=512controls. Exact same16 families/32 scene-actions; new families0. OnlyD=.0009765625 or.09921256574801246; oldP0/P1 offline only.0models/GPU/optimizer/train/R2B/formal collection/server.',
    r2a2_execution_authorized=True,r2a2_parent_release_sha256=sha(OUT/'PARENT_REVIEW.md'),
    r2a_diagnostic_evidence_accepted=True,formal_sampling_accepted=False,
    parent_review_received=True,future_stages_authorized=False,r2b_execution_authorized=False,
    r2_execution_authorized='R2A2_only512',
    next_action='Freeze inherited requests/new sole-D protocols. Actual four-arm initialization checks, first T-center A0 two-new-candidate gate, then remaining frozen slots only if safe. Report all dynamics, actual callback, STOP.')
s['budget'].update(r2a2_new_controls_cap=512,r2a2_max_slots=64,r2a2_controls_each=8,
    r2a2_max_substeps=5120,r2a2_new_independent_families=0,
    r2a2_prior_supervised_controls=40192,r2a2_prior_all_native_controls=74752)
s['callback']=dict(required=True,sent=False,attempts=[],thread_id=s['parent_thread_id'],
    host_id='local',message_file='r2a2/callback-r2a2.txt',result_file='r2a2/callback-result.json',
    standing_authorization_file='standing-callback-authorization.json',parent_review_received=False)
write(R/'stage-state.json',s)
shutil.copyfile(R/'native-rgb-cost-ledger.json',OUT/'prior-model-ledger.json')
shutil.copyfile(R/'r2a/runtime-identity.json',OUT/'runtime-identity.json')
old=read(R/'r2a/prior-resource-identities.json')
files=[Path(x['path']) for x in old['files']]+[R/'r2a/resource-ledger.json',
    R/'r2a/r2a-result.json',R/'r2a/index.json',R/'r2a/media/manifest.json']
write(OUT/'prior-resource-identities.json',dict(prior_supervised_controls=40192,
    prior_all_native_controls=74752,no_old_budget_reset=True,
    files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in files]))

source=read(R/'r2a/frozen-requests.json')
base=[x for x in source['requests'] if x['phase']=='R2A' and x['protocol']=='P0-frozen']
assert len(base)==32
arms=[('D-half-life-0.1s',.0009765625,.1),('D-half-life-0.3s',.09921256574801246,.3)]
rows=[]
for req in base:
    for arm,damping,half in arms:
        row=copy.deepcopy(req)
        row.update(request_id=req['request_id'].replace('R2A-','R2A2-',1).removesuffix('-P0-frozen')+'-'+arm,
            phase='R2A2',protocol=arm,source_request_id=req['request_id'],space_damping=damping,
            free_velocity_half_life_seconds=half,execution_enabled=True,
            requires_parent_release='ExplicitR2A2_512_saved_PARENT_REVIEW',new_independent_scene_families=0)
        rows.append(row)
families=copy.deepcopy([f for f in source['scene_families'] if f['phase']=='R2A'])
for family in families:family['phase']='R2A2'
approved=dict(kind='R2A2-parent-approved-same-scene-action-finite-damping-requests.v1',
    prior_requests_path=str((R/'r2a/frozen-requests.json').resolve()),
    prior_requests_sha256=sha(R/'r2a/frozen-requests.json'),source_manifest=source['source_manifest'],
    created_at_utc=datetime.now(timezone.utc).isoformat(),scene_families=families,requests=rows,
    arms=[dict(name=a,space_damping=d,free_velocity_half_life_seconds=h,retention_per_control=d**.1) for a,d,h in arms],
    counts=dict(slots=64,max_controls=512,new_independent_families=0,reused_independent_families=16),
    new_stage_prespecified_contact_descriptor=dict(AO_contact_substeps_range=[0,10],
        relatively_sustained_contact_substeps_gte=8,
        scope='R2A2 descriptive strata, not a sole pass criterion; old R2A reanalysis is explicitly new-stage not retroactive old acceptance'),
    old320_not_spent=True,replay_authorized=False,baseline_old_protocols_offline_only=True)
write(OUT/'approved-requests.json',approved)
idx=read(R/'r2a/index.json')
refs=[]
for row in idx['slots']:
    refs.append(dict(request_id=row['request_id'],family_id=row['family_id'],action_variant=row['action_variant'],
        protocol=row['protocol'],trajectory_file=str((R/'r2a'/row['trajectory_file']).resolve()),
        trajectory_sha256=sha(R/'r2a'/row['trajectory_file']),
        metadata_file=str((R/'r2a'/row['metadata_file']).resolve()),metadata_sha256=sha(R/'r2a'/row['metadata_file'])))
parentfiles=[p for p in (R/'r2a/parent-review').rglob('*') if p.is_file()]
write(OUT/'frozen-baseline-references.json',dict(old_root=str((R/'r2a').resolve()),
    old_media_manifest_sha256=sha(R/'r2a/media/manifest.json'),old_index_sha256=sha(R/'r2a/index.json'),
    slots=refs,parent_review_files=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in parentfiles]))
memo=('当前R2A2仅获批有限D对照64新槽/512控，D=0.0009765625或0.09921256574801246；'
    '复用原16family/32动作变体，0新增独立family，旧P0/P1只读不重跑。完整r2a2/PARENT_REVIEW.md。'
    '独立账本链接40192/74752，若全完成40704/75264。非D物理、实际Shape材料0/0、8absoluteactions和初态严格不变；'
    '先T-center A0新两臂16控gate。任一>2px/nonfinite/physicsIO/配对差异停全phase，不修补重跑或替换。'
    '不删滑行/小幅数据，AO子步0..10及>=8是本阶段预先描述组而非单独验收。'
    '新产物128MiB、共享3GiB含父审文件+256MiB预留逐槽审。0模型/GPU/训练/R2B/正式重采/未见/服务器/安装。'
    '完成实际主动父callback成功receipt后waiting_parent停止；人类持续回传授权有效无需再问。\n\n')
p=R/'SUPERVISION.md'
p.write_text(memo+'以下旧记录为历史：\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
p=ROOT/'docs/STATUS.md'
t=p.read_text(encoding='utf-8')
head=('## R2A2 finite-damping comparison authorized (2026-10-29)\n\n'
    'Only64 new slots/512 controls compare .1s/.3s free-velocity half-life candidates.\n'
    'Reuse the exact16 families/32 scene-action variants; no new independent family.\n'
    'OldP0/P1 are read-only offline baselines, with no rerun or re-encoding.\n'
    'All non-damping physics and actions remain identical. The16-control first gate\n'
    'and actual four-arm initialization checks precede remaining work. Coasting stays\n'
    'valid data; AO0..10 and>=8 are new-stage descriptive strata.\n'
    'R2A diagnostics passed; formal sampling and training have not. NoR2B/model/GPU/\n'
    'training/servers/unseen access. Details: r2a2/PARENT_REVIEW.md and stage-state.\n'
    'Complete, actively report, save receipt and stop.\n\n')
p.write_text(t.replace('# Shared project status\n\n','# Shared project status\n\n'+head,1),encoding='utf-8')
write(OUT/'bootstrap-preparation-error.json',dict(error='Initial inline bootstrap had a Python SyntaxError before any execution; replaced by this saved script.',
    partial_initial_bootstrap_mutations=False,new_physics_controls=0,models=0))
print(json.dumps(dict(stage='R2A2',requests=len(rows),families=len(families),new_families=0,
    new_control_cap=512,approved_requests_sha256=sha(OUT/'approved-requests.json'),parent_review_files=len(parentfiles)),ensure_ascii=False))
