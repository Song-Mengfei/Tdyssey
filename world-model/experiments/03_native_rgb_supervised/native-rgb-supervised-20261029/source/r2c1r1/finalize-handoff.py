"""Persist an actual successful callback receipt; no model operations."""
import json,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c1r1_budget import storage_for_config

EXP=ROOT/'artifacts/native-rgb-supervised-20261029'
OUT=EXP/'r2c1r1'
PARENT='__LOCAL_CHAT__'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(p,obj):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(p)

receipt=read(OUT/'callback-result.json')
assert receipt.get('isError') is False
assert any(PARENT in c.get('text','') for c in receipt.get('content',[]))
prepared=read(OUT/'callback-prepared.json')
assert sha(OUT/'callback-r2c1r1.txt')==prepared['prompt_sha256']
w=read(OUT/'model-work.json');assert w['closed'] and w['status']=='complete'
g=read(EXP/'native-rgb-cost-ledger.json')
assert g['active_stage'] is None and g['charged_updates']==g['completed_updates']==550
assert g['max_seconds'] is None
assert read(OUT/'report/page-QA.json')['finalized'] is True
for row in read(OUT/'source-identity.json')['files']:
    assert sha(row['source'])==sha(OUT/row['snapshot'])==row['sha256']
now=datetime.now(timezone.utc).isoformat()
history=dict(required=True,sent=True,status='sent',thread_id=PARENT,host_id='local',
    message_file='r2c1r1/callback-r2c1r1.txt',result_file='r2c1r1/callback-result.json',
    standing_authorization_file='standing-callback-authorization.json',
    parent_review_received=False,message_sha256=prepared['prompt_sha256'],
    prepared_prompt_sha256=prepared['prompt_sha256'],sent_prompt_sha256=prepared['prompt_sha256'],
    sent_at_utc=now,result_sha256=sha(OUT/'callback-result.json'),
    attempts=[dict(attempt=1,status='sent',started_at_utc=prepared['prepared_at_utc'],
        finished_at_utc=now,standing_human_authorization='standing-callback-authorization.json',
        result_file='r2c1r1/callback-result.json',result_sha256=sha(OUT/'callback-result.json'))])
save(OUT/'callback-result-record.json',history)
result=read(OUT/'r2c1r1-result.json')
result.update(status='waiting_parent',callback_sent=True,callback_result_file=str(OUT/'callback-result.json'),
    callback_result_sha256=history['result_sha256'],callback_prompt_sha256=prepared['prompt_sha256'],
    parent_review_received=False,stopped_at_utc=now)
save(OUT/'r2c1r1-result.json',result)
s=read(EXP/'stage-state.json')
if not (OUT/'stage-state-before-final-handoff.json').exists():save(OUT/'stage-state-before-final-handoff.json',s)
s.update(status='waiting_parent',current_stage='R2C1R1',r2c1r1_complete=True,
    r2c1r1_training_completed=True,r2c1r1_evaluation_completed=True,r2c1r1_reporting_complete=True,
    r2c1r1_model_work_closed=True,r2c1r1_execution_spent=True,r2c1r1_execution_authorized=False,
    training_authorized=False,future_stages_authorized=False,parent_review_received=False,
    r2c1r1_parent_review_received=False,r2c1r1_gray_motion_improved=True,
    r2c1r1_shape_accepted=False,r2c1r1_near_static_accepted=False,
    authorized_scope='R2C1R1 diagnostic batch complete and actually reported; no further execution authorized. Time record_only_v2, no user model-time budget.',
    next_action='STOP waiting_parent after successful actual callback; no polling or automatic model/physics/training continuation.',
    r2c1r1_result_file='r2c1r1/r2c1r1-result.json',r2c1r1_result_sha256=sha(OUT/'r2c1r1-result.json'),
    phase_r2c1r1_callback_history=history)
s.setdefault('revoked_time_limits_history',{})['new_training_global_max_seconds']=s['budget']['new_training_global_max_seconds']
s['budget']['new_training_global_max_seconds']=None
s['budget_time_fields_policy']='Older stage-specific seconds fields are historical only; record_only_v2 overrides all automatic time refusal/stops. Current R1 time limits are null.'
save(EXP/'stage-state.json',s)

status='''# R2C1R1 completed and actively reported; waiting for parent (2026-10-29)

The actual callback succeeded and its receipt is saved in supervised
r2c1r1/callback-result.json. This batch completed300/300 updates and4800 frozen
draws, saved and reloaded its full checkpoint, and retained288 raw four-step
predictions plus216 endpoint scores. Motion-enriched seen-val H1/H4 each won
12/12 against copy-last (H4 IoU0.258134 to0.531987), but natural-val each lost
6/12, near-static4/4 regressed, and H4 shape had12 failures/24 unknown. Shape,
static retention and unseen transfer are not accepted. Cold reviews passed
11177 integrity checks and24 runtime checks;24 readable pages preserve all
windows, original pages and scoring. Full report/evidence:
artifacts/native-rgb-supervised-20261029/r2c1r1/REPORT.md and report/index.html.

Model time is statistics only:486.375s this batch,1382.218s global/550 complete
updates, including the previous27.359s failed launch. All former time caps were
revoked before actual R1 work; the human set no model-time budget. The batch300
was a diagnostic plan, not a human lifetime limit. No new physics;41976/76536
physical counts remain. Stop waiting_parent with no polling or automatic extra
training. Next scope follows review of motion gains, static drift and shape
failures. Earlier status sections below are historical.

'''
p=ROOT/'docs/STATUS.md';p.write_text(status+p.read_text(encoding='utf-8'),encoding='utf-8')
supervision='''当前R2C1R1完整结束，实际主动父回传成功，receipt=r2c1r1/callback-result.json，当前waiting_parent STOP。300/300更新、4800冻结draw、完整checkpoint+实际reload、288原始四步帧/216运行内评分全部保存。明显运动seen-val H1/H4各12/12胜copy（H4 IoU .258134→.531987），natural各6/12退步、严格近静止4/4退步，H4形状12failure/24unknown，不能称保形或未见泛化通过。11177冷检查+24runtime检查通过，全部窗24可读页/原24页保留。完整r2c1r1/REPORT.md、r2c1r1-result.json、report/index.html。时间仅统计：本次486.375s/global1382.218s和550complete，旧27.359s失败保留；所有旧timecaps在启动前撤销，300仅本批诊断计划非用户总限额。0新physics，41976/76536不动。无更多执行许可，不轮询父、不自动续训；依结果父审后再定下一阶段。以下均为历史记录，以本段和record_only_v2计时政策为准。

'''
p=EXP/'SUPERVISION.md';p.write_text(supervision+p.read_text(encoding='utf-8'),encoding='utf-8')
decision='''
## D037 — Seen-motion improvement does not establish static or shape fidelity (2026-10-29)

R2C1R1 completed its300-update frozen diagnostic and real checkpoint reload.
All12 motion-enriched seen-val windows beat copy-last at H1/H4, with H4 gray
IoU0.258134 to0.531987. However natural-val lost6/12 at each horizon, all4
near-static auxiliary windows regressed, and H4 shape had12 failures/24 unknown.
Keep these failures alongside average gains; zero numeric/range failures and
unknown shape outcomes cannot establish rigidity or unseen-shape transfer.
Natural/motion cohorts reuse the same12 seen trajectories with temporal overlap.

The run used record-only model-time accounting from before launch;486.375s is
a statistic, not a stopping budget. All earlier costs and old failed artifacts
remain. The full report, raw predictions, runtime scores, unchanged-source/data
audits and full galleries were actually sent to the supervising chat under the
human's standing authorization. Wait for review without automatic extra work.

'''
p=ROOT/'docs/DECISIONS.md';p.write_text(p.read_text(encoding='utf-8')+decision,encoding='utf-8')
storage=storage_for_config(read(ROOT/'configs/native_rgb_r2c1r1.json')).check(force=True)
save(OUT/'storage-final.json',storage)
verification=dict(kind='R2C1R1-final-handoff-verification.v1',passed=True,checked_at_utc=now,
    model_work_closed=True,global_stage_closed=True,time_policy='record_only_v2',
    source_files_and_snapshots_verified=128,checkpoint_sha256=sha(OUT/'run/last.pt'),
    successful_actual_callback=True,callback_prompt_sha256=prepared['prompt_sha256'],
    callback_result_sha256=history['result_sha256'],final_state='waiting_parent',
    training_authorized=False,future_stages_authorized=False,parent_review_received=False,
    stage_bytes_before_this_record=storage['r2c1_artifact_bytes'],shared_bytes_before_this_record=storage['bytes'],
    atomic_reserve_bytes=268435456)
assert verification['checkpoint_sha256']=='3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0'
save(OUT/'final-state-verification.json',verification)
print(json.dumps(verification,ensure_ascii=False))
