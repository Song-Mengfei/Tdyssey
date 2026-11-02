"""Persist the successful authorized C2 callback and stop; no model operations."""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c2_inference import storage_for_config
EXP=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXP/'r2c2';PARENT='__LOCAL_CHAT__'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
receipt=read(OUT/'callback-result.json');assert receipt.get('isError') is False
assert any(PARENT in c.get('text','')for c in receipt.get('content',[]))
prepared=read(OUT/'callback-prepared.json');assert prepared['prompt_sha256']==sha(OUT/'callback-r2c2.txt')
now=datetime.now(timezone.utc).isoformat()
history=dict(required=True,sent=True,status='sent',thread_id=PARENT,host_id='local',
    standing_authorization_file='standing-callback-authorization.json',message_file='r2c2/callback-r2c2.txt',result_file='r2c2/callback-result.json',
    message_sha256=prepared['prompt_sha256'],prepared_prompt_sha256=prepared['prompt_sha256'],sent_prompt_sha256=prepared['prompt_sha256'],
    result_sha256=sha(OUT/'callback-result.json'),sent_at_utc=now,parent_review_received=False,
    attempts=[dict(attempt=1,status='sent',started_at_utc=prepared['prepared_at_utc'],finished_at_utc=now,
        result_file='r2c2/callback-result.json',result_sha256=sha(OUT/'callback-result.json'),standing_human_authorization='standing-callback-authorization.json')])
save(OUT/'callback-result-record.json',history)
result=read(OUT/'r2c2-result.json');result.update(status='waiting_parent',callback_sent=True,
    callback_result_file=str(OUT/'callback-result.json'),callback_result_sha256=history['result_sha256'],
    callback_prompt_sha256=prepared['prompt_sha256'],parent_review_received=False,R2C3_execution_authorized=False,stopped_at_utc=now)
save(OUT/'r2c2-result.json',result)
s=read(EXP/'stage-state.json');s.update(status='waiting_parent',current_stage='R2C2',r2c2_complete=True,
    r2c2_reporting_complete=True,r2c2_model_work_closed=True,r2c2_execution_spent=True,r2c2_execution_authorized=False,
    r2c2_training_authorized=False,training_authorized=False,future_stages_authorized=False,parent_review_received=False,
    r2c2_parent_review_received=False,r2c3_preparation_complete=True,r2c3_execution_authorized=False,
    r2c3_proposal_config='configs/native_rgb_r2c3_proposed.json',r2c3_proposal_config_sha256=sha(ROOT/'configs/native_rgb_r2c3_proposed.json'),
    r2c2_result_file='r2c2/r2c2-result.json',r2c2_result_sha256=sha(OUT/'r2c2-result.json'),phase_r2c2_callback_history=history,
    authorized_scope='R2C2 inference completed and actually reported; R2C3 proposal only. No execution until next parent stage instruction. No human time budget.',
    next_action='STOP waiting_parent after actual successful C2 callback including C3 proposal; no polling,training,physics,unseen,server or extra model calls.')
save(EXP/'stage-state.json',s)
status='''# R2C2 complete and actively reported; R2C3 proposed only (2026-10-29)

Same300-update checkpoint,native3/8/16 comparison is complete. Existing3/copy
outputs were reused;8/16 added72sampler batches,288raw float32 frames and144
endpoint scores,with864 observed denoiser forwards. All296 parameter and298
state hashes stayed fixed;0new optimizer updates or checkpoints. MotionH4 gray
IoU3/8/16=0.531987/0.504268/0.489620;nearstaticH4 error=0.235675/0.240562/
0.252964. H4 shape failures/unknown=12/24,13/23,12/24. More denoising did not
resolve these current-weight shape/static failures;local improvements remain
reported. This does not establish sufficient training,convergence,capacity
limits or uselessness of further training.300 is an early learning point.

All36windows/H1+H4 are in24 readable pages.4321 cold checks passed,including
133sources,2932old protected files,raw feedback/pairedseeds and closed metrics.
Actual callback receipt is supervised r2c2/callback-result.json. Full evidence:
artifacts/native-rgb-supervised-20261029/r2c2/REPORT.md and report/index.html.
86.907s this stage/global1469.125s are statistics only;550historical updates and
all old failed costs remain. No human model-time budget or old time cutoffs.
Known historical publication CRLF/LF check differences remain disclosed/unchanged.

R2C3_PROPOSAL.md and configs/native_rgb_r2c3_proposed.json prepare full resume
from lineage300 to1000/3000/6000 observations with fixed3/16 evaluations. Keep
cosine through3000 then hold eta_min;explicitly extend resumable data order and
old attempt scope without calling them human lifetime limits. No R2C3 execution
yet. Stop waiting_parent,no polling or automatic training;full server training
always runs manually by the user. Earlier sections below are historical.

'''
p=ROOT/'docs/STATUS.md';p.write_text(status+p.read_text(encoding='utf-8'),encoding='utf-8')
memo='''当前R2C2完整完成并实际主动父回传成功，receipt=r2c2/callback-result.json，waiting_parent STOP。固定300完整权重3/8/16，新增72sampler批/288raw/144scores，真实864forward，296参数/298state不变，无训练/重复权重。motionH4IoU .532/.504/.490，nearstaticH4error .236/.241/.253，shape fail/unknown12/24,13/23,12/24；更多步未救本权重形状，但不能说明300训练够/收敛/能力上限/续训无用。24页全窗图+4321冷审。86.907秒仅统计/global1469.125，旧550updates与失败cost保留，无旧时间帽。R2C3仅提案恢复300→1000/3000/6000 ×3/16eval，cos至3000后holdeta、扩展顺序/attemptscope、字节预算明确；未执行，等父下一阶段指令，无需重复找用户常规许可。无physics/unseen/server/自动续训，不轮询；完整server训练用户手动。完整r2c2/REPORT.md、R2C3_PROPOSAL.md、report/index.html。以下历史。

'''
p=EXP/'SUPERVISION.md';p.write_text(memo+p.read_text(encoding='utf-8'),encoding='utf-8')
decision='''
## D039 — Sampler comparison at300 does not settle training sufficiency (2026-10-29)

At fixed300-update weights,more native denoising did not resolve current shape
or static failures;motionH4 IoU3/8/16 was0.532/0.504/0.490. Preserve local metric
improvements and all failures/unknowns. This only measures sampler effects at
that checkpoint. The human explicitly rejects convergence,capacity-limit or
further-training-useless claims from a few hundred updates.

Prepare R2C3 full-state resume to1000/3000/6000 learning-curve observations with
fixed3/16 cross-evaluation and unchanged training data/math;cosine ends at3000
then holds eta_min instead of rebounding. Earlier sampling cursor4800 and
global3000 attempt guards need explicit versioned continuation,not silent resets
or human lifetime-budget claims. Byte-budget all retained weights and raw data.
Actual C2 callback precedes parent review and any R3 execution. No time hard cap.

'''
p=ROOT/'docs/DECISIONS.md';p.write_text(p.read_text(encoding='utf-8')+decision,encoding='utf-8')
for row in read(OUT/'source-identity.json')['files']:assert sha(row['source'])==sha(OUT/row['snapshot'])==row['sha256']
assert read(OUT/'model-work.json')['closed'] and read(EXP/'native-rgb-cost-ledger.json')['active_stage'] is None
storage=storage_for_config(read(ROOT/'configs/native_rgb_r2c2.json')).check(force=True);save(OUT/'storage-final.json',storage)
save(OUT/'final-state-verification.json',dict(kind='R2C2-final-handoff-verification.v1',passed=True,at_utc=now,
    state='waiting_parent',actual_callback_sent=True,callback_prompt_sha256=prepared['prompt_sha256'],
    callback_result_sha256=history['result_sha256'],model_work_closed=True,ledger_stage_closed=True,
    all133_sources_and_snapshots_verified=True,cold_integrity_checks=4321,raw_frames=288,new_endpoints=144,
    actual_denoiser_forwards=864,new_optimizer_updates=0,R2C3_proposal_only=True,R2C3_execution_authorized=False,
    time_policy='record_only_v2',model_time_budget=None,stage_bytes=storage['r2c1_artifact_bytes'],shared_bytes=storage['bytes'],atomic_reserve_bytes=268435456))
print(json.dumps(dict(state='waiting_parent',callback_sent=True,R2C3_execution=False,stage_bytes=storage['r2c1_artifact_bytes'],shared_bytes=storage['bytes'])))
