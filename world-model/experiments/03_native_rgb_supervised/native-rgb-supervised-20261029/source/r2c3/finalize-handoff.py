"""Save an actual final callback receipt and close the authorized stage."""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c3_continuation import storage_for_config
OUT=Path(__file__).resolve().parent;EXP=OUT.parent;PARENT='__LOCAL_CHAT__'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
receipt=read(OUT/'callback-result.json');assert receipt.get('isError')is False and any(PARENT in x.get('text','')for x in receipt.get('content',[]))
prepared=read(OUT/'callback-prepared.json');assert sha(OUT/'callback-final.txt')==prepared['prompt_sha256']
w=read(OUT/'model-work.json');assert w['closed'] and w['status']=='complete' and w['failure']is None
assert read(EXP/'native-rgb-cost-ledger.json')['active_stage']is None
assert read(OUT/'cold-integrity-review.json')['passed'] and read(OUT/'report/page-QA.json')['finalized'] and read(OUT/'report/focus-visual-review.json')['reviewed']
now=datetime.now(timezone.utc).isoformat()
history=dict(required=True,sent=True,status='sent',thread_id=PARENT,host_id='local',standing_authorization_file='standing-callback-authorization.json',
    message_file='r2c3/callback-final.txt',message_sha256=prepared['prompt_sha256'],prepared_prompt_sha256=prepared['prompt_sha256'],sent_prompt_sha256=prepared['prompt_sha256'],
    result_file='r2c3/callback-result.json',result_sha256=sha(OUT/'callback-result.json'),sent_at_utc=now,parent_review_received=False,
    attempts=[dict(attempt=1,status='sent',started_at_utc=prepared['prepared_at_utc'],finished_at_utc=now,result_file='r2c3/callback-result.json',result_sha256=sha(OUT/'callback-result.json'))])
save(OUT/'callback-result-record.json',history)
r=read(OUT/'r2c3-result.json');r.update(status='waiting_parent',callback_sent=True,callback_result_file=str(OUT/'callback-result.json'),callback_result_sha256=history['result_sha256'],callback_prompt_sha256=prepared['prompt_sha256'],parent_review_received=False,stopped_at_utc=now);save(OUT/'r2c3-result.json',r)
s=read(EXP/'stage-state.json');s.update(status='waiting_parent',current_stage='R2C3',r2c3_complete=True,r2c3_training_completed=True,r2c3_evaluation_completed=True,r2c3_reporting_complete=True,r2c3_focus_delivery_complete=True,r2c3_execution_spent=True,r2c3_model_work_closed=True,r2c3_execution_authorized=False,r2c3_training_authorized=False,training_authorized=False,future_stages_authorized=False,parent_review_received=False,r2c3_parent_review_received=False,phase_r2c3_final_callback_history=history,r2c3_result_file='r2c3/r2c3-result.json',r2c3_result_sha256=sha(OUT/'r2c3-result.json'),authorized_scope='Actual full-state300→6000 continuation and all requested galleries complete and actually reported. No further execution authorized;6000 is observation not convergence/user lifetime limit.',next_action='STOP waiting_parent after actual final callback;no polling/automatic10000+/model/loading/scoring/physics/unseen/server. Time statistics only.');save(EXP/'stage-state.json',s)
status='''# R2C3 actual6000 learning curve complete and actively reported (2026-10-29)

Full300-state resume completed5700 new actual updates,5700 attempts,0AMPskips,
91200new/96000total draws. No repeated300 evaluation. All296 parameter tensors
changed;57 periodic100-success full saves,2light milestones and1final full
payload retained;actual final full reload verified. New evaluations retained864
raw float32 frames/432scores/216sampler batches/2052sampling forwards;27fixed-
noise-plan loss batches.648 total endpoint rows reuse216 old copy/300 results.
30,895 cold checks passed,141 source identities and3222 old files unchanged.

At fixed3 denoising,H4 grayIoU TRAIN300→6000=.562509→.866609,motion=.531987→
.635105,natural=.658888→.727948. TRAIN keeps improving while validation gains
slow,natural is near flat after1000,and fixed-plan validation loss rises from
3000 to6000. These mixed signals require review,not a convergence/capacity claim
or a unique overfitting diagnosis. Nearstatic4/4 still lose to copy-last.
All failures/unknowns/per-window changes remain;unknown is not rigidity pass.

User-requested fixedO/B/C/X four-column truth/1000/3000/6000,all3denoising,
H1/H4 plus large O/C details are in r2c3/report/focus-gallery/index.html.
Independent visual review:O H1 partly improves but corners/square hole remain,
O H4 remains rounded;C gains arm height but bottom/turns/end cuts remain wrong.
Visible deformation cannot be hidden behind a proxy unknown. Full48pages and
curves are in artifacts/native-rgb-supervised-20261029/r2c3/report/index.html;
stage report is r2c3/REPORT.md. Actual final callback receipt is
r2c3/callback-result.json;1000/3000 were also actively reported without pausing.

Model work6876.985s/global8346.110s are statistics only,with no human time cap.
Global6250 completed updates retain unrelated250 history separately from this
6000 model lineage. All old weights/data/failures/costs remain;no new physics,
unseen letters,half-life checks,servers or geometry/motion modules. Known old
publishedJSON CRLF/LF differences remain disclosed and unchanged. Stop
waiting_parent;10000+ requires curve-based next scope,not automatic execution.
Earlier sections below are historical.

'''
p=ROOT/'docs/STATUS.md';p.write_text(status+p.read_text(encoding='utf-8'),encoding='utf-8')
memo='C3真实6000完整结束且最终实际主动回传成功，receipt=r2c3/callback-result.json；waiting_parent STOP。新增5700attempt/5700success/0skip、91200newdraw，全部状态完整resume，57periodicfull/1000-3000light/finalfull真实reload。864raw/432newscore+复用216=648；30895冷检查0异常，141源/3222旧文件不变。固定3步H4 TRAIN .563→.867、motion .532→.635、natural .659→.728；val增益慢、3000→6000fixedloss val回升，保留指标分歧/过拟合线索，不宣称收敛/能力上限/唯一原因。O H1局部恢复仍圆角/方孔未恢复，H4仍圆；C局部恢复但下缘/角/端部仍变形。严格4列真值/1000/3000/6000、固定四窗3去噪H1/H4及OCdetail全部报告/图片路径已父回传。full报告r2c3/REPORT.md、report/index.html、report/focus-gallery/index.html。6876.985s/global8346.110s仅统计，无用户时间帽。停止无更多模型/load/score/训练/physics/unseen/server，不轮询，10000+由父依据曲线决定。以下历史。\n\n'
p=EXP/'SUPERVISION.md';p.write_text(memo+p.read_text(encoding='utf-8'),encoding='utf-8')
decision='''
## D040 — A full6000 learning curve supports training effects,not shape convergence (2026-10-29)

Under direct human continuation authorization,C3 restored the full300 state and
completed5700 additional successful native updates without resetting data/RNG.
Fixed3 H4 motion IoU grew0.532 to0.635;TRAIN grew0.563 to0.867. Natural-val
slowed near0.728 and fixed-plan val loss rose3000→6000 while TRAIN loss fell.
Keep per-window/static/shape tradeoffs and unknown proxy semantics visible.
More training is a supported factor;neither convergence nor a unique root cause
for rounding is established. O/C requested raw fixed-window images still show
unrecovered corners/shape regions despite local improvements. Unknown does not
make visible deformation pass. Native3/16 matrix,physics,data/loss stayed fixed.

Normal AMP overflow is a separately charged skip and continues after scale
reduction,with no LR/success increment;the actual run had0skips. The old stage300,
global3000 and stream4800 scopes were explicitly superseded. Cosine stops at3000
then holds eta_min;time is only statistics. All states and periodic recovery
points are preserved within shared storage.1000/3000 and final6000 were actually
reported;wait for curve review before10000+,with no automatic extra training.

'''
p=ROOT/'docs/DECISIONS.md';p.write_text(p.read_text(encoding='utf-8')+decision,encoding='utf-8')
c=read(ROOT/'configs/native_rgb_r2c3.json');storage=storage_for_config(c).check(force=True);save(OUT/'storage-final.json',storage)
save(OUT/'final-state-verification.json',dict(kind='R2C3-final-handoff-verification.v1',passed=True,at_utc=now,status='waiting_parent',model_work_closed=True,ledger_stage_closed=True,actual_final_callback_sent=True,callback_prompt_sha256=prepared['prompt_sha256'],callback_result_sha256=history['result_sha256'],lineage_updates=6000,new_complete=5700,AMP_skips=0,cold_checks=30895,source_identity_file=c['source_identity'],focus_visual_reviewed=True,all48pages_QA=True,training_authorized=False,future_stages_authorized=False,parent_review_received=False,stage_bytes=storage['r2c1_artifact_bytes'],shared_bytes=storage['bytes'],atomic_reserve_bytes=268435456,time_policy='record_only_v2',model_time_hard_limit=None))
print(json.dumps(dict(status='waiting_parent',actual_callback=True,lineage_updates=6000,stage_bytes=storage['r2c1_artifact_bytes'],shared_bytes=storage['bytes'],atomic_reserve=268435456)))
