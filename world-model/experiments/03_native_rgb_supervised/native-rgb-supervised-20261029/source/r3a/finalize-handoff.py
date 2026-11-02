"""Close actual R3A reporting only after a real authorized callback receipt."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

stage=Path(__file__).resolve().parent;repo=stage.parents[2];base=stage.parent
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_stage_budget import native_storage_paths,storage_footprint
from tdyssey.native_rgb_training import atomic_json


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()


def storage_snapshot():
    config=read(repo/'configs/native_rgb_r3a.json')
    paths=native_storage_paths(repo,base,config['storage_roots'])
    extra=[repo/'configs/native_rgb_r3b_proposal.json',repo/'scripts/native_rgb_r3b_proposal_check.py']
    paths.extend(extra)
    sources=[q for folder,pattern in [('tdyssey','*r3a*.py'),('scripts','*r3a*.py'),('tests','*r3a*.py'),('configs','*r3a*.json')]for q in (repo/folder).glob(pattern)]+extra
    measured=storage_footprint(paths);size=storage_footprint([stage]+sources)['bytes']
    worst=measured['bytes']-size+96*1024**2+144*1024**2+256*1024**2+1*1024**2
    value=dict(kind='R3A-post-callback-final-storage.v1',checked_at_utc=utc(),
        actual_R3A_plus_source_bytes=size,R3A_cap_bytes=96*1024**2,
        shared_actual_bytes=measured['bytes'],shared_files=measured['files'],shared_cap_bytes=3*1024**3,
        future_R3B_reserved_bytes=144*1024**2,atomic_reserved_bytes=256*1024**2,
        nonstage_growth_reserved_bytes=1*1024**2,
        actual_plus_R3B_and_atomic_bytes=measured['bytes']+144*1024**2+256*1024**2,
        full_R3A_cap_pipeline_projection_bytes=worst,
        headroom_after_full_pipeline_reserves_bytes=3*1024**3-worst,
        callback_receipt_sha256=sha(stage/'callback-result.json'),
        passed=size<=96*1024**2 and worst<=3*1024**3)
    if not value['passed']:raise ValueError('Final exact storage cap/reserves failed: '+str(value))
    return value,paths


def main():
    callback=read(stage/'callback-result.json');verification=read(stage/'final-verification.json');result=read(stage/'r3a-result.json')
    if callback.get('sent')is not True or not verification['passed']or not result['completed']:
        raise ValueError('Real completed report, verification and actual callback are required')
    if callback['thread_id']!='__LOCAL_CHAT__':raise ValueError('Unexpected callback destination')
    storage_snapshot()
    sent=utc();history=dict(required=True,sent=True,status='sent',thread_id=callback['thread_id'],host_id='local',
        message_file='r3a/callback-final.txt',message_sha256=sha(stage/'callback-final.txt'),
        result_file='r3a/callback-result.json',result_sha256=sha(stage/'callback-result.json'),
        standing_authorization_file='standing-callback-authorization.json',recorded_after_actual_send_at_utc=sent,
        parent_review_received=False,attempts=[dict(attempt=1,status='sent',result_file='r3a/callback-result.json')])
    state=read(base/'stage-state.json')
    state.update(current_stage='R3A',status='waiting_parent',collection_complete=True,
        r3a_complete=True,r3a_reporting_complete=True,r3a_execution_authorized=False,
        r3a_collection_authorized=False,r3a_physics_closed=True,r3a_background_completed=True,
        r3a_active_worker_pid=None,r3a_active_collector_pid=None,
        r3a_numeric_integrity_passed=True,r3a_data_interface_passed=True,r3a_visual_review_complete=True,
        r3a_full_report_file='r3a/REPORT.md',r3a_result_file='r3a/r3a-result.json',
        r3a_result_sha256=sha(stage/'r3a-result.json'),r3a_reporting_model_calls=0,
        r3a_parent_review_received=False,r3b_preparation_complete=True,r3b_execution_authorized=False,
        r3b_training_authorized=False,future_stages_authorized=False,
        r3a_final_storage_after_callback_file='r3a/final-storage-after-callback.json',
        authorized_scope='R3A real collection and CPU checks/reporting are complete and actually reported. No further execution authorized in this stage.',
        next_action='STOP waiting_parent after actual full R3A callback. No parent polling, new physics/model/checkpoint loading/GPU/heldout/server work. R3B needs actual movement/video review and explicit next scope.',
        updated_at_utc=sent)
    state['callback']=history;state['phase_r3a_final_callback_history']=history
    atomic_json(base/'stage-state.json',state)
    prefix='''# R3A: 912 recorded trajectories audited and reported (2026-10-30)

Collection completed with 912 independent successful48-control trajectories:
128TRAIN and24newseen-validation per letter T/L/O/C/X/B,768/144 total. The
selected velocity half-life remains0.1s. The independent background worker and
collector both exited0; no collector owner remains active. All918 attempts and
6 failures remain:2 engineering interruptions and4 penetration-guard failures.
Costs reconcile to43,891 charged controls,43,886 complete controls and438,889
known physics substeps,unknown0;one historical unstarted reservation remains.

The44,589 numerical checks and94 failed-prefix checks passed. There are no exact
state/action/initial/RGB sequence duplicates or declared cross-split near pairs;
one within-TRAIN near pair is disclosed. Legal TRAIN H1 has34,560windows with
11,992 obvious-motion windows(34.699%). The selected pool contains23,984 distinct
windows,half motion andhalf real negatives,from all768TRAIN trajectories. One
audit epoch has1,499 batches of16,no repeated/dropped windows and exact RNG/
exposure restoration. New-library2worker throughput is30.453samples/s including
startup and74.020 steady,with bounded caches. Model training draws remain0.

Root reviewed35representative strips,6failed last states and3scientific figures.
Before encoding,1,715 representative and116failed-prefix control frames matched
collector RGB hashes exactly; MP4s are viewing copies. Later stalls,33TRAIN and
13val trajectories without a threshold-crossing H1 window,Xval joint-motion
support of2windows/1trajectory,missing outer arcs and untested closed holes are
reported. No model-quality improvement or complete boundary coverage is claimed.

Full evidence:artifacts/native-rgb-supervised-20261029/r3a/REPORT.md and
report/index.html,with real10Hz videos,failure clips,CSV,full distributions,
coverage,source identities,integrity and data-interface proof. The actual full
callback receipt isr3a/callback-result.json;standing human authorization was
used without another confirmation. Authorized historic cleanup receipts remain;
old audits were not rewritten. R3A remains below96MiB,with144MiB futureR3B and
256MiB atomic reserves under3GiB. Final live storage isfinal-storage-after-callback.json.

Clean/weak history code and the actual-size-bound8epoch R3Bproposal passed CPU
contracts. The proposal contains11,992 planned attempts/191,872newdata exposures,
starting from the preserved full6000checkpoint. GPU integration and actual
optimizer/scaler restore remain future work. No model/checkpoint/GPU/optimizer
or extra physics work occurred during post-collection checks. STOP waiting_parent;
review of real movement/videos precedes R3B. Do not poll or auto-train.
Earlier sections below are historical.

'''
    for path in [repo/'docs/STATUS.md',base/'SUPERVISION.md']:
        text=path.read_text(encoding='utf-8')
        if not text.startswith('# R3A: 912 recorded trajectories audited'):path.write_text(prefix+text,encoding='utf-8')
    decision='''
## D042 — A larger independent library is real data evidence, not model validation (2026-10-30)

R3A collected912 independent48-control trajectories with a pre-acquisition family
split:768TRAIN and144new seen-validation acrossT/L/O/C/X/B. Selected0.1s half-life
physics stayed fixed. All6failed attempts, partial substeps and one unstarted
reservation remain charged. Two engineering interruptions used audited versioned
ownership/I/O recovery; completed trajectories were never recollected. The
user-requested independent background process completed with both return codes0.

All44,589 numerical and94failed-prefix checks passed. Training selection keeps
11,992 distinct obvious-motion windows and11,992 distinct negatives; full legal
motion fraction34.699% is reported separately from selected50%. Epoch sampling
is without replacement, with trajectory separation per batch. The originalRGB
renderer is pixel-exact and state/geometry/motion metadata do not become network
features. Missing hole/outer-arc coverage, later stalls and weak joint-motion
validation support remain explicit; no unseen-shape or model-quality claim follows.

The clean/weak native-loss interface and8epoch,actual-library R3Bconfiguration
passed CPU contracts only. R3B requires explicit dataset/cursor/noise migration
from the complete6000checkpoint,actual GPU integration and subsequent review.
Actual reports and local paths were sent to the planningchat under standing
human authorization. Stop waiting_parent without polling or automatic training.
'''
    decisions=repo/'docs/DECISIONS.md';text=decisions.read_text(encoding='utf-8')
    if '## D042 —'not in text:decisions.write_text(text.rstrip()+'\n'+decision,encoding='utf-8')
    live_path=stage/'final-storage-after-callback.json'
    for _ in range(4):
        value,paths=storage_snapshot();atomic_json(live_path,value)
        after=storage_footprint(paths)
        if after['bytes']==value['shared_actual_bytes']and after['files']==value['shared_files']:break
    else:raise RuntimeError('Unexpected file growth during final storage snapshot')
    print(json.dumps(dict(stage_complete=True,status='waiting_parent',actual_callback_sent=True,
        R3A_bytes=value['actual_R3A_plus_source_bytes'],shared_bytes=value['shared_actual_bytes'],
        full_pipeline_headroom_bytes=value['headroom_after_full_pipeline_reserves_bytes'])))


if __name__=='__main__':main()
