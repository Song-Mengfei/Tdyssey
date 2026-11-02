"""Persist the current authorized CPU-only R3A stage, preserving history."""
import json
from pathlib import Path
from datetime import datetime, timezone

stage = Path(__file__).resolve().parent
root = stage.parents[2]
base = stage.parent
path = base / 'stage-state.json'
old = path.read_bytes()
snapshot = stage / 'stage-state-before-r3a.json'
if not snapshot.exists():
    snapshot.write_bytes(old)
state = json.loads(old)
state.update(current_stage='R3A', status='implementation_and_collection',
    authorized_scope='Implement CPU-only R3A and actually collect 912 independent 48-control trajectories; no R3B model work before real movement/video review.',
    authorized_stages=['R3A'], future_stages_authorized=False,
    r2c3_parent_review_received=True, r2c3_shape_accepted=False,
    r3a_execution_authorized=True, r3a_collection_authorized=True,
    r3a_parent_authorization_file='r3a/PARENT_REVIEW.md',
    r3a_execution_started=True, r3a_complete=False,
    r3a_renderer_consistency_passed=True,
    r3a_renderer_exact_frames=1325,
    r3a_storage_initial_preflight_passed=True,
    r3a_training_authorized=False, r3b_execution_authorized=False,
    r3b_training_authorized=False, collection_complete=False,
    next_action='Freeze new scene/split/source plan, collect six-letter first6 gate then all 912 slots, audit true movement/contact/coverage, render saved-state videos and measure loader; actively callback then STOP waiting_parent.',
    updated_at_utc=datetime.now(timezone.utc).isoformat())
state['r3a_scope'] = dict(seen_letters=list('TLOCXB'), heldout_letters_untouched=list('DGKNQRWZ'),
    train_per_letter=128, seen_val_per_letter=24, independent_successful_trajectories=912,
    controls_per_trajectory=48, expected_successful_controls=43776,
    max_replacement_candidates_per_slot=2, physics_substeps_per_control=10,
    stage_bytes_cap=96*1024**2, R3B_reserved_bytes=144*1024**2,
    atomic_reserved_bytes=256*1024**2, shared_bytes_cap=3*1024**3,
    obvious_motion_translation_image96px=2., obvious_motion_rotation_deg=5.,
    target_selected_distinct_motion_fraction=.5, GPU_model_work_allowed=False,
    no_user_model_time_limit=True, cleanup_coordination='Parent owns authorized historic cleanup; wait for exact deletion list before new storage/protection baseline.')
state['callback'] = dict(required=True, sent=False, status='not_prepared', attempts=[],
    thread_id=state['parent_thread_id'], host_id='local',
    message_file='r3a/callback-final.txt', result_file='r3a/callback-result.json',
    standing_authorization_file='standing-callback-authorization.json', parent_review_received=False)
path.write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
notice = '''# R3A new independent data and clean/weak history implementation (2026-10-30)

R2C3 really reached6000 full-lineage updates;300 is historical and is not the
current training limit or a convergence verdict. Parent review now requires a
larger independent data library and clearer history conditioning together.
R3A is CPU-only:912 new independent48-control trajectories acrossT/L/O/C/X/B,
768TRAIN plus144new seen-validation,split before collection. No heldout shapes,
model loading,GPU training/inference or servers. Failures and replacements count.

Saved-state original renderer proof passed1325/1325 oldR2B frames pixel-exact,
zero physics steps; six reusable renderers cover all letters,angles,walls,contact
and holes. Initial whole-pipeline storage preflight passed with96MiB R3A,
144MiB R3B and256MiB atomic reserve under3GiB;10,150,451B additional headroom.
Current evidence and full instruction:artifacts/native-rgb-supervised-20261029/r3a/.
The collector, bounded on-demand RGB loader and independent clean/weak/target
noise streams are being implemented in parallel. Actual data counts,amplitude,
contact and video evidence remain pending; a plan is not a completed dataset.

Next:freeze source/config/scene plan; collect first6 real trajectories and inspect
movement/physics, then the remaining target; audit and deliver real10Hz videos,
loader throughput and an executable R3B proposal. Actively report to the planning
chat using the standing human authorization, save its receipt, then STOP waiting_parent.
R3B waits for the retained human requirement of real movement/video review.
Parent owns the newly human-authorized historical cleanup; do not race deletion,
and create a recorded new storage baseline only after its exact cleanup list.
Earlier sections below are historical.

'''
for target in [root/'docs/STATUS.md', base/'SUPERVISION.md']:
    text = target.read_text(encoding='utf-8')
    if not text.startswith('# R3A new independent data'):
        target.write_text(notice+text, encoding='utf-8')
print(json.dumps(dict(current_stage=state['current_stage'], status=state['status'], training_authorized=False)))
