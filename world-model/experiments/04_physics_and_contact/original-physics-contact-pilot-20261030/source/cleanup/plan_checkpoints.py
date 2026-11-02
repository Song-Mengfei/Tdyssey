"""Prepare an explicit checkpoint deletion proposal; never delete or load models."""
import datetime
import json
from pathlib import Path

d=Path(__file__).resolve().parent
repo=d.parents[2]
base=json.loads((d/'inventory-base.json').read_text(encoding='utf-8'))
by_rel={row['relative_path'].replace('\\','/'):row for row in base['files']}
obsolete={
 'native-rgb-20261028/training/best.pt': ('obsolete_selected_legacy_checkpoint',
     'The stopped 20261028 finite-force native experiment selected best by validation loss. It is not byte-identical to last; both belong to the discarded pre-supervised experiment. Current UI binds7499, current before/after comparison binds6000, and neither current approved stage nor the frozen R3B evaluation uses this model. Preserve training/result.json, history, validation, existing raw evaluation and reports. scripts/native_rgb_world_model.py has a historical best.pt fallback; its old experiment command will require an explicit retained checkpoint after cleanup, not silently substitute another model.'),
 'native-rgb-20261028/training/last.pt': ('obsolete_complete_legacy_checkpoint',
     'Distinct complete checkpoint of the same stopped1668-update 20261028 experiment. No active UI/current approved comparison restores it. Its best and last are both superseded for the current line of work, not claimed duplicates. Preserve the full old experiment conclusion, counters, source, split/exposure/cost and raw predictions.'),
 'native-rgb-supervised-20261029/stage3b/run/last.pt': ('superseded_complete_microfit_checkpoint',
     '129-update TRAIN18 microfit parent, continued to250 at stage3c; the later R2C1R1 line explicitly restarted from original GPC rather than this lineage. Neither current6000/7499 comparison nor UI uses the old microfit weights. Preserve restore proof, parameter digests, costs and all54 raw comparisons.'),
 'native-rgb-supervised-20261029/stage3c/run/last.pt': ('obsolete_complete_microfit_checkpoint',
     '250-update TRAIN18 microfit with independentDEV/rigidity not accepted; R2C1R1 explicitly began a fresh initialization and new frozen data protocol. This abandoned lineage is not a retained current baseline. Preserve reports, truth/raw predictions, restore records and cost/provenance.'),
 'native-rgb-supervised-20261029/r2c1r1/run/last.pt': ('superseded_complete_300_checkpoint',
     '300-update full parent was exactly restored and extended by R2C3 to6000. Full6000 remains protected, and all300 fixed-noise outputs, metrics, draw/order/restore evidence remain. Current approved training-before reference is6000; no further300 checkpoint inference is approved.'),
 'native-rgb-supervised-20261029/r2c3/run/milestones/model-001000.pt': ('superseded_model_only_milestone',
     '1000-update model-only milestone superseded by complete6000 and7499. Its fixed paired raw predictions, metrics and O/C/B/X focus galleries remain, sufficient to inspect the recorded learning curve without new inference. No approved future use of1000 weights.'),
 'native-rgb-supervised-20261029/r2c3/run/milestones/model-003000.pt': ('superseded_model_only_milestone',
     '3000-update model-only milestone superseded by complete6000 and7499. Preserve fixed raw predictions, learning-curve metrics, focus galleries and provenance; do not claim the historical milestone file still exists after deletion.'),
}
rows=[]
for rel,(category,reason) in obsolete.items():
    row=dict(by_rel[rel]);row.update(category=category,reason=reason,
        approved_scope='Direct human authorization quoted in PARENT_HANDOFF.txt: delete old unused data/checkpoints; root owns actual deletion.',
        active_dependency=False,
        dependency_evidence=[str(repo/'apps/r3b-playground/server.py'),
            str(repo/'artifacts/native-rgb-supervised-20261029/r2c3/REPORT.md'),
            str(repo/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/REPORT.md')],
        deletion_requires_root_resolve_path_and_reparse_check=True,
        deleted=False)
    rows.append(row)
protected=[dict(by_rel[rel],reason=reason) for rel,reason in [
 ('native-rgb-supervised-20261029/r2c3/run/last.pt','Required complete6000 before-training comparison and recoverable parent.'),
 ('native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt','Current complete7499 model; actual active playground binds its SHA.')]]
result={
 'kind':'original-physics-pilot-explicit-checkpoint-delete-plan.v1',
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'proposed_for_root_literal_deletion_not_executed_by_inventory_agent',
 'verified_project_experiment_root':str(repo/'artifacts'),
 'files':rows,'file_count':len(rows),'proposed_bytes':sum(row['bytes'] for row in rows),
 'protected_checkpoints':protected,
 'outside_main_root_protected_initialization':'__LOCAL_WORKSPACE__/Tdyssey-world-model-validation/artifacts/frozen-wm-assets/denoiser.pth',
 'best_last_note':'Legacy best and last have different SHA and scientific roles in the historical experiment, but no active or approved future reference use. They are obsolete, not duplicates. Historical reports/raw scores are retained; old standalone model reload is deliberately retired and must be disclosed.',
 'live_dependencies_to_preserve':str(repo/'artifacts/original-physics-contact-pilot-20261030/live-before'),
 'process_inventory_limitation':'Default sandbox CIM unavailable; root saved actual health/session/active-processes.json and confirms only verified owned service is managed.',
 'model_calls':0,'physics_controls':0,'deleted_files':0,
}
(d/'delete-plan-checkpoints.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(rows),'bytes':result['proposed_bytes'],'path':str(d/'delete-plan-checkpoints.json')},ensure_ascii=False))
