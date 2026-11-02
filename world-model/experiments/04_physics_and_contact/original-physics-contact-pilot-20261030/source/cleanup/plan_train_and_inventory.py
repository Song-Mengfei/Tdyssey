"""Select retained legacy input files and explicit obsolete TRAIN candidates only."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re

d=Path(__file__).resolve().parent
repo=d.parents[2]
artifacts=repo/'artifacts'
base=json.loads((d/'inventory-base.json').read_text(encoding='utf-8'))
checkpoints=json.loads((d/'delete-plan-checkpoints.json').read_text(encoding='utf-8'))
rows=base['files']
train={Path(row['path']).stem:row for row in rows if '/r3a/trajectories/R3A-train-' in row['path'].replace('\\','/') and row['suffix']=='.npz'}
keep=collections.defaultdict(set)
pattern=re.compile(r'R3A-train-[TLOCXB]-\d+-a\d+')

def read_ids(path, reason):
    try:
        text=path.read_text(encoding='utf-8-sig')
    except (UnicodeError,OSError):
        return
    for ident in pattern.findall(text):
        if ident in train:keep[ident].add(reason+': '+str(path))

# Preserve every fixed TRAIN example that existing small visual/physics reports use.
for prefix in ['r3a/first-six-inspection','r3a/report']:
    for p in (artifacts/'native-rgb-supervised-20261029'/prefix).rglob('*'):
        if p.is_file() and p.suffix in {'.json','.md','.html','.csv'}:
            read_ids(p,'existing fixed report/physics example')

# Preserve directly frozen input paths in the current original-epoch and diagnostic protocols.
frozen_configs=[
 artifacts/'native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/background/runtime-config.json',
 artifacts/'r3b-playground-diagnostics-20261030/background/runtime-config.json',
]
binding_protection=[]
for p in frozen_configs:
    obj=json.loads(p.read_text(encoding='utf-8-sig'))
    bindings=obj.get('input_bindings',[])
    if isinstance(bindings,dict):iterable=[{'path':key, 'sha256':value} for key,value in bindings.items()]
    else:iterable=bindings
    for binding in iterable:
        path=Path(binding['path'])
        if path.stem in train:
            keep[path.stem].add('current frozen input binding: '+str(p))
            binding_protection.append({'config':str(p),'path':str(path),'sha256':binding.get('sha256')})

# CPU loader/venv/bootstrap records are preserved as reproducible input evidence.
supervised=artifacts/'native-rgb-supervised-20261029'
for p in supervised.rglob('*.json'):
    textpath=str(p).lower()
    if p.stat().st_size <= 2*1024*1024 and any(word in textpath for word in ('loader','diagnostic','benchmark','smoke','cpu-check')):
        read_ids(p,'CPU loader/diagnostic input evidence')

# Root review: retain the full59MB old TRAIN set because full original loader/audit
# depends on its manifest; no data deletion is claimed. The inventory distinguishes
# this conservative dependency retention from a genuinely minimal fixed subset.
retained=[]
candidates=[]
for ident,row in sorted(train.items()):
    if ident in keep:
        retained.append(dict(row,category='legacy_train_input_retained_for_full_historical_loader_and_audit',retention_reasons=sorted(keep[ident])))
    else:
        candidates.append(dict(row,category='retired_old_dynamic_train_trajectory',
            reason='Old dynamic-pusher/0.1s TRAIN data is retired. This full trajectory is outside the retained fixed visual/physics/CPU-loader inputs and both current frozen-input binding lists. Preserve its JSON sidecar, original manifest/split/family/seed/RGB-hash identifiers, training exposure/order/cost, and reports; do not silently claim the old full training set still exists.',
            active_dependency=False,deleted=False))
retention={
 'kind':'legacy-train-minimum-retention-index.v1',
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'policy':'Root explicitly retains the full old TRAIN set for now: full historical loader/trajectory audit manifest depends on it and seven deleted checkpoints already release569MiB. This is NOT claimed to be a minimal subset and no old training data was deleted. Keep exact files in place, no duplicate backup. All seen-validation, metadata/manifests/exposure/order, fixed O/C/B/X raw cohorts and prior prediction arrays remain untouched.',
 'source_protocol':'58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6',
 'retained_train_files':retained,'retained_train_count':len(retained),'retained_train_bytes':sum(r['bytes'] for r in retained),
 'current_frozen_configs':[str(p) for p in frozen_configs], 'current_direct_train_bindings':binding_protection,
 'seen_validation_policy':'Keep all original R3A seen_val NPZ and JSON (144 independent families), all old R2B trajectories/raw small inputs and all genuinely sealed test assets. No sealed test arrays decoded in this inventory.',
 'model_calls':0,'physics_controls':0,'extracted_new_arrays':0,
}
(d/'minimal-fixed-input-retention.json').write_text(json.dumps(retention,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
train_plan={
 'kind':'retired-old-dynamic-TRAIN-explicit-delete-plan.v1',
 'status':'no_old_training_data_deleted_or_proposed_after_root_dependency_review',
 'retention_index':str(d/'minimal-fixed-input-retention.json'),
 'files':candidates,'file_count':len(candidates),'proposed_bytes':sum(r['bytes'] for r in candidates),
 'do_not_delete_any_train_sidecar_or_manifests':True,
 'historical_manifest_disclosure_required':True,
 'no_full_old_training_resume_after_cleanup':False,
 'current_model_load_and_fixed_validation_inputs_remain':True,
 'model_calls':0,'physics_controls':0,'deleted_files':0,
}
(d/'delete-plan-old-train.json').write_text(json.dumps(train_plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

protected_binary=[r for r in rows if (r['suffix'] in {'.pt','.pth','.ckpt'} and r['path'] not in {x['path'] for x in checkpoints['files']})]
inventory={
 'kind':'original-physics-contact-pilot-cleanup-inventory.v1',
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'main_experiment_root':base['main_experiment_root'],
 'main_experiment_bytes_before':base['main_experiment_bytes'],'main_file_count_before':base['file_count'],
 'all_file_inventory':str(d/'inventory-base.json'),
 'shared_deduplicated_bytes_before':base['shared_deduplicated_bytes'],
 'shared_cap_bytes':base['shared_cap_bytes'],'atomic_reserve_bytes':base['atomic_reserve_bytes'],
 'filesystem_before':base['disk'],'reparse_points_skipped':base['reparse_points_skipped'],'scan_errors':base['errors'],
 'delete_plans':[
   {'path':str(d/'delete-plan-checkpoints.json'),'files':checkpoints['file_count'],'bytes':checkpoints['proposed_bytes']},
   {'path':str(d/'delete-plan-old-train.json'),'files':train_plan['file_count'],'bytes':train_plan['proposed_bytes']}],
 'proposed_total_bytes':checkpoints['proposed_bytes']+train_plan['proposed_bytes'],
 'proposed_total_files':checkpoints['file_count']+train_plan['file_count'],
 'protected_model_files':protected_binary,
 'protected_initialization_outside_main_root':str(repo.parent/'Tdyssey-world-model-validation/artifacts/frozen-wm-assets/denoiser.pth'),
 'all_outside_main_experiment_roots_readonly_not_candidates':base['outside_main_experiment_roots'],
 'protected_whole_roots':[
   str(artifacts/'r3b-playground-diagnostics-20261030'),
   str(artifacts/'r3b-playground'),
   str(artifacts/'original-physics-contact-pilot-20261030/live-before'),
 ],
 'protected_misc':'All source/Git/env/third_party/teammate original tdyssey_pushing_data/official initialization/sealed tests; all JSON/config/split/exposure/cost/reports/metrics, all current videos+raw, B penetration evidence, current recently savedclip and initial/session export. No unclassified unique file is a deletion candidate.',
 'retention_index':str(d/'minimal-fixed-input-retention.json'),
 'duplicates_zip_or_delivery_in_main_artifacts':0,
 'active_dependency_evidence':str(artifacts/'original-physics-contact-pilot-20261030/live-before'),
 'deletion_by_inventory_agent':False,'model_calls':0,'physics_controls':0,
 'snapshot_limit':'Inventory precedes new live-save clips, cleanup output metadata and concurrent current-pilot artifacts; root must re-stat/resolve/hash every target immediately before deletion and record actual before/after bytes.',
}
(d/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'train_total':len(train),'train_retained':len(retained),'train_retained_bytes':retention['retained_train_bytes'],
                 'old_train_candidates':len(candidates),'old_train_bytes':train_plan['proposed_bytes'],
                 'current_direct_train_bindings':len(binding_protection),'total_proposed_bytes':inventory['proposed_total_bytes']},indent=2))
