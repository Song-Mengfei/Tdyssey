"""Record authorized cleanup as a new baseline, never rewrite old audits."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys

stage=Path(__file__).resolve().parent
repo=stage.parents[2]
sys.path.insert(0,str(repo))
from tdyssey.native_rgb_stage_budget import native_storage_paths,storage_footprint
receipt=repo.parent/'tmp/project-cleanup-20261030/deletions.jsonl'
copy=stage/'authorized-cleanup-deletions.jsonl'
shutil.copyfile(receipt,copy)
records=[json.loads(line)for line in copy.read_text(encoding='utf-8-sig').splitlines()if line.strip()]
deleted=[r for r in records if r.get('status')=='deleted']
unexpected=[r['path']for r in deleted if Path(r['path']).exists()]
if unexpected:raise RuntimeError('Cleanup receipt path reappeared: '+str(unexpected))
old=json.loads((repo/'configs/native_rgb_r2c3.json').read_text(encoding='utf-8'))
paths=native_storage_paths(repo,stage.parent,old['storage_roots'])
current=storage_footprint(paths)
new_sources=list(repo.glob('tdyssey/*r3a*.py'))+list(repo.glob('scripts/*r3a*.py'))+list(repo.glob('tests/*r3a*.py'))+list(repo.glob('configs/*r3a*.json'))
stage_size=storage_footprint([stage]+new_sources)['bytes']
existing=current['bytes']-stage_size
upper=existing+96*1024**2+144*1024**2+256*1024**2+1*1024**2
result=dict(kind='R3A-authorized-post-cleanup-storage-baseline.v1',at_utc=datetime.now(timezone.utc).isoformat(),
    authorization='Supervising chat relayed new direct human project cleanup authorization; exact proposal confirmed as no current R3A read dependency, then actual deletion receipt received.',
    parent_thread_id='__LOCAL_CHAT__',
    receipt=str(copy),receipt_sha256=hashlib.sha256(copy.read_bytes()).hexdigest(),
    deleted_entries=len(deleted),deleted_paths_still_absent=True,
    binary_deleted_entries=[r for r in deleted if str(r['path']).endswith(('.pt','.zip'))],
    old_audits_immutable=True,old_omissions_authorized=True,
    caveat='Deleted historic files no longer exist. Retained historic audits describe their original state, not their current existence. Cache logical sizes may be hardlinks and are not disk-free gains.',
    current=current,new_stage_and_source_bytes=stage_size,existing_non_R3A_bytes=existing,
    R3A_cap_bytes=96*1024**2,R3B_reserved_bytes=144*1024**2,atomic_reserve_bytes=256*1024**2,
    nonstage_growth_reserve_bytes=1*1024**2,shared_cap_bytes=3*1024**3,
    full_pipeline_upper_bound_bytes=upper,headroom_bytes=3*1024**3-upper,
    passed=upper<=3*1024**3)
(stage/'post-cleanup-storage-baseline.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:result[k]for k in ('deleted_entries','existing_non_R3A_bytes','new_stage_and_source_bytes','headroom_bytes','passed')}))
