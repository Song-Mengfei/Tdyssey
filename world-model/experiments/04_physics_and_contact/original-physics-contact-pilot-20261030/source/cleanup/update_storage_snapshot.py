"""Refresh sizes only after new pilot outputs; retain prior asset-hash proof."""
import datetime
import json
import os
from pathlib import Path
import shutil
import stat

d=Path(__file__).resolve().parent
repo=d.parents[2]
artifacts=repo/'artifacts'
baseline=json.loads((artifacts/'r3b-playground-diagnostics-20261030/baseline-storage.json').read_text(encoding='utf-8'))
roots=[Path(p).resolve() for p in baseline['roots']]+[(artifacts/'original-physics-contact-pilot-20261030').resolve()]
directories=[p for p in roots if p.is_dir()]
minimal=[p for p in roots if not any(p!=q and p.is_relative_to(q) for q in directories)]
reparse=getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400)
skipped=[]
def walk(root):
    stack=[root]
    while stack:
        for entry in os.scandir(stack.pop()):
            p=Path(entry.path);s=entry.stat(follow_symlinks=False)
            if entry.is_symlink() or getattr(s,'st_file_attributes',0)&reparse:skipped.append(str(p))
            elif entry.is_dir(follow_symlinks=False):stack.append(p)
            elif entry.is_file(follow_symlinks=False):yield p,s.st_size
sizes={}
for p in minimal:
    if p.is_file():sizes[str(p)]=p.stat().st_size
    elif p.is_dir():sizes.update((str(q.resolve()),n) for q,n in walk(p))
main=list(walk(artifacts))
shared=sum(sizes.values())
result={'kind':'post-cleanup-final-storage-size-snapshot.v1',
        'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'prior_hash_proof':str(d/'retained-assets-verification.json'),
        'scope':'Sizes only; does not repeat or replace prior hash/content checks.',
        'main_artifact_bytes':sum(n for _,n in main),'main_artifact_files':len(main),
        'shared_bytes':shared,'shared_cap_bytes':3*1024**3,'atomic_reserve_bytes':256*1024**2,
        'headroom_after_atomic_reserve_bytes':3*1024**3-256*1024**2-shared,
        'filesystem':dict(zip(('total','used','free'),shutil.disk_usage(repo))),
        'reparse_skipped':skipped,'model_calls':0,'physics_controls':0,
        'snapshot_precedes_own_output':True}
(d/'storage-latest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
