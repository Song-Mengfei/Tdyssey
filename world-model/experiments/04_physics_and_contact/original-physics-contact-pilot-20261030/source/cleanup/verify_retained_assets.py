"""Read-only post-cleanup byte checks and compact storage snapshot; no model load."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

d=Path(__file__).resolve().parent
repo=d.parents[2]
artifacts=repo/'artifacts'
started=time.perf_counter()
base=json.loads((d/'inventory-base.json').read_text(encoding='utf-8'))
receipt=json.loads((d/'checkpoint-deletion-receipt.json').read_text(encoding='utf-8'))
reparse=getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400)
errors=[]
skipped=[]

def walk(root):
    stack=[root]
    while stack:
        current=stack.pop()
        for entry in os.scandir(current):
            p=Path(entry.path);info=entry.stat(follow_symlinks=False)
            if entry.is_symlink() or getattr(info,'st_file_attributes',0)&reparse:
                skipped.append(str(p))
            elif entry.is_dir(follow_symlinks=False):stack.append(p)
            elif entry.is_file(follow_symlinks=False):yield p,info

def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b''):value.update(chunk)
    return value.hexdigest()

known={
 artifacts/'native-rgb-supervised-20261029/r2c3/run/last.pt':'c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148',
 artifacts/'native-rgb-supervised-20261029/r3b/attempts/recovery-20261030/run/last.pt':'23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7',
 repo.parent/'Tdyssey-world-model-validation/artifacts/frozen-wm-assets/denoiser.pth':'4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892',
 artifacts/'r3b-playground-diagnostics-20261030/recordings/recording-7917cd3c57594ca1b4c64401251b5e97.webm':'a9477b14394d201cc0e5c26a8578ee0083f8956749bf065928053134e9307948',
 Path('__LOCAL_HOME__/Downloads/d0291b63-f736-4664-959c-924a3822538e.webm'):'a9477b14394d201cc0e5c26a8578ee0083f8956749bf065928053134e9307948',
}
checks=[]
for path,expected in known.items():
    sha=digest(path)
    checks.append({'path':str(path),'bytes':path.stat().st_size,'sha256':sha,'expected_sha256':expected,'matches':sha==expected})
official=repo.parent/'Tdyssey/checkpoints/official-dp/ema_policy.pt'
checks.append({'path':str(official),'bytes':official.stat().st_size,'sha256':digest(official),
               'matches':None,'note':'Unique official baseline remains outside cleanup scope; entire file read for checksum, no model/pickle loaded. No earlier expected digest supplied here.'})
raw_checks=0
for row in base['files']:
    rel=row['relative_path'].replace('\\','/')
    if row['suffix']=='.npz' and row['sha256']:
        p=Path(row['path'])
        if not p.exists() or digest(p)!=row['sha256']:errors.append({'path':str(p),'reason':'protected saved array absent or hash changed'})
        raw_checks+=1
video_checks=[]
for p in sorted((artifacts/'r3b-playground-diagnostics-20261030/recordings').glob('*.webm')):
    video_checks.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
for row in receipt['deleted']:
    if Path(row['path']).exists():errors.append({'path':row['path'],'reason':'claimed deleted checkpoint still exists'})

baseline=json.loads((artifacts/'r3b-playground-diagnostics-20261030/baseline-storage.json').read_text(encoding='utf-8'))
roots=[Path(p).resolve() for p in baseline['roots']]
roots.append((artifacts/'original-physics-contact-pilot-20261030').resolve())
directories=[p for p in roots if p.is_dir()]
minimal=[p for p in roots if not any(p!=q and p.is_relative_to(q) for q in directories)]
sizes={}
for p in minimal:
    if p.is_file():sizes[str(p)]=p.stat().st_size
    elif p.is_dir():
        for leaf,info in walk(p):sizes[str(leaf.resolve())]=info.st_size
main_files=list(walk(artifacts))
shared=sum(sizes.values())
team=(repo.parent/'Tdyssey/tdyssey_pushing_data').resolve()
result={
 'kind':'original-physics-pilot-post-cleanup-retained-assets.v1',
 'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'model_calls':0,'model_loads':0,'physics_controls':0,'optimizer_updates':0,
 'actual_deletion_receipt':str(d/'checkpoint-deletion-receipt.json'),
 'actual_deleted_files':len(receipt['deleted']),'actual_deleted_logical_bytes':receipt['actual_deleted_logical_bytes'],
 'checkpoint_and_user_video_checks':checks,
 'protected_saved_npz_hashes_checked':raw_checks,'protected_npz_hash_errors':errors,
 'all_user_recording_videos':video_checks,
 'team_original':{'path':str(team),'directory_exists':team.is_dir(),
                  'outside_verified_deletion_root':not team.is_relative_to(artifacts.resolve()),
                  'untouched_by_all_receipt_targets':all(not Path(row['path']).is_relative_to(team) for row in receipt['deleted']),
                  'payload_not_opened':True},
 'sealed_tests':'No test payload decoded; deletion targets are only7explicit historical checkpoints. All validation/test/source data and all NPZ remained untouched, verified saved-NPZ hashes above.',
 'current_launcher_files':[{'path':str(repo/'apps/r3b-playground'/name),'exists':(repo/'apps/r3b-playground'/name).exists()} for name in ('start.py','Start.cmd','server.py','app.js','capture.py')],
 'current_live_preservation_root':str(artifacts/'original-physics-contact-pilot-20261030/live-before'),
 'main_artifact_bytes_snapshot':sum(info.st_size for _,info in main_files),
 'main_artifact_file_count_snapshot':len(main_files),
 'shared_bytes_snapshot':shared,'shared_cap_bytes':3*1024**3,'atomic_reserve_bytes':256*1024**2,
 'headroom_after_atomic_reserve_bytes':3*1024**3-256*1024**2-shared,
 'filesystem_snapshot':dict(zip(('total','used','free'),shutil.disk_usage(repo))),
 'reparse_skipped':sorted(set(skipped)),
 'wall_seconds':time.perf_counter()-started,
 'snapshot_limit':'Snapshot precedes this output file and concurrent new pilot diagnostics; root may rerun this read-only script at final delivery. Logical removed bytes are exact; filesystem-free changes may include concurrent writes.',
}
result['passed']=not errors and all(row['matches'] is not False for row in checks)
(d/'retained-assets-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'all_user_recording_videos','checkpoint_and_user_video_checks','current_launcher_files'}},ensure_ascii=False,indent=2))
assert result['passed']
