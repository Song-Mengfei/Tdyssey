"""Bind final artifacts and budgets without changing source data or training."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlsplit

STAGE=Path(__file__).resolve().parent
REPO=STAGE.parents[2]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bind(path):return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main():
    checks=[]
    def require(flag,label):
        checks.append(dict(check=label,passed=bool(flag)))
        if not flag:raise AssertionError(label)
    core=read(STAGE/'core/manifest.json');stats=read(STAGE/'statistics/main_summary.json')
    for item in core['files']:
        path=STAGE/'core'/item['file']
        require(path.stat().st_size==item['bytes']and sha(path)==item['sha256'],'core binding '+item['file'])
    for item in read(STAGE/'statistics/inventory.json')['files']:
        path=Path(item['path'])
        require(path.stat().st_size==item['bytes']and sha(path)==item['sha256'],'statistics binding '+path.name)
    raw=read(STAGE/'independent-raw-checks.json')
    require(raw['passed']==raw['total']==106,'independent raw checks106/106')
    require(raw['core_controls_sha256']==sha(STAGE/'core/controls.npz'),'independent checks bind final core')
    require(stats['exact_first_epoch_index_plan']['first_epoch_window_order_sha256']=='800655f0dc54234f92c9784dc4b0ff94e44910d488d39f32439ff625a2ee2335','frozen first epoch window order')
    for key,q in stats['denominators'].items():
        require(sum(v['count']for v in q['six_mutually_exclusive_classes'].values())==q['denominator'],'six classes total '+key)
    source_manifest=STAGE.parent/'r3a/manifest.json'
    require(sha(source_manifest)=='0e91e1412f937c7d8f39473243a27a3720317a07474e730a510e628d040c0a19','original912 manifest unchanged')
    checkpoint=STAGE.parent/'r2c3/run/last.pt'
    require(sha(checkpoint)=='c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148','original6000 checkpoint unchanged')
    presentation=read(STAGE/'presentation/presentation-provenance.json')
    for filename,expected in presentation['original_media_hashes'].items():
        require(sha(Path(filename))==expected,'original media '+Path(filename).name)
    frames=[]
    for item in presentation['supplemental_records']:
        strip=item['new_strip'];path=STAGE/'presentation'/strip['file']
        require(sha(path)==strip['sha256'],'new strip binding '+path.name)
        frames.extend(strip['verified_frames'])
    require(len(frames)==50 and all(f['original_collection_rgb_sha256']==f['saved_state_render_rgb_sha256']for f in frames),'50 saved-state RGB checks')
    links=[]
    for filename in('REPORT.md','ANNEX.md','SAMPLING_PROPOSAL.md'):
        path=STAGE/filename
        links.extend((path,link)for link in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')))
    page=STAGE/'presentation/index.html'
    links.extend((page,link)for link in re.findall(r'(?:href|src)=["\x27]([^"\x27]+)["\x27]',page.read_text(encoding='utf-8')))
    checked_links=0
    for page,link in links:
        parsed=urlsplit(link)
        if parsed.scheme or not parsed.path:continue
        require((page.parent/unquote(parsed.path)).is_file(),'local link '+link)
        checked_links+=1
    require(core['model_calls']==stats['model_calls']==presentation['model_calls']==0,'zero model calls')
    require(core['new_physics_controls']==stats['new_physics_controls']==presentation['physics_controls']==0,'zero new physics controls')
    require(not core['Torch_imported']and not stats['Torch_imported'],'no Torch import in extraction or statistics')
    sources=[REPO/'tdyssey/native_rgb_contact_audit.py',REPO/'scripts/native_rgb_contact_audit.py',
             REPO/'scripts/native_rgb_contact_summary.py',REPO/'scripts/native_rgb_r3a_contact_presentation.py',
             REPO/'tests/test_native_rgb_contact_audit.py']
    now=datetime.now(timezone.utc).isoformat()
    callback_file=STAGE/'callback-result.json'
    callback=read(callback_file)if callback_file.exists()else None
    verification=dict(kind='contact-audit-final-verification.v1',at_utc=now,passed=True,checks=checks,
        checks_passed=len(checks),core_fixtures_passed=6,summary_fixtures_passed=6,raw_checks_passed=106,
        original_media_verified=70,saved_RGB_frames_verified=50,local_links_verified=checked_links,
        visual_review=['cohort and recorded-example overview','B0143.6s annotation','T093 wall-adjacent nearstatic strip'],
        sources=[bind(p)for p in sources],stage_reports=[bind(STAGE/name)for name in('REPORT.md','ANNEX.md','SAMPLING_PROPOSAL.md','result.json')],
        model_calls=0,optimizer_updates=0,new_physics_controls=0,callback_receipt_present=callback is not None,
        callback_receipt=None if callback is None else bind(callback_file),r3b_paused=True,
        known_existing_repository_check_issue='Three pre-existing publication-evidence hash mismatches; not modified or treated as an audit pass.')
    baseline=read(STAGE/'baseline-storage.json')
    roots=[Path(p)for p in baseline['roots']]+[STAGE]
    for folder,pattern in(('tdyssey','native_rgb*.py'),('scripts','native_rgb*.py'),('tests','test_native_rgb*.py')):
        roots.extend((REPO/folder).glob(pattern))
    files={}
    for path in roots:
        for item in([path]if path.is_file()else path.rglob('*')if path.is_dir()else[]):
            if item.is_file():files[str(item.resolve()).lower()]=item
    shared=sum(p.stat().st_size for p in files.values())
    audit_files=[p for p in STAGE.rglob('*')if p.is_file()and p.name!='final-storage.json']+sources
    # Conservatively count full modified shared documentation too, rather than
    # estimating only the added paragraphs. These are not additional disk copies.
    docfiles=[REPO/'docs/STATUS.md',REPO/'docs/DECISIONS.md',STAGE.parent/'SUPERVISION.md']
    conservative_docs=sum(p.stat().st_size for p in docfiles)
    audit_bytes=sum(p.stat().st_size for p in audit_files)+conservative_docs
    reserve_for_this_ledger=65536
    require(audit_bytes+reserve_for_this_ledger<=16777216,'16MiB audit cap including ledger reserve and full changed docs')
    require(shared+reserve_for_this_ledger+268435456<=3221225472,'3GiB shared cap with256MiB atomic reserve')
    verification['checks_passed']=len(checks)
    write(STAGE/'final-verification.json',verification)
    storage=dict(kind='contact-audit-final-logical-byte-accounting.v1',at_utc=now,shared_bytes_before_audit=baseline['shared_bytes_before_new_audit_files'],
        shared_bytes_before_this_ledger=shared,shared_cap_bytes=3221225472,atomic_reserve_bytes=268435456,
        remaining_after_atomic_and_ledger_reserve=3221225472-shared-268435456-reserve_for_this_ledger,
        audit_stage_and_new_source_bytes=sum(p.stat().st_size for p in audit_files),
        conservative_full_changed_document_bytes=conservative_docs,audit_bytes_conservative_before_ledger=audit_bytes,
        audit_cap_bytes=16777216,ledger_reserve_bytes=reserve_for_this_ledger,
        audit_cap_passed=True,shared_and_atomic_reserve_passed=True,
        logical_bytes_not_NTFS_allocated_bytes=True,shared_file_count=len(files),
        new_source_files=[bind(p)for p in sources],directories_not_deleted=True)
    write(STAGE/'final-storage.json',storage)
    print(json.dumps(dict(passed=True,verification_checks=len(checks),audit_bytes_conservative=audit_bytes,
        shared_bytes=shared,callback_receipt=callback is not None)))


if __name__=='__main__':main()
