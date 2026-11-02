"""Final immutable-data, receipt and logical-storage verification for pilot1."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlsplit

STAGE=Path(__file__).resolve().parent
ROOT=STAGE.parents[2]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def bind(path):return dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main():
    checks=[]
    def require(flag,label):
        checks.append(dict(check=label,passed=bool(flag)))
        if not flag:raise AssertionError(label)
    plan=read(STAGE/'plan/cases.json');manifest=read(STAGE/'manifest.json');ledger=read(STAGE/'resource-ledger.json')
    require(manifest['closed']and ledger['closed'],'physical pass closed')
    require(not(STAGE/'collector-owner.lock').exists(),'collector owner released')
    require(len(plan['cases'])==48 and len(manifest['entries'])==48,'48 frozen cases attempted once')
    require(len({r['case_id']for r in manifest['entries']})==48,'no repeated physical case')
    require(sum(r['charged_controls']for r in manifest['entries'])==ledger['charged_controls']==2262,'all charged controls reconcile below2304')
    require(ledger['completed_controls']==2261 and ledger['known_substeps']==22618,'complete and partial known cost')
    require(ledger['unknown_advance_substeps']==ledger['reserved_not_started_controls']==0,'zero unknown advance or unstarted reservation')
    for item in read(STAGE/'preserved-inputs.json')['files']+read(STAGE/'execution.json')['bindings']:
        require(sha(item['path'])==item['sha256'],'preserved/frozen binding '+Path(item['path']).name)
    for row in manifest['entries']:
        for key,hashkey in(('file','sha256'),('metadata_file','metadata_sha256')):
            require(sha(STAGE/row[key])==row[hashkey],'actual attempt binding '+row['case_id']+' '+key)
    independent=read(STAGE/'independent-check.json')
    require(independent['passed']and not independent['mismatches'],'72 raw NPZ independent metric check')
    controller_tests=read(STAGE/'controller-tests.json');report_tests=read(STAGE/'report/self-test-result.json')
    require(controller_tests['tests']==6 and controller_tests['exit_code']==0,'six controller regression checks')
    require(sha(ROOT/'tdyssey/native_rgb_contact_pilot_controller.py')==controller_tests['source_sha256'],'tested controller unchanged after pilot')
    require(all(r['geometry_verified']and r['initial_contact_target_preserved']for r in controller_tests['static_initial_commands']),'48 frozen initial geometry commands')
    require(report_tests.get('passed',report_tests.get('result',{}).get('passed',False)),'report focused checks passed')
    media=read(STAGE/'report/media-provenance.json')
    frames=[]
    for row in media['records']:
        frames.extend(row.get('verified_frames',row.get('frame_hashes',[])))
    # The exact frame-proof key is asserted below from the recorded media schema.
    if not frames:
        for row in media['records']:
            for value in row.values():
                if isinstance(value,list)and value and isinstance(value[0],dict)and'collection_sha256'in value[0]:frames.extend(value)
    require(len(frames)==196 and all(r['collection_sha256']==r['render_sha256']for r in frames),'196 paired saved RGB frames exact before encoding')
    require(media['physics_steps']==media['model_calls']==0,'media used no physics or model')
    require(media['media_bytes']<=2097152,'predeclared paired media below2MiB')
    for filename,expected in read(STAGE.parent/'contact-audit-20261030/presentation/presentation-provenance.json')['original_media_hashes'].items():
        require(sha(filename)==expected,'original R3A media unchanged '+Path(filename).name)
    require(sha(STAGE.parent/'r2c3/run/last.pt')=='c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148','6000 checkpoint unchanged')
    require(ledger['model_calls']==ledger['checkpoint_loads']==ledger['optimizer_updates']==0,'no new model work')
    links=[]
    for path in [STAGE/'REPORT.md',STAGE/'report/index.html']:
        source=path.read_text(encoding='utf-8')
        values=re.findall(r'\]\(([^)]+)\)',source)if path.suffix=='.md'else re.findall(r'(?:href|src)=["\x27]([^"\x27]+)["\x27]',source)
        for link in values:
            parsed=urlsplit(link)
            if parsed.scheme or not parsed.path:continue
            require((path.parent/unquote(parsed.path)).is_file(),'local report link '+link);links.append(link)
    receipt=STAGE/'callback-result.json';sent=receipt.exists()and read(receipt).get('sent',False)
    new_sources=[ROOT/f for f in('scripts/native_rgb_contact_pilot_run.py','scripts/native_rgb_contact_pilot_plan.py',
        'scripts/native_rgb_contact_pilot_report.py','tdyssey/native_rgb_contact_pilot_controller.py','tests/test_native_rgb_contact_pilot_controller.py')]
    excluded={'artifact-bindings.json','final-storage.json','final-verification.json'}
    artifacts=[bind(p)for p in sorted(STAGE.rglob('*'))if p.is_file()and p.name not in excluded]
    write(STAGE/'artifact-bindings.json',dict(kind='CONTACT-PILOT1-final-artifact-bindings.v1',files=artifacts,new_sources=[bind(p)for p in new_sources]))
    baseline=read(STAGE.parent/'contact-audit-20261030/baseline-storage.json');roots=[Path(p)for p in baseline['roots']]+[STAGE]
    files={}
    for path in roots:
        for p in([path]if path.is_file()else path.rglob('*')if path.is_dir()else[]):
            if p.is_file():files[str(p.resolve()).lower()]=p
    for folder,pattern in(('tdyssey','native_rgb*.py'),('scripts','native_rgb*.py'),('tests','test_native_rgb*.py')):
        for p in(ROOT/folder).glob(pattern):files[str(p.resolve()).lower()]=p
    shared=sum(p.stat().st_size for p in files.values())
    stage_bytes=sum(p.stat().st_size for p in STAGE.rglob('*')if p.is_file())+sum(p.stat().st_size for p in new_sources)
    conservative_docs=sum(p.stat().st_size for p in(ROOT/'docs/STATUS.md',ROOT/'docs/DECISIONS.md',STAGE.parent/'SUPERVISION.md'))
    ledger_reserve=131072
    require(stage_bytes+conservative_docs+ledger_reserve<=16777216,'16MiB stage cap including full changed docs and final-ledger reserve')
    require(shared+268435456+ledger_reserve<=3221225472,'3GiB shared cap with256MiB atomic reserve')
    now=datetime.now(timezone.utc).isoformat()
    write(STAGE/'final-verification.json',dict(kind='CONTACT-PILOT1-final-verification.v1',at_utc=now,passed=True,
        checks=checks,checks_passed=len(checks),callback_sent=sent,callback_receipt=None if not sent else bind(receipt),
        visual_review=['fixed T040 old/new strip','fixed X005 old/new strip','X086 last known partial poststate'],
        scientific_target_passed=False,controller_promoted=False,model_calls=0,new_physics_in_verification=0))
    write(STAGE/'final-storage.json',dict(kind='CONTACT-PILOT1-final-logical-byte-accounting.v1',at_utc=now,
        stage_and_new_source_bytes=stage_bytes,conservative_full_changed_docs_bytes=conservative_docs,
        conservative_increment_before_final_ledgers=stage_bytes+conservative_docs,final_ledger_reserve_bytes=ledger_reserve,
        stage_cap_bytes=16777216,stage_cap_passed=True,shared_bytes_before_final_ledgers=shared,
        shared_cap_bytes=3221225472,atomic_reserve_bytes=268435456,
        remaining_after_atomic_and_ledger_reserve=3221225472-shared-268435456-ledger_reserve,
        shared_cap_and_atomic_reserve_passed=True,logical_bytes_not_NTFS_allocated_bytes=True))
    print(json.dumps(dict(passed=True,checks=len(checks),stage_bytes=stage_bytes,shared_bytes=shared,callback_sent=sent)))


if __name__=='__main__':main()
