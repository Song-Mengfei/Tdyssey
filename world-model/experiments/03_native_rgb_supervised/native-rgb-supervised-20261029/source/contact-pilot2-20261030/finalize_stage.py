"""Bind CP2's negative efficiency result, immutable inputs and actual costs."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote,urlsplit
import numpy as np

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
    manifest=read(STAGE/'manifest.json');ledger=read(STAGE/'resource-ledger.json');summary=read(STAGE/'report/summary.json')
    require(manifest['closed']and ledger['closed'],'single physical pass closed')
    require(not(STAGE/'collector-owner.lock').exists(),'collector owner released')
    require(len(manifest['entries'])==60 and len({r['case_id']for r in manifest['entries']})==60,'60 frozen cases attempted once')
    require(all(r['status']=='completed'and r['completed_controls']==48 for r in manifest['entries']),'60 complete trajectories and no physical failures')
    require(ledger['charged_controls']==ledger['completed_controls']==2880 and ledger['known_substeps']==28800,'full actual physical cost')
    require(ledger['unknown_advance_substeps']==ledger['reserved_not_started_controls']==0,'zero unknown or unstarted reservation')
    for payload in('preserved-inputs.json','execution.json','regression-release.json'):
        data=read(STAGE/payload)
        for item in data.get('files',data.get('bindings',[])):
            require(sha(item['path'])==item['sha256'],'frozen input '+Path(item['path']).name)
    maximum_penetration=0.;maximum_entry_speed=0.;maximum_force=0.;telemetry_rows=0
    for item in manifest['entries']:
        path=STAGE/item['file'];meta_path=STAGE/item['metadata_file']
        require(sha(path)==item['sha256']and sha(meta_path)==item['metadata_sha256'],'NPZ and metadata binding '+item['case_id'])
        meta=read(meta_path);tpath=STAGE/meta['decision_telemetry_file']
        require(sha(tpath)==meta['decision_telemetry_sha256'],'lossless decision telemetry binding '+item['case_id'])
        with gzip.open(tpath,'rt',encoding='utf-8')as stream:decisions=json.load(stream)['decisions']
        with np.load(path,allow_pickle=False)as z:
            states=z['states'];substates=z['substep_states'];actions=z['actions'];status=z['substep_status']
            require(states.shape==(49,15)and actions.shape==(48,2)and np.array_equal(states,substates[::10])and np.all(status==1),
                    'complete100Hz state chain '+item['case_id'])
            require(np.array_equal(actions,np.asarray([d['final_action']for d in decisions],np.float64)),
                    'logged final action equals physically executed action '+item['case_id'])
            require(np.isfinite(actions).all()and np.all((actions>=0)&(actions<=511)),'original action bounds '+item['case_id'])
            maximum_penetration=max(maximum_penetration,float(np.max(z['substep_penetrations'])))
            maximum_entry_speed=max(maximum_entry_speed,float(np.linalg.norm(z['substep_velocity_after_cap'].astype(float),axis=1).max()))
            maximum_force=max(maximum_force,float(np.linalg.norm(z['substep_forces'][:,4:6].astype(float),axis=1).max()))
        telemetry_rows+=len(decisions)
    require(telemetry_rows==2880,'all2880 decisions and wall-risk evidence retained')
    require(maximum_penetration<=2.,'unchanged2worldpx penetration guard respected in every known substep')
    require(maximum_entry_speed<=200.0001 and maximum_force<=10000.01,'unchanged speed and force caps with float32 recording tolerance')
    independent=read(STAGE/'independent-check.json')
    require(independent['passed']and not independent['mismatches'],'132 original NPZ independent metric audit')
    root=read(STAGE/'root-regression-check.json');report_test=read(STAGE/'report/self-test-result.json')
    require(root['passed']and root['actual_tool_result']['exit_code']==0 and'Ran 13 tests'in root['actual_tool_result']['output'],'actual13 root regression checks passed')
    require(report_test['passed']and sha(ROOT/'scripts/native_rgb_contact_pilot2_report.py')==report_test['source_sha256'],'six focused report checks bind final reporter')
    for item in read(STAGE/'report/inventory.json')['files']:
        path=STAGE/'report'/item['file']
        require(path.stat().st_size==item['bytes']and sha(path)==item['sha256'],'report inventory '+item['file'])
    for item in read(STAGE/'report/csv-compression-proof.json')['files']:
        with gzip.open(STAGE/'report'/item['file'],'rb')as stream:data=stream.read()
        require(hashlib.sha256(data).hexdigest()==item['uncompressed_sha256']and len(data)==item['uncompressed_bytes'],
                'lossless CSV '+item['file'])
    media=read(STAGE/'report/media-provenance.json');proof=[]
    for record in media['records']:
        for version in record['versions']:proof.extend(version.get('verified_frames',[]))
    require(len(proof)==398 and all(r['collection_sha256']==r['render_sha256']for r in proof),'398 saved control RGB frames exact before encoding')
    require(media['media_bytes']<=2097152 and media['physics_steps']==media['model_calls']==0,'fixed3 version media within2MiB and no new physics/model')
    for filename,expected in read(STAGE.parent/'contact-audit-20261030/presentation/presentation-provenance.json')['original_media_hashes'].items():
        require(sha(filename)==expected,'original R3A media '+Path(filename).name)
    require(sha(STAGE.parent/'r2c3/run/last.pt')=='c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148','6000 checkpoint unchanged')
    require(ledger['model_calls']==ledger['checkpoint_loads']==ledger['optimizer_updates']==0,'no model work or training draws')
    require(not summary['groups']['CP2_all_planned']['all_engineering_targets_passed'],'negative engineering outcome retained')
    for path in(STAGE/'REPORT.md',STAGE/'report/index.html'):
        source=path.read_text(encoding='utf-8')
        links=re.findall(r'\]\(([^)]+)\)',source)if path.suffix=='.md'else re.findall(r'(?:href|src)=["\x27]([^"\x27]+)["\x27]',source)
        for link in links:
            parsed=urlsplit(link)
            if not parsed.scheme and parsed.path:require((path.parent/unquote(parsed.path)).is_file(),'local link '+link)
    new_sources=[ROOT/f for f in('scripts/native_rgb_contact_pilot2_run.py','scripts/native_rgb_contact_pilot2_plan.py',
        'scripts/native_rgb_contact_pilot2_report.py','tdyssey/native_rgb_contact_pilot2_controller.py',
        'tdyssey/native_rgb_contact_pilot2_safety.py','tests/test_native_rgb_contact_pilot2_controller.py','tests/test_native_rgb_contact_pilot2_safety.py')]
    excluded={'artifact-bindings.json','final-verification.json','final-storage.json'}
    write(STAGE/'artifact-bindings.json',dict(kind='CONTACT-PILOT2-final-artifact-bindings.v1',
        files=[bind(p)for p in sorted(STAGE.rglob('*'))if p.is_file()and p.name not in excluded],new_sources=[bind(p)for p in new_sources]))
    roots=[Path(p)for p in read(STAGE.parent/'contact-audit-20261030/baseline-storage.json')['roots']]+[STAGE];files={}
    for path in roots:
        for p in([path]if path.is_file()else path.rglob('*')if path.is_dir()else[]):
            if p.is_file():files[str(p.resolve()).lower()]=p
    for folder,pattern in(('tdyssey','native_rgb*.py'),('scripts','native_rgb*.py'),('tests','test_native_rgb*.py')):
        for p in(ROOT/folder).glob(pattern):files[str(p.resolve()).lower()]=p
    shared=sum(p.stat().st_size for p in files.values());new=sum(p.stat().st_size for p in STAGE.rglob('*')if p.is_file())+sum(p.stat().st_size for p in new_sources)
    docs=sum(p.stat().st_size for p in(ROOT/'docs/STATUS.md',ROOT/'docs/DECISIONS.md',STAGE.parent/'SUPERVISION.md'));reserve=262144
    require(new+docs+reserve<=16777216,'16MiB incremental cap including full changed docs and ledger reserve')
    require(shared+reserve+268435456<=3221225472,'3GiB shared cap and256MiB atomic reserve')
    receipt=STAGE/'callback-result.json';sent=receipt.exists()and read(receipt).get('sent',False);now=datetime.now(timezone.utc).isoformat()
    write(STAGE/'final-verification.json',dict(kind='CONTACT-PILOT2-final-verification.v1',at_utc=now,passed=True,
        checks=checks,checks_passed=len(checks),actual_decisions_verified=telemetry_rows,maximum_recorded_penetration_world_px=maximum_penetration,
        maximum_recorded_speed_after_cap_world_px_s=maximum_entry_speed,maximum_recorded_applied_force=maximum_force,
        scientific_targets_passed=False,controller_promoted=False,callback_sent=sent,callback_receipt=None if not sent else bind(receipt),
        visual_review=['X03 three-version strip and CP1unobserved N/A','T03 three-version strip','B04 three-version strip'],
        verification_new_physics=0,verification_model_calls=0))
    write(STAGE/'final-storage.json',dict(kind='CONTACT-PILOT2-final-logical-byte-accounting.v1',at_utc=now,
        stage_and_new_source_bytes=new,conservative_full_changed_document_bytes=docs,conservative_increment_before_ledgers=new+docs,
        final_ledger_reserve_bytes=reserve,increment_cap_bytes=16777216,increment_cap_passed=True,
        shared_bytes_before_final_ledgers=shared,shared_cap_bytes=3221225472,atomic_reserve_bytes=268435456,
        remaining_after_atomic_and_ledger_reserve=3221225472-shared-268435456-reserve,
        shared_cap_and_atomic_reserve_passed=True,logical_not_NTFS_allocated_bytes=True))
    print(json.dumps(dict(verified=True,checks=len(checks),stage_bytes=new,shared_bytes=shared,maximum_penetration=maximum_penetration,callback_sent=sent)))


if __name__=='__main__':main()
