"""In-place diagnostic exclusion inventory; no data copies or physics/model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

P = Path(__file__).resolve().parents[1]
ROOT = P.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def evidence(path):
    return dict(path=str(path), exists=path.is_file(),
                sha256=sha(path) if path.is_file() else None)


def inventory(root):
    files = bytes_ = npz = gzip = 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(directory)/name
            if not path.is_file():
                continue
            files += 1
            bytes_ += path.stat().st_size
            npz += path.suffix == '.npz'
            gzip += name.endswith('.json.gz')
    return dict(path=str(root), exists=root.is_dir(), files=files,
        bytes=bytes_, npz_files=npz, compressed_telemetry_files=gzip,
        training_eligible=False, treatment='diagnostic only; retained in place')


def main():
    folders = ['physics-design', 'physics-regression', 'collection',
               'ui-validation', 'live-before', 'live-before-update']
    rows = [inventory(P/name) for name in folders]
    phases = {}
    costs = []
    for phase, relative in (
            ('calibration-v1', 'collection/revisions/calibration-v1-ledger-frozen.json'),
            ('calibration-v2', 'collection/runs/calibration-v2/ledger.json')):
        path = P/relative
        ledger = read(path)
        phases[phase] = dict(attempts=len(ledger['attempts']),
            completed_attempts=sum(a['status'] == 'completed' for a in ledger['attempts']),
            actual_controls=ledger['actual_controls'],actual_substeps=ledger['actual_substeps'],
            simulated_duration_seconds=ledger['actual_physics_seconds'],
            cpu_elapsed_seconds_not_reported_by_this_ledger=True,
            model_calls=ledger['model_calls'],optimizer_updates=ledger['optimizer_updates'],
            cost_evidence=evidence(path),training_eligible=False)
        costs.append(evidence(path))
    for relative in ('physics-design/COST-SUMMARY.json',
            'physics-regression/v1/summary.json','physics-regression/v2/summary.json',
            'physics-regression/v3/summary.json','physics-regression/v4/summary.json',
            'collection/calibration-v1-completion.json',
            'collection/runs/calibration-v2/completion.json',
            'background/calibration-v2/completion.json'):
        costs.append(evidence(P/relative))
    source_archives = []
    for pattern in ('physics-design/candidate*-source*.txt',
                    'physics-regression/v*/candidate-source.py.txt',
                    'physics-regression/v*/original-source.py.txt'):
        source_archives += [evidence(path) for path in P.glob(pattern)]
    source_archives.append(evidence(ROOT/'tdyssey/native_rgb_contact_safe_physics.py'))
    # Summary and trajectory metadata are sufficient for identity inventory;
    # do not decompress/re-hash the large raw banks merely to quarantine them.
    protocol_records = []
    for path in sorted((P/'physics-design').glob('*.json')):
        if path.stat().st_size > 2*1024**2:
            continue
        try:
            value = read(path)
        except (ValueError, UnicodeError):
            continue
        digest = value.get('protocol_sha256') if isinstance(value, dict) else None
        if digest:
            protocol_records.append(dict(path=str(path),protocol_sha256=digest))
    for path in sorted((P/'collection/trajectories').glob('*/metadata.json')):
        value=read(path)
        protocol_records.append(dict(path=str(path),
            protocol_sha256=value.get('physics_protocol_sha256',value.get('protocol_sha256')),
            slot_id=value['slot']['slot_id'],phase=value['slot']['phase'],
            training_eligible=False))
    actual_phases = {item['phase'] for item in protocol_records if 'phase' in item}
    result = dict(kind='strict-original-diagnostic-exclusion.v1',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        authority=str(P/'STRICT_ORIGINAL_CORRECTION-20261031.txt'),
        training_eligible=False,diagnostic_only=True,
        excluded_roots=[str(P)],
        exclusion_scope='Entire legacy pilot root, including all v1-v6 prototypes, '
            'regressions, original comparison baselines, cal1/cal2, UI telemetry and '
            'any future legacy-nested formal/supplement paths. New strict data must be separate.',
        inventories=rows,physical_protocol_evidence=protocol_records,
        historical_source_archives=source_archives,phase_costs=phases,
        cost_evidence=costs,actual_collection_phases=sorted(actual_phases),
        formal_or_supplement_data_present=bool(actual_phases-{'calibration-v1','calibration-v2'}),
        raw_preserved_in_place=True,large_data_copies_created=0,
        raw_files_rewritten=0,raw_or_metadata_relabelled=0,
        model_calls_this_operation=0,optimizer_updates_this_operation=0,
        no_auto_retry=True,
        cost_note='Historical actual_physics_seconds is simulated integrated dt, '
            'not CPU/GPU elapsed. Actual elapsed is preserved separately in existing completions. '
            'No prior costs are reset or reused as strict-original outcomes.',
        new_loader='tdyssey.native_rgb_data_eligibility.StrictOriginalRGBDataset',
        human_training_approval_required=True)
    target=P/'quarantine/strict-original-exclusion-manifest.json'
    with target.open('x',encoding='utf8') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(path=str(target),sha256=sha(target),
        phase_costs=phases,formal_or_supplement_data_present=result['formal_or_supplement_data_present'],
        inventories=rows),indent=2))


if __name__=='__main__':
    main()
