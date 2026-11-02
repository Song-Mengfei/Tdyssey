"""Postprocess saved v2 measurements; never imports or invokes either decoder."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.fmean(values) if values else None


def equal_anchor(rows, key):
    branches, anchors = defaultdict(list), defaultdict(list)
    for row, value in rows:
        if value is not None:
            branches[row['anchor'], row.get('candidate'), row.get('horizon')].append(value)
    for (anchor, _, _), values in branches.items():
        anchors[anchor].append(mean(values))
    return mean([mean(values) for values in anchors.values()])


def distribution(pairs):
    values = sorted(value for _, value in pairs if value is not None)
    def percentile(q):
        if not values:
            return None
        index = (len(values)-1)*q
        lower = int(index)
        return values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(index-lower)
    return dict(measured_rows=len(values), anchor_equal_mean=equal_anchor(pairs, None),
                row_p50=percentile(.5), row_p90=percentile(.9), row_max=max(values) if values else None)


def csv_rows(path):
    def parse(value):
        if value == '':
            return None
        if value in ('True', 'False'):
            return value == 'True'
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return [{k:parse(v) for k,v in row.items()} for row in csv.DictReader(path.open(encoding='utf-8'))]


source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
if output.exists():
    raise SystemExit('Refusing to replace saved review metrics')
rows = [json.loads(line) for line in (source/'per-image.jsonl').read_text(encoding='utf-8').splitlines()]
metrics = ('position_error_pixels', 'angle_error_degrees', 'pusher_error_pixels', 'coverage_absolute_error')
groups = defaultdict(list)
for row in rows:
    groups[row['source'],row['shape'],row['horizon'] if row['source'] in ('forecast','physical_future') else None].append(row)
result = dict(development_only=True, decoding_or_model_calls=0,
              quantiles='Descriptive row quantiles; repeated noises/frames are correlated, no confidence interpretation',
              interval='Fixed development engineering assumption, not a generated-image guarantee',
              groups=[], calibration=[], sequences=[], baselines=[], failure_counts={})
for (kind, shape, horizon), members in groups.items():
    group = dict(source=kind,shape=shape,horizon=horizon,expected_rows=len(members),
                 available_rows=sum(r['input_available'] for r in members),versions={},common_subset={})
    for version in ('v1','v2'):
        stats = {field:sum(r[version]['view'][field] for r in members)
                 for field in ('object_valid','shape_valid','pusher_valid','coverage_valid')}
        stats['termination'] = dict(Counter(r[version]['view']['termination'] for r in members))
        stats['reasons'] = dict(Counter(r[version]['view']['reason'] for r in members))
        stats['errors'] = {metric:distribution([(r,r[version]['errors'][metric]) for r in members]) for metric in metrics}
        intervals = [(r,r[version]['view'].get('coverage_interval')) for r in members
                     if r[version]['view']['coverage_valid'] and r[version]['view'].get('coverage_interval')
                     and r[version]['errors']['truth_coverage'] is not None]
        stats['interval_containment'] = dict(eligible_rows=len(intervals),contained_rows=sum(
            bounds[0]<=r[version]['errors']['truth_coverage']<=bounds[1] for r,bounds in intervals))
        group['versions'][version] = stats
    for metric in metrics:
        common = [r for r in members if all(r[v]['errors'][metric] is not None for v in ('v1','v2'))]
        group['common_subset'][metric] = {v:distribution([(r,r[v]['errors'][metric]) for r in common]) for v in ('v1','v2')}
    result['groups'].append(group)
    if kind=='calibration':
        for r in members:
            result['calibration'].append(dict(id=r['id'],truth=r['truth'],versions={v:dict(
                view=r[v]['view'],errors=r[v]['errors']) for v in ('v1','v2')}))
seq = csv_rows(source/'sequence-scores.csv')
for shape in ('L','T'):
    for horizon in (1,2,3,4):
        for version in ('v1','v2'):
            members = [r for r in seq if r['shape']==shape and r['horizon']==horizon and r['version']==version]
            result['sequences'].append(dict(shape=shape,horizon=horizon,version=version,expected_rows=len(members),
                raw_valid=sum(r['raw_endpoint_valid'] for r in members),
                sequence_eligible=sum(r['sequence_score_eligible'] for r in members),
                certain_absorptions=sum(r['predicted_absorbing'] is True for r in members),
                unknown_history=sum(r['unknown_history_count']>0 for r in members)))
baseline = csv_rows(source/'baselines.csv')
for shape in ('L','T'):
    for horizon in (1,4):
        members = [r for r in baseline if r['shape']==shape and r['horizon']==horizon]
        result['baselines'].append(dict(shape=shape,horizon=horizon,expected_rows=len(members),
            coverage=distribution([(r,r['privileged_observed_pose_persistence_error']) for r in members]),
            position=distribution([(r,r['privileged_observed_pose_position_error']) for r in members]),
            angle=distribution([(r,r['privileged_observed_pose_angle_error']) for r in members])))
for version in ('v1','v2'):
    forecast = [r for r in rows if r['source']=='forecast']
    result['failure_counts'][version] = dict(
        reasons=dict(Counter(r[version]['view']['reason'] for r in forecast)),
        shape_reasons=dict(Counter((r[version].get('measurement') or {}).get('shape',{}).get('reason','unreported') for r in forecast)) if version=='v2' else None,
        pusher_reasons=dict(Counter((r[version].get('measurement') or {}).get('pusher',{}).get('reason','unreported') for r in forecast)) if version=='v2' else None)
result['input_sha256'] = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in (
    'per-image.jsonl','summary.json','sequence-scores.csv','baselines.csv','ranking-summary.csv')}
result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
output.parent.mkdir(parents=True,exist_ok=True)
with output.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(output=str(output),bytes=output.stat().st_size)))
