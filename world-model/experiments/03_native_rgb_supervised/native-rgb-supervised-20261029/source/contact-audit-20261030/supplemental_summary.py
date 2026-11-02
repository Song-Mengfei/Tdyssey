"""Small saved-core duration and letter-by-region supplements; no simulation."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import numpy as np

STAGE = Path(__file__).resolve().parent


def quantiles(values):
    values = np.asarray(values, float)
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), **dict(zip(("min", "p10", "median", "p90", "p95", "max"),
        map(float, np.quantile(values, [0,.1,.5,.9,.95,1])))))


def main():
    with np.load(STAGE/'core/controls.npz', allow_pickle=False) as z:
        c = {k:z[k] for k in z.files}
    metadata = json.loads((STAGE/'core/trajectories.json').read_text())['trajectories']
    with np.load(STAGE/'statistics/input-exposure.npz', allow_pickle=False) as z:
        selected = z['selected_target_control_rows']
    with gzip.open(STAGE/'core/contact-runs.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
        recorded_runs = list(csv.DictReader(stream))
    durations = {}
    for split, code in (('train',0),('seen_val',1)):
        rows = np.flatnonzero(c['split'] == code)
        lengths = [int(r['length_substeps']) for r in recorded_runs if r['split'] == split]
        assert sum(lengths) == int(c['contact_substeps'][rows].sum())
        positive_lengths = []
        for trajectory in np.unique(c['trajectory_index'][rows]):
            ix = np.flatnonzero(c['trajectory_index'] == trajectory)
            positive = (c['post_ao'][ix] & c['post_ao_impulse_known'][ix]
                        & (c['post_ao_impulse_max_norm'][ix] > 1e-8)).reshape(-1)
            current = 0
            for flag in list(positive)+[False]:
                if flag:
                    current += 1
                elif current:
                    positive_lengths.append(current)
                    current = 0
        histogram = {}
        lengths_array = np.asarray(lengths)
        for label, mask in (('1',lengths_array == 1),('2-4',(lengths_array>=2)&(lengths_array<=4)),
                            ('5-9',(lengths_array>=5)&(lengths_array<=9)),
                            ('10-49',(lengths_array>=10)&(lengths_array<=49)),('50+',lengths_array>=50)):
            histogram[label] = dict(runs=int(mask.sum()), contact_seconds=float(lengths_array[mask].sum()*.01))
        near = c['motion_class'][rows] == 0
        ao, wall = c['post_ao'][rows], c['post_ow'][rows]
        norms = c['post_ao_impulse_max_norm'][rows]
        known = ao & c['post_ao_impulse_known'][rows]
        durations[split] = dict(acquired_substeps=len(rows)*10, acquired_seconds=len(rows)*.1,
            postsolve_contact_run_seconds=quantiles(np.asarray(lengths)*.01),
            positive_impulse_run_seconds_epsilon_1e_8=quantiles(np.asarray(positive_lengths)*.01),
            contact_run_substep_histogram=histogram,
            postsolve_contact_seconds=float(ao.sum()*.01),
            positive_impulse_contact_seconds=float((known&(norms>1e-8)).sum()*.01),
            known_zero_impulse_substeps=int((known&(norms==0)).sum()),
            unknown_impulse_substeps=int((ao&~known).sum()),
            known_contact_impulse_norm_quantiles=quantiles(norms[known]),
            AO_in_nearstatic_control_seconds=float(ao[near].sum()*.01),
            simultaneous_AO_OW_in_nearstatic_control_seconds=float((ao[near]&wall[near]).sum()*.01),
            AO_without_simultaneous_OW_in_nearstatic_control_seconds=float((ao[near]&~wall[near]).sum()*.01))

    joint = []
    for name, rows in (('TRAIN_all',np.flatnonzero(c['split']==0)),('TRAIN_selected',selected),
                       ('VAL_all',np.flatnonzero(c['split']==1))):
        for letter in 'TLOCXB':
            for region in ('center','edges','corners'):
                ix = []
                for i in rows:
                    m = metadata[int(c['trajectory_index'][i])]
                    observed_region = ('center' if m['initial_space']=='center' else
                                       'corners' if '-' in m['initial_space'] else 'edges')
                    if m['letter']==letter and observed_region==region:
                        ix.append(int(i))
                ix = np.asarray(ix, np.int64)
                contact = c['contact_substeps'][ix]>0
                obvious = c['motion_class'][ix]==2
                near = c['motion_class'][ix]==0
                positive = contact & (c['signed_rotation_deg'][ix]>=5)
                negative = contact & (c['signed_rotation_deg'][ix]<=-5)
                moving = contact & obvious
                joint.append(dict(population=name,letter=letter,regiontype=region,controls=len(ix),
                    trajectories=len(np.unique(c['trajectory_index'][ix])),
                    contact_controls=int(contact.sum()),contact_substeps=int(c['contact_substeps'][ix].sum()),
                    nearstatic_controls=int(near.sum()),contact_obvious_controls=int(moving.sum()),
                    contact_obvious_trajectory_support=len(np.unique(c['trajectory_index'][ix[moving]])),
                    contact_positive_rotation_controls=int(positive.sum()),
                    contact_positive_rotation_trajectory_support=len(np.unique(c['trajectory_index'][ix[positive]])),
                    contact_negative_rotation_controls=int(negative.sum()),
                    contact_negative_rotation_trajectory_support=len(np.unique(c['trajectory_index'][ix[negative]]))))
    with (STAGE/'letter-region-support.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(joint[0]));writer.writeheader();writer.writerows(joint)
    result = dict(kind='contact-audit-duration-and-joint-support.v1',by_split=durations,
        joint_letter_region_rows=joint,
        timing='Runs are contiguous observed postsolve physics intervals, not exact within-step collision onset.',
        wall_note='Concurrent flags are observations. Nearstatic is classified over the containing control; wall contact alone does not establish a blocking cause.',
        impulse_note='Norms describe recorded event-vector existence/maxima, never summed point impulse or net physical force.',
        model_calls=0,physics_steps=0)
    (STAGE/'supplemental-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(durations,indent=2))


if __name__=='__main__':
    main()
