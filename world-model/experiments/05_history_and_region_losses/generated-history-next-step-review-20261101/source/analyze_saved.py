"""Saved-only explanatory statistics; no model, simulator, optimizer or rerender."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tdyssey.native_rgb_evaluation import palette_masks

RUN = ROOT / 'artifacts/generated-history-ab-20261101'
OUT = Path(__file__).parent
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

pool = read(RUN/'pool-index.json')['segments']
index = {r['sample_id']:r for r in read(ROOT/'artifacts/accelerated-rgb-run-20261031/selection/train-index.json')}
phase_rows = defaultdict(list)
for segment in pool:
    for phase, sid in enumerate(segment['supervised_H1_ids'], 1):
        row = index[sid]
        phase_rows[('ALL', phase)].append(row)
        phase_rows[(row['letter'], phase)].append(row)
phase_stats=[]
for (letter, phase), rows in sorted(phase_rows.items()):
    translation=np.array([r['translation_imagepx'] for r in rows])
    rotation=np.array([r['abs_rotation_deg'] for r in rows])
    phase_stats.append(dict(letter=letter, phase=phase, n=len(rows),
        contact_motion=sum(bool(r['diagnostic_contact_and_motion']) for r in rows),
        rotation_ge3deg=int((rotation>=3).sum()), translation_ge1px=int((translation>=1).sum()),
        translation_mean=float(translation.mean()), translation_p90=float(np.percentile(translation,90)),
        rotation_mean=float(rotation.mean()), rotation_p90=float(np.percentile(rotation,90))))

summary=read(RUN/'report/summary.json')
diag_groups=[r for r in summary['groups'] if r['scope']=='diagnostic32' and r['split']=='seen_val'
             and r['horizon'] in (1,4,8,16)]
diag_compact=[dict(mode=r['mode'], subset=r['subset'], arm=r['arm'], h=r['horizon'], n=r['denominator'],
    iou=r['metrics']['raw_mask_iou']['mean'],position=r['metrics']['object_position_error_image_px']['mean']) for r in diag_groups]

formal={arm:read(RUN/f'evaluation/{arm}/formal/metrics.json') for arm in ('baseline','A','B')}
hole_sequences=[]
for item in formal['B']['raw_files']:
    if item['letter']!='B': continue
    sid=item['sample_id']; seq={}
    for arm,content in formal.items():
        rows=sorted((r for r in content['rows'] if r['sample_id']==sid and r['method']==arm),key=lambda r:r['horizon'])
        assert len(rows)==16 and [r['horizon'] for r in rows]==list(range(1,17))
        mismatch=[r['horizon'] for r in rows if r['metrics']['holes_difference'] not in (None,0)]
        seq[arm]=dict(first_visible_hole_mismatch=min(mismatch) if mismatch else None,
            holes_by_h=[r['metrics']['predicted_holes'] for r in rows],
            truth_holes_by_h=[r['metrics']['truth_holes'] for r in rows])
    hole_sequences.append(dict(sample_id=sid,sequence=seq))

calibration=read(ROOT/'artifacts/native-rgb-20261028/frozen-renderer-calibration.json')
truth_regions=[]
for item in formal['B']['raw_files']:
    if item['letter'] not in 'OCB':continue
    with np.load(item['raw_path'],allow_pickle=False) as z: rgb=z['rgb']
    for h in (1,4,8,16):
        im=rgb[item['anchor']+h]
        gray,blue=palette_masks(im,calibration)
        labels,count=ndimage.label(~np.pad(gray,1),structure=ndimage.generate_binary_structure(2,1))
        outer=int(labels[0,0]);sizes=np.bincount(labels.ravel())
        holes=[int(sizes[i]) for i in range(1,count+1) if i!=outer and sizes[i]>=2]
        truth_regions.append(dict(sample_id=item['sample_id'],letter=item['letter'],h=h,
             gray_pixels=int(gray.sum()),blue_pixels=int(blue.sum()),hole_sizes=holes,
             gray_fraction=float(gray.mean()),visible_holes_fraction=sum(holes)/9216))
region_summary={}
for letter in 'OCB':
    rr=[r for r in truth_regions if r['letter']==letter]
    region_summary[letter]=dict(frames=len(rr),gray_pixels_min=min(r['gray_pixels'] for r in rr),
        gray_pixels_max=max(r['gray_pixels'] for r in rr),gray_fraction_mean=float(np.mean([r['gray_fraction'] for r in rr])),
        hole_pixels_min=min(sum(r['hole_sizes']) for r in rr),hole_pixels_max=max(sum(r['hole_sizes']) for r in rr))

result=dict(kind='generated-history-explanation-saved-only.v1',model_calls=0,physics_steps=0,
    changed_training_settings=False,phase_stats=phase_stats,diagnostic_groups=diag_compact,
    formal_B_hole_sequences=hole_sequences,formal_truth_region_summary=region_summary,formal_truth_regions=truth_regions,
    limits=['Pixel counts do not measure loss or gradient dominance.',
            'Hole counts are visible-pixel topology, affected by occlusion and rasterization; not a rigidity certificate.',
            'Diagnostic trajectories include feedback-recorded actions, unlike the committed four-step training segments.',
            'Post-hoc explanatory statistics; old evaluation definitions, outputs and data remain unchanged.'])
(OUT/'SAVED_ANALYSIS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(phase_stats=[r for r in phase_stats if r['letter']=='O'],
    formal_B_hole_onsets={r['sample_id']:{a:s['first_visible_hole_mismatch'] for a,s in r['sequence'].items()} for r in hole_sequences},
    formal_truth_region_summary=region_summary),ensure_ascii=False,indent=2))
