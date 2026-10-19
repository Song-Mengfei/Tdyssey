"""Compare max/final/sum reward from saved candidate trajectories; no simulation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey.world_model_diagnostics import candidate_score_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose a new file')
    payload=args.probe.read_bytes();report=json.loads(payload)
    if report['status']!='complete':parser.error('Probe is not complete')
    future=len(report['anchors'][0]['candidates'][0]['predicted_rewards'])
    result={'kind':'posthoc_score_views_from_saved_probe_no_new_physics',
            'source_probe_sha256':hashlib.sha256(payload).hexdigest(),
            'note':'Original selection used max reward; final/sum views are posthoc descriptive diagnostics.',
            'anchors':[{'episode_id':row['episode_id'],'anchor':row['anchor'],
                        'scores':candidate_score_diagnostics(row['candidates'],future)} for row in report['anchors']]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'anchors':len(result['anchors']),'new_physical_steps':0}))


if __name__=='__main__':main()
