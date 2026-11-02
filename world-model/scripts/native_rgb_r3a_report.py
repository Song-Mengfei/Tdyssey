"""CPU-only real-data audit and saved-state10Hz video entry point."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tdyssey.native_rgb_r3a_report import audit,videos,failure_videos


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--videos',action='store_true')
    args=parser.parse_args()
    if args.videos and args.source_root is None:parser.error('--videos requires --source-root')
    result,rows,_=audit(args.manifest,args.output)
    records=videos(args.manifest,rows,args.source_root,args.output/'videos') if args.videos else []
    failures=failure_videos(args.manifest.parent/'index.json',args.source_root,args.output/'failure-videos') if args.videos else []
    print(json.dumps(dict(trajectories=result['trajectories'],completed_controls=result['completed_controls'],
        max_penetration_worldpx=result['max_penetration_worldpx'],videos=len(records),failure_videos=len(failures),model_calls=0,physics_replay_controls=0)))


if __name__=='__main__':main()
