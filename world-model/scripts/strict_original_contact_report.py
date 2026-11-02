"""Saved-array-only strict-original report CLI; no physics/model execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resource-policy',type=Path,required=True)
    parser.add_argument('--storage-roots-json',type=Path,required=True)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--videos',action='store_true')
    args=parser.parse_args()
    from tdyssey.strict_original_contact_report import build_report
    roots=json.loads(args.storage_roots_json.read_text(encoding='utf-8-sig'))['roots']
    result=build_report(args.output,resource_policy=args.resource_policy,shared_roots=roots,
        videos=args.videos,source_root=args.source_root)
    print(json.dumps(dict(report=str((args.output/'REPORT.md').resolve()),
        all_attempts=result['all_attempts'],actual_controls=result['actual_controls'],
        valid_controls=result['valid_controls'],model_calls=0,optimizer_updates=0)))


if __name__=='__main__':main()
