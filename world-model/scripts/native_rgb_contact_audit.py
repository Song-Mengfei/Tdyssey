"""Extract actual contact/motion from the complete saved library, CPU only."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_contact_audit import extract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',default=str(ROOT/'artifacts/native-rgb-supervised-20261029/contact-audit-20261030'));a=p.parse_args()
    result=extract(a.stage)
    print(json.dumps({key:result[key] for key in ('closed','successful_trajectories','successful_controls','failed_summary','core_bytes_before_manifest','elapsed_seconds','new_physics_controls','model_calls')},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
