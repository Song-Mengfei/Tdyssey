"""Build report figures/videos from saved R3B diagnostic clips, never a model."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT)not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_playground_report import build_report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--no-videos',action='store_true');a=p.parse_args()
    print(json.dumps(build_report(a.stage,videos=not a.no_videos),ensure_ascii=False))
