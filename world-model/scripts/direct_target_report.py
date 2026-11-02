"""Generate a guarded saved-only direct-target acceptance report."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdyssey.direct_target_report import generate_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = generate_report(args.root)
    print(json.dumps(dict(run_id=result['run_id'], all=result['all'], model_calls=0,
                          optimizer_updates=0, report_physics_calls=0), ensure_ascii=False))
