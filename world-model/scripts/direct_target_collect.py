"""Approved finite acceptance collector; no training/model entry point."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tdyssey.direct_target_collection import collect

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--config-sha',required=True)
    a=p.parse_args();result=collect(a.config,a.config_sha,require_detached=True)
    return 1 if result['failed'] else 0
if __name__=='__main__':raise SystemExit(main())
