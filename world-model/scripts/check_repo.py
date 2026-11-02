"""Run source-only release validation with the standard library."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from check_release import main

if __name__=='__main__':
    raise SystemExit(main())
