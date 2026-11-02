"""Bind an actual completed912-scene CPU gate into an executable CPU proposal."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.native_rgb_r3b_proposal_check import build_proposal,write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=ROOT/'artifacts/native-rgb-supervised-20261029/r3a'
    p.add_argument('--manifest',type=Path,default=base/'manifest.json')
    p.add_argument('--gate',type=Path,default=base/'data-interface/interface-gate.json')
    p.add_argument('--outdir',type=Path,default=base/'r3b-proposal')
    p.add_argument('--config',type=Path,default=ROOT/'configs/native_rgb_r3b_proposal.json')
    a=p.parse_args();result=build_proposal(a.manifest,a.gate,a.outdir,a.config)
    write(a.outdir/'CPU-validation.json',result)
    print(json.dumps(result,ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
