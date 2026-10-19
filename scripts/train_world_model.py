"""Train a bounded visual L world-model pilot on CUDA; no installation/downloads."""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey.world_model_protocol import WorldModelConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--future',type=int,default=4)
    parser.add_argument('--max-steps',type=int,default=750)
    parser.add_argument('--seconds',type=int,default=300)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--learning-rate',type=float,default=.0003)
    args=parser.parse_args()
    try:
        config=WorldModelConfig(**{k:v for k,v in vars(args).items() if k not in ('catalog','output')})
    except ValueError as exc:parser.error(str(exc))
    from tdyssey.world_model_training import train
    report=train(args.catalog,args.output,config)
    print(f"Completed {report['steps']} CUDA updates; best validation epoch {report['best_epoch']}; model-test untouched.")


if __name__=='__main__':main()
