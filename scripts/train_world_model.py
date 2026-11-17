"""Train a bounded official visual world-model pilot on CUDA; no new installs."""

import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey.world_model_protocol import WorldModelConfig
from tdyssey.tracking import add_tracking_arguments, tracking_from_args


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
    parser.add_argument('--max-bytes',type=int,default=64*1024**2)
    parser.add_argument('--checkpoint-every',type=int,default=250)
    parser.add_argument('--resume',type=Path,help='last.pt recovery; copy best.pt beside it; NEW output directory')
    add_tracking_arguments(parser)
    args=parser.parse_args()
    try:
        from dataclasses import fields
        config=WorldModelConfig(**{f.name:getattr(args,f.name) for f in fields(WorldModelConfig)})
    except ValueError as exc:parser.error(str(exc))
    from tdyssey.world_model_training import train
    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    report=train(args.catalog,args.output,config,tracking_from_args(args),args.resume)
    print(f"Completed {report['steps']} CUDA updates; best validation epoch {report['best_epoch']}; model-test untouched.")


if __name__=='__main__':main()
