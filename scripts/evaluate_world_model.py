"""Evaluate a saved visual model on CUDA; validation by default, no model updates."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=('validation','model_test'),default='validation')
    parser.add_argument('--seconds',type=int,default=120)
    args=parser.parse_args()
    if not 1<=args.seconds<=600:parser.error('Use a wall-time cap of 1–600 seconds')
    if args.output.exists():parser.error('Output exists; choose a new file')
    from tdyssey.dataset import digest,json_bytes
    from tdyssey.training_data import WindowSpec
    from tdyssey.world_model_data import VisualDynamicsDataset
    from tdyssey.world_model_training import cuda_setup,evaluate,load_checkpoint
    started=time.monotonic()
    device=cuda_setup(0)
    model,payload=load_checkpoint(args.checkpoint,device)
    if digest(args.catalog.read_bytes())!=payload['catalog_sha256']:
        parser.error('Catalog differs from the frozen checkpoint training catalog')
    dataset=VisualDynamicsDataset(args.catalog,args.split,WindowSpec(2,payload['config']['future']))
    if not len(dataset):parser.error('Selected split has no windows')
    report={'kind':'saved_visual_world_model_evaluation','split':args.split,
            'checkpoint_sha256':digest(args.checkpoint.read_bytes()),'catalog_sha256':payload['catalog_sha256'],
            'trained_epoch':payload['epoch'],'metrics':evaluate(model,dataset,payload['normalization'],deadline=started+args.seconds),
            'elapsed_seconds':time.monotonic()-started,'model_test_used':args.split=='model_test'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:stream.write(json_bytes(report))
    print(json.dumps({'status':'complete','split':args.split,'windows':report['metrics']['windows']}))


if __name__=='__main__':main()
