"""Raw T/L H1 click-to-push viewer; explicit user-provided trained assets."""
import argparse
from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--checkpoint',type=Path,default=os.environ.get('TDYSSEY_WM_CHECKPOINT'))
    parser.add_argument('--checkpoint-sha256',default=os.environ.get('TDYSSEY_WM_SHA256'))
    parser.add_argument('--sidecar',type=Path,default=os.environ.get('TDYSSEY_WM_SIDECAR'))
    parser.add_argument('--sidecar-sha256')
    parser.add_argument('--gpc-root',type=Path,default=os.environ.get('TDYSSEY_GPC_ROOT'))
    parser.add_argument('--source-root',type=Path,default=os.environ.get('TDYSSEY_OFFICIAL_SOURCE'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=5478)
    parser.add_argument('--seconds',type=int,default=1800)
    parser.add_argument('--check-assets',action='store_true',help='Read hashes/sidecar only; no Torch, model, simulator or window')
    args=parser.parse_args(argv)
    from tdyssey.official_world_model import model_profile,inspect_profile,OfficialWorldModel,RAW_CONTRACT
    if any(value is None for value in (args.checkpoint,args.checkpoint_sha256,args.sidecar,args.gpc_root,args.source_root)):
        parser.error('Supply checkpoint, SHA256, sidecar and both source roots (flags or TDYSSEY_* variables)')
    if not 0<=args.seed<2**32 or not 1<=args.seconds<=1800:parser.error('Use uint32 seed and seconds1..1800')
    try:
        profile=model_profile(args.checkpoint,args.checkpoint_sha256,sidecar=args.sidecar,
            expected_sidecar_sha256=args.sidecar_sha256)
        inspected=inspect_profile(profile)
        sources={name:dict(path=str(path.resolve()),present=path.is_dir()) for name,path in
            (('gpc',args.gpc_root),('official',args.source_root))}
        ready=inspected['ready'] and all(item['present'] for item in sources.values())
        if args.check_assets:
            print(json.dumps(dict(ready=ready,asset=inspected,sources=sources,contract=RAW_CONTRACT,
                model_loads=0,forecasts=0,physics=0),indent=2))
            return 0 if ready else 2
        if not ready:parser.error(inspected['reason'] or 'Source root is missing')
    except (OSError,ValueError) as error:parser.error(str(error))
    from tdyssey.manual_world_model import ManualSession,ModelPredictor,make_environment
    from scripts.manual_world_model import App
    import pygame
    output=args.output or Path('artifacts/manual-world-model')/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    model=OfficialWorldModel(profile,args.gpc_root,execute=True);session=app=None
    try:
        session=ManualSession(partial(make_environment,source_root=args.source_root),noise_seed=args.seed)
        app=App(output,args.seconds,session=session,predictor=ModelPredictor(model))
        clock=pygame.time.Clock()
        while not app.closed:
            app.poll();app.draw();clock.tick(30)
    finally:
        if app is not None:app.close()
        else:
            if session is not None:session.close()
            model.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
