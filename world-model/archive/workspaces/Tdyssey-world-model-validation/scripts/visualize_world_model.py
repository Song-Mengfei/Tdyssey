"""Official original-colour H1 viewer; explicit assets, no legacy T1 fallback."""
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
    parser.add_argument('--checkpoint',type=Path,help='Explicit official denoiser.pt; never inferred from a video/default')
    parser.add_argument('--checkpoint-sha256',help='Current weight-file SHA256, not the sidecar init_sha256')
    parser.add_argument('--checkpoint-source',default=None,help='Human-readable origin/run label; initial weights retain their released-model label')
    parser.add_argument('--model-label',default='Official T',help='Adapted-model button label, for example Raw T+L; does not alter checkpoint identity')
    parser.add_argument('--sidecar',type=Path,help='Official denoiser.json (default: next to checkpoint)')
    parser.add_argument('--sidecar-sha256')
    parser.add_argument('--checkpoint-kind',choices=('official','initial'),default='official',help='Initial is an explicit released-model control, never a teammate-trained model')
    parser.add_argument('--initial-checkpoint',type=Path,help='Optional released initial control for C comparison')
    parser.add_argument('--gpc-root',type=Path,default=Path(os.environ.get('TDYSSEY_GPC_ROOT',ROOT/'third_party/gpc')))
    parser.add_argument('--source-root',type=Path,default=Path(os.environ.get('TDYSSEY_OFFICIAL_SOURCE',ROOT/'third_party/diffusion_policy')))
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--seed',type=int,default=5478,help='Sampling seed; local recorded-hold scene setup is preserved')
    parser.add_argument('--seconds',type=int,default=1800)
    parser.add_argument('--check-assets',action='store_true',help='Read metadata/hash only; never create a simulator, import Torch or open a window')
    args=parser.parse_args(argv)
    if not args.model_label.strip() or len(args.model_label)>48 or any(c in args.model_label for c in '\r\n'):
        parser.error('Choose a nonempty model label of at most48 characters')
    from tdyssey.official_world_model import model_profile,inspect_profile,RAW_CONTRACT
    try:
        primary=model_profile(args.checkpoint_kind,args.checkpoint,expected_sha256=args.checkpoint_sha256,
            sidecar=args.sidecar,expected_sidecar_sha256=args.sidecar_sha256,label=args.checkpoint_source)
        if args.checkpoint_kind=='official':
            profiles={'adapted':primary}
            if args.initial_checkpoint is not None:
                profiles['frozen']=model_profile('initial',args.initial_checkpoint)
            selected='adapted'
        else:
            profiles={'frozen':primary}
            selected='frozen'
        inspections={role:inspect_profile(profile) for role,profile in profiles.items()}
        sources={name:dict(path=str(path.resolve()),present=path.is_dir()) for name,path in
                 (('gpc',args.gpc_root),('official_environment',args.source_root))}
        ready=inspections[selected]['status']=='available' and all(item['present'] for item in sources.values())
        if args.check_assets:
            print(json.dumps(dict(profile='official-original-color',ready=ready,selected_role=selected,
                contract=RAW_CONTRACT,assets=inspections,sources=sources,model_loads=0,physics_controls=0),indent=2))
            return 0 if ready else 2
        if not ready:
            reason=inspections[selected]['reason'] if inspections[selected]['status']!='available' else 'Pinned upstream source directory missing; supply --gpc-root and --source-root'
            parser.error(reason)
        if not 1<=args.seconds<=1800 or not 0<=args.seed<2**32:
            parser.error('Choose seconds1..1800 and uint32 sampling seed')
    except (OSError,ValueError) as error:
        parser.error(str(error))
    # Imports capable of UI/simulation/model work occur only after explicit assets pass.
    from scripts.manual_world_model import App
    from tdyssey.manual_world_model import ManualSession,make_environment,OFFICIAL_DOMAIN
    from tdyssey.manual_official_predictor import OfficialPredictor
    import pygame
    output=args.output or Path(os.environ.get('TDYSSEY_WM_OUTPUT',ROOT/'artifacts/manual-official-world-model'))/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(output.resolve()/'runtime-cache'))
    session=ManualSession(partial(make_environment,source_root=args.source_root),image_domain=OFFICIAL_DOMAIN)
    session.noise_seed=args.seed
    session.configure(role=selected,horizon=1)
    app=None
    try:
        predictor=OfficialPredictor(profiles,args.gpc_root,inspections=inspections)
        app=App(output.resolve(),args.seconds,session=session,predictor=predictor,
            role_labels={'frozen':'Initial','adapted':args.model_label},monitor_host=False,
            available_roles=tuple(profiles))
        clock=pygame.time.Clock()
        while not app.closed:
            app.poll();app.draw();clock.tick(30)
    finally:
        if app is not None:app.close()
        else:session.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
