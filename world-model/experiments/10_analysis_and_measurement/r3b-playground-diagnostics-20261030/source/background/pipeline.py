"""One approved frozen-model diagnostic pass and saved-frame report, no training."""
from pathlib import Path
import argparse,json,os,sys,time,traceback,hashlib
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tdyssey/native_rgb_original_physics.py').is_file())
sys.path.insert(0,str(ROOT))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True,type=Path);args=parser.parse_args()
    os.chdir(ROOT)
    for key,value in(('SDL_VIDEODRIVER','dummy'),('SDL_AUDIODRIVER','dummy'),('PYGAME_HIDE_SUPPORT_PROMPT','1'),('PYTHONDONTWRITEBYTECODE','1')):os.environ.setdefault(key,value)
    from tdyssey.native_rgb_playground_diagnostic_runner import execute,write,utc
    config=json.loads(args.config.read_text(encoding='utf-8'));output=Path(config['output']);failure=None;report=None;start=time.perf_counter()
    try:
        proof=json.loads(Path(config['physics_validation_summary']).read_text(encoding='utf-8'))
        if proof['passed']is not True:raise RuntimeError('Actual original-source physics alignment must pass first')
        result=execute(args.config)
        if result:raise RuntimeError('Diagnostic inference stopped; preserve inference completion/failure without retry')
        write(output/'pipeline-status.json',dict(run_id=config['run_id'],phase='saved_frame_report',at_utc=utc(),model_calls_in_report=0))
        from tdyssey.native_rgb_playground_report import build_report
        report=build_report(output,videos=True)
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        write(output/'pipeline-completion.json',dict(run_id=config['run_id'],completed=failure is None,closed=True,
            report=report,failure=failure,wall_seconds=time.perf_counter()-start,finished_at_utc=utc(),optimizer_updates=0,
            next_action='Owner verifies saved figures, completes actual UI recording/browser checks, sends real parent report/receipt, then STOP waiting_user_approval.'))
    return 0 if failure is None else 2
if __name__=='__main__':raise SystemExit(main())
