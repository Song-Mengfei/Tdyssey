"""Start/reuse exactly the requested local model and report real startup errors."""
from pathlib import Path
import argparse,json,os,socket,subprocess,sys,time,urllib.request,webbrowser
from runtime_config import APP,REPO,SERVICE,load_runtime,validate_assets


def health(url):
    try:
        with urllib.request.urlopen(url+'/api/health',timeout=2) as response:return json.load(response)
    except (OSError,ValueError):return None


def matches(existing,config):
    return bool(existing and existing.get('ready') and existing.get('service')==SERVICE
                and existing.get('repository')==str(REPO.resolve())
                and existing.get('model_id')==config['model_id']
                and existing.get('checkpoint_sha256')==config['profile']['checkpoint_sha256'])


def interpreter():
    if os.name!='nt':return sys.executable
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];kernel.GetModuleFileNameW.restype=wintypes.DWORD
    buffer=ctypes.create_unicode_buffer(32768)
    if not kernel.GetModuleFileNameW(None,buffer,len(buffer)):raise ctypes.WinError(ctypes.get_last_error())
    return str(Path(buffer.value).resolve())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('hmf500','hmf3000'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--open',action='store_true')
    parser.add_argument('--check',action='store_true',help='Verify external assets without loading a model or starting a server')
    args=parser.parse_args(argv);config=load_runtime(args.model,args.config)
    if not 1<=args.port<=65535:parser.error('Port must be between1 and65535')
    receipt=validate_assets(config)
    if args.check:print(json.dumps(receipt,ensure_ascii=False));return 0
    runtime=REPO/'artifacts/r3b-playground';runtime.mkdir(parents=True,exist_ok=True)
    url=f'http://127.0.0.1:{args.port}';existing=health(url)
    if matches(existing,config):print(json.dumps(dict(url=url,reused=True,model=config['model_id'])),flush=True)
    else:
        if existing:
            raise RuntimeError('This port serves another model/application. Close that app first, or choose --port. No existing process was stopped.')
        # Detect an occupied/unresponsive port before constructing another GPU model.
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            try:probe.bind(('127.0.0.1',args.port))
            except OSError as exc:raise RuntimeError('Local port is occupied or unavailable. Close the previous app or choose --port.') from exc
        command=[interpreter(),'-X','utf8','-B',str(APP/'server.py'),'--port',str(args.port),'--model',config['model_id'],'--config',str(config['config_path'])]
        env=os.environ.copy();env['__PYVENV_LAUNCHER__']=sys.executable;env['PYTHONUNBUFFERED']='1'
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):env[key]='1'
        flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB) if os.name=='nt' else 0
        startup=None
        if os.name=='nt':
            startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        error_file=runtime/'stderr.log'
        with (runtime/'stdout.log').open('ab') as out,error_file.open('ab') as err:
            offset=error_file.stat().st_size
            process=subprocess.Popen(command,cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags,close_fds=True,startupinfo=startup)
        print(json.dumps(dict(pid=process.pid,url=url,starting=True,model=config['model_id'])),flush=True)
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            current=health(url)
            if matches(current,config):break
            if process.poll() is not None:
                with error_file.open('rb') as file:file.seek(offset);detail=file.read()[-8192:].decode('utf-8',errors='replace')
                raise RuntimeError('Server exited. Actual error:\n'+detail+'\nFull log: '+str(error_file))
            time.sleep(.5)
        else:raise RuntimeError('Server is still starting. Inspect '+str(error_file)+' before retrying; no duplicate model was launched.')
        print(json.dumps(dict(url=url,ready=True,model=config['model_id'])),flush=True)
    if args.open:webbrowser.open(url)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:
        print('启动失败：'+str(exc),file=sys.stderr);raise SystemExit(1)
