"""Start/reuse the local inference server independently of the launcher."""
from pathlib import Path
import sys,os,json,subprocess,time,urllib.request,webbrowser
APP=Path(__file__).resolve().parent;REPO=APP.parents[1];RUNTIME=REPO/'artifacts/r3b-playground'
RUNTIME.mkdir(parents=True,exist_ok=True)
def healthy(url):
    try:
        with urllib.request.urlopen(url+'/api/health',timeout=1) as r:d=json.load(r)
        return d.get('service')=='tdyssey-r3b-playground-v1' and d.get('ready')
    except Exception:return False
url='http://127.0.0.1:8767'
if healthy(url):
    print(json.dumps({'url':url,'reused':True}),flush=True)
else:
    flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_BREAKAWAY_FROM_JOB) if os.name=='nt' else 0
    env=os.environ.copy();env['__PYVENV_LAUNCHER__']=sys.executable;env['PYTHONUNBUFFERED']='1'
    exe=getattr(sys,'_base_executable',sys.executable)
    with (RUNTIME/'stdout.log').open('ab') as out,(RUNTIME/'stderr.log').open('ab') as err:
        p=subprocess.Popen([exe,'-X','utf8','-B',str(APP/'server.py')],cwd=str(REPO),env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=flags,close_fds=True)
    print(json.dumps({'pid':p.pid,'url':url,'starting':True}),flush=True)
    # Readiness deadline is an engineering check, not a model-work or training cap.
    for _ in range(120):
        if healthy(url):break
        if p.poll() is not None:raise RuntimeError('Server exited. See artifacts/r3b-playground/stderr.log')
        time.sleep(.5)
    else:raise RuntimeError('Server is still starting. Inspect stdout/stderr before retrying; do not start a duplicate.')
    print(json.dumps({'url':url,'ready':True}),flush=True)
if '--open' in sys.argv:webbrowser.open(url)
