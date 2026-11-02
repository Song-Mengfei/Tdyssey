import json,os,sys,time
from pathlib import Path
time.sleep(30)
Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'argv':sys.argv}))
