import json,os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'argv':sys.argv}))
