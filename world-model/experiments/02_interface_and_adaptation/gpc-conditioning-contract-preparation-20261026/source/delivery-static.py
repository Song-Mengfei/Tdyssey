"""Final doc/message checks only, not another synthetic or native experiment."""
import json
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parent
root=base.parent.parent
started=time.monotonic()
rows=[]
for args in ([sys.executable,'-S','-B',str(root/'scripts/check_repo.py')],
        [sys.executable,'-S','-B',str(root/'scripts/check_messages.py'),'--commit-message',str(base/'result-commit-message.txt')]):
    result=subprocess.run(args,cwd=root,capture_output=True,text=True,timeout=20)
    rows.append(dict(arguments=args,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
    if result.returncode:break
with (base/'delivery-static.json').open('x',encoding='utf-8') as stream:
    json.dump(dict(seconds=time.monotonic()-started,checks=rows,torch=0,models=0,physics=0),stream,indent=2)
if result.returncode:raise SystemExit(result.returncode)
