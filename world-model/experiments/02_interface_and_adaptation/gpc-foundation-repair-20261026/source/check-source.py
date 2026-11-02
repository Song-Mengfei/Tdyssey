"""Bounded stdlib-only source/message validation; imports no project runtime."""
import ast
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent
for filename in ('run-real.py','review-video.py','verify-offline.py'):
    ast.parse((base/filename).read_text(encoding='utf-8'),filename=filename)
for command in ([sys.executable,'-S','-B',str(root/'scripts/check_repo.py')],
        [sys.executable,'-S','-B',str(root/'scripts/check_messages.py'),'--commit-message',str(base/'commit-message.txt')]):
    result=subprocess.run(command,cwd=root,timeout=45,text=True,capture_output=True)
    print(result.stdout,end='')
    print(result.stderr,end='',file=sys.stderr)
    if result.returncode:raise SystemExit(result.returncode)
