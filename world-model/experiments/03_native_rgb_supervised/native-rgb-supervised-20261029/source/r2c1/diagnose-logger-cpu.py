"""Artifact-only CPU reproduction/proposed fix; never modifies frozen source."""
import sys,json,ast,difflib,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c1_training import event,safe_json,sha,atomic_json
OUT=Path(__file__).resolve().parent;FIX=OUT/'offline-repair';FIX.mkdir(exist_ok=True)
source=ROOT/'tdyssey/native_rgb_r2c1_training.py';before=source.read_text(encoding='utf-8')
assert sha(source)=='e882799be46ee02f7f1c0ffa5d69c2d3c826f2a06b92ab8cbb546b5feb9c683c'
info=json.loads((OUT/'model-work.json').read_text())['training_result']['checkpoint']
try:event(FIX/'legacy-should-not-be-written.jsonl','checkpoint',label='t0-reproduction',**info,model_seconds=0.)
except TypeError as exc:reason=str(exc)
else:raise AssertionError('Expected original argument binding error')
assert reason=="event() got multiple values for argument 'path'"
assert not (FIX/'legacy-should-not-be-written.jsonl').exists()
after=before.replace('def event(path,kind,**values):','def event(log_path,kind,**values):',1).replace('with Path(path).open("a",encoding="utf-8") as stream:','with Path(log_path).open("a",encoding="utf-8") as stream:',1)
assert before!=after;compile(after,str(source),'exec')
oldtree,newtree=ast.parse(before),ast.parse(after)
def without_event(tree):return ast.dump(ast.Module(body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='event'],type_ignores=[]),include_attributes=False)
assert without_event(oldtree)==without_event(newtree)
(FIX/'proposed-logger-parameter-fix.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/tdyssey/native_rgb_r2c1_training.py',tofile='b/tdyssey/native_rgb_r2c1_training.py')),encoding='utf-8')
# Independent stand-alone equivalent for binding verification; not injected
# into the running/frozen module and not used to resume or replay any work.
def fixed_event(log_path,kind,**values):
 import os
 with Path(log_path).open('a',encoding='utf-8') as stream:
  stream.write(json.dumps(safe_json(dict(kind=kind,**values)),ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
fixed=FIX/'proposed-helper-CPU-binding.jsonl';assert not fixed.exists()
fixed_event(fixed,'checkpoint',label='t0-reproduction',**info,model_seconds=0.)
record=json.loads(fixed.read_text());assert record['path']==info['path'] and record['sha256']==info['sha256'] and record['kind']=='checkpoint'
assert 'torch'not in sys.modules and sha(source)=='e882799be46ee02f7f1c0ffa5d69c2d3c826f2a06b92ab8cbb546b5feb9c683c'
atomic_json(FIX/'repair-record.json',dict(kind='R2C1-artifact-only-logger-diagnosis.v1',original_error_reproduced=True,error=reason,
 root_cause='checkpoint metadata includes path, which collides with event(path,kind,**values) positional log-file parameter before function body executes',
 proposed_change='Rename only the logging helper first argument path to log_path and its one Path(log_path) reference; retain JSON checkpoint path field.',
 original_source_sha256=sha(source),proposed_source_sha256=hashlib.sha256(after.encode()).hexdigest(),
 patch_file='proposed-logger-parameter-fix.patch',patch_sha256=sha(FIX/'proposed-logger-parameter-fix.patch'),
 every_non_event_function_AST_identical=True,proposed_source_syntax_valid=True,independent_binding_probe_passed=True,
 actual_source_modified=False,frozen_snapshot_modified=False,checkpoint_deserialized=False,model_calls=0,new_optimizer_attempts=0,
 no_retry_or_resume_performed=True,proposal_needs_parent_next_stage_instruction=True))
print(json.dumps(dict(error_reproduced=reason,patch_sha256=sha(FIX/'proposed-logger-parameter-fix.patch'),source_unchanged=True,new_model_calls=0)))
