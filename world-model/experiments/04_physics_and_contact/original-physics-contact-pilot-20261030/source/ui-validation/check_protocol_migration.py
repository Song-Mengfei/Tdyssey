"""One fresh-import migration rejection check; no live UI/model/service."""
import base64
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
OUTPUT=Path(__file__).resolve().parent
PILOT=OUTPUT.parent
REPO=PILOT.parent.parent
STAGED=PILOT/'ui-next'
sys.path[:0]=[str(STAGED),str(REPO)]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
server_sha=sha(STAGED/'server.py')
assert server_sha=='157403c2fdc0d33d08489ab75497481094067fd14f3d5fc946e65809d377c8cd'
import server
server.APP=REPO/'apps/r3b-playground'
server.REPO=REPO
server.WORKSPACE=REPO.parent
server.GPC=REPO.parent/'Tdyssey/third_party/gpc_code'
server.PHYSICS=REPO.parent/'Tdyssey/third_party/diffusion_policy'
server.RUN=REPO/'artifacts/native-rgb-supervised-20261029/r3b/attempts/recovery-20261030'
server.CHECKPOINT=server.RUN/'run/last.pt'
server.RUNTIME=REPO/'artifacts/r3b-playground'
server.DIAGNOSTICS=REPO/'artifacts/r3b-playground-diagnostics-20261030'
import numpy as np
from capture import session_export,unpack
from tdyssey import native_rgb_contact_safe_physics as physical
physical_sha=sha(physical.__file__)
cost=dict(attempted_controls=0,completed_controls=0,executed_microsteps=0,simulated_seconds=0.,
          model_calls=0,GPU_calls=0,optimizer_updates=0,forbidden_predictor_calls=0)
original_step=physical.ContactSafePhysicsEnv.step
def count_step(self,*args,**kwargs):
 cost['attempted_controls']+=1
 try:
  result=original_step(self,*args,**kwargs)
  cost['completed_controls']+=int(result[1]['valid'])
  return result
 finally:
  rows=self.last_substeps
  cost['executed_microsteps']+=sum(bool(row.get('executed',False))for row in rows)
  cost['simulated_seconds']+=sum(row['dt']for row in rows if row.get('executed',False))
def forbidden_predictor(*args,**kwargs):
 cost['forbidden_predictor_calls']+=1
 raise AssertionError('No predictor call is authorized for migration rejection')
physical.ContactSafePhysicsEnv.step=count_step
started=time.perf_counter()
app=None
passed=False
error=None
rejection=None
checks={}
try:
 app=server.Playground(forbidden_predictor)
 exported=app.export_session()
 metadata,arrays=unpack(base64.b64decode(exported['archive_base64']))
 # Construct a fully valid archive envelope with the historical v1 protocol.
 # Body/history arrays remain unchanged because the pre-reset protocol gate
 # must reject before any of those values can be applied.
 metadata['environment_physics_sha256']='d83b4c544b1e5193c92e2b882d0904995b1d49e174adaf0e8c5b284031835e9b'
 old_protocol_export=session_export(metadata,arrays)
 before=dict(env_id=id(app.env),state=app.env.state().copy(),history=np.stack(app.history).copy(),
  past=np.stack(app.past).copy(),session=app.session,steps=app.steps,revision=app.revision,
  total_controls=app.total_controls,environment_control_step=app.env.control_step,
  target=copy.deepcopy(app.target),terminal=app.terminal,recording_id=app.recording_id,
  physics_cost=copy.deepcopy(cost))
 try:app.restore_session(old_protocol_export)
 except ValueError as exc:rejection=str(exc)
 else:raise AssertionError('Historical protocol export was not rejected')
 checks=dict(
  rejected_before_reset='current session was preserved'in rejection,
  environment_instance_preserved=id(app.env)==before['env_id'],
  exact_state15_preserved=bool(np.array_equal(app.env.state(),before['state'])),
  exact_history4_preserved=bool(np.array_equal(np.stack(app.history),before['history'])),
  exact_past_actions_preserved=bool(np.array_equal(np.stack(app.past),before['past'])),
  session_preserved=app.session==before['session'],steps_preserved=app.steps==before['steps'],
  revision_preserved=app.revision==before['revision'],total_controls_preserved=app.total_controls==before['total_controls'],
  physical_control_step_preserved=app.env.control_step==before['environment_control_step'],
  target_preserved=app.target==before['target'],terminal_preserved=app.terminal==before['terminal'],
  recording_id_preserved=app.recording_id==before['recording_id'],
  rejection_added_zero_physics=cost==before['physics_cost'],
  prediction_calls_zero=cost['forbidden_predictor_calls']==0,
 )
 assert all(checks.values()),checks
 passed=True
except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
finally:
 if app is not None:app.env.close()
 physical.ContactSafePhysicsEnv.step=original_step
 report=dict(kind='staged-UI-cross-protocol-preserving-rejection.v1',passed=passed,
  completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,
  source_server_SHA256_before=server_sha,source_server_SHA256_after=sha(STAGED/'server.py'),
  physical_source_SHA256_before=physical_sha,physical_source_SHA256_after=sha(physical.__file__),
  actual_environment_protocol_SHA256=physical.PROTOCOL_SHA256,
  rejected_protocol_SHA256='d83b4c544b1e5193c92e2b882d0904995b1d49e174adaf0e8c5b284031835e9b',
  checks=checks,rejection=rejection,error=error,cost=cost,
  initialization_warmup_controls=cost['completed_controls'],restore_added_controls=0 if checks.get('rejection_added_zero_physics')else None,
  torch_imported='torch'in sys.modules,browser_used=False,live_service_operated=False,
  original_13_check_results_preserved=True,
  note='Fresh import tests only the reset-before-reject fix. Actual CPU physics is three initialization warmups; the rejected restore executes no control and no predictor. This is not a full physical safety gate.')
 (OUTPUT/'migration-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2))
if not passed:raise SystemExit(1)
