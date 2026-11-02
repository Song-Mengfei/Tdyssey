"""Bounded diagnostic capture. No model, optimizer or simulator construction."""
from collections import deque
from dataclasses import asdict
from datetime import datetime,timezone
import base64,hashlib,io,json,os,shutil,uuid
from pathlib import Path
import numpy as np

MAX_RING_BYTES=32*1024**2
MAX_RING_BATCHES=16
MAX_VIDEO_BYTES=64*1024**2
MAX_RECORDING_RAW_BYTES=128*1024**2
SHARED_CAP=3*1024**3
ATOMIC_RESERVE=256*1024**2

def utc():return datetime.now(timezone.utc).isoformat()
def safe(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list,deque)):return [safe(v) for v in value]
    return value
def digest(data):return hashlib.sha256(data).hexdigest()
def json_bytes(value):return (json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
def physics_snapshot(env):
    """Public body properties, controller target/count and recorder boundary state."""
    return dict(schema='r3b-playground-public-control-boundary.v2',snapshot=asdict(env.capture_state()),
        state15=env.state().tolist(),controller=dict(name=env.controller,kp=float(env.k_p),kv=float(env.k_v),
            latest_action=None if env.latest_action is None else np.asarray(env.latest_action).tolist(),
            control_step=int(env.control_step),physics_hz=int(env.sim_hz),control_hz=int(env.control_hz),solver_iterations=int(env.space.iterations),substep_dt=.01,substeps_per_control=10,pending_action=None if not hasattr(env,'_action') else np.asarray(env._action).tolist()),
        observer=dict(terminal=bool(env._terminal),in_step=bool(env._in_step)),
        solver_cache_serialized=False,restore_limitation='Public body/controller state is retained; native contact-solver cache and hidden position-bias velocities are not serialized. Saved RGB/actions/raw model inputs replay exactly; future mid-contact physics requires separate validation.')

def validate_original_controller(env,controller):
    """Reject nonoriginal or incomplete controller state before body writes/reset."""
    expected=dict(name=env.controller,kp=100.,kv=20.,physics_hz=100,control_hz=10,solver_iterations=10,substep_dt=.01,substeps_per_control=10)
    if env.initial_identity()['protocol']['name']!='original-kinematic-pusht-goalfree-rgb.v1':raise ValueError('Only strict original protocol restores are allowed')
    if any(controller.get(key)!=value for key,value in expected.items()):raise ValueError('Saved original-controller state is incomplete or differs; current session was preserved')
    if controller.get('integration_microstep_ticks') is not None:raise ValueError('Modified integration-controller state cannot be restored')

def restore_physics(env,value):
    """Restore a complete saved boundary to an already initialized same-physics env."""
    from tdyssey.state import Snapshot
    snapshot=Snapshot.from_dict(value['snapshot'])
    if snapshot.shape!=env.letter or snapshot.physics_hash!=env._physics_hash():raise ValueError('Saved geometry/physics differs')
    validate_original_controller(env,value['controller'])
    for body,state in ((env.block,snapshot.object_body),(env.agent,snapshot.pusher_body)):
        body.angle=state.pose.angle;body.position=state.pose.position;body.velocity=state.velocity
        body.angular_velocity=state.angular_velocity;body.force=state.force;body.torque=state.torque
        env.space.reindex_shapes_for_body(body)
    env.goal_pose[:]=(*snapshot.goal.position,snapshot.goal.angle)
    env.latest_action=None if snapshot.pusher_target is None else np.asarray(snapshot.pusher_target,dtype=float)
    env.control_step=snapshot.control_step
    env._terminal=bool(value['observer']['terminal']);env._in_step=False
    if value['controller'].get('pending_action') is not None:env._action=np.asarray(value['controller']['pending_action'],dtype=float)
    if not np.array_equal(env.state(),np.asarray(value['state15'],dtype=np.float64)):
        raise RuntimeError('Saved complete body state did not restore exactly')

class ClipRing:
    def __init__(self,max_bytes=MAX_RING_BYTES,max_batches=MAX_RING_BATCHES):
        self.records=deque();self.bytes=0;self.sequence=0;self.dropped=0;self.max_bytes=max_bytes;self.max_batches=max_batches
    def append(self,metadata,arrays):
        arrays={k:np.ascontiguousarray(v).copy() for k,v in arrays.items()}
        size=sum(v.nbytes for v in arrays.values())+len(json_bytes(metadata))
        self.sequence+=1
        if size>self.max_bytes:raise RuntimeError('A diagnostic batch exceeds the capture ring budget')
        self.records.append(dict(sequence=self.sequence,metadata=safe(metadata),arrays=arrays,bytes=size));self.bytes+=size
        while len(self.records)>self.max_batches or self.bytes>self.max_bytes:
            self.bytes-=self.records.popleft()['bytes'];self.dropped+=1
    def stats(self):return dict(batches=len(self.records),bytes=self.bytes,max_bytes=self.max_bytes,max_batches=self.max_batches,dropped_batches=self.dropped,last_sequence=self.sequence)
    def pack(self,session,recording_id=None,first_sequence=None,events=None,session_arrays=None):
        selected=[r for r in self.records if first_sequence is None or r['sequence']>=first_sequence]
        metadata=dict(kind='R3B-playground-diagnostic-clip.v1',created_at_utc=utc(),recording_id=recording_id,
            ring=self.stats(),first_requested_sequence=first_sequence,empty_capture=not selected,history_truncated=bool(first_sequence is not None and first_sequence<=self.sequence and (not selected or selected[0]['sequence']>first_sequence)),
            session=session,events=safe(events or[]),diagnostic_development_material=True,training_pool_member=False,records=[])
        arrays={('session_'+key):np.asarray(value) for key,value in (session_arrays or{}).items()}
        metadata['session_arrays']={key:'session_'+key for key in (session_arrays or{})}
        for number,row in enumerate(selected):
            refs={key:f'b{number:03d}_{key}' for key in row['arrays']}
            arrays.update({refs[key]:value for key,value in row['arrays'].items()})
            metadata['records'].append(dict(sequence=row['sequence'],**row['metadata'],arrays=refs))
        arrays['metadata_json']=np.asarray(json.dumps(metadata,ensure_ascii=False,allow_nan=False))
        stream=io.BytesIO();np.savez_compressed(stream,**arrays)
        return stream.getvalue(),metadata

def unpack(data):
    if len(data)>MAX_RING_BYTES*2:raise ValueError('Clip archive is too large')
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        if sum(v.nbytes for k in archive.files if k!='metadata_json' for v in [archive[k]])>MAX_RING_BYTES*2:raise ValueError('Expanded archive is too large')
        meta=json.loads(str(archive['metadata_json']));arrays={k:archive[k].copy() for k in archive.files if k!='metadata_json'}
    return meta,arrays

def storage_check(repo,new_bytes=0):
    """Exact original native experiment roots plus this playground and diagnostics."""
    baseline=repo/'artifacts/native-rgb-supervised-20261029/r3b/baseline-storage.json'
    # Source-only installations have no historical storage-baseline file.
    # Always bound this app's outputs; an optional external shared plan retains
    # the existing experiment budget without requiring data/weights in Git.
    roots=[Path(p) for p in json.loads(baseline.read_text(encoding='utf-8'))['roots']] if baseline.is_file() else []
    roots += [repo/'artifacts/r3b-playground-diagnostics-20261030',repo/'artifacts/r3b-playground',repo/'apps/r3b-playground',repo/'artifacts/original-physics-contact-pilot-20261030',repo/'tdyssey/native_rgb_contact_safe_physics.py',repo/'tdyssey/original_contact_pilot.py',repo/'tdyssey/original_contact_pilot_report.py',repo/'artifacts/strict-original-contact-pilot-20261031',repo/'tdyssey/strict_original_contact_pilot.py',repo/'tdyssey/native_rgb_data_eligibility.py',repo/'scripts/strict_original_contact_pilot.py']
    files={}
    for root in roots:
        for p in ([root] if root.is_file() else root.rglob('*') if root.is_dir() else []):
            if p.is_file():files[str(p.resolve()).lower()]=p
    used=sum(p.stat().st_size for p in files.values());free=shutil.disk_usage(repo).free
    result=dict(shared_bytes=used,projected_additional_bytes=int(new_bytes),shared_cap_bytes=SHARED_CAP,atomic_reserve_bytes=ATOMIC_RESERVE,filesystem_free_bytes=free)
    if used+new_bytes+ATOMIC_RESERVE>SHARED_CAP or free<new_bytes+ATOMIC_RESERVE:raise RuntimeError('Local experiment storage/atomic reserve is insufficient')
    from runtime_config import load_runtime
    plan_path=load_runtime()['shared_storage_plan']
    if plan_path is not None:
        plan=json.loads(plan_path.read_text(encoding='utf-8-sig'));shared={}
        for root in [*(Path(p) for p in plan['shared_roots']),repo]:
            for p in ([root] if root.is_file() else root.rglob('*') if root.is_dir() else []):
                if p.is_file() and '.git' not in p.parts:shared[str(p.resolve()).lower()]=p
        shared_used=sum(p.stat().st_size for p in shared.values());guard=int(plan['atomic'])+int(plan.get('growth',0));cap=int(plan['shared_cap'])
        result.update(external_shared_bytes=shared_used,external_shared_cap_bytes=cap,external_guard_bytes=guard)
        if shared_used+new_bytes+guard>cap:raise RuntimeError('External shared experiment storage reserve is insufficient')
    return result

def persist_clip(repo,data,metadata):
    output=repo/'artifacts/r3b-playground-diagnostics-20261030/captures';output.mkdir(parents=True,exist_ok=True)
    identity='clip-'+uuid.uuid4().hex;path=output/(identity+'.npz');meta=output/(identity+'.json')
    resource=storage_check(repo,len(data)+len(json_bytes(metadata)))
    with path.with_suffix('.npz.tmp').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(path.with_suffix('.npz.tmp'),path);meta.write_bytes(json_bytes(metadata))
    return dict(saved=True,clip_id=identity,recording_id=metadata.get('recording_id'),path=str(path),metadata_path=str(meta),
        bytes=len(data),sha256=digest(data),download_url='/captures/'+path.name,metadata_url='/captures/'+meta.name,
        batches=len(metadata['records']),history_truncated=metadata['history_truncated'],ring_dropped_batches=metadata['ring']['dropped_batches'],storage=resource)

def session_export(session,arrays):
    stream=io.BytesIO();np.savez_compressed(stream,metadata_json=np.asarray(json.dumps(safe(session),ensure_ascii=False,allow_nan=False)),**arrays)
    data=stream.getvalue();return dict(kind='R3B-playground-session-export.v1',archive_base64=base64.b64encode(data).decode('ascii'),bytes=len(data),sha256=digest(data),session=session['session'],physics_state_complete=False,public_physics_state_complete=True,native_solver_and_bias_state_complete=False,solver_cache_serialized=False)

class RecordingWriter:
    """User-started synchronous raw stream; the bounded ring is not its archive."""
    def __init__(self,repo,identity,initial_export):
        self.repo=repo;self.identity=identity;self.output=repo/'artifacts/r3b-playground-diagnostics-20261030/recordings'/identity/'raw'
        self.stage=repo/'artifacts/r3b-playground-diagnostics-20261030'
        self.initial_resource=storage_check(repo,MAX_RECORDING_RAW_BYTES+MAX_VIDEO_BYTES)
        self.external_bytes=self.initial_resource['shared_bytes']-sum(p.stat().st_size for p in self.stage.rglob('*')if p.is_file())
        self.output.mkdir(parents=True,exist_ok=False);self.bytes=0;self.rows=[];self.failures=[];self.closed=False;self.complete=True
        initial=base64.b64decode(initial_export['archive_base64'],validate=True)
        self._write('initial-session.npz',initial);self.initial=dict(file='initial-session.npz',bytes=len(initial),sha256=digest(initial))
        self._manifest()
    def _write(self,name,data):
        path=self.output/name;temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('xb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);self.bytes+=len(data)
    def resource_check(self,additional):
        current=sum(p.stat().st_size for p in self.stage.rglob('*')if p.is_file())
        estimated=self.external_bytes+current+8*1024**2
        return self.bytes+additional<=MAX_RECORDING_RAW_BYTES and estimated+additional+MAX_VIDEO_BYTES+ATOMIC_RESERVE<=SHARED_CAP and shutil.disk_usage(self.repo).free>=additional+MAX_VIDEO_BYTES+ATOMIC_RESERVE
    def can_accept(self,horizon):
        # Exact raw arrays are bounded by the requested inputs/predictions/truth;
        # extra1MiB conservatively covers full body/substep JSON and archive framing.
        upper=int(horizon)*(4*3*96*96*4+3*96*96*4+3*96*96+4*1024**2)+1024**2
        return not self.closed and self.resource_check(upper)
    def append(self,record):
        if self.closed:raise RuntimeError('Recording raw stream already closed')
        meta=dict(record['metadata']);meta.update(kind='R3B-playground-recording-raw-batch.v1',recording_id=self.identity,sequence=record['sequence'])
        arrays=dict(record['arrays']);arrays['metadata_json']=np.asarray(json.dumps(safe(meta),ensure_ascii=False,allow_nan=False))
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);data=stream.getvalue();name=f'batch-{record["sequence"]:08d}.npz'
        try:
            if not self.resource_check(len(data)+8192):raise RuntimeError('Recording raw resource reserve exhausted')
            self._write(name,data)
            self.rows.append(dict(sequence=record['sequence'],file=name,sha256=digest(data),bytes=len(data),session=meta['session'],letter=meta['letter'],
                requested_horizon=meta['requested_horizon'],completed_controls=meta['completed_controls'],ui_step_before=meta['ui_step_before']))
        except BaseException as exc:
            self.complete=False;self.failures.append(dict(sequence=record['sequence'],reason=repr(exc)))
            self._manifest();raise
        self._manifest()
    def _manifest(self):
        value=dict(kind='R3B-complete-recording-raw-stream.v1',recording_id=self.identity,closed=self.closed,
            raw_complete=self.complete,initial_session=self.initial,raw_bytes=self.bytes,raw_cap_bytes=MAX_RECORDING_RAW_BYTES,
            batches=self.rows,missing_batches=self.failures,ring_buffer_is_not_the_full_archive=True,
            stream_created_only_after_user_start=True,optimizer_updates=0,updated_at_utc=utc())
        target=self.output/'manifest.json';temporary=target.with_suffix('.json.tmp');temporary.write_bytes(json_bytes(value));os.replace(temporary,target)
    def finish(self):
        self.closed=True;self._manifest()
        base='/diagnostics-root/recordings/'+self.identity+'/raw/'
        rows=''.join('<li><a href="'+row['file']+'">'+row['file']+'</a> · '+row['letter']+' · '+str(row['completed_controls'])+' controls</li>'for row in self.rows)
        html=('<!doctype html><meta charset="utf-8"><title>Recording raw archive</title><h1>'+self.identity+'</h1><p>Full raw stream complete: '+str(self.complete)+
            '. This is not only the recent ring buffer. No optimizer updates.</p><p><a href="initial-session.npz">Initial full session</a> · <a href="manifest.json">Checksums and per-batch index</a></p><ol>'+rows+'</ol>')
        (self.output/'index.html').write_text(html,encoding='utf-8')
        return dict(raw_complete=self.complete,raw_batches=len(self.rows),raw_bytes=self.bytes,missing_batches=self.failures,
            manifest_path=str(self.output/'manifest.json'),manifest_url=base+'manifest.json',index_url=base+'index.html')
