"""Audit actual UI/download files only: zero model/physics calls."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
p=Path(__file__).resolve().parent
repo=p.parents[2]
d=repo/'artifacts/r3b-playground-diagnostics-20261030/recordings'
rid='recording-b1c2689362b345baa268d0689cbefeaf'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
meta=json.loads((d/(rid+'.json')).read_text(encoding='utf-8'))
raw_root=d/rid/'raw'
manifest=json.loads((raw_root/'manifest.json').read_text(encoding='utf-8'))
download=Path('__LOCAL_HOME__/Downloads')/(rid+'.webm')
assert sha(download)==sha(d/(rid+'.webm'))==meta['sha256']
rows=[]
for item in manifest['batches']:
 path=raw_root/item['file'];assert sha(path)==item['sha256']
 with np.load(path,allow_pickle=False)as z:
  m=json.loads(z['metadata_json'].item())
  for i in range(1,m['completed_controls']):np.testing.assert_array_equal(z['input_histories_float32'][i,-1],z['raw_predictions_float32'][i-1])
  infos=m['control_infos']
  for info in infos:
   assert info['valid'] and info['protocol_sha256']=='2ef7a9562c7451b502e23a5da08e5110349d25c8c2cfac37886bb990086b3d69'
   assert len(info['substeps'])==200 and abs(sum(r['dt']for r in info['substeps'])-.1)<1e-8
  rows.append(dict(sequence=item['sequence'],letter=m['letter'],session=m['session'],horizon=m['requested_horizon'],completed=m['completed_controls'],
   microsteps=sum(len(i['substeps'])for i in infos),contact_seconds=sum(i['agent_object_contact_seconds']for i in infos),
   requested_target=m['requested_target'],max_overlap=np.max([i['penetration']for i in infos],axis=0).tolist(),
   input_predictions_sha256=hashlib.sha256(z['raw_predictions_float32'].tobytes()).hexdigest()))
cap=cv2.VideoCapture(str(download));counts=[]
while True:
 ok,frame=cap.read()
 if not ok:break
 assert frame.shape[:2]==(776,1224)
 counts.append([int((np.min(frame[100:676,24:600],axis=2)<220).sum()),int((np.min(frame[100:676,624:1200],axis=2)<220).sum())])
cap.release()
play=json.loads((p/'browser-video-playback.json').read_text(encoding='utf-8'))
client_seconds=meta['client']['duration_ms']/1000 if 'client'in meta else meta['server_metadata']['client']['duration_ms']/1000
health=json.loads((p/'after-browser-health.json').read_text(encoding='utf-8-sig'))
state=json.loads((p/'after-browser-state.json').read_text(encoding='utf-8-sig'))
clip_meta=json.loads(Path(meta['clip']['metadata_path']).read_text(encoding='utf-8'))
result=dict(kind='actual-cua-browser-v6-recording-check.v1',actual_browser=True,
 passed=bool(manifest['closed']and manifest['raw_complete']and not manifest['missing_batches']and len(rows)==2 and sum(r['completed']for r in rows)==5 and play['ended']and play['duration']>0 and any(a>20 and b>20 for a,b in counts)),
 protocol_sha256=health['physics_sha256'],checkpoint_sha256=health['checkpoint_sha256'],
 recording_id=rid,downloaded_path=str(download),server_video_path=str(d/(rid+'.webm')),sha256=sha(download),
 decoded_frames=len(counts),frames_with_both_nonempty=sum(a>20 and b>20 for a,b in counts),
 actual_browser_playback=play,client_duration_seconds=client_seconds,tail_gap_seconds=client_seconds-play['duration'],
 raw_complete=manifest['raw_complete'],raw_closed=manifest['closed'],raw_batches=rows,
 model_calls_before=0,model_calls_after=health['calls'],recorded_inference_calls=sum(r['completed']for r in rows),
 final_letter=state['letter'],final_steps=state['steps'],recording_active=state['recording_id'],
 measured_recorded_H4_inference_ms=clip_meta['session']['metrics']['latency_ms'],total_model_seconds_not_exposed=True,
 concurrent_live_use_detected=health['calls']>sum(r['completed']for r in rows),
 model_calls_not_attributed_to_recorded_test=health['calls']-sum(r['completed']for r in rows),
 live_session_not_reset_or_reloaded_by_audit=True,
 optimizer_updates=0,new_model_calls_from_audit=0,new_physics_controls_from_audit=0,
 proof_images=['saved-recording-browser.png','opened-video-browser.png'],
 notes=['Real IAB button/canvas interactions: B H1 then reset C and H4, stop/save/download/open/play to end.',
 'Video records94s actual UI wall time including pauses; physical transitions remain5x.1s. This is not a collection10Hz trajectory video.',
 'One selector wait expired during asynchronous persistence; subsequent actual UI showed saved links. No save action was repeated.',
 'The user continued using/resetting the live UI. Total service counter85 is not charged as85 test calls; the immutable recording contains exactly5 test calls. Final live state and timing are not the recorded test end state.',
 'Model remains old-physics trained; this verifies interface/capture, not improved prediction accuracy.'])
(p/'actual-browser-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2));assert result['passed']
