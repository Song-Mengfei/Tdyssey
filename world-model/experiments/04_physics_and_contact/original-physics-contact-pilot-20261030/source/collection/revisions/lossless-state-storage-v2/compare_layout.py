"""Lossless array-order comparison on the already saved V2 comparison file."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
repo=out.parents[4];sys.path.insert(0,str(repo))
from tdyssey.original_contact_pilot_storage import load_attempt_archive,canonical
source=out/'first-complete-T-raw-v2-comparison.npz'
destination=out/'first-complete-T-raw-v2-column-major-comparison.npz'
started=time.perf_counter()
with np.load(source,allow_pickle=False)as archive:arrays={k:archive[k]for k in archive.files}
for key in('telemetry_microstates_float64','telemetry_microstate_refs'):arrays[key]=np.asfortranarray(arrays[key])
with destination.open('xb')as stream:np.savez_compressed(stream,**arrays)
decoded=load_attempt_archive(destination)
actual_hash=hashlib.sha256(canonical({k:decoded[k]for k in('telemetry','decisions')})).digest()
assert actual_hash==arrays['telemetry_sha256'].tobytes()
original=json.loads((out/'comparison.json').read_text())
result=dict(kind='lossless-v2-column-major-layout.v1',passed=True,
 source_comparison=str(source),destination=str(destination),destination_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
 legacy_bytes=original['legacy_bytes'],v2_row_major_bytes=source.stat().st_size,v2_column_major_bytes=destination.stat().st_size,
 ratio_column_over_legacy=destination.stat().st_size/original['legacy_bytes'],
 original_payload_sha256=original['original_canonical_payload_sha256'],reconstructed_payload_sha256=actual_hash.hex(),
 wall_seconds=time.perf_counter()-started,physics_controls=0,model_calls=0,GPU_calls=0,optimizer_updates=0,
 statement='Only .npy array storage order changes; public shapes/dtypes/float64 values and canonical evidence hash stay exact. Uses the existing comparison archive; no original physical replay.')
(out/'layout-comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
