"""Compare one completed saved T trajectory. No simulator/model import."""
import datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np

out=Path(__file__).resolve().parent
repo=out.parents[4]
sys.path.insert(0,str(repo))
from tdyssey.original_contact_pilot_storage import (
    STATE_FIELDS,TELEMETRY_STORAGE_KEYS,canonical,pack_telemetry,
    unpack_telemetry,load_attempt_archive,
)
collection=repo/'artifacts/original-physics-contact-pilot-20261030/collection'
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
started=time.perf_counter()
samples=[]
for metadata_path in sorted((collection/'trajectories').glob('*/metadata.json')):
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['complete'] and metadata['slot']['phase']=='calibration-v1' and metadata['slot']['letter']=='T':
        samples.append((metadata_path,metadata))
if not samples:raise RuntimeError('No complete T calibration trajectory exists; no physics is run here')
metadata_path,metadata=samples[0]
source=metadata_path.parent/metadata['raw']['file']
before=sha(source)
assert before==metadata['raw']['sha256']
with np.load(source,allow_pickle=False)as archive:
    original={key:archive[key]for key in archive.files}
load_start=time.perf_counter();payload=unpack_telemetry(original);legacy_load_seconds=time.perf_counter()-load_start
pack_start=time.perf_counter();packed=pack_telemetry(payload['telemetry'],payload['decisions']);pack_seconds=time.perf_counter()-pack_start
destination=out/'first-complete-T-raw-v2-comparison.npz'
primary={key:value for key,value in original.items()if key not in TELEMETRY_STORAGE_KEYS}
with destination.open('xb')as handle:np.savez_compressed(handle,**primary,**packed)
unpack_start=time.perf_counter();reconstructed=unpack_telemetry(packed);unpack_seconds=time.perf_counter()-unpack_start
assert canonical(payload)==canonical(reconstructed)
with np.load(destination,allow_pickle=False)as archive:
    assert all(archive[key].dtype.kind!='O'for key in archive.files)
decoded=load_attempt_archive(destination)
array_proof={key:dict(dtype=value.dtype.str,shape=list(value.shape),
    exact_dtype_shape_bytes=decoded[key].dtype==value.dtype and decoded[key].shape==value.shape and decoded[key].tobytes()==value.tobytes(),
    sha256=hashlib.sha256(value.tobytes()).hexdigest())for key,value in primary.items()}
assert all(row['exact_dtype_shape_bytes']for row in array_proof.values())
microsteps=0;state_fields=0
for control,restored_control in zip(payload['telemetry'],reconstructed['telemetry']):
    for original_row,restored_row in zip(control.get('substeps',[]),restored_control.get('substeps',[])):
        microsteps+=1
        for field in STATE_FIELDS:
            if field in original_row:
                assert np.asarray(original_row[field],dtype='<f8').tobytes()==np.asarray(restored_row[field],dtype='<f8').tobytes()
                state_fields+=1
assert sha(source)==before
result=dict(kind='saved-T-lossless-telemetry-v2-comparison.v1',checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
    source=str(source),source_sha256_before=before,source_sha256_after=sha(source),source_not_modified=True,
    comparison_file=str(destination),comparison_sha256=sha(destination),legacy_bytes=source.stat().st_size,
    v2_bytes=destination.stat().st_size,bytes_saved=source.stat().st_size-destination.stat().st_size,
    size_ratio_v2_over_legacy=destination.stat().st_size/source.stat().st_size,
    original_canonical_payload_sha256=hashlib.sha256(canonical(payload)).hexdigest(),
    reconstructed_canonical_payload_sha256=hashlib.sha256(canonical(reconstructed)).hexdigest(),
    canonical_payload_exact=True,primary_arrays=array_proof,
    actual_saved_controls=len(payload['telemetry']),actual_saved_microsteps=microsteps,state_fields_bitwise_compared=state_fields,
    unique_float64_vectors=len(packed['telemetry_microstates_float64']),float64_references=len(packed['telemetry_microstate_refs']),
    timings_seconds=dict(legacy_json_load=legacy_load_seconds,pack_v2=pack_seconds,unpack_v2=unpack_seconds,whole_comparison=time.perf_counter()-started),
    new_physics_controls=0,new_model_calls=0,GPU_calls=0,optimizer_updates=0,Torch_imported='torch'in sys.modules,Pymunk_imported='pymunk'in sys.modules,
    helper_sha256=sha(repo/'tdyssey/original_contact_pilot_storage.py'),tests_sha256=sha(repo/'tests/test_original_contact_pilot_storage.py'),
    limitation='One existing T sample, not a dataset-wide compression estimate. V2 comparison file is an extra small experiment artifact; root owns any later deletion. Running collector/report/physics/UI files were not edited.')
(out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
