"""Correct reporting metadata only; preserve the complete original diagnosis."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
source=root/'artifacts/wm-interface-v2-development'
output=Path(__file__).resolve().parent
corrections={}
for name in ('termination-confusion','per-image'):
    with (source/(name+'.csv')).open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        columns=list(reader.fieldnames)
        rows=list(reader)
    before=len(rows)
    if name=='termination-confusion':
        rows=[row for row in rows if not (row['version']=='v1' and row['method']=='declared_interval')]
        corrections['v1_interval_rows_removed_as_not_applicable']=before-len(rows)
    else:
        columns.append('ambiguity_assessed')
        for row in rows:
            row['ambiguity']=''
            row['ambiguity_assessed']='False'
            if row['input_available']=='False':
                row['reason']='missing_input'
        corrections['ambiguity']='Unassessed, not evidence of a unique pose. Numeric pose/coverage fields are unchanged.'
    with (output/(name+'-reviewed.csv')).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
corrections.update(decoder_calls=0,model_calls=0,real_operations=0,
    original_source_commit='ab01b9e',
    original_output_manifest_sha256=hashlib.sha256((source/'output-manifest.json').read_bytes()).hexdigest(),
    numeric_measurement_changes=0,
    current_source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        'tdyssey/wm_image_measurement_v2.py','tdyssey/wm_interface_diagnostics.py')})
with (output/'reporting-corrections.json').open('x',encoding='utf-8') as stream:
    json.dump(corrections,stream,indent=2,sort_keys=True)
    stream.write('\n')
print(json.dumps(corrections,indent=2))
