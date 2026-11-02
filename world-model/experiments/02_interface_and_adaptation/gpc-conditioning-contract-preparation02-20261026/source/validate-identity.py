"""Stdlib-only identity/configuration proof; no synthetic or model execution."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent.parent
started=time.monotonic()
sha=lambda value:hashlib.sha256(value).hexdigest()
old_raw=(ROOT/'configs/gpc_conditioning_contract.json').read_bytes()
assert sha(old_raw)=='043339f6e1950970b54e3c084311a8c325bfac7ae4bb66540711c7fefff6c327'
old=json.loads(old_raw)
new=json.loads((ROOT/'configs/gpc_conditioning_contract02.json').read_bytes())
changes={key:dict(before=old[key],after=new[key]) for key in old if old[key]!=new[key]}
assert set(old)==set(new) and set(changes)=={'candidate_id','output_dir'}
assert new['candidate_id']=='gpc-conditioning-contract-diagnostic-02'
auth='auth20261026-user-autonomous-gpc-conditioning-contract-diagnostic-02'
assert new['output_dir']=='artifacts/gpc-conditioning-contract-diagnostic-20261026/'+auth
old_science={key:value for key,value in old.items() if key not in changes}
new_science={key:value for key,value in new.items() if key not in changes}
assert old_science==new_science

def normalized(tree,names):
    result=deepcopy(tree)
    for node in ast.walk(result):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            node.value=ast.Constant(value='<identity>')
    return ast.dump(result,include_attributes=False)

source_proofs={}
for relative,names in (
    ('tdyssey/gpc_conditioning_entry.py',{'CONFIG_RELATIVE','CANDIDATE','AUTHORIZATION','OUTPUT'}),
    ('tdyssey/gpc_conditioning_diagnostic.py',{'CANDIDATE_ID','AUTHORIZATION_ID'})):
    prior=subprocess.check_output(['git','-C',str(ROOT),'show','8f41bcf84cc3cf42f5aff7ad0d4ed86ff4996566:'+relative],timeout=5)
    current=(ROOT/relative).read_bytes()
    assert normalized(ast.parse(prior),names)==normalized(ast.parse(current),names)
    source_proofs[relative]=dict(only_identity_assignments_differ=True,sha256=sha(current))
entry=ast.parse((ROOT/'tdyssey/gpc_conditioning_entry.py').read_bytes())
values={node.targets[0].id:node.value for node in entry.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
assert ast.literal_eval(values['AUTHORIZATION'])==auth
assert ast.literal_eval(values['CANDIDATE'])==new['candidate_id']
assert ast.literal_eval(values['CONFIG_RELATIVE'])=='configs/gpc_conditioning_contract02.json'
budget=next(node for node in ast.parse((ROOT/'tdyssey/gpc_conditioning_diagnostic.py').read_bytes()).body if isinstance(node,ast.ClassDef))
budget_values={node.targets[0].id:ast.literal_eval(node.value) for node in budget.body if isinstance(node,ast.Assign)}
assert budget_values['AUTHORIZATION_ID']==auth and budget_values['CANDIDATE_ID']==new['candidate_id']
for item in json.loads((BASE/'start.json').read_bytes())['preserved01']:
    assert (ROOT/item['path']).stat().st_size==item['bytes'] and sha((ROOT/item['path']).read_bytes())==item['sha256']
plan=json.loads((ROOT/'artifacts/gpc-conditioning-contract-preparation-20261026/run-plan.json').read_bytes())
plan.update(config=new,parent_attempt='gpc-conditioning-contract-diagnostic-01',parent_model_loads=0,parent_forecasts=0,
    parent_stop='startup RAM2.6919GiB<4GiB',new_authorization=auth,user_trigger='现在应该够了')
with (BASE/'run-plan.json').open('x',encoding='utf-8') as stream:json.dump(plan,stream,indent=2,allow_nan=False)
result=dict(changed_config_fields=changes,scientific_config_equal=True,source_proofs=source_proofs,
    old_claim_and_config_preserved=True,new_config_sha256=sha((ROOT/'configs/gpc_conditioning_contract02.json').read_bytes()),
    seconds=time.monotonic()-started,synthetic_checks=0,torch=0,models=0,physics=0)
with (BASE/'identity-validation.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result,indent=2))
