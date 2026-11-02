"""CPU/driver-only final freeze. Never imports Torch or executes a model."""
import sys,json,os,shutil,subprocess,ast,platform
from pathlib import Path
from importlib import metadata
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c1_training import sha,atomic_json
from scripts.native_rgb_r2c1_run import config_contract
from tdyssey.native_rgb_r2c1_budget import storage_for_config
EXPERIMENT=ROOT/'artifacts/native-rgb-supervised-20261029';OUT=EXPERIMENT/'r2c1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
assert not (OUT/'source-identity.json').exists() and not (OUT/'run-claim.json').exists() and not (OUT/'run').exists(),'No freeze overwrite/relaunch'
cfgpath=ROOT/'configs/native_rgb_r2c1.json';cfg=read(cfgpath);config_contract(cfg)
assert read(EXPERIMENT/'stage-state.json')['r2c1_execution_authorized']
assert sha(EXPERIMENT/'native-rgb-cost-ledger.json')=='2729b06c8e565618624c65e1835701f2a37eabbcc9da0722c5bccad8bd7c40e6'
for row in read(OUT/'input-identities.json')['files']:assert sha(row['path'])==row['sha256'],row['path']
for p,h in cfg['input_hashes'].items():assert sha(p)==h,p
assert sha(cfg['init_checkpoint'])==cfg['init_checkpoint_sha256']
newfiles=sorted({p.resolve() for pat in ['tdyssey/native_rgb_r2c1*.py','scripts/native_rgb_r2c1*.py','tests/test_native_rgb_r2c1*.py','configs/native_rgb_r2c1*.json'] for p in ROOT.glob(pat)})
roots={Path(p).resolve() for p in read(OUT/'storage-roots.json')['roots']};roots.update(newfiles);roots.update(ROOT/n for n in ['tdyssey','scripts','tests','configs']);roots.add(OUT.resolve())
cfg['storage_roots']=[str(p) for p in sorted(roots)];cfg['new_stage_source_files']=[str(p) for p in newfiles];atomic_json(cfgpath,cfg);config_contract(cfg)
exe=shutil.which('nvidia-smi')
if not exe:raise RuntimeError('Cannot record required GPU/driver preflight: nvidia-smi unavailable')
gpu=subprocess.run([exe,'--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
atomic_json(OUT/'gpu-preflight.json',dict(command=[exe,'--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version','--format=csv,noheader,nounits'],returncode=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr,driver_query_only=True,torch_imported='torch'in sys.modules,model_calls=0))
assert gpu.returncode==0 and '3060 Ti' in gpu.stdout,'The authorized3060Ti GPU/driver was not confirmed'
gpus=[line.split(',') for line in gpu.stdout.splitlines() if line.strip()];assert len(gpus)==1
assert float(gpus[0][4])>=2560,'Less than2.5GiB free GPU memory; do not launch against inadequate headroom'
versions={name:metadata.version(name) for name in ['torch','numpy','scipy','Pillow','matplotlib']}
atomic_json(OUT/'runtime-identity.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,distribution_metadata_only=True,torch_imported='torch'in sys.modules,new_model_calls=0))
assert 'torch'not in sys.modules
files=set(newfiles);files.update(p.resolve() for p in (ROOT/'tdyssey').rglob('*.py'));files.update(p.resolve() for p in (ROOT/'scripts').glob('native_rgb*.py'));files.update(p.resolve() for p in OUT.glob('*.py'))
files.update([ROOT/'configs/native_rgb_stage3_score.json',Path(cfg['calibration'])])
upstream=read(EXPERIMENT/'r2c0/source-settings-audit.json')['upstream_source']['files']
for r in upstream:
 assert sha(r['path'])==r['expected_sha256'],r['path']
 files.add(Path(r['path']).resolve())
source=[]
for p in sorted(files):
 assert p.is_file(),p
 if p.suffix=='.py':compile(ast.parse(p.read_text(encoding='utf-8')),str(p),'exec')
 if p.is_relative_to(ROOT):rel='repo/'+p.relative_to(ROOT).as_posix()
 elif p.is_relative_to(Path(cfg['source_root'])):rel='upstream-gpc/'+p.relative_to(Path(cfg['source_root'])).as_posix()
 else:raise ValueError('Unexpected source root: '+str(p))
 target=OUT/'source-snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 source.append(dict(source=str(p),sha256=sha(p),snapshot=target.relative_to(OUT).as_posix(),bytes=p.stat().st_size))
atomic_json(OUT/'source-identity.json',dict(kind='R2C1-pre-model-source-config-freeze.v1',files=source,config_sha256=sha(cfgpath),weight_sha256=sha(cfg['init_checkpoint']),R2C0_inputs=cfg['input_hashes'],global_ledger_before=sha(EXPERIMENT/'native-rgb-cost-ledger.json'),model_calls_before_freeze=0,torch_imported=False))
resources=storage_for_config(cfg).check(force=True);atomic_json(OUT/'launch-resource-preflight.json',resources)
assert 'torch'not in sys.modules
print(json.dumps(dict(source_files=len(source),config_sha256=sha(cfgpath),source_identity_sha256=sha(OUT/'source-identity.json'),gpu=gpu.stdout.strip(),storage_bytes=resources['bytes'],conservative_shared_upper_bound_bytes=resources['conservative_shared_upper_bound_bytes'],Torch_imported=False)))
