"""Source-only publication checks. Never import scientific/model modules."""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
import subprocess
import tomllib
from urllib.parse import unquote,urlsplit

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','.pytest_cache'}
SECRET=[re.compile(x) for x in [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',r'\bgithub_pat_[A-Za-z0-9_]{50,}\b',r'\bhf_[A-Za-z0-9]{30,}\b',r'\bAKIA[0-9A-Z]{16}\b']]
PERSONAL=re.compile(r'(?i)(?:[A-Z]:[/\\]+Users[/\\]+[^/\\\s]+|/h[o]me/(?!runner\b)[^/\s]+|c[o]dex://threads/|01a[0-9a-f]{5}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})')
BANNED={'.pt','.pth','.ckpt','.safetensors','.npz','.npy','.pyc','.pkl','.pickle','.mp4','.avi','.zip','.gz','.exe','.dll'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Write a validation receipt inside this repository')
    args=parser.parse_args(argv)
    errors=[];counts=collections.Counter();total=0
    # Check exactly the publication candidate, not ignored runtime logs/local paths.
    # Tracked files are still scanned even if their name matches an ignore rule.
    try:
        listing=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z'],capture_output=True,check=True).stdout.decode('utf-8')
        files=[ROOT/name for name in sorted(set(filter(None,listing.split('\0')))) if (ROOT/name).is_file()]
    except (OSError,subprocess.CalledProcessError):
        ignored_roots={'artifacts','checkpoints','datasets','data','third_party','.cache','cache','workspaces','runs','output'}
        files=[p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts)
               and p.relative_to(ROOT).parts[0] not in ignored_roots and '.local.' not in p.name]
    rawfiles={p.relative_to(ROOT).as_posix():p for p in files}
    for rel,p in rawfiles.items():
        total+=p.stat().st_size
        if p.suffix.lower() in BANNED or p.name=='.env' or p.name.startswith('.env.'):
            errors.append('Prohibited release asset: '+rel)
        if p.stat().st_size>5_000_000:errors.append('Oversize release file: '+rel)
        if p.suffix.lower()=='.png':counts['figures']+=1;continue
        try:content=p.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            errors.append('Unexpected non-UTF8 file: '+rel);continue
        if any(pattern.search(content) for pattern in SECRET):errors.append('Potential secret: '+rel)
        if PERSONAL.search(content):errors.append('Personal path or internal chat reference: '+rel)
        try:
            if p.suffix=='.py':ast.parse(content,filename=rel);counts['python_syntax']+=1
            if p.suffix=='.json':json.loads(content);counts['json']+=1
            if p.suffix in {'.toml','.lock'}:tomllib.loads(content);counts['toml_or_lock']+=1
        except (SyntaxError,ValueError) as exc:errors.append(f'Parse error {rel}: {exc}')
        if p.suffix=='.md':
            counts['markdown']+=1
            for target in re.findall(r'\[[^\]]+\]\(([^\s)]+)\)',content):
                u=urlsplit(target.strip('<>'))
                if u.scheme or u.netloc or not u.path:continue
                dest=(p.parent/unquote(u.path)).resolve()
                if not dest.exists():errors.append(f'Broken documentation link {rel}: {target}')
                counts['local_links']+=1
    for name in ['source-map','evidence-map','documentation-map']:
        rows=json.loads((ROOT/f'manifests/{name}.json').read_text(encoding='utf-8'))
        for row in rows:
            dest=row.get('repository_path')
            if not dest:continue
            p=(ROOT/dest).resolve()
            if ROOT not in p.parents or not p.is_file():errors.append('Missing/invalid mapped file: '+dest);continue
            actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=row['public_sha256']:errors.append('Mapped hash mismatch: '+dest)
            counts['mapped_hashes']+=1
    project=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))
    lock=tomllib.loads((ROOT/'uv.lock').read_text(encoding='utf-8'))
    requirements=list(project['project']['dependencies'])
    for extra in project['project']['optional-dependencies'].values():requirements+=extra
    for requirement in requirements:
        name,version=requirement.split('==')
        if not any(x['name']==name.replace('_','-') and x['version'].split('+')[0]==version for x in lock['package']):errors.append('Pin absent from lock: '+requirement)
        counts['direct_dependency_pins']+=1
    upstream=json.loads((ROOT/'manifests/upstream.json').read_text(encoding='utf-8'))['repositories']
    for item in upstream:
        if not re.fullmatch('[0-9a-f]{40}',item['commit']):errors.append('Invalid upstream pin: '+item['name'])
    if total>150_000_000:errors.append('Release exceeds source-only 150MB preparation limit')
    result=dict(passed=not errors,scope='Standard-library source checks only; no model/CUDA/simulator execution or clean-machine reproduction.',files=len(files),bytes=total,checks=dict(counts),errors=errors,training_calls=0,model_calls=0,physics_steps=0,secret_scan='Heuristic high-confidence token/private-key and personal-path patterns; not an exhaustive security audit.')
    if args.report:
        dest=args.report.resolve()
        if ROOT not in dest.parents:parser.error('Report must be inside this repository')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return int(bool(errors))


if __name__=='__main__':raise SystemExit(main())
