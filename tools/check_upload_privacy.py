"""Scan actual files or Git publication candidates without exposing matched values."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ACTIVE_LOGS={'artifacts/r3b-playground/stdout.log','artifacts/r3b-playground/stderr.log'}
PATTERNS={
    'personal_directory':re.compile(r'''(?i)(?:\b[A-Z]:(?:\\+|/)(?:Users|Documents and Settings)(?:\\+|/)[^/\\\s"'<>]+|/(?:home|Users)/[^/\s"'<>]+/)'''),
    'internal_chat':re.compile(r'c[o]dex://threads/|01a[0-9a-f]{5}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'),
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'known_secret':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})\b'),
    'url_password':re.compile(r'https?://[^/\s:@]+:[^/\s@]+@'),
}


def publication_files():
    value=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'ls-files','--cached','--others','--exclude-standard','-z'],capture_output=True,check=True).stdout.decode('utf-8')
    return {name for name in value.split('\0') if name and (ROOT/name).is_file()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('all','git'),default='all')
    parser.add_argument('--allow-active-logs',action='store_true',help='Explicitly allow only the two ignored logs retained for an owner-requested running app')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args(argv);published=publication_files()
    files=sorted(published) if args.scope=='git' else sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts)
    findings=[];exceptions=[]
    for relative in files:
        raw=(ROOT/relative).read_bytes()
        try:value=raw.decode('utf-8-sig')
        except UnicodeDecodeError:value=raw.decode('latin1')
        hits=[]
        for name,pattern in PATTERNS.items():
            lines=sorted({value.count('\n',0,m.start())+1 for m in pattern.finditer(value)})
            if lines:hits.append(dict(category=name,lines=lines))
        if not hits:continue
        item=dict(path=relative,categories=hits,in_publication_set=relative in published)
        if args.allow_active_logs and relative in ACTIVE_LOGS and relative not in published:exceptions.append(item)
        else:findings.append(item)
    result=dict(passed=not findings,scope=args.scope,files_scanned=len(files),publication_files=len(published),findings=findings,explicit_owner_retained_active_logs=exceptions,matched_values_omitted=True,limits='Heuristic personal-directory/key/reference scan; system font paths and synthetic test paths are not personal directories.')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));return int(bool(findings))


if __name__=='__main__':raise SystemExit(main())
