"""Explicitly fetch one upstream repository at its recorded immutable revision."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    sources=json.loads((ROOT/'manifests/upstream.json').read_text(encoding='utf-8'))['repositories']
    byname={r['name']:r for r in sources}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',choices=sorted(byname),required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();item=byname[args.name];dest=args.destination.resolve()
    if dest.exists():parser.error('Destination exists; no existing checkout will be modified')
    if not re.fullmatch('[0-9a-f]{40}',item['commit']):raise ValueError('Expected full Git SHA')
    if not item['url'].startswith('https://github.com/'):raise ValueError('Unexpected upstream host')
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout',item['url'],str(dest)],check=True)
    subprocess.run(['git','-C',str(dest),'checkout','--detach',item['commit']],check=True)
    actual=subprocess.run(['git','-C',str(dest),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if actual!=item['commit']:raise RuntimeError('Revision mismatch')
    print(args.name,actual)


if __name__=='__main__':main()
