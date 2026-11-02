"""Check an explicitly supplied external asset; never download or load it."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    assets=json.loads((ROOT/'manifests/external-assets.json').read_text(encoding='utf-8'))['assets']
    byid={a['id']:a for a in assets}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id',choices=sorted(byid),required=True)
    parser.add_argument('--path',type=Path,required=True)
    args=parser.parse_args();asset=byid[args.id]
    if args.path.stat().st_size!=asset['bytes']:raise SystemExit('FAIL: size mismatch')
    with args.path.open('rb') as file:actual=hashlib.file_digest(file,'sha256').hexdigest()
    if actual!=asset['sha256']:raise SystemExit('FAIL: SHA-256 mismatch')
    print('PASS:',args.id,actual)


if __name__=='__main__':main()
