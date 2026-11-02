"""Reconstruct a historical first-party layout without executing its code."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, help='Workspace name in manifests/source-profiles.json')
    parser.add_argument('--destination', required=True, type=Path, help='A new empty directory outside this repository')
    parser.add_argument('--include-experiments', action='store_true')
    parser.add_argument('--local-workspace', type=Path, help='Optional replacement for normalized historical path tokens')
    args = parser.parse_args(argv)
    profiles = json.loads((ROOT/'manifests/source-profiles.json').read_text(encoding='utf-8'))
    if args.profile not in profiles:
        parser.error('Unknown profile. Choices: '+', '.join(profiles))
    dest = args.destination.resolve()
    if dest == ROOT or ROOT in dest.parents or dest in ROOT.parents:
        parser.error('Destination must be outside this repository and its parents')
    if dest.exists():
        parser.error('Destination must not exist; reconstruction never overwrites a directory')
    rows = json.loads((ROOT/'manifests/source-map.json').read_text(encoding='utf-8'))
    chosen = [r for r in rows if r['workspace'] == args.profile and r.get('repository_path')
              and (args.include_experiments or not r['original_path'].startswith('artifacts/'))]
    files = []
    for row in chosen:
        source = ROOT/row['repository_path']
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['public_sha256']:
            raise ValueError('Source hash mismatch: '+row['repository_path'])
        relative = Path(row['original_path'])
        target = (dest/relative).resolve()
        if dest not in target.parents:
            raise ValueError('Invalid target path')
        if args.local_workspace:
            # Use forward slashes so the replacement is valid inside Python/JSON literals.
            raw = raw.replace(b'__LOCAL_WORKSPACE__',args.local_workspace.resolve().as_posix().encode('utf-8'))
        files.append((target,raw))
    for filename, source in profiles[args.profile].get('environment_files',{}).items():
        files.append((dest/filename,(ROOT/source).read_bytes()))
    dest.mkdir(parents=True)
    for target,raw in files:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    receipt = dict(profile=args.profile,files_written=len(files),model_calls=0,
                   scope='Source layout only. No data, checkpoints, archived approvals, local launchers or frozen evidence were restored.',
                   warning='Historical hash bindings and run authorization must be regenerated. This is not an exact binary reproduction.')
    (dest/'SOURCE_RECONSTRUCTION.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
