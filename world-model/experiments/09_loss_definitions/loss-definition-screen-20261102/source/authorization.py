"""Preparation is inert. Heavy entry points require a later frozen release."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
@dataclass
class Permit:
    config: dict
    config_sha256: str
    root: Path
    scoring_parity_passed: bool=False

def issue_permit(root=ROOT):
    root=Path(root).resolve()
    release=root/'execution-release.json';config=root/'frozen-config.json'
    if not release.exists() or not config.exists():
        raise RuntimeError('PREPARATION ONLY: final formula, frozen budget and release absent')
    raw=config.read_bytes();value=json.loads(raw);r=json.loads(release.read_text())
    sha=hashlib.sha256(raw).hexdigest()
    if not r.get('execution_authorized') or r.get('config_sha256')!=sha or r.get('run_id')!=value.get('run_id'):
        raise RuntimeError('Execution release/config identity mismatch')
    return Permit(value,sha,root)

def require_permit(permit):
    if not isinstance(permit,Permit) or permit.root!=ROOT:
        raise RuntimeError('No execution permit; no model/library work allowed')
    return permit.config

