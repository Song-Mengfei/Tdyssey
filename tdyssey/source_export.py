"""Verify pinned upstream Git checkouts or explicitly inventoried source exports."""
import json
from pathlib import Path
import subprocess

from .policy import hash_file

EXPORT_MANIFEST = "tdyssey-source-export.json"


def verify_source(root, spec, label="upstream"):
    root = Path(root).resolve()
    expected = spec["source_sha256"]
    if (root / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        mode = "git"
    else:
        provenance = json.loads((root / EXPORT_MANIFEST).read_text(encoding="utf-8"))
        if (provenance.get("kind") != "tdyssey.pinned-source-export.v1"
                or provenance.get("source_sha256") != expected):
            raise ValueError(f"{label} export inventory differs from pinned source")
        revision = provenance.get("source_commit")
        mode = "hash_verified_export"
    if revision != spec["source_commit"]:
        raise ValueError(f"{label} source revision differs from manifest")
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or hash_file(path) != digest:
            raise ValueError(f"Modified/missing {label} source: {relative}")
    return dict(source_commit=revision, verification=mode, files=len(expected))
