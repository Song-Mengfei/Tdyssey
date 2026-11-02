"""Hash existing evidence without importing runtime or owning any ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def snapshot(root: Path) -> dict:
    trees = [
        "artifacts/milestone-b-20261023",
        "artifacts/milestone-c-20261024",
        "artifacts/policy-c-registry",
        "artifacts/milestone-c-evaluation-20261024",
        "artifacts/policy-c-evaluation-registry",
        "artifacts/milestone-c-load-memory-20261024",
        "artifacts/milestone-c-evaluation-20261024-meta",
    ]
    inventory = {}
    files = {}
    for name in trees:
        members = sorted(p for p in (root / name).rglob("*") if p.is_file())
        inventory[name] = [p.relative_to(root).as_posix() for p in members]
        for p in members:
            files[p.relative_to(root).as_posix()] = digest(p)
    files["configs/policy_c.example.json"] = digest(root / "configs/policy_c.example.json")
    original = root.parent / "Tdyssey"
    base = original / "checkpoints/official-dp/ema_policy.pt"
    files[str(base)] = digest(base)
    git_command = ["git", "-c", f"safe.directory={original.as_posix()}", "-C", str(original)]
    status = subprocess.check_output(git_command + ["status", "--porcelain", "-z"])
    entries = status.decode("utf-8").split("\0")
    research_files = {}
    for entry in entries:
        if not entry:
            continue
        # Existing checkout has ordinary modified/untracked paths, no renames.
        p = original / entry[3:]
        if p.is_file():
            research_files[p.relative_to(original).as_posix()] = digest(p)
    return {
        "kind": "offline_input_inventory",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "trees": inventory,
        "research_checkout_status": entries,
        "research_checkout_files": research_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--before", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing inventory")
    result = snapshot(args.root.resolve())
    if args.before:
        before = json.loads(args.before.read_text(encoding="utf-8"))
        fields = ["files", "trees", "research_checkout_status", "research_checkout_files"]
        result["unchanged"] = {field: result[field] == before[field] for field in fields}
        result["all_unchanged"] = all(result["unchanged"].values())
        result["before_identity"] = digest(args.before)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "file_count": len(result["files"]),
                      "all_unchanged": result.get("all_unchanged")}))
    if result.get("all_unchanged") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
