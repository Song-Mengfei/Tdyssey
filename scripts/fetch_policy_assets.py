"""Fetch only the pinned EMA policy checkpoint, with size/hash checks and no cache."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_checkpoint(spec, destination, max_bytes):
    destination = Path(destination)
    if spec["bytes"] > max_bytes:
        raise ValueError("Pinned checkpoint exceeds the download byte budget")
    if destination.exists():
        if destination.stat().st_size != spec["bytes"] or file_hash(destination) != spec["sha256"]:
            raise ValueError("Existing checkpoint does not match its pinned size/hash")
        return "already_verified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < spec["bytes"] + 64 * 1024 * 1024:
        raise ValueError("Insufficient free disk space for the pinned checkpoint")
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Incomplete download exists: {temporary}; inspect/remove it before retrying")
    url = f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{spec['path']}"
    digest, count = hashlib.sha256(), 0
    created = False
    try:
        with temporary.open("xb") as target:
            created = True
            source = urllib.request.urlopen(url, timeout=60)
            with source:
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > min(max_bytes, spec["bytes"]):
                        raise ValueError("Download exceeds its declared size/budget")
                    target.write(block)
                    digest.update(block)
        if count != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
            raise ValueError("Downloaded checkpoint failed size/hash verification")
        temporary.replace(destination)
    except BaseException:
        # This command exclusively created this incomplete file, never an older asset.
        if created:
            temporary.unlink(missing_ok=True)
        raise
    return "downloaded_verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/gpc-policy")
    parser.add_argument("--max-mib", type=int, default=400)
    parser.add_argument("--download", action="store_true", help="Without this flag, only print the small download plan")
    args = parser.parse_args()
    spec = json.loads((ROOT / "manifests/gpc_policy.json").read_text())["checkpoint"]
    if not 1 <= args.max_mib <= 512:
        parser.error("Use a download budget of 1–512 MiB")
    destination = args.output / spec["filename"]
    status = fetch_checkpoint(spec, destination, args.max_mib * 1024 * 1024) if args.download else "plan_only"
    print(json.dumps({"status": status, "bytes": spec["bytes"], "sha256": spec["sha256"],
                      "repo_id": spec["repo_id"], "revision": spec["revision"], "file": spec["path"],
                      "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
