"""Plan/download the official CNN checkpoint, retain compact EMA, remove staging data."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tdyssey.official import official_spec
from tdyssey.policy import hash_file


def compact_checkpoint(source, destination, spec):
    import dill
    import torch
    from omegaconf import OmegaConf

    if source.stat().st_size != spec["checkpoint"]["bytes"] or hash_file(source) != spec["checkpoint"]["sha256"]:
        raise ValueError("Original official checkpoint differs from its pinned size/hash")
    if destination.exists():
        raise FileExistsError(destination)
    # The original published training checkpoint contains OmegaConf/dill metadata.
    # Deserialize only this fixed, hash-verified publisher asset. All later policy
    # loads use the compact tensor/native-dict asset with weights_only=True.
    payload = torch.load(source, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    config = OmegaConf.to_container(payload["cfg"].policy, resolve=True)
    if config != spec["policy_config"]:
        raise ValueError("Saved policy configuration differs from the publisher config")
    state = payload["state_dicts"]["ema_model"]
    if not state or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Expected a complete finite EMA tensor state dictionary")
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.touch(exist_ok=False)
    try:
        torch.save({"state_dict": state, "policy_config": config,
                    "source_checkpoint_sha256": spec["checkpoint"]["sha256"]}, temporary)
        if temporary.stat().st_size != spec["compact"]["bytes"] or hash_file(temporary) != spec["compact"]["sha256"]:
            raise ValueError("Compact EMA file differs from its recorded serialization; staging source retained")
        # Check the saved tensor-only representation before deleting any source.
        verified = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
        if set(verified["state_dict"]) != set(state):
            raise ValueError("Compaction lost policy/normalizer entries")
        temporary.rename(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/official-dp")
    parser.add_argument("--download", action="store_true", help="Download and compact one pinned checkpoint; no cache retained")
    parser.add_argument("--max-mib", type=int, default=5120, help="Staging checkpoint budget, at most 6144 MiB")
    args = parser.parse_args()
    spec = official_spec()
    if not 1 <= args.max_mib <= 6144:
        parser.error("Use a staging download budget of 1–6144 MiB")
    output = args.output.resolve()
    compact = output / spec["compact"]["filename"]
    status = "plan_only"
    if args.download:
        if compact.exists():
            if compact.stat().st_size != spec["compact"]["bytes"] or hash_file(compact) != spec["compact"]["sha256"]:
                raise ValueError("Existing compact checkpoint differs from its pinned size/hash")
            status = "already_verified"
        else:
            asset = spec["checkpoint"]
            if asset["bytes"] > args.max_mib * 1024**2:
                parser.error("Pinned source checkpoint exceeds the staging budget")
            output.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(output).free < asset["bytes"] + spec["compact"]["bytes"] + 64 * 1024**2:
                parser.error("Insufficient free space for source plus compact checkpoint")
            source = output / "original.ckpt"
            if not source.exists():
                temporary = source.with_name(source.name + ".part")
                digest, count = hashlib.sha256(), 0
                # Exclusive creation preserves earlier partial files.
                with temporary.open("xb") as target:
                    try:
                        with urllib.request.urlopen(asset["url"], timeout=60) as response:
                            while block := response.read(8 * 1024**2):
                                count += len(block)
                                if count > asset["bytes"]:
                                    raise ValueError("Download exceeds pinned size")
                                target.write(block)
                                digest.update(block)
                        if count != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
                            raise ValueError("Official source checkpoint failed hash/size verification")
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                temporary.rename(source)
            compact_checkpoint(source, compact, spec)
            source.unlink()  # Verified staging asset only; avoid retaining optimizer/duplicate weights.
            status = "downloaded_compacted_verified"
    print(json.dumps({"status": status, "source_bytes": spec["checkpoint"]["bytes"],
                      "retained_bytes": spec["compact"]["bytes"], "destination": str(compact)}, indent=2))


if __name__ == "__main__":
    main()
