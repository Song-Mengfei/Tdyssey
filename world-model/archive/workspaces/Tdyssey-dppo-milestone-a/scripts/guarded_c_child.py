"""Wait for the parent's actual-process ownership ACK before running C code."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handshake", type=Path, required=True)
    parser.add_argument("--ack", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = args.target[1:] if args.target[:1] == ["--"] else args.target
    if not target:
        parser.error("one target script is required")
    identity = {"pid": os.getpid(), "parent_pid": os.getppid(), "nonce": args.nonce}
    temporary = args.handshake.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as out:
        json.dump(identity, out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, args.handshake)
    began = time.monotonic()
    while time.monotonic() - began < 30:
        if args.ack.exists():
            accepted = json.loads(args.ack.read_text(encoding="utf-8"))
            if accepted != {"pid": identity["pid"], "nonce": args.nonce}:
                raise ValueError("Parent ACK identity differs")
            sys.argv = target
            sys.path.insert(0, str(Path(target[0]).resolve().parent.parent))
            runpy.run_path(target[0], run_name="__main__")
            return 0
        time.sleep(0.05)
    raise TimeoutError("No parent ownership ACK; no target code started")


if __name__ == "__main__":
    raise SystemExit(main())
