"""Plan a future compatibility guard; --execute is a separate explicit gate.

The target entry also verifies its own authorization and the native guardian
handoff. Default planning never samples resources, imports a model, or creates
an attempt/lock/output. Extra entry arguments follow a literal ``--``.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdyssey.wm_compat_guard import DEFAULT_LOCK, plan, supervise
from tdyssey.wm_compat_identity import REGISTRY as REGISTRY_NAME
REGISTRY = ROOT/"artifacts"/REGISTRY_NAME


def build_command(output_dir, ledger_dir=None, entry_arguments=(), *, execute=False):
    command = [str(ROOT/"scripts/validate_wm_compat.py"), "--output-dir", str(Path(output_dir).resolve())]
    if ledger_dir is not None:
        command += ["--ledger-dir", str(Path(ledger_dir).resolve())]
    if execute:
        command.append("--execute")
    command.extend(entry_arguments)
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path)
    parser.add_argument("--deadline-seconds", type=float, default=1800)
    parser.add_argument("entry_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    extra = args.entry_arguments[1:] if args.entry_arguments[:1] == ["--"] else args.entry_arguments
    # Prevent a pass-through argument from overriding the explicit launcher gate.
    if any(value in ("--execute", "--output-dir", "--ledger-dir")
            or value.startswith(("--execute=", "--output-dir=", "--ledger-dir=")) for value in extra):
        parser.error("Execution and output/ledger paths must be declared on the launcher")
    command = build_command(args.output_dir, args.ledger_dir, extra, execute=args.execute)
    if not args.execute:
        result = plan(command, output_dir=args.output_dir, lock_path=DEFAULT_LOCK,
                      deadline_seconds=args.deadline_seconds)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.ledger_dir is None:
        parser.error("--execute requires an explicit new --ledger-dir")
    if not args.python.is_file() or not Path(command[0]).is_file():
        parser.error("Explicit Python interpreter and reviewed entry file must exist")
    result = supervise(command, output_dir=args.output_dir, python_executable=args.python,
        deadline_seconds=args.deadline_seconds, ledger_dir=args.ledger_dir,
        telemetry_dirs=(args.ledger_dir, REGISTRY), attempt_registry=REGISTRY)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "exited" else 2


if __name__ == "__main__":
    raise SystemExit(main())
