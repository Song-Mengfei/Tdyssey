"""One native stdlib-only ownership/termination check; no model or ledger."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_guarded_c_policy import claim_worker, write_json

directory = Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "worker-control-check")
directory.mkdir(exist_ok=False)
results = []
for case in ("complete", "terminate", "job_close"):
    output = directory / case
    output.mkdir()
    target = output / "target.py"
    target.write_text("import json,os,sys,time\nfrom pathlib import Path\n"
        + ("time.sleep(30)\n" if case != "complete" else "")
        + "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'argv':sys.argv}))\n", encoding="utf-8")
    nonce = uuid.uuid4().hex
    marker = output / "marker.json"
    command = [sys.executable, "-S", "-B", str(ROOT / "scripts/guarded_c_child.py"),
        "--handshake", str(output / "worker.json"), "--ack", str(output / "worker-ack.json"),
        "--nonce", nonce, "--", str(target), str(marker), "token"]
    with (output / "stdout.log").open("w") as log:
        launcher = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        owned = None
        try:
            owned = claim_worker(launcher, output, nonce)
            time.sleep(0.2)
            if case == "terminate":
                owned.terminate_and_wait(timeout=5)
            elif case == "job_close":
                owned.close()
            launcher.wait(timeout=5)
            actual_exit = owned.exit_code()
            evidence = {"case": case, "launcher_pid": launcher.pid, "worker_pid": owned.pid,
                "launcher_exit": launcher.returncode, "actual_exit": actual_exit,
                "marker_exists": marker.exists()}
            if case == "complete":
                payload = json.loads(marker.read_text())
                assert payload["pid"] == owned.pid and payload["argv"][2] == "token"
                assert actual_exit == 0 and launcher.returncode == 0
                evidence["target"] = payload
            else:
                assert not marker.exists() and actual_exit is not None
                if case == "terminate":
                    assert owned.poll_exit() and launcher.returncode != 0
            results.append(evidence)
        finally:
            if owned is not None:
                owned.close()
            if launcher.poll() is None:
                launcher.terminate()
                launcher.wait(timeout=5)
write_json(directory / "report.json", {"status": "passed", "scope": "three tiny stdlib targets only; no model/ledger/physical operations", "results": results})
print(json.dumps(results, indent=2))
