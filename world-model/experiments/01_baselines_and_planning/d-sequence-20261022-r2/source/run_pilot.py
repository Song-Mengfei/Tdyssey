"""Predeclared three-state recovery pilot; preserve every outcome and source."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SOURCES = ["tdyssey/simulator_sequence.py", "tdyssey/contact_geometry.py",
           "tdyssey/simulator_search.py", "tdyssey/simulator_planning.py",
           "scripts/evaluate_contact_sequence.py", "tdyssey/envs/official.py",
           "tdyssey/official.py", "tdyssey/geometry.py", "tdyssey/state.py", "tdyssey/evaluation.py",
           "pyproject.toml", "uv.lock", "manifests/official_dp.json"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = BASE/"protocol.json"
    if path.exists():
        raise ValueError("Pilot output exists; do not overwrite")
    snapshot = BASE/"source"
    for name in SOURCES:
        dest = snapshot/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    hashes = {name: digest(ROOT/name) for name in SOURCES}
    jobs = []
    for scene in (100000, 100100, 100200):
        prior = ROOT/f"artifacts/d-teacher-20261021/matrix-v2/scene-{scene}-search-42/report.json"
        data = json.loads(prior.read_text(encoding="utf-8"))
        row = data["steps"][159]
        state = row["state"]
        delta = state["goal"]["angle"]-state["object_body"]["pose"]["angle"]
        output = BASE/f"scene-{scene}-resume160-search42"
        command = [sys.executable, "-B", "scripts/evaluate_contact_sequence.py", "--resume-report", str(prior),
            "--resume-step", "160", "--steps", "300", "--seconds", "600", "--horizon", "80",
            "--max-evaluations", "40", "--max-global-searches", "8", "--search-seed", "42", "--output", str(output)]
        jobs.append(dict(scene=scene, search_seed=42, resume_step=160, source=str(prior),
            source_sha256=digest(prior), command=command, output=str(output), status="queued",
            start_coverage=row["metrics"]["coverage"], start_position_error=math.dist(state["goal"]["position"],state["object_body"]["pose"]["position"]),
            start_angle_error_deg=abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta))))))
    protocol = dict(kind="three_predeclared_partial_overlap_recoveries", local_date="2026-10-22", status="running", revision=2, parent_protocol="artifacts/d-sequence-20261022/protocol.json",
        not_reset_success=True, development_only=True, selection="Three previously named V2 scene/search42 rows at control160, before observing sequence outcomes",
        training=False, checkpoint_loaded=False, source_sha256=hashes, jobs=jobs,
        bounds=dict(total_episode_controls=300, imported_history_controls=160, new_control_cap=140,
            seconds_per_run=600, forecast_total_including_hold=80, max_attempts_per_global=40,
            max_global_searches=8, max_concurrent=2, native_success="coverage >0.95; immediate stop"))
    began = time.monotonic()
    def save():
        path.write_text(json.dumps(protocol, indent=2)+"\n",encoding="utf-8")
    save()
    def run(job):
        if any(digest(ROOT/n) != value for n,value in hashes.items()):
            raise ValueError("Source changed during pilot")
        if digest(job["source"]) != job["source_sha256"]:
            raise ValueError("Prior report changed")
        output = Path(job["output"])
        log = BASE/(output.name+".log")
        started = time.monotonic()
        job["status"] = "running"
        print(f"START {output.name}", flush=True)
        with log.open("w",encoding="utf-8") as stream:
            process = subprocess.Popen(job["command"],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONUTF8="1"))
            job["pid"] = process.pid
            code = process.wait()
        job.update(status="process_complete" if code == 0 else "process_failed", exit_code=code,
                   process_seconds=time.monotonic()-started)
        if (output/"report.json").is_file():
            report = json.loads((output/"report.json").read_text(encoding="utf-8"))
            job["result"] = {k: report.get(k) for k in ("status","success","stop_reason","counts","search_counts",
                "max_coverage","max_iou","final_metrics","initial_pose_rms","final_pose_rms","global_searches","torch_loaded","error")}
        print("FINISH "+output.name+" "+json.dumps(job.get("result",{})),flush=True)
        return job
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        while any(not f.done() for f in futures):
            save()
            time.sleep(1)
        for future in futures:
            future.result()
    protocol["status"] = "complete"
    protocol["elapsed_seconds"] = time.monotonic()-began
    protocol["sources_unchanged"] = all(digest(ROOT/n)==value for n,value in hashes.items())
    save()
    print("PILOT COMPLETE",protocol["elapsed_seconds"],flush=True)


if __name__ == "__main__":
    main()
