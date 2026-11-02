"""Fixed one-shot T/L A/B collection, with no model loading or training."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tdyssey.dataset import json_bytes
from tdyssey.gpc_collection_runtime import (CollectionBudget, Heartbeat, stage_paths,
    status, supervise, verify_worker, _verify_authority)

PREPARATION=ROOT/"artifacts/wm-data-preparation-20261026"
EXCLUSIONS=PREPARATION/"exclusions.json"
PLAN=PREPARATION/"independent-benchmark-plan.json"
SCHEDULE=ROOT/"artifacts/wm-sampling-b-only-20261026/training-schedule.json"
ADAPTATION_CONFIG=ROOT/"configs/gpc_supervised_adaptation.json"
PLAN_SHA256="829eeb5c911025618c670fcb8538befc14e07e1988c24c802a006b1e2a2159cf"
EXCLUSIONS_SHA256="1533f622e79e5e9c4f650df151cbbed05d3bd33b63e5f3ab6be2f91ff540104e"
A_MANIFEST_SHA256="2a28136ca9c3b111d202f507ebab16c992b4d26a9c48d388956e390c6b2b069e"


class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split(".")[0] in ("torch","diffusers","gpc_code") or
                fullname.startswith("diffusion_policy.policy") or fullname=="tdyssey.gpc_world_model"):
            raise ImportError("Model/policy imports are forbidden during A/B data collection")
        return None


def read_exclusions():
    payload=EXCLUSIONS.read_bytes();value=json.loads(payload)
    value["sha256"]=hashlib.sha256(payload).hexdigest()
    if len(set(value["reviewed_scene_sha256"]))<316:
        raise ValueError("All 316 reviewed full-state exclusions are required")
    return value


def frozen_b2_inputs(exclusions):
    """Metadata-only gate: the new train order must exist before B2 can start."""
    if (exclusions["sha256"]!=EXCLUSIONS_SHA256 or len(exclusions["reviewed_scene_sha256"])!=779
            or len(exclusions["excluded_reset_seeds"])!=109 or
            hashlib.sha256(PLAN.read_bytes()).hexdigest()!=PLAN_SHA256):
        raise ValueError("B2 must preserve the reviewed scene/seed/construction plans")
    from tdyssey.gpc_adaptation import read_config
    cfg,config_sha256=read_config(ADAPTATION_CONFIG);declared=cfg.get("training_schedule",{})
    path=ROOT/declared.get("path","<missing>")
    if (path.resolve()!=SCHEDULE.resolve() or cfg.get("train",{}).get("order")!="frozen_train_schedule_v1"
            or hashlib.sha256(SCHEDULE.read_bytes()).hexdigest()!=declared.get("sha256")):
        raise ValueError("A complete frozen 500-exposure schedule is required before B2")
    from tdyssey.gpc_training_schedule import load_schedule
    schedule=load_schedule(SCHEDULE,declared["sha256"])
    source=stage_paths("A")[0]/"adaptation-data.json"
    if ((ROOT/cfg["data_manifest"]).resolve()!=source.resolve() or
            schedule.get("source_manifest_sha256")!=A_MANIFEST_SHA256 or
            hashlib.sha256(source.read_bytes()).hexdigest()!=A_MANIFEST_SHA256):
        raise ValueError("B2 may not change or recollect the reviewed A data")
    for stage,expected in (("A","28c873d8f32ccd066938b9d1d65dea1f68b6405f48e1fc338365b016f95020be"),
                           ("B","e3ddffa52e7aec5308e8c737b59a7dde479069b8eb2cb672bc737adc8e85c100")):
        previous=(stage_paths(stage)[0]/"guardian/stop-record.json").read_bytes()
        if hashlib.sha256(previous).hexdigest()!=expected:
            raise ValueError("The reviewed prior A/B stop record changed")
        claim=json.loads(previous)["claim"]
        if hashlib.sha256(Path(claim["path"]).read_bytes()).hexdigest()!=claim["sha256"]:
            raise ValueError("Prior A/B accounts must remain spent and unchanged")
    return dict(training_schedule_sha256=declared["sha256"],
                training_config_sha256=config_sha256)


def worker(path):
    handoff=verify_worker(path)
    sys.meta_path.insert(0,NoModelImports())
    budget=CollectionBudget.create(handoff["stage"],handoff["binding"],claim=handoff["claim"])
    phase={"value":"setup"}
    with Heartbeat(path,payload={"phase":"setup"},refresh=lambda:status(budget,phase["value"])) as heartbeat:
        try:
            _verify_authority(handoff["binding"])
            if hashlib.sha256(PLAN.read_bytes()).hexdigest()!=handoff["binding"]["benchmark_plan_sha256"]:
                raise ValueError("Frozen benchmark construction plan changed")
            exclusions=read_exclusions()
            if exclusions["sha256"]!=handoff["binding"]["exclusions_sha256"]:
                raise ValueError("Frozen scene exclusions changed")
            if handoff["stage"]=="B2":
                frozen=frozen_b2_inputs(exclusions)
                if any(handoff["binding"].get(key)!=value for key,value in frozen.items()):
                    raise ValueError("The train schedule/config changed after B2 launch binding")
            from tdyssey.gpc_data_collection import collect_fixed_runs, real_environment_factory
            factory=real_environment_factory(handoff["worker_options"]["source_root"])
            phase["value"]="running";heartbeat.update()
            if handoff["stage"]=="A":
                spec=importlib.util.spec_from_file_location("collection_entry",ROOT/"scripts/collect_official_play.py")
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                from tdyssey.envs.official import official_shape
                cached={}
                def provenance(config):
                    if not cached:cached.update(module.provenance(config))
                    value=deepcopy(cached);value["geometry_hash"]=official_shape(config.shape).geometry_hash
                    return value
                result=collect_fixed_runs(budget,factory,exclusions=exclusions,provenance_factory=provenance)
            else:
                from tdyssey.gpc_adaptation_data import read_adaptation_manifest
                from tdyssey.gpc_independent_benchmark import run_benchmark
                source=stage_paths("A")[0]/"adaptation-data.json"
                if hashlib.sha256(source.read_bytes()).hexdigest()!=handoff["binding"]["stage_a_manifest_sha256"]:
                    raise ValueError("Completed stage A source changed")
                data=read_adaptation_manifest(source)
                scenes=set(exclusions["reviewed_scene_sha256"])|{row["initial_state_sha256"] for row in data["episodes"]}
                seeds=set(exclusions["excluded_reset_seeds"])|{
                    row["entry"]["collection_metadata"]["reset_seed"] for row in data["episodes"]}
                budget.begin_run("B")
                result=run_benchmark(budget.output,budget,env_factory=factory,excluded_scene_sha256=scenes,
                    excluded_reset_seeds=seeds,plan_path=PLAN)
                budget.finish_run()
            phase["value"]="export";heartbeat.update()
            budget.write_bytes("completion.json",json_bytes(dict(status="complete",stage=handoff["stage"],
                result_file="collection-result.json" if handoff["stage"]=="A" else "benchmark.json",
                counts=budget.counts,held_payload_reads=0,model_loads=0,optimizer_updates=0)))
            budget.finish();phase["value"]="complete";heartbeat.update()
        except BaseException as error:
            budget.fail(type(error).__name__+": "+str(error));phase["value"]="error"
            heartbeat.update();raise
        finally:
            budget.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path)
    parser.add_argument("--stage",choices=("A","B","B2"))
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--source-root",type=Path)
    args=parser.parse_args()
    if args.worker:
        if args.stage or args.execute or args.source_root:parser.error("Worker accepts only its verified handoff")
        worker(args.worker);return
    if not args.execute:
        print(json.dumps(dict(kind="fixed_gpc_collection_plan",stage=args.stage,execution=False,
            models=0,training=0,automatic_retry=False)));return
    if args.stage is None or args.source_root is None:parser.error("Execution requires fixed --stage and --source-root")
    exclusions=read_exclusions()
    binding=dict(authorization_id="20261026-user-authorized-fixed-A-B-no-retry",
        source_commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        exclusions_sha256=exclusions["sha256"],benchmark_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest())
    if args.stage in ("B","B2"):
        output=stage_paths("A")[0]
        stopped=json.loads((output/"guardian/stop-record.json").read_bytes())
        if stopped["status"]!="complete" or not stopped["native_integration_verified"]:
            raise ValueError("Stage B requires completed native stage A; failure stops the plan")
        binding["stage_a_manifest_sha256"]=hashlib.sha256((output/"adaptation-data.json").read_bytes()).hexdigest()
    if args.stage=="B2":
        binding["authorization_id"]="20261026-user-authorized-B-only-02-after-fb1bb20"
        binding.update(frozen_b2_inputs(exclusions))
    report=supervise(args.stage,binding,python_executable=sys.executable,script_path=Path(__file__),
        worker_options=dict(source_root=str(args.source_root.resolve())))
    print(json.dumps(report,indent=2))
    if report["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
