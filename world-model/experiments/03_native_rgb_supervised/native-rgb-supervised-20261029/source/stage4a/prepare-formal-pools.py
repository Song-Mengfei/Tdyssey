"""Single CPU-only metadata preparation; exactly360 dry draws, no dataset/render."""
from collections import Counter
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.native_rgb_stage2b_report import read_merged_records
from tdyssey import native_rgb_formal_sampling as sampling


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")


def main():
    if sys.argv[1:] != ["--execute-cpu-preparation"]:
        raise SystemExit("Explicit --execute-cpu-preparation required; no model work exists here")
    output = Path(__file__).resolve().parent
    supervised = output.parent
    marker = output/"FORMAL_POOL_CPU_STARTED.json"
    if marker.exists():
        raise FileExistsError("The real metadata two-cycle dryrun may not be silently replayed")
    write(marker, dict(scope="CPU metadata only; at most360 drydraws, not training exposures",
        started_unix=time.time(), real_training_draws=0, optimizer_attempts=0, ledger_touched=False))
    began = time.monotonic()
    joint = supervised/"stage2c/joint-data"
    joint_manifest = json.loads((joint/"manifest.json").read_text(encoding="utf-8"))
    reference = supervised/"stage2b/report/merged-train-manifest.json"
    if sampling._file_hash(reference) != sampling.TRAIN_REFERENCE_SHA256:
        raise ValueError("Frozen TRAIN reference changed")
    records = read_merged_records(reference, supervised)
    train_entries = [entry for entry in joint_manifest["entries"] if entry["split"] == "train"]
    if len(records) != 396 or [r["entry"]["trajectory_id"] for r in records] != [e["trajectory_id"] for e in train_entries]:
        raise ValueError("Joint TRAIN entry order differs from the formal merged reference")
    for record, entry in zip(records, train_entries):
        for key in ("trajectory_id", "shape", "sha256", "metadata_sha256"):
            if record["entry"][key] != entry[key]:
                raise ValueError("Joint TRAIN record binding differs: "+key)
    rows = sampling.candidate_rows_from_records(records)
    expected = dict(moving=14692, intermediate=1017, near_static_contact=1176, near_static_no_contact=7271)
    if len(rows) != 24156 or dict(Counter(row["index_stratum"] for row in rows)) != expected:
        raise ValueError("Original formal H1 strata changed")
    identity = dict(manifest_sha256=sampling._file_hash(joint/"manifest.json"),
        audit_sha256=sampling._file_hash(joint/"physics-audit.json"),
        split_sha256=sampling.canonical_hash(joint_manifest["splits"]),
        train_reference_manifest_sha256=sampling._file_hash(reference),
        motion_statistics_sha256=sampling._file_hash(ROOT/"tdyssey/native_rgb_motion_statistics.py"))
    pools = sampling.build_pool_manifest(rows, data_identity=identity)
    support = sampling.support_summary(rows)
    support.update(recorded_trajectories=len(records), original_real_controls=sum(len(r["arrays"]["actions"]) for r in records),
        raw_data_or_gate_filtered=False, dataset_constructed=False, render_calls=0,
        initial_space_counts=dict(Counter(row["requested_spatial_class"] for row in rows)),
        actual_spatial_before_counts=dict(Counter(row["actual_spatial_class_before"] for row in rows)),
        boundary_behavior_counts=dict(Counter(row["boundary_behavior"] for row in rows)),
        contact_by_motion={name:dict(Counter(str(row["contact"]) for row in rows if row["motion_class"]==name))
                           for name in ("moving", "intermediate", "near_static")},
        pool_identity=pools["identity"], data_identity=identity)
    write(output/"formal-pools.json", pools)
    write(output/"formal-pool-support.json", support)
    original = sampling.FormalWindowOrder(pools)
    prefix = [index for _ in range(11) for index in original.take(16)]
    state = json.loads(json.dumps(original.state_dict()))
    write(output/"formal-sampling-dryrun-state.json", dict(artifact_role="CPU_DRYRUN_NOT_TRAINING_CHECKPOINT",
        actual_training_draws=0, simulated_cursor=176, state=state))
    resumed = sampling.FormalWindowOrder(pools)
    resumed.load_state_dict(state)
    after_restore_at176_before_draw = resumed.state_dict() == state
    crossing = resumed.take(16)
    indices = prefix+crossing+[index for _ in range(10) for index in resumed.take(16)]+resumed.take(8)
    assert len(indices) == 360 and resumed.cursor == 360 and original.cursor == 176
    by_index = {row["dataset_index"]:row for row in rows}
    draws = [by_index[i] for i in indices]
    cycles = []
    for cycle in range(2):
        selected = draws[cycle*180:(cycle+1)*180]
        per_letter = {letter:dict(Counter(row["index_stratum"] for row in selected if row["letter"]==letter))
                      for letter in sampling.LETTERS}
        assert all(counts == sampling.QUOTAS for counts in per_letter.values())
        cycles.append(dict(cycle=cycle, draws=len(selected), strata=dict(Counter(row["index_stratum"] for row in selected)),
                           per_letter=per_letter))
    dryrun = dict(kind="formal-sampling-real-metadata-two-cycle-cpu-dryrun.v1", simulated_draws=360,
        actual_training_draws=0, optimizer_attempts=0, ledger_touched=False, model_loads=0, model_calls=0,
        native_sampler_calls=0, physics_controls=0, dataset_constructed=False, render_calls=0,
        unique_windows=len(set(indices)), independent_trajectories=len({row["trajectory_id"] for row in draws}),
        restored_full_state_exact_at176=after_restore_at176_before_draw,
        cross_cycle_batch_positions=[176,191], cross_cycle_batch_indices=crossing,
        cycles=cycles, identity=pools["identity"], final_state_sha256=resumed.state_dict()["state_sha256"],
        future500_draws_estimate_only=8000,
        future500_note="44complete180-slot cycles plus80tail slots; exact70/10/10/10 applies to complete cycles, not each batch16. Future actual repeats/exposures are unmeasured.",
        wall_seconds=time.monotonic()-began)
    write(output/"formal-sampling-dryrun.json", dryrun)
    fields=["draw", "dataset_index", "letter", "trajectory_id", "anchor", "index_stratum", "requested_spatial_class",
            "actual_spatial_class_before", "boundary_behavior", "contact", "motion_class"]
    with (output/"formal-sampling-dryrun.csv").open("x", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for index,row in enumerate(draws):
            writer.writerow(dict(draw=index, **{key:row[key] for key in fields if key!="draw"}))
    lines=["# 正式TRAIN分层采样CPU准备", "", "仅使用冻结状态/动作metadata；0Dataset构造、0渲染、0模型/权重/原生采样/优化器/物理。原始数据与静止帧保留。", "",
        "正式TRAIN396轨迹/25344真实控制，24156个原有效H1候选；独立validity与原dataset catalogue逐anchor一致，joint顺序/hash与2A+2B引用一致。",
        "", "四层唯一窗口：moving14692、intermediate1017、near-static-contact1176、near-static-no-contact7271。",
        "", "|字母|M窗口/轨迹|I窗口/轨迹|SC窗口/轨迹|SN窗口/轨迹|", "|---|---:|---:|---:|---:|"]
    for letter in sampling.LETTERS:
        values=support["per_letter_stratum"][letter]
        lines.append("|"+letter+"|"+"|".join(f"{values[name]['unique_windows']}/{values[name]['independent_trajectories']}" for name in sampling.STRATA)+"|")
    lines += ["", "每180槽严格每字母7M+1I+1SC+1SN，由独立seed5478 slot RNG洗牌；每层先trajectory均衡队列，再该轨迹层内window无放回轮转，耗尽才洗牌。较小轨迹层可能更常重复，属于训练曝光；不增加物理证据。",
        "", f"真实metadata只模拟360draws：{len(set(indices))}unique窗口、{dryrun['independent_trajectories']}独立轨迹。176处完整JSON状态恢复完全相同，下个16batch跨180边界，两个周期各126/18/18/18；无训练账本消费。合成测试额外流只来自假metadata。",
        "", "原始初始9空间H1分母："+json.dumps(support['initial_space_counts'],ensure_ascii=False)+"。",
        "", "实际边界行为："+json.dumps(support['boundary_behavior_counts'],ensure_ascii=False)+"。",
        "", f"letter×stratum×初始空间空格共{len(support['empty_letter_stratum_initial_space_cells'])}，详见JSON；全72个letter-stratum均非空，绝不向别层借样本。边界静止/墙阻挡为保存状态和接触proxy，不是wall arbiter或成功证明。",
        "", "低窗口支持（窗口/独立轨迹）："+"；".join(f"{v['letter']} {v['stratum']} {v['unique_windows']}/{v['independent_trajectories']}" for v in support['lowest_support'])+"。",
        "", "未来500次有效更新预估8000draws只是假设；44完整周期+80尾部，尾部和实际尝试/完成/unique/repeats将另记。此准备不授权模型执行或下一阶段。"]
    (output/"formal-pool-support.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(candidate_windows=len(rows),counts=expected,identity=pools['identity'],cpu_drydraws=360,
        unique_dry_windows=len(set(indices)),restore_exact=after_restore_at176_before_draw,cycle_quota_checks_passed=True,
        runtime_model_calls=0,optimizer_attempts=0,physics_controls=0,wall_seconds=dryrun['wall_seconds']),ensure_ascii=False))


if __name__ == "__main__":
    main()
