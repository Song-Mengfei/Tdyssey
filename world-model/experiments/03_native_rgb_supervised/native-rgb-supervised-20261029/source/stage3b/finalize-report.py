"""CPU-only evidence/state writer. Does not import torch or scientific runtime."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


work = read(HERE / "model-work.json")
run = work["training_result"]
progress = run["progress"]
budget = work["budget"]
ledger = read(ROOT / "native-rgb-cost-ledger.json")
score = read(HERE / "score/score-report.json")
repeat = read(HERE / "baseline-prefix-comparison.json")
integrity = read(HERE / "review-postrun-integrity.json")
budget_review = read(HERE / "review-postrun-budget.json")
assert work["failure"] is None and budget["closed"] and ledger["active_stage"] is None
assert integrity["integrity_checks_passed"] and integrity["finalized"] and not integrity["pending"]
assert budget_review["verdict"]["budget_and_training_chain_checks_passed"]
assert repeat["all_four_raw_bytes_identical"] and not repeat["review_required"]
assert ledger["charged_updates"] == ledger["completed_updates"] == progress["updates"] == 129
assert ledger == budget["ledger"] and ledger["charged_seconds"] <= 600
assert score["train"]["denominator"] == 18 and score["development"]["denominator"] == 36
assert not score["train"]["dynamic_fit_target"]["passed"]

storage = read(HERE / "final-storage.json")
seen = set()
total = 0
for root in storage["roots"]:
    path = Path(root)
    assert path.exists(), path
    paths = path.rglob("*") if path.is_dir() else [path]
    for p in paths:
        if not p.is_file():
            continue
        key = str(p.resolve()).casefold()
        if key not in seen:
            seen.add(key)
            total += p.stat().st_size
now = datetime.now(timezone.utc).isoformat()
storage_snapshot = {
    "kind": "stage3b-storage-at-callback.v1", "checked_at_utc": now,
    "root_list_source": "final-storage.json", "roots": storage["roots"],
    "files": len(seen), "bytes": total, "reserved_atomic_bytes": 268435456,
    "projected_bytes": total + 268435456, "max_bytes": 3221225472,
    "passed": total + 268435456 <= 3221225472,
    "qualification": "Metadata snapshot before final result/callback receipt small files; no cleanup or deletion."
}
assert storage_snapshot["passed"]
write(HERE / "storage-at-callback.json", storage_snapshot)

def cohort_summary(cohort):
    return {k: cohort[k] for k in ["denominator", "methods", "fitted_vs_hold", "fitted_vs_frozen"]}

evidence_names = [
    "REPORT.md", "DIAGNOSIS.md", "model-work.json", "run/result.json", "run/last.pt",
    "predictions/frozen-train18.npz", "predictions/frozen-dev36.npz",
    "predictions/fitted-train18.npz", "predictions/fitted-dev36.npz",
    "run/history.jsonl", "run/sampling-history.jsonl", "source-identity.json",
    "entry-from-stage3.diff", "reused-inputs.json", "entry-cpu-checks.json",
    "baseline-prefix-comparison.json", "score/score-report.json", "score/per-window.csv",
    "range-accounting.json", "score-execution.json", "DIAGNOSIS.json",
    "score/files.json", "review-postrun-budget.json", "review-postrun-integrity.json",
    "root-visual-review.json", "final-storage.json", "storage-at-callback.json"
]
evidence = {name: {"path": str(HERE / name), "sha256": digest(HERE / name)} for name in evidence_names}
result = {
    "kind": "native-rgb-stage3b-corrective-microfit-result.v1", "stage": "3B", "created_at_utc": now,
    "outcome": "actual_training_chain_verified_bounded_motion_fit_target_not_met",
    "execution_cycle_finished": True, "one_actual_execution": True, "failure": work["failure"],
    "runtime_status": run["status"], "stop_reason": progress["stop_reason"],
    "training_chain_verified": True, "model_work_clock_closed": True,
    "original_initialization_sha256": run["identity"]["initialization"]["sha256"],
    "native_parameter_count": run["native_parameter_count"],
    "optimizer_attempts": ledger["charged_updates"], "optimizer_updates": progress["updates"],
    "training_draws": progress["samples"], "unique_training_windows_selected": 18,
    "selected_window_exposures": progress["selected_window_exposures"],
    "gradient_parameter_tensors_finite_nonzero": len(progress["first_gradient_norms"]),
    "changed_parameter_tensors": progress["parameter_update_report"]["changed_parameter_tensors"],
    "loss_before": progress["fixed_train_loss_before"], "loss_after": progress["fixed_train_loss_after"],
    "denoising_validation": progress["validation"], "checkpoint": run["last"],
    "checkpoint_semantic_review": work["checkpoint_review"],
    "frozen_prediction_frames_generated": 54, "fitted_prediction_frames_generated": 54,
    "raw_outputs_unchanged": True, "raw_range_summaries": integrity["raw_summaries"],
    "offline_scoring_wall_seconds": None, "offline_scoring_duration_instrumented": False,
    "repeated_first4_identical": True, "baseline_max_absolute_difference": 0,
    "train_dynamic_fit_target": score["train"]["dynamic_fit_target"],
    "train": cohort_summary(score["train"]), "development": cohort_summary(score["development"]),
    "development_H1_motion": score["development"]["groups"]["H1_motion"],
    "shape_interpretation": "Rounded contours remain visible. TRAIN18 legacy unknown; DEV2fail/3assessable+33unknown. Unknown is not a pass.",
    "model_work_seconds": budget["elapsed_model_work_seconds"],
    "prior_stage3_model_work_seconds": budget["aggregate_seconds_before"],
    "aggregate_new_round_model_seconds": ledger["charged_seconds"],
    "aggregate_new_round_attempts": ledger["charged_updates"], "model_work_overrun_seconds": budget["overrun_seconds"],
    "peak_cuda_allocated_bytes": run["peak_cuda_allocated"], "peak_cuda_reserved_bytes": run["peak_cuda_reserved"],
    "remaining_new_round_attempts": 3000-ledger["charged_updates"],
    "remaining_new_round_seconds": 5400-ledger["charged_seconds"], "remaining_is_not_retry_authorization": True,
    "historical_native_optimizer_attempts": 1899, "historical_native_model_seconds": 5352.516,
    "physical_controls": 0, "supervised_physical_controls": 39680, "all_native_physical_controls": 74240,
    "no_H4": True, "no_full319": True, "no_unseen": True, "no_server": True, "no_packaging": True,
    "integrity_and_protocol_audit_passed": True, "budget_and_training_audit_passed": True,
    "storage_at_callback": {k:v for k,v in storage_snapshot.items() if k != "roots"},
    "evidence": evidence, "next_action": "Actual parent callback then waiting_parent and stop; later stages require new release."
}
write(HERE / "stage3b-result.json", result)
state = read(ROOT / "stage-state.json")
state.update({
    "current_stage": "3B", "status": "ready_to_report", "stage3b_execution_cycle_complete": True,
    "stage3b_training_chain_verified": True, "stage3b_dynamic_fit_demonstrated": False,
    "stage3b_outcome": result["outcome"], "stage3b_result_file": "stage3b/stage3b-result.json",
    "stage3b_result_sha256": digest(HERE / "stage3b-result.json"),
    "stage3b_training_completed_updates": 129, "future_stages_authorized": False,
    "stage3b_remaining_new_round_budget": {"attempts": 2871, "seconds": 5400-ledger["charged_seconds"], "authorized_to_spend": False},
    "next_action": "Actual stage3B parent result callback; save receipt then waiting_parent and stop."
})
state["resources"].update({"optimizer_attempts":129,"stage3b_completed_optimizer_updates":129,
    "stage3b_training_draws":2064,"stage3b_model_work_seconds":budget["elapsed_model_work_seconds"],
    "aggregate_model_work_seconds":ledger["charged_seconds"],"stage3b_native_prediction_frames":108,
    "stage3b_native_prediction_batch_calls":28,"model_inference_calls":29,
    "supervised_native_prediction_batch_calls":29,"combined_persistent_bytes_at_report":total})
state["callback"].update({"sent":False,"message_file":"stage3b/callback-stage3b.txt","result_file":"stage3b/callback-result.json"})
write(ROOT / "stage-state.json", state)
callback = f'''[阶段3B完成回传] 单次纠正执行已冷态结束；真实完成129次优化器尝试/129更新，2064draws，原54+拟合54个H1全部生成。链路验证通过，但18训练动态拟合目标未达；没有第三次模型启动。请父审查后决定下一步，本实施回传成功后waiting_parent并停止。

实际学习证据：原发布SHA4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892严格重新初始化，8,729,827全参数、296首梯度均finite>0、296参数hash均变化；18固定窗各114/115次，fixed-noise loss .0342575386→.00242708654，dev denoise .00265394265。实际129更新在尾部预留截止处正常停，不是执行200步或OOM。last105247534B SHA e94f9349e1affb8372aacc4613bd57ade496704a50b63d257f77cb4c5ad3626a，10完整字段CPU语义load(.125s)于416.266s完成且纳入时钟。没有额外resume/GPU探针。

核心结论：冻结20%+12/18目标未达。TRAIN fitted原始灰1-IoU=.2280759621，hold=.2325831710，只改善1.93789%，9/18strictwins，0hardfail。外观由原红色前景转灰/蓝，但轮廓圆化/细槽和棱角受损；TRAIN legacy18全unknown，不能称形状通过。DEV36 fitted=.274351965 vs hold=.196032459，4/36wins；moving31=.287241605 vs .223749623，4/31wins、1hardfail（J seen-val-2c-097:38推杆空mask）；static4 drift error=.174189315/centroid .582984463 imagepx，2/3assessable shape失败+1unknown，不要求胜过零。全DEV形状2/3失败+33unknown。loss降低/比原错误颜色baseline好不替代运动门槛，也不足据此否定全训练路线。

原18/36/真值/批次/seed/协议完全不变；旧首4与新baseline逐值byteSHA完全相等，maxabs0。唯一finite越界不abort修改按授权执行，原始值/status仍为failure。original TRAIN5/18与DEV13/36越界，302+850负channel，无>1，逐张min/max/计数已保存；fitted54全部finite[0,1]。score继续保留所有失败/全分母，原TRAIN11hardfail/DEV26，拟合0/1。DIAGNOSIS单列惩罚与实际像素测量，不另调阈值/clip/映射/校准/几何loss/头。全54行与9页固定图库齐。

计账：旧stage3 28.3440000000046s/0完整保留；3B416.2810000000027s，合计444.6250000000073<600，129<200，globalledgerclosed且overrun0。最后更新390.797<391.656截止，baseline/finalnative/save/CPUcheckpointload/cleanup均计费。峰allocated1790047744B/reserved2025848832B。新轮3000/5400数值余额2871/4955.375，不是后续授权；旧native1899/5352.516另列不重置。CPU评分/审计单列，不计作模型秒。物理新增0，监督39680/全native74240不变。旧stage3全部152、原保护381、64冻结源/快照均保持。现存储{total}B+268435456原子预留={total+268435456}B<3221225472；未删旧数据/权重、未下载或安装。

实际命令：../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_stage3b_run.py --execute（唯一模型执行exit0）；既有compare-baseline-prefix.py（CPU exit0）和未变native_rgb_stage3_score.py score（CPU exit0，只运行一次，完整18/36）。完整参数见REPORT与命令log。5聚焦CPU检查、26预算审查、独立源/输入/实际预测审查通过；审查未再次加载模型。

绝对证据目录：{HERE}
关键文件：REPORT.md、stage3b-result.json、DIAGNOSIS.md、model-work.json、run/result.json、run/last.pt、run/history.jsonl、run/sampling-history.jsonl、predictions/frozen-train18.npz、frozen-dev36.npz、fitted-train18.npz、fitted-dev36.npz、score/score-report.json、score/per-window.csv、score/files.json、baseline-prefix-comparison.json、review-postrun-budget.json/.md、review-postrun-integrity.json/.md、entry-from-stage3.diff、source-identity.json、storage-at-callback.json。

请审查链路、训练拟合未达、DEV回退和形状缺陷，再明确下阶段。未执行全319/H4/未见8/新物理/服务器/ZIP；不自动扩大训练或第三次模型执行。
'''
(HERE / "callback-stage3b.txt").write_text(callback, encoding="utf-8")
state["callback"]["message_sha256"] = digest(HERE / "callback-stage3b.txt")
write(ROOT / "stage-state.json",state)
print(json.dumps({"ready_to_report":True,"result_sha256":state["stage3b_result_sha256"],"storage_bytes":total,"callback_characters":len(callback)},ensure_ascii=True))
