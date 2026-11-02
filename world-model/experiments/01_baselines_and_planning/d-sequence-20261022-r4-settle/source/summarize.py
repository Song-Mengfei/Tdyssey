"""Independent saved-report accounting and local video gallery; no simulation."""
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tdyssey.envs.official import official_shape

BASE = Path(__file__).resolve().parent
OLD = ROOT/"artifacts/d-sequence-20261022-r2"
MID = ROOT/"artifacts/d-sequence-20261022-r3"
FIRST = ROOT/"artifacts/d-sequence-20261022-r4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_errors(state):
    pose, goal = state["object_body"]["pose"], state["goal"]
    return math.dist(pose["position"],goal["position"]), abs(math.degrees(math.atan2(
        math.sin(goal["angle"]-pose["angle"]),math.cos(goal["angle"]-pose["angle"]))))


def independent_rms(state):
    points = sorted({v for component in official_shape("D").components for v in component.vertices})
    def transform(pose,p):
        return (pose["position"][0]+math.cos(pose["angle"])*p[0]-math.sin(pose["angle"])*p[1],
                pose["position"][1]+math.sin(pose["angle"])*p[0]+math.cos(pose["angle"])*p[1])
    return math.sqrt(sum(math.dist(transform(state["object_body"]["pose"],p),transform(state["goal"],p))**2 for p in points)/len(points))


def main():
    rows, issues, sources, total_audit_controls = [], [], [], 0
    for revision, base in ((2,OLD),(3,MID),(4,FIRST),(5,BASE)):
        protocol = json.loads((base/"protocol.json").read_text(encoding="utf-8"))
        assert protocol["status"] == "complete"
        source_ok = all(digest(base/"source"/name)==value for name,value in protocol["source_sha256"].items())
        sources.append(dict(revision=revision,frozen_source_copies_match=source_ok))
        if not source_ok:
            issues.append(f"revision{revision} source snapshot mismatch")
        for index, job in enumerate(protocol["jobs"]):
            if job["scene"] not in (100000,100200):
                continue
            assert job["status"] == "process_complete"
            path = Path(job["output"])/"report.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            name = path.parent.name
            audit_dir = (base/"videos"/f"scene-{job['scene']}"/f"00-{name}" if revision==2 else
                         base/"videos"/f"{index:02d}-{name}")
            audit = json.loads((audit_dir/"audit.json").read_text(encoding="utf-8"))
            if audit["status"] != "passed":
                issues.append(f"{revision}/{name}: media replay failed")
            total_audit_controls += audit["counts"]["prefix_controls"]+audit["counts"]["source_controls"]
            first = data["search_start_state"]
            if revision in (3,4,5):
                previous=json.loads((OLD/name/"report.json").read_text(encoding="utf-8"))
                if first != previous["search_start_state"]:
                    issues.append(f"{name}: comparison start differs")
                for k in ("search_seed","steps","seconds","horizon","max_evaluations","max_global_searches","resume_step"):
                    if data["config"][k] != previous["config"][k]:
                        issues.append(f"{name}: comparison cap differs: {k}")
            last = data["steps"][-1]["state"] if data["steps"] else first
            position,angle = pose_errors(last)
            start_position,start_angle = pose_errors(first)
            if digest(job["source"]) != job["source_sha256"]:
                issues.append(f"{name}: source report changed")
            if len(data["steps"]) != data["counts"]["control_steps"] or len(data["steps"])>140:
                issues.append(f"{name}: actual control accounting mismatch")
            for offset, step in enumerate(data["steps"],161):
                if step["state"]["control_step"] != offset or step["prediction_state_error"] > 1e-7:
                    issues.append(f"{name}: transition/prediction mismatch")
                if abs(independent_rms(step["state"])-step["pose_rms"])>1e-7:
                    issues.append(f"{name}: RMS mismatch")
            for key in ("restores","replay_steps","candidate_steps"):
                count = sum(item["counts"][key] for item in data["planning_attempts"])
                if count != data["search_counts"][key]:
                    issues.append(f"{name}: {key} total mismatch")
            for decision in data["decisions"]:
                if decision["evaluations"]>40 or decision["global_searches"]>8 or decision["forecast_budget"]>80:
                    issues.append(f"{name}: search cap exceeded")
                if decision["continuation"] and any(decision["counts"].values()):
                    issues.append(f"{name}: continuation unexpectedly simulates branches")
                for key in ("restores","replay_steps","candidate_steps"):
                    if sum(c["counts"][key] for c in decision["candidates"]) != decision["counts"][key]:
                        issues.append(f"{name}: candidate accounting mismatch")
            if data["success"] != bool(data["steps"] and data["steps"][-1]["upstream_done"]):
                issues.append(f"{name}: success flag mismatch")
            if any(step["upstream_done"] for step in data["steps"][:-1]):
                issues.append(f"{name}: controls after native success")
            if data["success"] and data["final_metrics"]["coverage"]<=.95:
                issues.append(f"{name}: non-native success threshold")
            if data["torch_loaded"] or not data["resume_sources_unchanged"]:
                issues.append(f"{name}: scope/source changed")
            row = dict(revision=revision,scene=job["scene"],success=data["success"],stop_reason=data["stop_reason"],
                new_controls=len(data["steps"]),total_control=last["control_step"],
                start_coverage=data["initial_metrics"]["coverage"],max_coverage=data["max_coverage"],
                final_coverage=data["final_metrics"]["coverage"],max_iou=data["max_iou"],final_iou=data["final_metrics"]["iou"],
                start_position_px=start_position,final_position_px=position,start_angle_deg=start_angle,final_angle_deg=angle,
                start_rms=independent_rms(first),final_rms=independent_rms(last),global_searches=data["global_searches"],
                restores=data["search_counts"]["restores"],replay=data["search_counts"]["replay_steps"],
                forecast=data["search_counts"]["candidate_steps"],elapsed_seconds=data["elapsed_seconds"],
                planning_seconds=sum(p["seconds"] for p in data["planning_attempts"]),
                audit="passed",report=str(path),video=str(audit_dir/"actual-controls-paused.mp4"),
                audit_report=str(audit_dir/"audit.json"))
            rows.append(row)
    result = dict(status="passed" if not issues else "failed",issues=issues,sources=sources,rows=rows,
        resumed_development_not_reset=True, training=False,checkpoint_loaded=False,
        independent_audit_controls_including_prefix=total_audit_controls,
        independent_audit_reset_substeps=8,
        video_timing="0.3 seconds per new actual control; no interpolation; planning latency separately recorded",
        revisions={str(v):dict(outcomes=dict(Counter(r["stop_reason"] for r in rows if r["revision"]==v)),
            successes=sum(r["success"] for r in rows if r["revision"]==v),
            costs={k:sum(r[k] for r in rows if r["revision"]==v) for k in
                   ("new_controls","restores","replay","forecast","planning_seconds")}) for v in (2,3,4,5)})
    (BASE/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    with (BASE/"summary.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    def relative(value):
        import os
        return html.escape(Path(os.path.relpath(value,BASE)).as_posix(),quote=True)
    cards=[]
    for row in rows:
        state = "成功：覆盖率超过95%" if row["success"] else "未成功："+row["stop_reason"]
        cards.append(f'''<article><h2>修订 {row['revision']} · 场景 {row['scene']} · {state}</h2>
<video controls preload="none" src="{relative(row['video'])}"></video>
<p>最高 / 最终覆盖率：{100*row['max_coverage']:.2f}% / {100*row['final_coverage']:.2f}%</p>
<p>中心误差：{row['start_position_px']:.2f} → {row['final_position_px']:.2f} px；角度误差：{row['start_angle_deg']:.2f} → {row['final_angle_deg']:.2f}°</p>
<p>联合误差：{row['start_rms']:.2f} → {row['final_rms']:.2f} px；新增控制 {row['new_controls']}，总控制 {row['total_control']}；全局搜索 {row['global_searches']} 次</p>
<p>历史回放 {row['replay']:,}，预测 {row['forecast']:,}；运行 {row['elapsed_seconds']:.1f}s</p>
<a href="{relative(row['report'])}">原始报告</a> · <a href="{relative(row['audit_report'])}">独立回放核对</a></article>''')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>D 卡点恢复实验</title>
<style>body{max-width:1300px;margin:auto;padding:30px;background:#f4f6fa;color:#17283a;font:16px/1.7 system-ui,"Microsoft YaHei"}h1{font-size:28px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;padding:20px;border:1px solid #dae1eb;border-radius:12px}h2{font-size:18px}video{width:100%;aspect-ratio:1;background:#152537}a{color:#275c91}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>D 联合纠偏：两个失败场景的四轮同预算比较</h1>
<p>全部从旧 V2 场景第160步恢复；不是从重置状态成功。保留修订2、3的四次失败，以及本轮修订4和执行修正5的全部四次结果。单次最多新增140步，完整回合上限300步，600秒，8轮全局搜索，每轮40次尝试，总预测视野80步。</p>
<p>每个新增实际控制步固定停留0.3秒，无插帧；播放速度不代表规划耗时。绿色为目标，灰色为D，蓝色为推杆。没有加载策略权重或训练学生。</p>
<p>修订2和3为历史失败；修订4约束中心与角度，保护不同候选类型并允许三段组合。修订5修复按压后保持尚未执行就重新搜索的时序问题，并按实际执行前缀评价终点。它改变多个部分，结果不构成单机制消融；没有延长预算或追加种子。</p>
<div class="grid">'''+"\n".join(cards)+"</div></html>"
    for row in rows:
        if not Path(row["video"]).is_file():
            raise ValueError("Missing video")
    (BASE/"gallery.html").write_text(page,encoding="utf-8")
    (BASE/"summary.md").write_text("# Two failed D states: matched revision 2/3/4/5 recovery comparison\n\n"+
        "| Revision | Scene | Result | Controls new/total | Peak/final coverage | Position px start/final | Angle deg start/final | RMS start/final |\n|---|---|---|---|---|---|---|---|\n"+
        "\n".join(f"| {r['revision']} | {r['scene']} | {r['stop_reason']} | {r['new_controls']}/{r['total_control']} | {r['max_coverage']:.6f}/{r['final_coverage']:.6f} | {r['start_position_px']:.3f}/{r['final_position_px']:.3f} | {r['start_angle_deg']:.3f}/{r['final_angle_deg']:.3f} | {r['start_rms']:.3f}/{r['final_rms']:.3f} |" for r in rows)+
        "\n\nIndependent audit: "+result["status"]+". Frozen source copies checked per version; no current-code substitution for the first version. Not reset or held-out successes.\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","issues","revisions","independent_audit_controls_including_prefix")},indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
