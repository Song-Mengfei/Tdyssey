"""Build the complete local V1/V2 research gallery from saved evidence only.

No simulation, replay, search, training, package installation, or network access.
Re-run after reports/audits finish; refreshes the two saved matrix summaries.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from html import escape
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


BASE = Path(__file__).resolve().parent
OUTCOMES = {
    "queued": "待运行", "running": "运行中 · 暂定", "control_failure": "用完控制预算 · 未成功",
    "timeout": "达到时间预算 · 未成功", "native_success": "达到原生成功条件", "protocol_error": "协议或运行错误",
}
PARITY = {"passed": "通过", "pending": "待补全", "failed": "未通过", "unknown": "未知", "not_applicable": "不适用"}
CSS = """
:root{color-scheme:light;--ink:#182839;--muted:#607082;--line:#dce3eb;--paper:#f3f6fa;--blue:#245e91}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 'Microsoft YaHei','PingFang SC',system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:36px 28px 64px}h1{margin:0 0 12px;font-size:32px;line-height:1.3}h2{font-size:25px;margin:0}h3{font-size:19px;margin:0}
p{margin:8px 0}a{color:var(--blue);text-underline-offset:3px}nav{display:flex;gap:18px;margin:16px 0}.intro,.matrix-head,.card{background:#fff;border:1px solid var(--line);border-radius:14px}
.intro{padding:24px 28px}.notice{background:#edf3fa;border-left:4px solid #5486ad;padding:10px 14px;margin:16px 0}.muted{color:var(--muted)}.small{font-size:13px}.fixed{font-variant-numeric:tabular-nums}
.matrix{margin-top:32px}.matrix-head{padding:20px 24px;margin-bottom:16px}.counts{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.pill{padding:3px 10px;background:#edf1f5;border-radius:6px;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{overflow:hidden}.card-header{padding:18px 20px 12px;display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.status{font-size:13px;background:#eeeef1;border-radius:6px;padding:4px 9px;white-space:normal;max-width:190px}.status.running,.status.queued{background:#e9f1fa;color:#295a86}.status.timeout{background:#fff0dc;color:#87520f}.status.protocol_error{background:#fae4e5;color:#9c3039}.status.native_success{background:#deefe8;color:#246248}
video{display:block;width:100%;max-height:620px;aspect-ratio:1;background:#101c2a;object-fit:contain}.placeholder{display:flex;min-height:190px;align-items:center;justify-content:center;background:#edf1f6;color:var(--muted);padding:24px;text-align:center}
.content{padding:16px 20px 20px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px 9px;margin:0}.metrics div{min-width:0}.metrics dt{font-size:12px;color:var(--muted)}.metrics dd{margin:1px 0;font-size:18px;line-height:1.5;font-weight:650;font-variant-numeric:tabular-nums}.metrics .unit{font-size:11px;font-weight:400;color:var(--muted)}
.evidence{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 0;font-size:13px}details{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}summary{cursor:pointer;color:var(--blue)}.costs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:3px 15px;margin-top:8px;font-size:13px}.costs p{margin:1px 0}.issues{color:#98333d;background:#fff0f0;padding:10px;font-size:13px}.preview{display:block;max-width:760px;width:100%;height:auto;margin:14px auto 0}.footer{margin-top:32px;padding-top:16px;border-top:1px solid var(--line)}
@media(max-width:820px){main{padding:20px 14px 40px}.grid{grid-template-columns:1fr}h1{font-size:26px}.intro{padding:20px}.metrics dd{font-size:17px}.card-header{padding:16px}.content{padding:16px}}
@media print{video{display:none}.grid{grid-template-columns:1fr}.card{break-inside:avoid}details{display:block}}
"""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_url(path):
    path = Path(path).resolve()
    relative = path.relative_to(BASE)
    if not path.is_file():
        raise ValueError(f"Linked asset is missing: {path}")
    return quote(relative.as_posix(), safe="/")


def asset_link(path, label):
    return f'<a href="{escape(local_url(path), quote=True)}">{escape(label)}</a>'


def number(value, decimals=0):
    return "—" if value is None else f"{value:,.{decimals}f}"


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def metric(label, value, unit=""):
    return f'<div><dt>{escape(label)}</dt><dd>{escape(value)} <span class="unit">{escape(unit)}</span></dd></div>'


def audited_video(row):
    options = []
    for name in row.get("audit_paths", []):
        path = Path(name)
        if not path.is_file():
            raise ValueError(f"Recorded audit is missing: {path}")
        audit = read_json(path)
        video = audit.get("video", {})
        if (audit.get("status") == "passed" and video.get("verified")
                and video.get("timing") == "fixed_pause_per_actual_control_not_wall_time_or_interpolated_motion"):
            options.append((path, audit, Path(video["path"])))
    return options[-1] if options else None


def render_card(version, row):
    outcome = row["outcome"]
    key = f"v{version}-{row['scene_request']}-{row['search_seed']}"
    role = "已用于调试的开发场景" if row["scene_role"] == "development" else "预先固定的补充场景"
    parts = [f'<article class="card run-card" id="{key}" data-run="{key}" data-outcome="{outcome}">',
        '<div class="card-header"><div>', f'<h3>场景 {row["scene_request"]} · 搜索种子 {row["search_seed"]}</h3>',
        f'<p class="muted small">{role}</p></div>',
        f'<span class="status {outcome}">{escape(OUTCOMES[outcome])}</span></div>']
    audited = audited_video(row)
    evidence = []
    if audited:
        audit_path, audit, video = audited
        poster = video.parent / "final.png"
        parts.append(f'<video controls preload="none" playsinline poster="{local_url(poster)}" aria-label="V{version} 场景 {row["scene_request"]} 种子 {row["search_seed"]} 实际动作回放">'
                     f'<source src="{local_url(video)}" type="video/mp4">浏览器无法播放时，{asset_link(video, "打开本地视频")}。</video>')
        evidence += [asset_link(video, "实际动作视频"), asset_link(audit_path, "回放核对：通过")]
        timing = f'固定回放：每个实际控制步暂停 {audit["video"].get("pause_seconds", .3):g} 秒；非计算等待时长。'
    else:
        pending = "该实验尚未结束；视频与核对结果待生成。" if outcome in ("queued", "running") else "实验已结束；实际动作视频或独立回放核对仍待补全。"
        parts.append(f'<div class="placeholder">{pending}</div>')
        timing = "视频未就绪，不以候选预测或未完成片段代替实际回放。"
    parts += ['<div class="content">', '<dl class="metrics">',
        metric("实际控制步", f'{row["actual_controls"]} / 300'),
        metric("历史最高覆盖率", percent(row["max_coverage"])),
        metric("最终覆盖率", percent(row["final_coverage"])),
        metric("最高 IoU", percent(row["max_iou"])),
        metric("最终 IoU", percent(row["final_iou"])),
        metric("最终中心误差", number(row["final_position_error_px"], 2), "像素"),
        metric("最终角度误差", number(row["final_angle_error_deg"], 2), "度"),
        metric("发生接触的控制步", number(row["contact_controls"])), '</dl>',
        f'<p class="muted small">{timing}</p>']
    if row["provisional"]:
        parts.append('<p class="muted small">暂定值：仅包含最近一次保存的实际状态和计算成本，当前规划调用尚未记入。</p>')
    report = Path(row["report_path"])
    if report.is_file():
        evidence.append(asset_link(report, "原始报告 JSON"))
    parts.append('<div class="evidence">' + " · ".join(evidence) + '</div>')
    source = PARITY.get(row["source_parity"], row["source_parity"])
    initial = PARITY.get(row["initial_parity"], row["initial_parity"])
    config = PARITY.get(row["config_parity"], row["config_parity"])
    parts += ['<details><summary>计算成本与证据</summary><div class="costs">',
        f'<p>分支恢复：{number(row["restores"])}</p><p>历史重放控制：{number(row["replay_controls"])}</p>',
        f'<p>候选预测控制：{number(row["forecast_controls"])}</p><p>累计规划：{number(row["planning_seconds"], 3)} 秒</p>',
        f'<p>接触计数之和：{number(row["contact_count_sum"])}</p><p>截止时间规划调用：{number(row["deadline_attempt_count"])}</p>',
        f'<p>成本状态：{"最终累计" if row["costs_complete"] else "已保存下限"}</p><p>来源核对：{source}</p>',
        f'<p>初始状态核对：{initial}</p><p>配置核对：{config}</p>', '</div>',
        '<p class="muted small">角度误差为实际 D 状态与目标的最小包角；中心误差使用实际刚体位置。接触计数不是独立接触点的数量。</p></details>']
    if row["issues"]:
        parts.append('<p class="issues">' + escape("；".join(row["issues"])) + '</p>')
    parts.append('</div></article>')
    return "\n".join(parts)


def render_matrix(version, summary, protocol):
    descriptions = {1: "V1 · 初始通用动作搜索", 2: "V2 · 短推、旋转与复合恢复"}
    counts = summary["outcome_counts"]
    ended = sum(counts[name] for name in ("control_failure", "timeout", "native_success", "protocol_error"))
    totals = summary["totals"]
    parts = [f'<section class="matrix" id="v{version}"><div class="matrix-head">',
        f'<h2>{descriptions[version]}</h2>',
        f'<p>{"实验结果已全部保存" if not summary["provisional"] else "结果暂定，保留所有待运行与运行中实验"} · 已结束 {ended}/8 · 原生成功 {counts["native_success"]}/8</p>',
        '<div class="counts">' + "".join(f'<span class="pill">{escape(OUTCOMES[name])}：{counts[name]}</span>' for name in OUTCOMES) + '</div>',
        f'<p class="small fixed">累计成本：实际 {number(totals["actual_controls"])} 控制 · 重放 {number(totals["replay_controls"])} 控制 · 预测 {number(totals["forecast_controls"])} 控制 · 规划 {number(totals["planning_seconds"], 3)} 秒。{"最终成本" if totals["costs_complete"] else "仅为已保存下限"}。</p>',
        f'<p class="small fixed">矩阵实际经过时间：{number(protocol.get("elapsed_seconds"), 3)} 秒（最多 3 个 CPU 进程并发；未结束时留空）。</p>',
        '<p class="small muted">规划秒数为各实验之和；并发运行时不能视为真实经过的墙钟时间。失败和超时成本均已计入；独立回放核对成本另列于 JSON 汇总。</p>',
        '<div class="evidence">' + " · ".join(asset_link(BASE / f"matrix-v{version}" / name, label) for name, label in (
            ("summary.md", "完整汇总"), ("summary.json", "汇总 JSON"), ("summary.csv", "汇总 CSV"), ("protocol.json", "固定实验协议"))) + '</div>',
        '</div><div class="grid">']
    parts += [render_card(version, row) for row in summary["rows"]]
    parts.append('</div></section>')
    return "\n".join(parts)


class GalleryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards, self.refs, self.videos = [], [], 0

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if "run-card" in attributes.get("class", "").split():
            self.cards.append(attributes["data-run"])
        if tag == "video":
            self.videos += 1
            if attributes.get("preload") != "none" or "controls" not in attributes:
                raise ValueError("Videos must use explicit controls and preload=none")
        if tag == "script":
            raise ValueError("Gallery must not load JavaScript")
        for key in ("src", "href", "poster"):
            if attributes.get(key):
                self.refs.append(attributes[key])


def validate(html, expected):
    parser = GalleryParser()
    parser.feed(html)
    if len(parser.cards) != 16 or len(set(parser.cards)) != 16 or set(parser.cards) != expected:
        raise ValueError("Gallery must contain all 16 unique predeclared V1/V2 cards")
    checked = set()
    for reference in parser.refs:
        url = urlsplit(reference)
        if url.scheme or url.netloc or reference.startswith(("/", "\\")):
            raise ValueError(f"Non-relative/local asset URL: {reference}")
        if not url.path:
            continue
        path = (BASE / unquote(url.path)).resolve()
        path.relative_to(BASE)
        if not path.is_file():
            raise ValueError(f"Missing linked asset: {path}")
        checked.add(str(path.relative_to(BASE)))
    return {"status": "passed", "cards": len(parser.cards), "unique_cards": len(set(parser.cards)),
            "card_ids": parser.cards, "embedded_verified_videos": parser.videos,
            "all_asset_urls_relative_within_base": True, "all_linked_files_exist": True,
            "linked_files": sorted(checked), "external_js_or_network_resources": False}


def main():
    helper_path = BASE / "summarize_matrix.py"
    spec = importlib.util.spec_from_file_location("teacher_matrix_summary", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    summaries, protocols, expected = {}, {}, set()
    for version in (1, 2):
        matrix = BASE / f"matrix-v{version}"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = helper.write_outputs(matrix, matrix / "summary", BASE / "scene_inventory.json", BASE)
        protocol = read_json(matrix / "protocol.json")
        pairs = {(scene, seed) for scene in protocol["requested_scene_seeds"] for seed in protocol["search_seeds"]}
        rows = [(row["scene_request"], row["search_seed"]) for row in summary["rows"]]
        if len(rows) != 8 or len(set(rows)) != 8 or set(rows) != pairs:
            raise ValueError(f"V{version} omitted or duplicated declared cases")
        summaries[version], protocols[version] = summary, protocol
        expected.update(f"v{version}-{scene}-{seed}" for scene, seed in pairs)
    generated = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8")
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>D 形状教师验证 · 全量实验回放</title>', f'<style>{CSS}</style></head><body><main>',
        '<header class="intro"><h1>D 形状教师验证 · 全量实验回放</h1>',
        '<p>完整保留 V1、V2 各 8 次预定实验：4 个固定场景 × 搜索种子 42 / 43，共 16 条。结果由已保存的实际动作与状态生成。</p>',
        '<p>每次实验至多 300 个实际控制步、600 秒；最多 3 个 CPU 进程并发。</p>',
        '<div class="notice"><strong>成功标准：实际覆盖率严格大于 95%，随后立即停止。</strong><br>使用已知形状、完整物理状态和真实模拟器搜索动作；未进行学生策略训练。场景 100000 已用于调试，当前结果是开发验证，不代表未见形状泛化。</div>',
        '<p><strong>回放速度：</strong>每个实际控制步显示 0.3 秒，并保留首尾停顿。视频时长不代表规划延迟或真实运行时长。绿色为目标，灰色为 D 物体，蓝色为实际推杆。</p>',
        f'<p class="small muted">页面生成：{generated}。运行中数字为最近一次保存的下限；待完成行不会计为成功。本页不访问网络，也不会自动播放或预加载视频。</p>',
        '<nav><a href="#v1">查看 V1 全部 8 次</a><a href="#v2">查看 V2 全部 8 次</a></nav>']
    if all(not summary["provisional"] for summary in summaries.values()):
        successes = sum(summary["outcome_counts"]["native_success"] for summary in summaries.values())
        parts.append(f'<p class="notice"><strong>本轮最终结果：这 16 次预定实验中，{successes} 次达到原生成功条件。</strong>'
                     '所有未成功、超时和错误结果均保留在下方。</p>')
    preview = BASE / "initial-scenes.png"
    if preview.is_file():
        parts += ['<details><summary>4 个固定初始场景与清单</summary>',
            f'<img class="preview" src="{local_url(preview)}" alt="四个预定 D 场景的原生初始状态">',
            '<p class="small">' + asset_link(BASE / "scene_inventory.json", "初始状态完整清单") + '</p></details>']
    parts.append('</header>')
    parts += [render_matrix(version, summaries[version], protocols[version]) for version in (1, 2)]
    parts += ['<footer class="footer small muted">数据与视频均为本地研究产物。中心/角度误差取最后实际执行状态，不取候选预测。全部成本包含失败和超时实验；独立核对回放的控制成本单独记录。</footer>',
              '</main></body></html>']
    html = "\n".join(parts) + "\n"
    validation = validate(html, expected)
    validation.update(generated_local=generated, training=False, simulator_or_search_started=False,
                      matrix_outcomes={f"v{version}": summaries[version]["outcome_counts"] for version in (1, 2)})
    (BASE / "gallery.html").write_text(html, encoding="utf-8")
    (BASE / "gallery-validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gallery": str(BASE / "gallery.html"), "status": validation["status"],
                      "cards": validation["cards"], "verified_videos": validation["embedded_verified_videos"],
                      "linked_files": len(validation["linked_files"]), "matrix_outcomes": validation["matrix_outcomes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
