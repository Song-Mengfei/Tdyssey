"""R2B selected-config media from saved RGB only; no simulator/model calls.

Dynamic executed actions are checked against each persisted policy decision.
All54 slots retain their planned denominator, including blocked/partial cases.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import native_rgb_r2a_media as shared

ORIGINAL_HELPER_SHA256 = "04a5570fa86d1a78180a0a6a3a1dd2b7dd29dc8ea2f1c096d3926cae6cdded59"
SELECTED_PROTOCOL_SHA256 = "58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6"
FPS, SCALE = 10, 6
WALLS = shared.WALLS
sha, json_write, inside, safe_id = shared.sha, shared.json_write, shared.inside, shared.safe_id
_first, _finite_list, measures = shared._first, shared._finite_list, shared.measures
encode, verify_lossless = shared.encode, shared.verify_lossless
text, short, frame_fields = shared.text, shared.short, shared.frame_fields


def planned_phase(control):
    if control < 4:return "接近/建立接触"
    if control < 14:return "持续反馈推送"
    if control < 20:return "撤回/自然减速观察"
    return "有限重接触"


def guard(root, cap):
    # Shared helper is immutable; its arithmetic cap applies only to this root.
    return shared.storage_guard(root, cap)


def load_slot(root, row, request):
    sid = safe_id(row.get("slot_id", row.get("request_id", request["request_id"])))
    result = dict(slot_id=sid, request=request, index=row, attempted=row.get("attempted"), status=row.get("status", "unknown"), available=False,
        dispatchable=request.get("dispatchable"), blocked_reason=_first(row,("blocked_reason","reason","error"),_first(request,("blocked_reason","static_blocked_reason","issues"))))
    npz = inside(root, _first(row, ("trajectory_file", "npz_file", "file"), f"trajectories/{sid}/trajectory.npz"))
    meta = inside(root, row.get("metadata_file", f"trajectories/{sid}/metadata.json"))
    if not npz.exists():
        if row.get("attempted") is True:
            result["missing_reason"] = "attempted_but_no_saved_trajectory"
        else:
            result["missing_reason"] = "not_executed_or_no_saved_trajectory"
        return result
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    if metadata.get("protocol_sha256") != SELECTED_PROTOCOL_SHA256:
        raise ValueError(f"{sid}: saved protocol is not the selected0.1s configuration")
    with np.load(npz, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    rgb, states, actions = arrays["rgb"], arrays["states"], arrays["actions"]
    n = len(actions)
    if rgb.dtype != np.uint8 or rgb.shape != (n + 1, 3, 96, 96):
        raise ValueError(f"{sid}: uint8 RGB N+1 CHW96 contract failed")
    if states.shape != (n + 1, 15) or actions.shape != (n, 2) or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"{sid}: finite state/action timeline contract failed")
    if (actions < 0).any() or (actions > 511).any():
        raise ValueError(f"{sid}: absolute action contract failed")
    if n > int(request["max_controls"]):
        raise ValueError(f"{sid}: recorded controls exceed released slot cap")
    if metadata.get("request", {}).get("request_id", request["request_id"]) != request["request_id"]:
        raise ValueError("Metadata request identity mismatch")
    records = metadata.get("control_records", [])
    completed_records = {}
    failed_tail_records = []
    unclassified_records = []
    for record in records:
        if record.get("completed") is True:
            control = record.get("control_index")
            if type(control) is not int or control < 0 or control >= n or control in completed_records:
                raise ValueError(f"{sid}: completed record index outside saved boundary actions or duplicated")
            completed_records[control] = record
        elif record.get("completed") is False:
            failed_tail_records.append(record)
        else:
            unclassified_records.append(record)
    missing_completed_records = [t for t in range(n) if t not in completed_records]
    declared = _first(metadata, ("frame_sha256", "frameSHA", "frame_sha256_chw_uint8"))
    frame_hashes = [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in rgb]
    if isinstance(declared, dict):
        declared = declared.get("sha256", declared.get("frames"))
    if declared is not None:
        declared = [v.get("sha256") if isinstance(v, dict) else v for v in declared]
        if declared != frame_hashes:
            raise ValueError(f"{sid}: saved CHW uint8 frame hash mismatch")
    centers, gaps = measures(request["letter"], states)
    transitions = []
    for t in range(n):
        # Raw boundary frames remain usable even if a failed IO left metadata
        # incomplete; unknown contact/penetration is never fabricated as zero.
        record = completed_records.get(t, {})
        decision = record.get("decision")
        if not isinstance(decision, dict) or "proposed_action" not in decision:
            raise ValueError(f"{sid}: completed control {t} lacks explicit dynamic decision.proposed_action")
        proposed = np.asarray(decision["proposed_action"], dtype=np.float64)
        if proposed.shape != (2,) or not np.isfinite(proposed).all() or not np.array_equal(actions[t], proposed.astype(actions.dtype)):
            raise ValueError(f"{sid}: saved action differs from dynamic decision {t}")
        expected_state_sha = hashlib.sha256(np.ascontiguousarray(states[t],dtype=np.float64).tobytes()).hexdigest()
        if decision.get("state_before_sha256") != expected_state_sha or decision.get("control_index") != t:
            raise ValueError(f"{sid}: decision state-before/control index binding differs at {t}")
        contact = _first(record, ("contact", "contact_count", "object_pusher_contact_count"))
        if contact is None and "contact" in arrays:
            contact = int(arrays["contact"][t])
        arc = _first(record, ("contact_arc_bins", "actual_contact_arc_bins", "actual_arc_bins", "contacted_bin_ids"))
        pen = _first(record, ("max_substep_penetrations", "max_substep_penetrations_world_px", "penetration", "max_penetrations"))
        if pen is None and "substep_penetrations" in arrays:
            pen = np.asarray(arrays["substep_penetrations"][t]).max(0)
        delta = (centers[t + 1] - centers[t]) * 96 / 512
        transitions.append(dict(slot_id=sid, letter=request["letter"], protocol="0.1秒半衰期（已选）", protocol_sha256=SELECTED_PROTOCOL_SHA256, control=t,
            from_state=t, to_state=t + 1, time_start=t / FPS, time_end=(t + 1) / FPS,
            dx_image_px=float(delta[0]), dy_image_px=float(delta[1]), translation_image_px=float(np.linalg.norm(delta)),
            signed_rotation_deg=float(np.degrees(states[t + 1, 4] - states[t, 4])),
            area_centroid_world_xy=centers[t + 1].tolist(), actual_contact=bool(contact) if contact is not None else None,
            contact_count=contact, contact_arc_bins=arc, max_substep_penetrations=_finite_list(pen),
            actual_contact_point_samples=record.get("actual_contact_point_samples"),
            penetration_max=float(np.max(pen)) if pen is not None and np.isfinite(pen).all() else None,
            actual_four_wall_gaps_world_px=dict(zip(WALLS, map(float, gaps[t + 1]))),
            actual_two_nearest_walls=[dict(wall=WALLS[i], gap_world_px=float(gaps[t + 1, i])) for i in np.argsort(gaps[t + 1])[:2]],
            absolute_action_xy=actions[t].tolist(), action_target_distance_world_px=float(np.linalg.norm(actions[t] - states[t, :2])),
            action_phase=_first(decision, ("actual_phase", "phase"), _first(record, ("action_phase", "phase"), "unknown")),
            planned_phase=planned_phase(t), decision=decision, decision_action_exact=True,
            valid=_first(record, ("valid", "valid_transition")), failure=record.get("failure")))
    result.update(available=True, attempted=True, arrays=arrays, metadata=metadata, frame_hashes=frame_hashes,
        frames=rgb, controls=n, transitions=transitions, gaps=gaps, centers=centers,
        npz_file=npz.relative_to(root).as_posix(), npz_sha256=sha(npz), metadata_file=meta.relative_to(root).as_posix(), metadata_sha256=sha(meta),
        declared_frame_hashes_verified=declared is not None, initial_state_raw_sha256=hashlib.sha256(np.ascontiguousarray(states[0]).tobytes()).hexdigest(),
        initial_identity=metadata.get("initial_identity"), protocol_sha256=_first(metadata, ("protocol_sha256", "protocolSHA")),
        failure=_first(metadata, ("failure", "error"), row.get("failure")),
        control_records_total=len(records), completed_record_count=len(completed_records),
        missing_completed_record_controls=missing_completed_records,
        failed_tail_records=failed_tail_records, unclassified_control_records=unclassified_records,
        partial_endpoint=metadata.get("partial_endpoint"),
        partial_control_span_seconds=metadata.get("partial_control_span_seconds"),
        partial_time_exact=metadata.get("partial_time_exact"),
        partial_state_substep_indices=metadata.get("partial_state_substep_indices"),
        unavailable_partial_state_substeps=metadata.get("unavailable_partial_state_substeps"),
        partial_states_present="partial_states" in arrays,
        partial_states_saved_count=len(arrays["partial_states"]) if "partial_states" in arrays else None,
        partial_rgb_available=bool(np.asarray(arrays["partial_rgb"]).size) if "partial_rgb" in arrays else None,
        telemetry_file=metadata.get("telemetry_file"), telemetry_sha256=metadata.get("telemetry_sha256"),
        decisionlogs_file=metadata.get("decisionlogs_file"), decisionlogs_sha256=metadata.get("decisionlogs_sha256"),
        policy_identity=metadata.get("policy_identity"),
        partial_endpoint_is_boundary_frame=False,
        physical_span_seconds=n / FPS, encoded_duration_seconds=(n + 1) / FPS, final_endpoint_hold_seconds=.1)
    return result


def curves(slot):
    image = Image.new("RGB", (960, 280), "white")
    draw = ImageDraw.Draw(image)
    rows = slot.get("transitions", [])
    controls = int(slot["request"]["max_controls"])
    specs = (("translation_image_px", "单步面积质心 Δ图像px", (0., max([8.] + [r["translation_image_px"] for r in rows]))),
             ("signed_rotation_deg", "单步带符号 Δ角度°", None),
             ("penetration_max", "穿透worldpx；红线2px", (0., max([2.] + [r["penetration_max"] for r in rows if r["penetration_max"] is not None]))))
    angular = max([15.] + [abs(r["signed_rotation_deg"]) for r in rows])
    for j, (key, title, bounds) in enumerate(specs):
        x = 16 + j * 316
        text(draw, (x, 0), title, 16)
        shared._points(draw, (x, 26, x + 296, 118), [r.get(key) for r in rows], "#196fc0",
            bounds=bounds or (-angular, angular), controls=controls, threshold=2. if key == "penetration_max" else None)
        for step in (4, 14, 20):
            xx = x + step / controls * 296
            draw.line((xx, 26, xx, 118), fill="#b8c4cd", width=1)
    text(draw, (16, 150), "实际四墙带符号gap(worldpx)：左蓝/右橙/上绿/下紫；不等于墙接触次数", 16)
    rect = (16, 179, 940, 248)
    draw.rectangle(rect, outline="#b2bdc7")
    gaps = slot.get("gaps", np.empty((0, 4)))
    if len(gaps):
        lo, hi = min(-2., float(gaps.min())), max(64., float(gaps.max()))
        for k, color in enumerate(("#196fc0", "#d65c16", "#278c44", "#8c4bb4")):
            ps = [(rect[0] + i / controls * (rect[2] - rect[0]), rect[3] - (float(v) - lo) / (hi - lo) * (rect[3] - rect[1])) for i, v in enumerate(gaps[:, k])]
            if len(ps) > 1:draw.line(ps, fill=color, width=2)
        text(draw, (16, 252), f"完整原值见CSV/JSON；gap范围{lo:.2f}～{hi:.2f}；竖线为计划阶段边界，不代表实际接触成功", 14)
    return image


def single_frame(slot, i, chart):
    image = Image.new("RGB", (960, 1120), "#edf1f4")
    draw = ImageDraw.Draw(image)
    req = slot["request"]
    text(draw, (16, 3), "R2B | 0.1秒半衰期（已选） | 10Hz / 1×真实时间", 24)
    text(draw, (16, 38), f"{req['letter']} / {req.get('initial_space',req.get('spatial_class','未知空间'))} / {slot['slot_id']} / {slot['status']}", 16)
    text(draw, (16, 68), f"状态帧{i}  t={i/FPS:.1f}s；物理跨度{slot.get('physical_span_seconds',0):.1f}s；完整终帧额外显示0.1s不计运动", 17)
    if not slot.get("available") or i >= len(slot["frames"]):
        draw.rectangle((16, 108, 944, 820), fill="#253242")
        text(draw, (70, 330), "未执行 / 受阻 / 早停后无完整记录", 28, "white")
        text(draw, (70, 380), "审阅占位牌，不是物理数据帧；不得伪造25帧", 22, "#ffd078")
        image.paste(chart, (0, 840))
        return np.asarray(image)
    raw = np.moveaxis(slot["frames"][i], 0, -1)
    source = Image.fromarray(raw)
    image.paste(source, (16, 110))
    image.paste(source.resize((576, 576), Image.Resampling.NEAREST), (192, 244))
    text(draw, (136, 108), "左为原96×96；下为仅显示放大6×nearest；上一控制的接触/幅度另列", 16)
    if i == 0:
        text(draw, (136, 135), "初始帧：没有前一控制，接触/动作/幅度均不作成功声明", 17)
    else:
        r = slot["transitions"][i - 1]
        text(draw, (136, 131), f"控制{i-1}  计划:{r['planned_phase']} / 实际decision:{r['action_phase']}", 16)
        text(draw, (136, 154), f"Δ面积质心={r['translation_image_px']:.3f}px  Δθ={r['signed_rotation_deg']:+.3f}°  valid={r['valid']}", 17)
        text(draw, (136, 177), f"AO命中={r['contact_count']}/10子步(非独立事件)  实际弧段={r['contact_arc_bins']}", 16)
        ax, ay = r["absolute_action_xy"]
        text(draw, (136, 200), f"实际目标=({ax:.2f},{ay:.2f}) 距推杆{r['action_target_distance_world_px']:.2f}worldpx  穿透max={short(r['penetration_max'])}", 15)
        samples = r["actual_contact_point_samples"]
        if samples is None:
            label = "真实接触点：未记录（不伪造）"
        else:
            preferred = sorted(samples, key=lambda p: 0 if p.get("phase") == "post_solve" else 1)
            xy = [p["object_world"] for p in preferred if p.get("object_world") is not None][:2]
            label = f"真实点样本共{len(samples)}个；world " + " / ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in xy)
        text(draw, (136, 223), label, 15)
    for k, (start, end, label) in enumerate(((0,3,"接近/建立"),(4,13,"反馈推送"),(14,19,"撤回/减速"),(20,23,"有限重接触"))):
        yy = 290 + k * 85
        color = "#196fc0" if i > 0 and start <= i-1 <= end else "#687580"
        text(draw, (16, yy), f"计划{start}–{end}", 16, color)
        text(draw, (16, yy + 25), label, 16, color)
    text(draw, (16, 678), "计划≠接触成功", 15)
    text(draw, (16, 704), "无AO/滑行保留", 15)
    text(draw, (16, 730), "无额外物理渲染", 15)
    text(draw, (16, 824), "实际4墙gap(worldpx): " + " / ".join(f"{k}={v:.2f}" for k,v in zip(WALLS,slot['gaps'][i])), 15)
    image.paste(chart, (0, 840))
    if not np.array_equal(np.asarray(image.crop((16,110,112,206))),raw):
        raise RuntimeError("Native96 pixels changed during annotation")
    return np.asarray(image)


def verify_single_timing(path, frames):
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path),pix_fmt="rgb24")
    metadata = next(reader)
    count = sum(1 for _ in reader)
    if tuple(metadata["size"])!=(960,1120) or metadata["fps"]!=10 or count!=frames:
        raise RuntimeError("Single-slot review video fps/size/frame count mismatch")
    return dict(decoded_frames=count,fps=metadata["fps"],duration_seconds=count/FPS,codec=metadata.get("codec"))


def _csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys or ["slot_id"]);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def build_media(r2b_root, *, index_path=None,requests_path=None,output=None,confirmed_cold=False,max_r2b_bytes=128*1024**2):
    started=time.monotonic();root=Path(r2b_root).resolve()
    if sha(ROOT/"scripts/native_rgb_r2a_media.py")!=ORIGINAL_HELPER_SHA256:
        raise RuntimeError("Original frozen helper source changed")
    index_path=Path(index_path).resolve() if index_path else root/"index.json"
    requests_path=Path(requests_path).resolve() if requests_path else root/"frozen-requests.json"
    index=json.loads(index_path.read_text(encoding="utf-8"))
    if not confirmed_cold and not (index.get("collection_closed") or index.get("closed")):
        raise ValueError("Cold collection confirmation required; never read hot trajectories")
    source=json.loads(requests_path.read_text(encoding="utf-8"));requests=source["requests"] if isinstance(source,dict) else source
    requests=[r for r in requests if r.get("phase")=="R2B"]
    if len(requests)!=54 or len({r["request_id"] for r in requests})!=54:
        raise ValueError("R2B requires all54 planned slots, including unexecuted/blocked")
    if any(int(r["max_controls"])!=24 for r in requests):raise ValueError("Only24-control released slots")
    indexed=index["slots"];indexed=list(indexed.values()) if isinstance(indexed,dict) else indexed
    lookup={str(r.get("slot_id",r.get("request_id"))):r for r in indexed}
    if len(lookup)!=len(indexed):raise ValueError("Duplicate index slot identities")
    output=inside(root,output or "media")
    if output.exists():raise FileExistsError("Preserve existing media attempt; output must be new")
    guard(root,max_r2b_bytes);output.mkdir(parents=True);(output/"slots").mkdir()
    slots=[];controls=[];sections=[]
    for req in requests:
        sid=req["request_id"];row=lookup.get(sid,dict(slot_id=sid,attempted=False,status="not_executed"))
        slot=load_slot(root,row,req);directory=output/"slots"/safe_id(slot["slot_id"]);directory.mkdir()
        if slot.get("available"):
            chart=curves(slot);chart.save(directory/"curves.png")
            raw=[]
            for i,frame in enumerate(slot["frames"]):
                p=directory/f"frame-{i:03d}.png";Image.fromarray(np.moveaxis(frame,0,-1)).save(p)
                raw.append(dict(frame_index=i,state_index=i,time_seconds=i/FPS,chw_uint8_sha256=slot["frame_hashes"][i],png_file=p.name,png_sha256=sha(p)))
            native=directory/"native96-10fps-1x-lossless.mkv"
            encode(native,(np.moveaxis(frame,0,-1) for frame in slot["frames"]),lossless=True)
            slot["lossless_decoded_frames_verified"]=verify_lossless(native,slot["frames"])
            review=directory/"selected-0p1s-1x-6x.mp4"
            encode(review,(single_frame(slot,i,chart) for i in range(len(slot["frames"]))))
            slot["review_timing"]=verify_single_timing(review,len(slot["frames"]))
            Image.fromarray(single_frame(slot,0,chart)).save(directory/"poster.png")
            slot.update(native_video=native.relative_to(output).as_posix(),native_video_sha256=sha(native),review_video=review.relative_to(output).as_posix(),review_video_sha256=sha(review))
            json_write(directory/"frames.json",dict(frames=raw,physical_span_seconds=slot["physical_span_seconds"],encoded_duration_seconds=slot["encoded_duration_seconds"],final_endpoint_hold_seconds=.1))
            json_write(directory/"controls.json",slot["transitions"]);_csv(directory/"controls.csv",slot["transitions"])
            fields=frame_fields(slot);json_write(directory/"frame-fields.json",fields);_csv(directory/"frame-fields.csv",fields)
            partial={k:slot[k] for k in ("failure","failed_tail_records","partial_control_span_seconds","partial_time_exact","partial_state_substep_indices","unavailable_partial_state_substeps","partial_states_present","partial_states_saved_count","partial_rgb_available","partial_endpoint_is_boundary_frame","metadata_file","npz_file","telemetry_file","decisionlogs_file","decisionlogs_sha256","policy_identity")}
            json_write(directory/"partial-evidence.json",partial);controls.extend(slot["transitions"])
            source_meta=Path(os.path.relpath(inside(root,slot["metadata_file"]),output)).as_posix()
            partial_text="未知（不补零）" if slot["partial_control_span_seconds"] is None else f"{slot['partial_control_span_seconds']:.3f}s，独立partial证据，不增加完整帧"
            content=f'<video controls preload="none" src="{review.relative_to(output).as_posix()}" poster="{(directory/"poster.png").relative_to(output).as_posix()}"></video><p>完整控制{slot["controls"]}；物理跨度{slot["physical_span_seconds"]:.1f}s；容器{slot["encoded_duration_seconds"]:.1f}s（末帧显示.1s）；partial {partial_text}</p>'
            for filename,label in (("native96-10fps-1x-lossless.mkv","原96无损视频"),("frames.json","全部原始帧SHA"),("frame-fields.json","完整接触点"),("controls.csv","逐控制/decision CSV"),("curves.png","曲线"),("partial-evidence.json","失败/partial证据")):
                content+=f'<a href="{(directory/filename).relative_to(output).as_posix()}">{label}</a> '
            content+=f'<a href="{html.escape(source_meta,quote=True)}">原metadata</a>'
            if slot.get("decisionlogs_file"):
                decision_href=Path(os.path.relpath(inside(root,slot["decisionlogs_file"]),output)).as_posix()
                content+=f' <a href="{html.escape(decision_href,quote=True)}">原始decision日志</a>'
        else:
            content='<p class="missing">未执行/受阻/缺少原始记录。无视频，不伪造25帧；保留计划分母。原因：'+html.escape(str(slot.get("blocked_reason") or slot.get("missing_reason")))+'</p>'
        sections.append(f'<section><h2>{html.escape(sid)} | {html.escape(req["letter"])} | {html.escape(str(slot["status"]))}</h2>{content}</section>')
        slots.append(slot);guard(root,max_r2b_bytes)
    _csv(output/"controls.csv",controls)
    _csv(output/"slots.csv",[dict(slot_id=s["slot_id"],letter=s["request"]["letter"],planned=True,dispatchable=s.get("dispatchable"),blocked_reason=s.get("blocked_reason"),attempted=s.get("attempted"),status=s["status"],available=s["available"],complete_controls=s.get("controls"),physical_span_seconds=s.get("physical_span_seconds"),partial_span_seconds=s.get("partial_control_span_seconds"),review_video=s.get("review_video"),failure=s.get("failure")) for s in slots])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>R2B 已选0.1秒半衰期：全部54槽</title><style>body{font-family:Microsoft YaHei,sans-serif;background:#edf1f4;margin:24px}section{padding:18px;background:white;margin:20px 0}video{width:100%;max-width:960px}a{margin-right:14px}.missing{color:#9e2033}</style><h1>R2B：0.1秒半衰期（已选）</h1><p>全部54计划槽。10Hz/1×真实时间；原96px+明确6×nearest显示；原PNG/FFV1无损，H264仅审阅编码。计划阶段不是接触成功。动态action与保存decision逐值核对，未发生AO如实保留；partial不是完整0.1s帧。未执行/失败不替换、不伪造25帧。</p><p><a href="manifest.json">清单</a><a href="slots.csv">全部54槽分母</a><a href="controls.csv">完整逐控制表</a></p>'+''.join(sections)+'</html>'
    (output/"index.html").write_text(page,encoding="utf-8")
    result=dict(kind="R2B-selected0p1-saved-media.v1",source_script_sha256=sha(__file__),original_helper_sha256=ORIGINAL_HELPER_SHA256,index_sha256=sha(index_path),requests_sha256=sha(requests_path),
        selected_label="0.1秒半衰期（已选）",selected_protocol_sha256=SELECTED_PROTOCOL_SHA256,space_damping=.0009765625,planned_slots=54,attempted_slots=sum(s.get("attempted") is True for s in slots),saved_slots=sum(s["available"] for s in slots),
        complete_controls=sum(s.get("controls",0) for s in slots),slots=[shared._without_arrays(s) for s in slots],fps=FPS,display_scale=SCALE,display_interpolation="nearest",raw_codec="FFV1 decoded RGB bitwise verified",review_codec="H264 display only",dynamic_actions_equal_recorded_decisions=True,
        old_videos_reencoded=0,new_physics_controls=0,model_calls=0,extra_physics_render_calls=0,cpu_wall_seconds=time.monotonic()-started)
    result["stage_bytes_before_final_manifest"]=guard(root,max_r2b_bytes);json_write(output/"manifest.json",result);guard(root,max_r2b_bytes)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True);p.add_argument("--index");p.add_argument("--requests");p.add_argument("--output",default="media");p.add_argument("--confirmed-cold",action="store_true");p.add_argument("--max-r2b-bytes",type=int,default=128*1024**2);a=p.parse_args()
    r=build_media(a.root,index_path=a.index,requests_path=a.requests,output=a.output,confirmed_cold=a.confirmed_cold,max_r2b_bytes=a.max_r2b_bytes)
    print(json.dumps({k:r[k] for k in ("planned_slots","attempted_slots","saved_slots","complete_controls","cpu_wall_seconds")},ensure_ascii=False))


if __name__=="__main__":main()

