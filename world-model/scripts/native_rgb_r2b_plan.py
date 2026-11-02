"""Write the54-slot R2B static plan; never construct/step an environment."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2b_sampling import build_plan


def write_plan(plan,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,value in (("sampling-rules.json",plan["rules"]),("proposed-requests.json",plan),("static-review.json",plan["static_review"])):
        with (output/name).open("x",encoding="utf-8") as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    rows=[]
    for q in plan["requests"]:
        c=q.get("selected_contact",{});i=q.get("initialization",{})
        rows.append(dict(execution_order=q["execution_order"],request_id=q["request_id"],letter=q["letter"],initial_space=q["initial_space"],role=q["role"],process_goal=q.get("process_goal"),
            dispatchable=q["dispatchable"],requested_gap=q["requested_gap_world_px"],selected_gap=q.get("selected_gap_world_px"),gap_request_met=q.get("gap_request_met"),
            yaw=i.get("angle_rad"),origin=i.get("object_origin"),pusher=i.get("pusher_start"),actual_four_gaps=q.get("actual_initial_wall_gaps"),
            arc=c.get("arc_id"),local_point=c.get("point_local"),local_normal=c.get("outward_normal_local"),normal_signed_lever=c.get("signed_lever_world_px"),
            resultant_signed_lever=c.get("resultant_signed_lever_world_px"),tangent_ratio=c.get("drive_tangent_ratio"),free_approach_path=c.get("free_approach_path"),
            source_family=q["source_family_id"],old_missing=q["old_static_missing_goals"],resolution=q["old_static_gap_resolution"],missing=q["static_missing_reasons"]))
    with (output/"proposed-requests.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    lines=["# R2B新采样计划：0.1秒半衰期（已选）","",f"54新槽×最多24=1296控制，仅静态规划；可派发{plan['static_review']['dispatchable_slots']}，不可派发保留54分母且不补槽。旧R1B请求不直接执行，源仅提供初始pose出处。协议、引擎PD/force/speed/geometry不变。","",
        "首6固定为T正旋转、C平移、X平移、B平移、L负旋转、O沿墙；均在54内。表中角色是意图，不是实际接触或运动成功。","",
        "0–3接近，4–13当前位姿固定local arc跟随推送，14–19合法撤回并固定释放目标观察，20–23同arc有限重接触。恢复不延长24；连续推送内部H4单列，不把withdraw混作持续接触。","",
        "法向目标在当前contact center向内48worldpx；approach/recontact24；平移切向比例仅按静态力矩消除公式且绝对≤.5，mixed固定±.35。每步目标距当前推杆≤64worldpx、相邻目标变化≤96、目标墙内22.5..488.5。free route与允许进入物体的contact-push虚拟目标严格区分：后者验证外部prefix与指定arc入口，不称整条路径无碰撞；真实接触由100Hz日志决定。","",
        "失联2控后最多4控同arc恢复，当前几何不可证则hold并记录miss，不换arc/场景、不瞬移。纯几何有限yaw/gap/arc选择不是动力学前瞻；封闭洞不派发，窄槽/墙堵只称几何未证。任何真实>2px/nonfinite/IO或规则不一致由root停整个阶段。","",
        "|字母|可派发|平移不同arc|正/负旋转意图|oblique/wall意图|计划最少支持|","|---|---:|---|---|---:|---|"]
    for l,s in plan["static_review"]["per_letter_support"].items():lines.append(f"|{l}|{s['dispatchable']}|{s['translation_intent_distinct_arcs']}|{s['positive_rotation_intents']}/{s['negative_rotation_intents']}|{s['oblique_or_wall_intents']}|{s['planned_minima_met']}|")
    lines += ["","旧4静态缺口逐项列static-review.json；新点/pose/gap的静态解法不代表旧实际目标已实现。四signed gaps、双墙与4/18/48请求是否兑现逐槽CSV，不为比例强摆。全部54/失败/小运动/无接触/墙角数据和连续视频都保留；只root可按1296实际预算执行，模型/GPU/训练0。"]
    with (output/"PLAN.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-plan",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();plan=build_plan(args.source_plan);write_plan(plan,args.output)
    print(json.dumps(plan["static_review"],ensure_ascii=False))


if __name__=="__main__":main()
