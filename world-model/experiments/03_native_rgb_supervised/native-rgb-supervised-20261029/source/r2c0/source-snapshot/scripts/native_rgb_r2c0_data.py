"""R2C0 CPU-only split/reference/draw-plan preparation. No models or physics."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_r2c0_data import (POLICY,PROTOCOL_SHA256,R2C0Dataset,R2C0WindowOrder,build_pool_manifest,choose_split,file_sha,read_json,read_source_catalogue)


def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def prepare(r2b_root,output):
    root,out=Path(r2b_root).resolve(),Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    catalogue=read_source_catalogue(root);split=choose_split(catalogue)
    for entry in catalogue["entries"]:entry["split"]=split["assignments"][entry["trajectory_id"]]
    for name in ("h1","h4"):
        counters=Counter()
        for row in catalogue[name]:
            row["split"]=split["assignments"][row["trajectory_id"]];row["dataset_index"]=counters[row["split"]];counters[row["split"]]+=1
        write(out/(name+"-windows.json"),catalogue[name])
    write(out/"split-plan.json",split)
    manifest=dict(kind="R2C0-R2B-saved-RGB-reference.v1",source_root=os.path.relpath(root,out).replace('\\','/'),
        protocol_sha256=PROTOCOL_SHA256,source_identity=catalogue["source_identity"],entries=catalogue["entries"],
        catalogues={k:dict(file=n,sha256=file_sha(out/n)) for k,n in (("h1","h1-windows.json"),("h4","h4-windows.json"),("split","split-plan.json"))},
        counts=dict(trajectories=53,train=41,seen_val=12,H1=1113,H4=954,train_H1=861,val_H1=252,train_H4=738,val_H4=216),
        input_semantics="image5CHW=4inputhistory+H1truth;action4=3historyactions+current; onlyimage/action enter network; rawuint8storedRGB/255 and action/255.5-1 withoutclipping",
        source_scope="R2B only; source file/hash references, no RGB copies; one blockedslot has no datasetrow; no oldD0/D1/R2A/R2A2 mixed",models_or_physics=0)
    write(out/"manifest.json",manifest)
    return dict(manifest=str(out/"manifest.json"),sha256=file_sha(out/"manifest.json"),split_sha256=file_sha(out/"split-plan.json"),counts=manifest["counts"],val=split["selections"])


def draw_summary(indices,rows):
    selected=[rows[i] for i in indices];families=Counter(r["family_id"] for r in selected);windows=Counter(r["window_id"] for r in selected)
    distribution=lambda key:dict(min=min(r[key] for r in selected),median=float(np.median([r[key] for r in selected])),p95=float(np.quantile([r[key] for r in selected],.95)),max=max(r[key] for r in selected))
    return dict(draws=len(selected),letter_group=dict(Counter(r["letter"]+"/"+("moving" if r["moving"] else "other") for r in selected)),
        category=dict(Counter(r["category"] for r in selected)),phase=dict(Counter(r["phase"] for r in selected)),contact=dict(Counter(str(r["agent_object_contact"]) for r in selected)),
        rotation_sign=dict(Counter("positive>=5" if r["signed_rotation_deg"]>=5 else "negative<=-5" if r["signed_rotation_deg"]<=-5 else "smaller" for r in selected)),
        boundary=dict(Counter(r["boundary_process"] for r in selected)),actual_arc_ids=sorted({a for r in selected for a in r["actual_contact_arc_ids"]}),
        actual_letter_arc_ids=sorted({r["letter"]+"/"+str(a) for r in selected for a in r["actual_contact_arc_ids"]}),
        letter_category=dict(Counter(r["letter"]+"/"+r["category"] for r in selected)),
        actual_lever_signs=dict(Counter(s for r in selected for s in r["signed_lever_signs"])),independent_families=len(families),family_exposures=dict(families),
        unique_windows=len(windows),maximum_window_exposures=max(windows.values()),repeated_exposures=len(selected)-len(windows),
        translation_image96_px=distribution("translation_image96_px"),abs_rotation_deg=distribution("abs_rotation_deg"))


def draw_plan(manifest_path):
    dataset=R2C0Dataset(manifest_path,split="train",cache_bytes=1024*1024);out=dataset.root
    pools=build_pool_manifest(dataset.window_rows,dataset.identity());write(out/"pools.json",pools)
    order=R2C0WindowOrder(pools);first=order.take(480);state480=order.state_dict();allindices=first+order.take(4320)
    for count,values in ((480,first),(4800,allindices)):
        write(out/f"draws{count}.json",dict(kind="R2C0-pre-model-proposed-TRAIN-draws.v1",pool_identity=order.identity(),seed=5478,indices=values,
            window_ids=[dataset.window_rows[i]["window_id"] for i in values],draws_are_exposures_not_independent_samples=True))
    write(out/"order-state480.json",state480);write(out/"order-state4800.json",order.state_dict())
    audit=dict(kind="R2C0-pre-model-draw-audit.v1",policy=POLICY,pool_identity=order.identity(),
        source_support={l:{g:dict(unique_windows=sum(len(v) for cats in pools["groups"][l][g].values() for v in cats.values()),independent_families=len(pools["groups"][l][g])) for g in ("moving","other")} for l in POLICY["letters"]},
        first_cycle=draw_summary(first,dataset.window_rows),full_proposal=draw_summary(allindices,dataset.window_rows),
        cycles=[dict(cycle=i,**draw_summary(allindices[i*480:(i+1)*480],dataset.window_rows)) for i in range(10)],
        batches=[dict(batch=i,**draw_summary(allindices[i*16:(i+1)*16],dataset.window_rows)) for i in range(300)],
        cycles_are_not_identical=all(allindices[i*480:(i+1)*480]!=first for i in range(1,10)),
        model_optimizer_physics_calls=0,training_permission=False)
    write(out/"draw-audit.json",audit)
    lines=["# R2C0引用数据与抽样准备","","R2B只通过小规模学习诊断数据准入。53轨迹按完整family切为TRAIN41/seen-val12，只有新选定0.1秒协议；没有RGB副本、模型或物理调用。seen-val刻意含运动，不是无偏自然测试或未见字母。","",
        "TRAIN861 H1/738 H4，val252 H1/216 H4。明显运动定义为H1面积质心≥2图像px或abs角≥5°，其它合法窗保留。Dataset五帧是四历史输入和一个H1真值，四动作是三个历史动作和当前动作；仅image/action允许进入网络。","",
        "480周期每字母64明显+16其它；每6draw随机字母轮转，再letter/group family均衡，再family互斥运动类别/window队列。两运动条件重叠只归both一次。family类别不齐时如实列支持，不能借别的字母补类。后续RNG延续，4800不是机械复制首周期。所有重复只计exposure。","",
        "|字母|明显unique/family|其它unique/family|","|---|---|---|"]
    for l,s in audit["source_support"].items():lines.append(f"|{l}|{s['moving']['unique_windows']}/{s['moving']['independent_families']}|{s['other']['unique_windows']}/{s['other']['independent_families']}|")
    lines += ["",f"首480使用{audit['first_cycle']['unique_windows']}个不同窗；拟4800使用{audit['full_proposal']['unique_windows']}个不同窗、最大单窗曝光{audit['full_proposal']['maximum_window_exposures']}。逐16draw batch、各cycle的真实phase/contact/幅度/方向/arc/family/repeat见draw-audit.json。","",
        "split-plan.json记录seed/hash选点、TRAIN保留平移/旋转/可用正负、初态near检查与完整signature重复披露；原始source duplicate审计另见父目录source-duplicate-audit.json。H1/H4catalogue保持全部源窗口，抽样不改自然分母或原值。","",
        "300×16/4800draw只是冻结提案；300尝试/1200模型秒尚未授权，closed250/868.484旧账本不动。"]
    with (out/"DATA_REPORT.md").open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    dataset.close();return dict(pools_identity=order.identity(),first_cycle=audit["first_cycle"],full_proposal=audit["full_proposal"],cycles_not_identical=audit["cycles_are_not_identical"])


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("prepare");s.add_argument("--r2b-root",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("draw-plan");s.add_argument("--manifest",type=Path,required=True)
    a=p.parse_args();result=prepare(a.r2b_root,a.output) if a.command=="prepare" else draw_plan(a.manifest)
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
