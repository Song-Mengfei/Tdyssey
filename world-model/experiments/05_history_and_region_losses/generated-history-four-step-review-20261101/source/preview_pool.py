"""Review-only grouping of saved H1 indices; no simulator/model/Torch calls."""
from pathlib import Path
from collections import defaultdict, Counter
import json, hashlib
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"artifacts/accelerated-rgb-run-20261031/selection/train-index.json"
OUT=Path(__file__).resolve().parent
payload=SRC.read_bytes(); rows=json.loads(payload)
if isinstance(rows,dict): rows=rows["rows"]
groups=defaultdict(list)
for r in rows:
    assert r["split"]=="train" and r["horizon"]==1
    groups[(r["family_id"],r["raw_path"])].append(r)
segments=[]; leftovers=[]; meta_hashes={}
for (family,path),rs in sorted(groups.items()):
    mp=Path(path).with_suffix(".json"); mb=mp.read_bytes(); m=json.loads(mb)
    meta_hashes[str(mp)]=hashlib.sha256(mb).hexdigest()
    assert m["split"]=="train" and m["family_id"]==family
    rs=sorted(rs,key=lambda r:r["anchor"]); runs=[]; current=[]
    for r in rs:
        if current and r["anchor"]!=current[-1]["anchor"]+1: runs.append(current);current=[]
        current.append(r)
    if current: runs.append(current)
    for run in runs:
        end=len(run)//4*4
        for i in range(0,end,4):
            block=run[i:i+4]; a=block[0]["anchor"]
            assert a>=3 and a+4<=m["clean_control_count"]
            segs=m.get("segments",m.get("blocks",[]))
            committed=any(int(s.get("start_control",-1))<=a and a+4<=int(s.get("end_control",-1)) for s in segs)
            segments.append(dict(segment_id=f"{family}:a{a}:four-step-review",family_id=family,
                letter=block[0]["letter"],raw_path=path,anchor=a,supervised_H1_ids=[x["sample_id"] for x in block],
                committed_future=committed,contact_motion_targets=sum(x["diagnostic_contact_and_motion"] for x in block)))
        leftovers.extend(r["sample_id"] for r in run[end:])
ids=[x for s in segments for x in s["supervised_H1_ids"]]
assert len(ids)==len(set(ids)) and len(ids)+len(leftovers)==len(rows)
summary=dict(review_only=True,execution_authorized=False,source_sha256=hashlib.sha256(payload).hexdigest(),
    source_H1_windows=len(rows),four_step_segments=len(segments),supervised_target_frames=len(ids),
    independent_families=len({s["family_id"] for s in segments}),uncovered_H1_ids_count=len(leftovers),
    optimizer_batches_per_arm=(len(segments)+3)//4,sequence_microbatch=4,effective_target_batch=16,
    committed_sequences=sum(s["committed_future"] for s in segments),
    feedback_recorded_sequences=sum(not s["committed_future"] for s in segments),
    per_letter={l:dict(segments=sum(s["letter"]==l for s in segments),
        targets=4*sum(s["letter"]==l for s in segments),
        families=len({s["family_id"] for s in segments if s["letter"]==l}),
        contact_motion_targets=sum(s["contact_motion_targets"] for s in segments if s["letter"]==l)) for l in "TLOCXB"},
    new_physics_calls=0,new_model_calls=0,new_training_updates=0)
out=dict(summary=summary,segments=segments,uncovered_H1_ids=leftovers,metadata_sha256=meta_hashes)
(OUT/"POOL_PREVIEW.json").write_text(json.dumps(out,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False))
