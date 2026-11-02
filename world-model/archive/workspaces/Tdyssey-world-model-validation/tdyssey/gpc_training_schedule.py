"""Frozen source-train JSON sampling; no tensor, image, model or physics work."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

KIND="tdyssey.gpc_training_schedule.v1"
ORDER="frozen_train_schedule_v1"
RULE={"version":ORDER,"seed":5478,"examples":500,"shape_order":["T","L"],
    "examples_per_shape":250,"scenes_per_shape":12,"scene_order":"trajectory_id ascending",
    "scene_selection":"round_robin; first ten scenes get21 and last two get20 per shape",
    "rotation":"absolute wrapped target-control turn strictly >5 degrees, first up to scene quota",
    "fill":"alternate contact-only then other; skip exhausted/empty buckets",
    "bucket_order":"sha256(compact_json([version,5478,shape,trajectory_id,anchor])) ascending",
    "repeats":"only if total legal scene windows <scene quota; deterministic cycle of same scene order",
    "targets":"transition anchor=3..N-1; unchanged five-image/four-action window",
    "roles":["train"],"source_split":"train","benchmark_feedback":False,
    "distribution":"scene equal, rotation oversampled, remaining contact balanced; privileged train-label selection"}


def _encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


RULE_SHA256=_hash(RULE)


def schedule_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def _train_rows(data):
    if not isinstance(data,dict) or not _sha(data.get("sha256")) or not isinstance(data.get("episodes"),list):
        raise ValueError("A frozen normalized adaptation manifest is required")
    rows=[];offset=0;seen=set();scenes=set()
    for row in data["episodes"]:
        if row.get("role")!="train":continue
        if (row.get("split")!="train" or row.get("shape") not in ("T","L")
                or not isinstance(row.get("trajectory_id"),str) or not row["trajectory_id"]
                or row["trajectory_id"] in seen or not _sha(row.get("initial_state_sha256"))
                or row["initial_state_sha256"] in scenes or type(row.get("control_steps")) is not int
                or row["control_steps"]<4 or row.get("windows")!=row["control_steps"]-3
                or row.get("payloads_present") is not True):
            raise ValueError("Need complete distinct source-train scenes/windows")
        rows.append((row,offset));offset+=row["windows"]
        seen.add(row["trajectory_id"]);scenes.add(row["initial_state_sha256"])
    if any(sum(row["shape"]==shape for row,_ in rows)!=12 for shape in ("T","L")):
        raise ValueError("Exactly twelve source-train scenes per shape are required")
    return rows


def _record_file(row):
    records=[item for item in row["entry"]["files"] if Path(item["name"]).suffix==".json"]
    if len(records)!=1:raise ValueError("Exactly one training record JSON is required")
    item=records[0]
    if Path(item["name"]).name!=item["name"] or not _sha(item.get("sha256")) or type(item.get("bytes")) is not int:
        raise ValueError("Invalid training record identity")
    return item


def _read_record(row):
    if row.get("role")!="train" or row.get("split")!="train":raise ValueError("Nontrain payload access forbidden")
    return (Path(row["root"])/_record_file(row)["name"]).read_bytes()


def _inventory(row,offset,payload):
    item=_record_file(row)
    if not isinstance(payload,bytes) or len(payload)!=item["bytes"] or hashlib.sha256(payload).hexdigest()!=item["sha256"]:
        raise ValueError("Bound source training JSON changed")
    record=json.loads(payload);trajectory=record.get("trajectory",{})
    transitions=trajectory.get("transitions",[]);diagnostics=record.get("step_diagnostics",[]);n=row["control_steps"]
    if (trajectory.get("trajectory_id")!=row["trajectory_id"] or trajectory.get("collector_version")!="official-waypoint-play-v1"
            or trajectory.get("control_dt")!=.1 or len(transitions)!=n or len(diagnostics)!=n):
        raise ValueError("Training record timing/identity/count differs")
    windows=[];contact_total=0
    for anchor,(transition,diagnostic) in enumerate(zip(transitions,diagnostics)):
        contacts=diagnostic.get("n_contacts");before,after=transition.get("before",{}),transition.get("after",{})
        if (type(contacts) is not int or contacts<0 or before.get("shape")!=row["shape"] or after.get("shape")!=row["shape"]
                or type(before.get("control_step")) is not int or after.get("control_step")!=before["control_step"]+1):
            raise ValueError("Invalid source control/contact/shape labels")
        angles=[state.get("object_body",{}).get("pose",{}).get("angle") for state in (before,after)]
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in angles):
            raise ValueError("Missing finite source rotation labels")
        contact_total+=contacts>0
        if anchor<3:continue
        delta=math.degrees(math.atan2(math.sin(angles[1]-angles[0]),math.cos(angles[1]-angles[0])))
        contact=contacts>0;rotation=abs(delta)>5
        stratum="contact_rotation" if contact and rotation else "contact_only" if contact else "rotation_only" if rotation else "other"
        windows.append(dict(anchor=anchor,stratum=stratum,contact=contact,rotation_degrees=delta))
    if contact_total!=row["contact_steps"]:raise ValueError("Source contact summary differs")
    return dict(trajectory_id=row["trajectory_id"],shape=row["shape"],role="train",split="train",
        initial_state_sha256=row["initial_state_sha256"],dataset_offset=offset,record_sha256=item["sha256"],
        record_bytes=item["bytes"],catalog_sha256=row["catalog_sha256"],source_manifest_sha256=row["source_manifest_sha256"],windows=windows)


def _entries(inventory):
    scenes={shape:sorted([row for row in inventory if row["shape"]==shape],key=lambda row:row["trajectory_id"]) for shape in ("T","L")}
    if any(len(rows)!=12 for rows in scenes.values()):raise ValueError("Frozen scene count differs")
    orders={}
    for scene in inventory:
        identifier=scene["trajectory_id"]
        sort=lambda rows:sorted(rows,key=lambda w:_hash([ORDER,5478,scene["shape"],identifier,w["anchor"]]))
        rotation=sort([w for w in scene["windows"] if abs(w["rotation_degrees"])>5])
        contact=sort([w for w in scene["windows"] if abs(w["rotation_degrees"])<=5 and w["contact"]])
        other=sort([w for w in scene["windows"] if abs(w["rotation_degrees"])<=5 and not w["contact"]])
        fill=[]
        for i in range(max(len(contact),len(other))):
            if i<len(contact):fill.append(contact[i])
            if i<len(other):fill.append(other[i])
        orders[identifier]=rotation+fill
        if not orders[identifier]:raise ValueError("Empty scene cannot be scheduled")
    result=[]
    for cursor in range(500):
        shape=("T","L")[cursor%2];local=cursor//2;scene=scenes[shape][local%12]
        identifier=scene["trajectory_id"];ordered=orders[identifier];window=ordered[(local//12)%len(ordered)]
        result.append(dict(cursor=cursor,step=cursor+1,shape=shape,trajectory_id=identifier,
            initial_state_sha256=scene["initial_state_sha256"],anchor=window["anchor"],
            dataset_index=scene["dataset_offset"]+window["anchor"]-3,
            stratum=window["stratum"],contact=window["contact"],rotation_degrees=window["rotation_degrees"]))
    return result


def _coverage(inventory,entries):
    output={}
    for shape in ("T","L"):
        source=[r for r in inventory if r["shape"]==shape];selected=[r for r in entries if r["shape"]==shape];per_scene=[]
        for scene in source:
            rows=[r for r in selected if r["trajectory_id"]==scene["trajectory_id"]]
            available=Counter(w["stratum"] for w in scene["windows"]);used=Counter(w["stratum"] for w in rows)
            per_scene.append(dict(trajectory_id=scene["trajectory_id"],draws=len(rows),unique_windows=len({r["anchor"] for r in rows}),
                source_strata=dict(available),selected_strata=dict(used),
                source_rotation_windows=sum(abs(w["rotation_degrees"])>5 for w in scene["windows"]),
                sampled_unique_rotation_windows=len({r["anchor"] for r in rows if abs(r["rotation_degrees"])>5})))
        unique=len({(r["trajectory_id"],r["anchor"]) for r in selected})
        output[shape]=dict(examples=len(selected),distinct_scenes=len({r["trajectory_id"] for r in selected}),
            unique_windows=unique,repeated_exposures=len(selected)-unique,contact_examples=sum(r["contact"] for r in selected),
            rotation_examples=sum(abs(r["rotation_degrees"])>5 for r in selected),
            clockwise_examples=sum(r["rotation_degrees"]>5 for r in selected),counterclockwise_examples=sum(r["rotation_degrees"]< -5 for r in selected),
            contact_rotation_examples=sum(r["contact"] and abs(r["rotation_degrees"])>5 for r in selected),
            source_rotation_windows=sum(row["source_rotation_windows"] for row in per_scene),
            sampled_unique_rotation_windows=sum(row["sampled_unique_rotation_windows"] for row in per_scene),per_scene=per_scene,
            all_scenes_exposed=all(row["draws"] for row in per_scene))
    return output


def generate_schedule(data,*,record_reader=None):
    """One deterministic JSON pass; nontrain records/images are never opened."""
    rows=_train_rows(data);reader=record_reader or _read_record
    inventory=[_inventory(row,offset,reader(row)) for row,offset in rows]
    inventory.sort(key=lambda row:(row["shape"],row["trajectory_id"]))
    entries=_entries(inventory)
    return dict(kind=KIND,rule=deepcopy(RULE),rule_sha256=RULE_SHA256,source_manifest_sha256=data["sha256"],
        source_episodes=inventory,source_inventory_sha256=_hash(inventory),entries=entries,order_sha256=_hash(entries),
        coverage=_coverage(inventory,entries),selected_roles=["train"],image_payload_reads=0,
        excluded_role_counts=dict(Counter(row.get("role") for row in data["episodes"] if row.get("role")!="train")))


def validate_schedule(value,*,data=None):
    if (not isinstance(value,dict) or value.get("kind")!=KIND or value.get("rule")!=RULE
            or value.get("rule_sha256")!=RULE_SHA256 or not _sha(value.get("source_manifest_sha256"))
            or value.get("selected_roles")!=["train"] or value.get("image_payload_reads")!=0):
        raise ValueError("Schedule rule/source/role contract differs")
    inventory=value.get("source_episodes",[]);seen=set();scenes=set();offsets=[]
    for scene in inventory:
        identifier=scene.get("trajectory_id");windows=scene.get("windows",[])
        if (not isinstance(identifier,str) or not identifier or identifier in seen or scene.get("shape") not in ("T","L")
                or scene.get("role")!="train" or scene.get("split")!="train"
                or any(not _sha(scene.get(key)) for key in ("initial_state_sha256","record_sha256","catalog_sha256","source_manifest_sha256"))
                or scene["initial_state_sha256"] in scenes or type(scene.get("dataset_offset")) is not int or scene["dataset_offset"]<0
                or not windows or [w.get("anchor") for w in windows]!=list(range(3,3+len(windows)))):
            raise ValueError("Invalid source-train inventory")
        for window in windows:
            contact=window.get("contact");turn=window.get("rotation_degrees")
            if type(contact) is not bool or type(turn) not in (int,float) or not math.isfinite(turn):
                raise ValueError("Invalid schedule contact/rotation label")
            expected="contact_rotation" if contact and abs(turn)>5 else "contact_only" if contact else "rotation_only" if abs(turn)>5 else "other"
            if window.get("stratum")!=expected:raise ValueError("Schedule stratum disagrees with source labels")
        seen.add(identifier);scenes.add(scene["initial_state_sha256"]);offsets.append((scene["dataset_offset"],len(windows)))
    cursor=0
    for offset,count in sorted(offsets):
        if offset!=cursor:raise ValueError("Source dataset offsets overlap or have gaps")
        cursor+=count
    if value.get("source_inventory_sha256")!=_hash(inventory):raise ValueError("Schedule source inventory hash differs")
    entries=_entries(inventory)
    if value.get("entries")!=entries or value.get("order_sha256")!=_hash(entries) or value.get("coverage")!=_coverage(inventory,entries):
        raise ValueError("Schedule ordered entries/coverage differ from fixed construction")
    if data is not None:
        if data.get("sha256")!=value["source_manifest_sha256"]:raise ValueError("Schedule is for another source manifest")
        source={row["trajectory_id"]:(row,offset) for row,offset in _train_rows(data)}
        if set(source)!=seen:raise ValueError("Schedule scene membership differs from train dataset")
        for scene in inventory:
            row,offset=source[scene["trajectory_id"]];item=_record_file(row)
            if (scene["dataset_offset"]!=offset or len(scene["windows"])!=row["windows"] or scene["shape"]!=row["shape"]
                    or scene["initial_state_sha256"]!=row["initial_state_sha256"] or scene["record_sha256"]!=item["sha256"]
                    or scene["record_bytes"]!=item["bytes"] or scene["catalog_sha256"]!=row["catalog_sha256"]
                    or scene["source_manifest_sha256"]!=row["source_manifest_sha256"]):
                raise ValueError("Schedule physical scene/window/source identity differs")
    return value


def load_schedule(path,sha256,*,data=None):
    payload=Path(path).read_bytes()
    if not _sha(sha256) or hashlib.sha256(payload).hexdigest()!=sha256:raise ValueError("Frozen schedule file SHA256 differs")
    value=json.loads(payload)
    if payload!=schedule_bytes(value):raise ValueError("Schedule must use its fixed canonical encoding")
    return validate_schedule(value,data=data)


def binding_fields(value,sha256):
    validate_schedule(value)
    if hashlib.sha256(schedule_bytes(value)).hexdigest()!=sha256:raise ValueError("Schedule binding hash differs")
    return dict(training_schedule_sha256=sha256,training_schedule_order_sha256=value["order_sha256"],
                training_schedule_rule_sha256=value["rule_sha256"])


class ScheduleConsumer:
    """Read-only finite order; cursor is the account's spent example highwater."""
    def __init__(self,value,sha256,dataset):
        if getattr(dataset,"split",None)!="train" or getattr(dataset,"role",None)!="train":
            raise ValueError("Schedule consumer accepts the train dataset only")
        validate_schedule(value,data=dataset.manifest)
        expected=[row for row in dataset.manifest["episodes"] if row["role"]=="train"]
        if dataset.episodes!=expected:raise ValueError("Dataset scene order/content was reassigned")
        offsets=[0]
        for row in expected:offsets.append(offsets[-1]+row["windows"])
        if dataset.offsets!=offsets:raise ValueError("Dataset window offsets differ")
        self.value=deepcopy(value);self.binding=binding_fields(value,sha256)
        self._indices=tuple(row["dataset_index"] for row in value["entries"])
        self.identity=dict(sha256=sha256,order_sha256=value["order_sha256"],rule_sha256=value["rule_sha256"])
    def indices(self,cursor,count):
        if type(cursor) is not int or type(count) is not int or cursor<0 or count<1 or cursor+count>500:
            raise ValueError("Schedule cursor/count exceeds its frozen 500 examples")
        return list(self._indices[cursor:cursor+count])
    def checkpoint_state(self,cursor):
        if type(cursor) is not int or not 0<=cursor<=500:raise ValueError("Invalid cumulative schedule cursor")
        return dict(self.identity,cursor=cursor)
