"""Fixed raw-RGB train pass and development-only H1 metadata selection.

Planning writes nothing and reads no image arrays or independent/held records.
The source catalogs/manifests retain their original episode roles.
"""
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .gpc_adaptation_data import GPCAdaptationDataset, read_adaptation_manifest
from . import gpc_training_schedule as legacy

SOURCE_SHA256 = "2a28136ca9c3b111d202f507ebab16c992b4d26a9c48d388956e390c6b2b069e"
PREFIX_SHA256 = "f7217a2d09393798529b00e72fcd971e8d7be11cab7a9c769704a9b30ddd004a"
IMAGE_DOMAIN = "official_raw_rgb"
KIND = "tdyssey.gpc_raw_fulltrain_schedule.v1"
RULE = dict(version="prefix500_then_unused_round_robin_raw_v1", examples=2328,
    prefix_examples=500, appended_examples=1828, examples_per_shape=1164,
    appended_per_shape=914, scenes_per_shape=12, windows_per_scene=97,
    shape_order=["T", "L"], scene_order="trajectory_id ascending",
    append_anchor_order="ascending within scene; round-robin across scenes; skip exhausted",
    repeats=False, epochs=1, selected_roles=["train"], source_split="train",
    image_domain=IMAGE_DOMAIN, t1=False,
    actions="pilot float32 /511*2-1 unchanged; original float64 targets retained for shared H1 prediction",
    benchmark_feedback=False)


def schedule_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


canonical_bytes = schedule_bytes


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _entry(scene,window,cursor):
    return dict(cursor=cursor,step=cursor+1,shape=scene["shape"],trajectory_id=scene["trajectory_id"],
        initial_state_sha256=scene["initial_state_sha256"],anchor=window["anchor"],
        dataset_index=scene["dataset_offset"]+window["anchor"]-3,
        stratum=window["stratum"],contact=window["contact"],rotation_degrees=window["rotation_degrees"])


def build_full_schedule(prefix):
    """Pure construction; plan_data separately enforces approved file identities."""
    legacy.validate_schedule(prefix)
    inventory=prefix["source_episodes"]
    if len(inventory)!=24 or any(len(scene["windows"])!=97 for scene in inventory):
        raise ValueError("Full pass requires exactly24 train scenes with97 windows each")
    entries=deepcopy(prefix["entries"])
    used={(row["trajectory_id"],row["anchor"]) for row in entries}
    if len(entries)!=500 or len(used)!=500 or Counter(row["shape"] for row in entries)!=dict(T=250,L=250):
        raise ValueError("The500-entry prefix must be unique and balanced")
    appended={}
    for shape in ("T","L"):
        scenes=sorted((scene for scene in inventory if scene["shape"]==shape),key=lambda row:row["trajectory_id"])
        queues=[deque(window for window in sorted(scene["windows"],key=lambda row:row["anchor"])
                      if (scene["trajectory_id"],window["anchor"]) not in used) for scene in scenes]
        ordered=[]
        while any(queues):
            for scene,queue in zip(scenes,queues):
                if queue: ordered.append((scene,queue.popleft()))
        if len(scenes)!=12 or len(ordered)!=914:
            raise ValueError("Append must contain914 unused windows per shape")
        appended[shape]=ordered
    for index in range(914):
        for shape in ("T","L"):
            scene,window=appended[shape][index]
            entries.append(_entry(scene,window,len(entries)))
    keys={(row["trajectory_id"],row["anchor"]) for row in entries}
    expected={(scene["trajectory_id"],window["anchor"]) for scene in inventory for window in scene["windows"]}
    if len(entries)!=2328 or keys!=expected or len(keys)!=len(entries):
        raise ValueError("Full schedule must visit all2328 train windows exactly once")
    coverage={}
    for shape in ("T","L"):
        selected=[row for row in entries if row["shape"]==shape]
        coverage[shape]=dict(examples=len(selected),unique_windows=len({(row["trajectory_id"],row["anchor"]) for row in selected}),
            distinct_scenes=len({row["initial_state_sha256"] for row in selected}),
            distinct_trajectories=len({row["trajectory_id"] for row in selected}),prefix_examples=250,
            appended_examples=914,repeated_exposures=0,contact_examples=sum(row["contact"] for row in selected),
            rotation_examples=sum(abs(row["rotation_degrees"])>5 for row in selected))
    return dict(kind=KIND,rule=deepcopy(RULE),rule_sha256=_hash(RULE),
        source_manifest_sha256=prefix["source_manifest_sha256"],prefix_schedule=deepcopy(prefix),
        prefix_sha256=hashlib.sha256(legacy.schedule_bytes(prefix)).hexdigest(),
        entries=entries,order_sha256=_hash(entries),coverage=coverage,selected_roles=["train"],image_payload_reads=0)


def validate_full_schedule(value,*,data=None,require_fixed=True):
    if not isinstance(value,dict) or value.get("kind")!=KIND:
        raise ValueError("Expected raw fulltrain schedule")
    prefix=value.get("prefix_schedule")
    legacy.validate_schedule(prefix,data=data)
    if require_fixed and (value.get("source_manifest_sha256")!=SOURCE_SHA256 or value.get("prefix_sha256")!=PREFIX_SHA256):
        raise ValueError("Raw fulltrain source/prefix identity differs")
    if value!=build_full_schedule(prefix):
        raise ValueError("Raw fulltrain ordered entries/rule/coverage were changed")
    return value


def load_full_schedule(path,sha256,*,data=None):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=sha256:
        raise ValueError("Frozen fulltrain schedule SHA256 differs")
    value=json.loads(payload)
    if schedule_bytes(value)!=payload: raise ValueError("Fulltrain schedule must use canonical encoding")
    return validate_full_schedule(value,data=data)


def _development_record(row):
    if row.get("role")!="development" or row.get("split")!="validation":
        raise ValueError("Development planning refuses all other record roles")
    item=legacy._record_file(row)
    return (Path(row["root"])/item["name"]).read_bytes()


def development_plan(data,*,record_reader=None):
    """Select first distinct development windows by class, never borrow held data."""
    reader=record_reader or _development_record
    rows=[row for row in data["episodes"] if row.get("role")=="development"]
    inventories=[];offset=0;sources=[]
    for row in rows:
        if row.get("split")!="validation" or row.get("shape") not in ("T","L") or row.get("payloads_present") is not True:
            raise ValueError("Development episodes must preserve complete official validation split")
        labels=legacy._inventory(row,offset,reader(row))
        labels.update(role="development",split="validation")
        inventories.append(labels)
        item=legacy._record_file(row)
        sources.append(dict(trajectory_id=row["trajectory_id"],shape=row["shape"],role="development",
            record_file=item["name"],record_sha256=item["sha256"],record_bytes=item["bytes"],
            catalog_sha256=row["catalog_sha256"],source_manifest_sha256=row["source_manifest_sha256"]))
        offset+=row["windows"]
    entries=[];coverage={}
    for shape in ("T","L"):
        scenes=sorted((row for row in inventories if row["shape"]==shape),key=lambda row:row["trajectory_id"])
        available=[(scene,window) for scene in scenes for window in scene["windows"]]
        contact=[pair for pair in available if pair[1]["contact"]]
        noncontact=[pair for pair in available if not pair[1]["contact"]]
        chosen=contact[:4]+noncontact[:4]
        used={(scene["trajectory_id"],window["anchor"]) for scene,window in chosen}
        remainder=[pair for pair in available if (pair[0]["trajectory_id"],pair[1]["anchor"]) not in used]
        chosen+=remainder[:max(0,8-len(chosen))]
        chosen.sort(key=lambda pair:(pair[0]["trajectory_id"],pair[1]["anchor"]))
        for scene,window in chosen:
            row=_entry(scene,window,len(entries))
            row.update(role="development",source_split="validation",image_domain=IMAGE_DOMAIN,
                case_id=f'{shape}-{scene["trajectory_id"]}-a{window["anchor"]:03d}',
                record_sha256=scene["record_sha256"],record_bytes=scene["record_bytes"])
            entries.append(row)
        coverage[shape]=dict(requested_windows=8,selected_windows=len(chosen),source_windows=len(available),
            source_scenes=len({scene["initial_state_sha256"] for scene in scenes}),source_trajectories=len(scenes),
            selected_scenes=len({scene["initial_state_sha256"] for scene,_ in chosen}),
            contact_windows=sum(window["contact"] for _,window in chosen),
            noncontact_windows=sum(not window["contact"] for _,window in chosen),
            available_contact=len(contact),available_noncontact=len(noncontact),
            contact_quota_shortage=max(0,4-len(contact)),noncontact_quota_shortage=max(0,4-len(noncontact)),
            shape_window_shortage=max(0,8-len(chosen)))
    return dict(kind="tdyssey.gpc_raw_fulltrain_development.v1",image_domain=IMAGE_DOMAIN,
        source_manifest_sha256=data["sha256"],selected_roles=["development"],requested_windows=16,
        entries=entries,order_sha256=_hash(entries),coverage=coverage,source_records=sources,image_payload_reads=0,
        independent_validation_payload_reads=0,held_payload_reads=0,
        sampling="first4contact+first4noncontact per shape in trajectory/anchor order; shortages fill same-role remainder",
        independence="development only; overlapping windows are not independent scenes")


def plan_data(manifest_path,prefix_path):
    """Return bound plans without freezing files or opening any image archive."""
    data=read_adaptation_manifest(manifest_path)
    if data["sha256"]!=SOURCE_SHA256: raise ValueError("Approved source adaptation manifest identity differs")
    prefix=legacy.load_schedule(prefix_path,PREFIX_SHA256,data=data)
    full=build_full_schedule(prefix)
    validate_full_schedule(full,data=data)
    development=development_plan(data)
    return dict(data=data,full_schedule=full,development=development,image_domain=IMAGE_DOMAIN,
                train_examples=2328,development_examples=len(development["entries"]))


class RawDataset(GPCAdaptationDataset):
    def __init__(self,manifest_path,role="train"):
        if role not in ("train","development"): raise ValueError("RawDataset only permits train/development")
        super().__init__(manifest_path,role,image_domain=IMAGE_DOMAIN)
        if self.manifest["sha256"]!=SOURCE_SHA256: raise ValueError("RawDataset source identity differs")


class RawScheduleConsumer:
    def __init__(self,value,sha256,dataset):
        if (getattr(dataset,"role",None)!="train" or getattr(dataset,"split",None)!="train"
                or getattr(dataset,"image_domain",None)!=IMAGE_DOMAIN):
            raise ValueError("Fulltrain consumer needs the explicit raw train dataset")
        validate_full_schedule(value,data=dataset.manifest)
        if hashlib.sha256(schedule_bytes(value)).hexdigest()!=sha256: raise ValueError("Fulltrain schedule binding hash differs")
        episodes=[row for row in dataset.manifest["episodes"] if row["role"]=="train"]
        offsets=[0]
        for row in episodes: offsets.append(offsets[-1]+row["windows"])
        if dataset.episodes!=episodes or dataset.offsets!=offsets:
            raise ValueError("Raw train scene membership/order was reassigned")
        self.value=deepcopy(value)
        self._indices=tuple(row["dataset_index"] for row in value["entries"])
        self.identity=dict(sha256=sha256,order_sha256=value["order_sha256"],rule_sha256=value["rule_sha256"],
                           prefix_sha256=value["prefix_sha256"],image_domain=IMAGE_DOMAIN)
    def indices(self,cursor,count):
        if type(cursor) is not int or type(count) is not int or cursor<0 or count<1 or cursor+count>2328:
            raise ValueError("Raw schedule cursor/count exceeds one2328-example pass")
        return list(self._indices[cursor:cursor+count])
    def checkpoint_state(self,cursor):
        if type(cursor) is not int or not 0<=cursor<=2328: raise ValueError("Raw fulltrain cursor is outside its single pass")
        return dict(self.identity,cursor=cursor)
