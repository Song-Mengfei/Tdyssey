"""R2C0 saved-RGB reference data and resumable TRAIN-only80/20 draw order.

Only R2B accepted files are referenced. No environment, renderer, torch, model,
optimizer, or future-state-conditioned action generation is used here.
"""
from __future__ import annotations
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

LETTERS=tuple("TLOCXB")
PROTOCOL_SHA256="58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6"
POLICY=dict(kind="R2C0-letter-family-motion80-other20.v1",seed=5478,
    letters=list(LETTERS),cycle_slots=480,per_letter=dict(moving=64,other=16),maximum_frozen_draws=4800,
    moving="H1 area-centroid translation>=2 image96px OR abs unwrapped angle>=5deg",
    slot_order="80 rounds; each round shuffled6letters, per-letter64moving16other shuffled each cycle",
    family_order="within letter/group balanced shuffled family queue, exhaust before reshuffle",
    window_order="moving family: balanced nonempty mutually-exclusive translation_only/rotation_only/both queues; category window queue exhausts before reshuffle; other family: all-other window queue",
    overlap="both belongs to one category and one moving pool, never counted twice",
    repeat_scope="repeated exposures, not new independent data; nextcycle uses continuing independent RNGs")
SPLIT_POLICY=dict(kind="R2C0-pre-model-family-split.v1",seed=5478,val_per_letter=2,
    candidate_rule="distinct families: first has H1translation>=2, second absH1rotation>=5, both have H4legalanchor<=20; remaining TRAIN keeps H1/H4legal translation/rotation and availablepositive/negative>=5deg",
    tie_break="lowest SHA256(domain R2C0-split-v1,seed,letter,translationfamily,rotationfamily)",
    near_duplicate=dict(initial_area_centroid_distance_world_px_lte=8.,initial_symmetry_angle_deg_lte=5.,initial_pusher_local_distance_world_px_lte=16.),
    near_policy="reject a candidate pair if any declared initial near-scene component crosses split",
    scope="seen-shape motion-containing diagnostic split, not unseen-letter or unbiased natural test")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):raise ValueError("Reference path escapes R2B root")
    return path


def _rot(a):return np.array(((math.cos(a),-math.sin(a)),(math.sin(a),math.cos(a))))


def read_source_catalogue(r2b_root):
    """Verify cold R2B inputs and derive metadata only; never copy RGB files."""
    from .envs.official import official_shape
    root=Path(r2b_root).resolve();index=read_json(root/"index.json");report=read_json(root/"report/r2b-report.json")
    if index.get("collection_closed") is not True or report.get("data_errors"):
        raise ValueError("Only completed accepted cold R2B data are supported")
    published={(r["request_id"],r["anchor"],r["horizon"]):r for name in ("h1_windows","h4_windows") for r in report["populations"][name]}
    entries=[];h1=[];h4=[];signatures={}
    for slot in index["slots"]:
        if slot["status"]=="static_blocked":continue
        if slot["status"]!="completed" or slot["completed_controls"]!=24:raise ValueError("R2C0 only the53 complete24-control R2B families")
        npz,meta=_safe(root,slot["trajectory_file"]),_safe(root,slot["metadata_file"])
        m=read_json(meta)
        if file_sha(npz)!=slot["trajectory_sha256"] or file_sha(meta)!=slot["metadata_sha256"] or m["protocol_sha256"]!=PROTOCOL_SHA256:
            raise ValueError("Source hash/protocol differs")
        with np.load(npz,allow_pickle=False) as z:states=z["states"].copy();actions=z["actions"].copy()
        if states.shape!=(25,15) or actions.shape!=(24,2) or not np.isfinite(states).all() or not np.isfinite(actions).all() or (actions<0).any() or (actions>511).any():raise ValueError("Invalid source timeline/action")
        records=m["control_records"]
        valid=[r.get("completed") is True and r.get("valid_transition") is True and r.get("substep_count")==10 and len(r.get("max_penetrations",[]))==3 and all(x is not None and math.isfinite(x) and x<=2 for x in r["max_penetrations"]) for r in records]
        if len(valid)!=24 or not all(valid):raise ValueError("R2B accepted transition validity changed")
        letter=slot["letter"];local=np.asarray(official_shape(letter).centroid);R=np.asarray([_rot(s[4]) for s in states]);area=states[:,2:4]+np.einsum('nij,j->ni',R,local)
        entry=dict(trajectory_id=slot["request_id"],family_id=slot["family_id"],letter=letter,shape=letter,
            file=slot["trajectory_file"],sha256=slot["trajectory_sha256"],metadata_file=slot["metadata_file"],metadata_sha256=slot["metadata_sha256"],
            controls=24,initial_space=slot["initial_space"],role=slot["role"],protocol_sha256=PROTOCOL_SHA256,
            initial_state15=states[0].tolist(),initial_area_centroid=area[0].tolist(),initial_pusher_local=(_rot(-states[0,4])@(states[0,:2]-states[0,2:4])).tolist(),
            symmetry=int(official_shape(letter).symmetry_order),source_frame_sha256=m["frame_sha256"])
        entries.append(entry)
        signatures[entry["trajectory_id"]]=dict(states=hashlib.sha256(states.tobytes()).hexdigest(),actions=hashlib.sha256(actions.tobytes()).hexdigest(),initial_state=hashlib.sha256(states[0].tobytes()).hexdigest(),initial_RGB=m["frame_sha256"][0])
        for horizon,out in ((1,h1),(4,h4)):
            for anchor in range(3,25-horizon):
                if not all(valid[anchor-3:anchor+horizon]):continue
                x=float(np.linalg.norm(area[anchor+horizon]-area[anchor])*96/512);angle=float((states[anchor+horizon,4]-states[anchor,4])*180/math.pi)
                old=published[(entry["trajectory_id"],anchor,horizon)]
                if not old["valid_window"] or abs(old["translation_image96_px"]-x)>1e-8 or abs(old["signed_rotation_rad"]*180/math.pi-angle)>1e-8:raise ValueError("Independent saved-state amplitude differs from accepted report")
                translation=x>=2;rotation=abs(angle)>=5
                out.append(dict(window_id=f"{entry['trajectory_id']}:{anchor}:H{horizon}",trajectory_id=entry["trajectory_id"],family_id=entry["family_id"],letter=letter,
                    anchor=anchor,horizon=horizon,translation_image96_px=x,signed_rotation_deg=angle,abs_rotation_deg=abs(angle),
                    translation=translation,rotation=rotation,moving=translation or rotation,
                    category="both" if translation and rotation else "translation_only" if translation else "rotation_only" if rotation else "other",
                    initial_space=entry["initial_space"],phase=old["phase"],planned_phases=old.get("planned_phases",[]),
                    agent_object_contact=old["agent_object_contact"],agent_object_contact_substeps=old["agent_object_contact_substeps"],
                    boundary_process=old["boundary_process"],actual_contact_arc_ids=old["actual_contact_arc_ids"],signed_lever_signs=old["signed_lever_signs"],
                    valid_window=True))
    if len(entries)!=53 or len(h1)!=1113 or len(h4)!=954:raise ValueError("R2B accepted fixed denominators changed")
    return dict(entries=entries,h1=h1,h4=h4,signatures=signatures,source_identity=dict(index_sha256=file_sha(root/"index.json"),
        accepted_report_sha256=file_sha(root/"report/r2b-report.json"),source_identity_sha256=file_sha(root/"source-identity.json"),frozen_requests_sha256=file_sha(root/"frozen-requests.json")))


def choose_split(catalogue,seed=5478):
    """Freeze whole-family split before models; no motion window crosses it."""
    if seed!=5478:raise ValueError("Frozen split seed5478 required")
    entries,rows=catalogue["entries"],catalogue["h1"];support={e["trajectory_id"]:dict(translation=False,rotation=False,positive=False,negative=False,translation_h4eligible=False,rotation_h4eligible=False) for e in entries}
    for r in rows:
        s=support[r["trajectory_id"]];s["translation"]|=r["translation"];s["rotation"]|=r["rotation"];s["positive"]|=r["signed_rotation_deg"]>=5;s["negative"]|=r["signed_rotation_deg"]<=-5
        if r["anchor"]<=20:s["translation_h4eligible"]|=r["translation"];s["rotation_h4eligible"]|=r["rotation"]
    near=[];exact=[]
    for i,a in enumerate(entries):
        for b in entries[:i]:
            sa,sb=catalogue["signatures"][a["trajectory_id"]],catalogue["signatures"][b["trajectory_id"]]
            same=[k for k in sa if sa[k]==sb[k]]
            if same:exact.append(dict(first=a["trajectory_id"],second=b["trajectory_id"],equal_fields=same))
            if a["letter"]!=b["letter"]:continue
            distance=float(np.linalg.norm(np.asarray(a["initial_area_centroid"])-b["initial_area_centroid"]))
            period=2*math.pi/max(1,a["symmetry"]);angle=abs((a["initial_state15"][4]-b["initial_state15"][4]+period/2)%period-period/2)*180/math.pi
            pusher=float(np.linalg.norm(np.asarray(a["initial_pusher_local"])-b["initial_pusher_local"]))
            if distance<=8 and angle<=5 and pusher<=16:near.append(dict(first=a["trajectory_id"],second=b["trajectory_id"],initial_area_distance_world_px=distance,symmetry_angle_deg=angle,pusher_local_distance_world_px=pusher))
    selections=[];val=set()
    for letter in LETTERS:
        ids=[e["trajectory_id"] for e in entries if e["letter"]==letter];possible=[]
        available={k:any(support[t][k] for t in ids) for k in ("translation","rotation","positive","negative","translation_h4eligible","rotation_h4eligible")}
        for trans in ids:
            for rotation in ids:
                if trans==rotation or not support[trans]["translation_h4eligible"] or not support[rotation]["rotation_h4eligible"]:continue
                selected={trans,rotation};train=set(ids)-selected
                if any(available[k] and not any(support[t][k] for t in train) for k in available):continue
                if any((n["first"] in selected)!=(n["second"] in selected) for n in near if n["first"] in ids):continue
                if any((n["first"] in selected)!=(n["second"] in selected) and ("states" in n["equal_fields"] or "initial_RGB" in n["equal_fields"]) for n in exact if n["first"] in ids and n["second"] in ids):continue
                rank=canonical_hash(["R2C0-split-v1",seed,letter,trans,rotation]);possible.append((rank,trans,rotation))
        if not possible:raise ValueError("Cannot preserve declared TRAIN/val/near-duplicate support: "+letter)
        rank,trans,rotation=min(possible);val.update((trans,rotation))
        selections.append(dict(letter=letter,translation_family=trans,rotation_family=rotation,rank_sha256=rank,legal_ordered_candidate_pairs=len(possible),train_kept_support={k:any(support[t][k] for t in set(ids)-{trans,rotation}) for k in available}))
    split={e["trajectory_id"]:"seen_val" if e["trajectory_id"] in val else "train" for e in entries}
    return dict(kind="R2C0-frozen-family-split.v1",policy=deepcopy(SPLIT_POLICY),selections=selections,assignments=split,
        counts=dict(Counter(split.values())),support=support,near_duplicate_pairs=near,exact_signature_matches=exact,
        cross_split_near_pairs=[n for n in near if split[n["first"]]!=split[n["second"]]],
        scope="family-isolated, seen-shape, deliberately motion-containing diagnostic; no model outputs consulted")


def verify_reference(manifest_path,verify_files=True):
    path=Path(manifest_path).resolve();m=read_json(path);root=(path.parent/m["source_root"]).resolve()
    if m.get("kind")!="R2C0-R2B-saved-RGB-reference.v1" or root.name!="r2b" or m["protocol_sha256"]!=PROTOCOL_SHA256:raise ValueError("Only new selected-physics R2B reference supported")
    index=read_json(root/"index.json")
    if file_sha(root/"index.json")!=m["source_identity"]["index_sha256"] or index.get("collection_closed") is not True:raise ValueError("R2B index identity changed")
    for key,relative in (("accepted_report_sha256","report/r2b-report.json"),("source_identity_sha256","source-identity.json"),("frozen_requests_sha256","frozen-requests.json")):
        if file_sha(root/relative)!=m["source_identity"][key]:raise ValueError("Accepted R2B source identity changed")
    for key,name in (("h1","h1-windows.json"),("h4","h4-windows.json"),("split","split-plan.json")):
        if file_sha(path.parent/name)!=m["catalogues"][key]["sha256"]:raise ValueError("Frozen split/catalogue differs")
    assignments=read_json(path.parent/"split-plan.json")["assignments"]
    registry={r["request_id"]:r for r in index["slots"] if r["status"]=="completed"}
    if len({e["family_id"] for e in m["entries"]})!=len(m["entries"]) or {e["trajectory_id"] for e in m["entries"]}!=set(registry) or set(assignments)!=set(registry):raise ValueError("Only complete original R2B families may appear once")
    for e in m["entries"]:
        original=registry[e["trajectory_id"]]
        if (e["split"]!=assignments[e["trajectory_id"]] or e["split"] not in ("train","seen_val") or e["letter"] not in LETTERS
                or e["file"]!=original["trajectory_file"] or e["sha256"]!=original["trajectory_sha256"]
                or e["metadata_file"]!=original["metadata_file"] or e["metadata_sha256"]!=original["metadata_sha256"]
                or e["family_id"]!=original["family_id"] or e["letter"]!=original["letter"]):raise ValueError("Entry/split not bound to original R2B registry")
    for horizon,key in ((1,"h1"),(4,"h4")):
        rows=read_json(path.parent/m["catalogues"][key]["file"]);expected=[];counters=Counter()
        for e in m["entries"]:
            for anchor in range(3,25-horizon):
                expected.append((e["trajectory_id"],anchor,e["split"],counters[e["split"]]));counters[e["split"]]+=1
        if [(r["trajectory_id"],r["anchor"],r["split"],r["dataset_index"]) for r in rows]!=expected:raise ValueError("Window catalogue reorders/mixes/drops original H1/H4 timeline")
    if verify_files:
        for e in m["entries"]:
            for relative,expected in ((e["file"],e["sha256"]),(e["metadata_file"],e["metadata_sha256"])):
                if file_sha(_safe(root,relative))!=expected:raise ValueError("Referenced R2B artifact differs")
    return m,root


class R2C0Dataset:
    """FiveCHW frames=4 input+H1target;4 normalized actions, no state inputs."""
    def __init__(self,manifest_path,split="train",cache_bytes=1024*1024):
        if split not in ("train","seen_val") or type(cache_bytes)is not int or cache_bytes<0:raise ValueError("Invalid split/cache")
        self.manifest_path=Path(manifest_path).resolve();self.manifest,self.source_root=verify_reference(self.manifest_path)
        self.root=self.manifest_path.parent;self.split=split;self.entries=[e for e in self.manifest["entries"] if e["split"]==split]
        rows=read_json(self.root/"h1-windows.json");self.window_rows=[r for r in rows if r["split"]==split]
        byid={e["trajectory_id"]:i for i,e in enumerate(self.entries)}
        self.windows=[(byid[r["trajectory_id"]],r["anchor"]) for r in self.window_rows]
        self.h4_windows=[(byid[r["trajectory_id"]],r["anchor"]) for r in read_json(self.root/"h4-windows.json") if r["split"]==split]
        self.cache_bytes=cache_bytes;self.cache_nbytes=0;self._cache=OrderedDict()

    def __len__(self):return len(self.windows)
    def identity(self):
        return dict(manifest_sha256=file_sha(self.manifest_path),split_plan_sha256=self.manifest["catalogues"]["split"]["sha256"],h1_catalogue_sha256=self.manifest["catalogues"]["h1"]["sha256"],source_index_sha256=self.manifest["source_identity"]["index_sha256"],split=self.split)
    def trajectory(self,index):
        if type(index)is not int or not 0<=index<len(self.entries):raise IndexError(index)
        if index in self._cache:
            self._cache.move_to_end(index);return self.entries[index],self._cache[index]
        e=self.entries[index];path=_safe(self.source_root,e["file"])
        if file_sha(path)!=e["sha256"]:raise ValueError("Source changed during dataset use")
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in ("rgb","actions","states")}
        if arrays["rgb"].shape!=(25,3,96,96) or arrays["rgb"].dtype!=np.uint8:raise ValueError("Stored rawRGB layout changed")
        size=sum(a.nbytes for a in arrays.values())
        while self._cache and self.cache_nbytes+size>self.cache_bytes:
            _,old=self._cache.popitem(last=False);self.cache_nbytes-=sum(a.nbytes for a in old.values())
        if size<=self.cache_bytes:self._cache[index]=arrays;self.cache_nbytes+=size
        return e,arrays
    def __getitem__(self,index):
        entry_index,anchor=self.windows[index];entry,arrays=self.trajectory(entry_index)
        image=arrays["rgb"][anchor-3:anchor+2].astype(np.float32)/np.float32(255)
        action=arrays["actions"][anchor-3:anchor+1].astype(np.float32)/np.float32(255.5)-np.float32(1)
        return dict(image=image,action=action,trajectory_id=entry["trajectory_id"],family_id=entry["family_id"],shape=entry["letter"],anchor=anchor,dataset_index=index)
    def close(self):self._cache.clear();self.cache_nbytes=0


def build_pool_manifest(rows,data_identity):
    rows=deepcopy(rows);groups={l:{g:{} for g in ("moving","other")} for l in LETTERS}
    for i,row in enumerate(rows):
        if row["split"]!="train" or row["dataset_index"]!=i or row["horizon"]!=1 or row["moving"]!=(row["translation"] or row["rotation"]):raise ValueError("Pools require complete ordered TRAIN-only H1 catalogue")
        translation=row["translation_image96_px"]>=2;rotation=row["abs_rotation_deg"]>=5
        category="both" if translation and rotation else "translation_only" if translation else "rotation_only" if rotation else "other"
        if row["translation"]!=translation or row["rotation"]!=rotation or row["category"]!=category:raise ValueError("Image96px/angle motion definition differs; oldworldpx labels forbidden")
        group="moving" if row["moving"] else "other";categories=groups[row["letter"]][group].setdefault(row["family_id"],{})
        categories.setdefault(row["category"],[]).append(i)
    if any(not groups[l][g] for l in LETTERS for g in ("moving","other")):raise ValueError("Empty letter/group; never borrow or fabricate")
    value=dict(kind="R2C0-TRAIN-pools.v1",policy=deepcopy(POLICY),data_identity=deepcopy(data_identity),windows=rows,groups=groups)
    value["identity"]=dict(policy_sha256=canonical_hash(POLICY),data_sha256=canonical_hash(data_identity),pool_sha256=canonical_hash(groups),catalogue_sha256=canonical_hash(rows))
    value["manifest_sha256"]=canonical_hash(value);return value


class R2C0WindowOrder:
    def __init__(self,pools,seed=5478):
        if seed!=5478 or pools!=build_pool_manifest(pools["windows"],pools["data_identity"]):raise ValueError("Pool/data/policy changed or wrong seed")
        self.pools=deepcopy(pools);self.cursor=0;self.cycle=0;self.position=0;self.rng=np.random.default_rng(seed);self.queues={};self.slot_order=self._slots()
    def identity(self):return deepcopy(self.pools["identity"])
    def _slots(self):
        patterns={l:self.rng.permutation(["moving"]*64+["other"]*16).tolist() for l in LETTERS}
        return [(str(l),patterns[str(l)][i]) for i in range(80) for l in self.rng.permutation(LETTERS).tolist()]
    def _draw(self,values,label):
        key=json.dumps(label,separators=(",",":"));q=self.queues.get(key)
        if q is None:
            rng=np.random.default_rng(int(canonical_hash([5478,self.identity(),label])[:32],16));q=dict(order=rng.permutation(values).tolist(),position=0,epoch=0,rng=rng);self.queues[key]=q
        if q["position"]==len(q["order"]):q.update(order=q["rng"].permutation(values).tolist(),position=0,epoch=q["epoch"]+1)
        value=q["order"][q["position"]];q["position"]+=1;return value
    def take(self,count):
        if type(count)is not int or count<1 or self.cursor+count>4800:raise ValueError("Frozen proposal at most4800 TRAIN exposures")
        result=[]
        for _ in range(count):
            if self.position==480:self.cycle+=1;self.position=0;self.slot_order=self._slots()
            letter,group=self.slot_order[self.position];families=self.pools["groups"][letter][group]
            family=self._draw(sorted(families),["family",letter,group]);categories=families[family]
            category=self._draw(sorted(categories),["category",letter,group,family])
            result.append(int(self._draw(categories[category],["window",letter,group,family,category])))
            self.cursor+=1;self.position+=1
        return result
    def state_dict(self):
        q={k:dict(order=deepcopy(v["order"]),position=v["position"],epoch=v["epoch"],rng=deepcopy(v["rng"].bit_generator.state)) for k,v in self.queues.items()}
        state=dict(kind="R2C0-draw-state.v1",identity=self.identity(),cursor=self.cursor,cycle=self.cycle,position=self.position,slot_order=[list(x) for x in self.slot_order],slot_rng=deepcopy(self.rng.bit_generator.state),queues=q)
        state["state_sha256"]=canonical_hash(state);return state
    def load_state_dict(self,state):
        value=deepcopy(state);checksum=value.pop("state_sha256",None)
        if checksum!=canonical_hash(value) or value.get("identity")!=self.identity() or type(value.get("cursor"))is not int or not 0<=value["cursor"]<=4800:raise ValueError("Invalid order/pool/data identity or state checksum")
        rebuilt=type(self)(self.pools)
        if value["cursor"]:rebuilt.take(value["cursor"])
        if rebuilt.state_dict()!=state:raise ValueError("Saved queues/cursors/RNG do not match frozen stream")
        self.__dict__.update(rebuilt.__dict__)


def metadata_only_sampler_factory(dataset,seed=5478,indices=None):
    if dataset.split!="train" or indices is not None:raise ValueError("All R2C0 TRAIN H1 windows, no legacy subset/val draw")
    pools=read_json(dataset.root/"pools.json")
    if pools["data_identity"]!=dataset.identity() or len(pools["windows"])!=len(dataset):raise ValueError("Dataset/pool identity or indices differ")
    for (entry,anchor),row in zip(dataset.windows,pools["windows"]):
        if dataset.entries[entry]["family_id"]!=row["family_id"] or anchor!=row["anchor"]:raise ValueError("Dataset catalogue differs")
    return R2C0WindowOrder(pools,seed)
