"""Saved-data-only review. No environment, torch, model construction or inference."""
import json, random, statistics, hashlib
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/accelerated-rgb-run-20261031"
OUT = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
train = read(RUN/"selection/train-index.json")
if isinstance(train, dict): train = train["rows"]
losses = [json.loads(s) for s in (RUN/"training/losses.jsonl").read_text().splitlines() if s.strip()]
cfg = read(RUN/"runtime-config.json")
rng = random.Random(int(cfg.get("order_seed", 20261031)))
pools = defaultdict(list)
for i, r in enumerate(train): pools[(r["family_id"], str(r.get("event","unknown")))].append(i)
keys = sorted(pools); rng.shuffle(keys)
for pool in pools.values(): rng.shuffle(pool)
order = []
while keys:
    remaining = []
    for key in keys:
        order.append(pools[key].pop())
        if pools[key]: remaining.append(key)
    keys = remaining
batches = [order[i:i+16] for i in range(0,len(order),16)]
identity = read(RUN/"training/identity.json")
canonical = json.dumps([[train[i]["sample_id"] for i in b] for b in batches], sort_keys=True, separators=(",",":"), allow_nan=False).encode()
order_hash = hashlib.sha256(canonical).hexdigest()
assert order_hash == identity["order_sha256"], "Order reconstruction mismatch"
assert sum(r["draws"] for r in losses) == len(train) and len(losses) == len(batches)
letters = {}
for letter in "OCBTLX":
    rs = [r for r in train if r["letter"] == letter]
    def quant(field):
        a = sorted(float(r[field]) for r in rs)
        return {str(q):a[round((len(a)-1)*q)] for q in (.5,.9,.99)}
    letters[letter] = dict(windows=len(rs),families=len({r["family_id"] for r in rs}),
        events=dict(Counter(r["event"] for r in rs)),
        translation_imagepx=quant("translation_imagepx"),abs_rotation_deg=quant("abs_rotation_deg"),
        rotation_ge_3deg=sum(r["abs_rotation_deg"]>=3 for r in rs),
        rotation_ge_5deg=sum(r["abs_rotation_deg"]>=5 for r in rs),
        translation_ge_1px=sum(r["translation_imagepx"]>=1 for r in rs))
quarters = []
for start,end in ((0,484),(484,968),(968,1452),(1452,len(batches))):
    rs = [train[i] for batch in batches[start:end] for i in batch]
    quarters.append(dict(batch_range=[start,end],draws=len(rs),
        mean_loss=statistics.mean(r["loss"] for r in losses[start:end]),
        median_loss=statistics.median(r["loss"] for r in losses[start:end]),
        mean_H1_rotation_deg=statistics.mean(r["abs_rotation_deg"] for r in rs),
        mean_H1_translation_imagepx=statistics.mean(r["translation_imagepx"] for r in rs),
        events=dict(Counter(r["event"] for r in rs))))
spec=read(RUN/"evaluation-spec.json")
metrics=[]
for phase in ("before","after"):
    for r in read(RUN/f"evaluation/{phase}/metrics.json")["rows"]:
        if r["method"]==phase: metrics.append(r)
groups=[]
for phase in ("before","after"):
    for letter in "OC":
        for h in (1,4,8,16):
            rs=[r for r in metrics if r["method"]==phase and r["letter"]==letter and r["horizon"]==h]
            groups.append(dict(method=phase,letter=letter,horizon=h,n=len(rs),
                gray_iou=statistics.mean(r["metrics"]["raw_mask_iou"] for r in rs),
                visible_centroid_error_imagepx=statistics.mean(r["metrics"]["object_position_error_image_px"] for r in rs),
                shape_assessable=sum(r["metrics"]["legacy_shape_assessable"] for r in rs)))
sources=["selection/train-index.json","training/losses.jsonl","training/identity.json","evaluation-spec.json",
         "evaluation/before/metrics.json","evaluation/after/metrics.json","training/source-recipe.json"]
result=dict(kind="saved-only-OC-training-review.v1",physics_calls=0,model_calls=0,training_updates=0,
    verified_training_order_sha256=order_hash,letters=letters,loss_quarters=quarters,
    evaluation_anchor_counts=dict(Counter(r["anchor"] for r in spec["rows"])),OC_metrics=groups,
    limitations=["Loss quarters differ in samples, motion mixture and random diffusion noise; no convergence claim.",
        "Current H1 evaluation is below showcase movement threshold; not a test of all moving H1 transitions.",
        "Image mask registration and corner counts are affected by occlusion/symmetry; no pose-independent rigidity pass."],
    source_sha256={s:hashlib.sha256((RUN/s).read_bytes()).hexdigest() for s in sources})
(OUT/"SAVED_DATA_ANALYSIS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(saved=str(OUT/"SAVED_DATA_ANALYSIS.json"),order_verified=True,
    train_windows=len(train),loss_batches=len(losses),O_windows=letters["O"]["windows"],
    C_windows=letters["C"]["windows"],new_physics=0,new_inference=0,new_training=0),ensure_ascii=False))
