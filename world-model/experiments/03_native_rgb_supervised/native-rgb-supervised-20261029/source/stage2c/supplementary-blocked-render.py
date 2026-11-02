"""Authorized one-case supplement from already frozen extra evaluation indices."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.native_rgb_stage2c_indices import read_json,sha,verify_frozen_inputs,write_json
from scripts.native_rgb_stage2c_report import render_examples
from tdyssey.native_rgb_motion_statistics import read_records

stage=Path(__file__).resolve().parent
indices_path=stage/"report/indices/indices.json"
indices=read_json(indices_path)
verify_frozen_inputs(stage/"data",ROOT/"configs/native_rgb_stage2c_indices.json")
available=[row for row in indices["extra"] if row["conditions"]["likely_wall_blocked_proxy"]]
if not available:raise RuntimeError("No frozen extra blocked-proxy window; no fallback allowed for this supplement")
chosen=min(available,key=lambda row:(row["selection_hash"],row["trajectory_id"],row["anchor"]))
manifest=read_json(stage/"data/manifest.json")
entry=next(row for row in manifest["entries"] if row["trajectory_id"]==chosen["trajectory_id"])
assert entry["sha256"]==chosen["source"]["npz_sha256"]
assert entry["metadata_sha256"]==chosen["source"]["metadata_sha256"]
assert sha(stage/"data"/entry["metadata_file"])==entry["metadata_sha256"]
records=read_records(stage/"data",dict(entries=[entry]))
case=dict(letter=chosen["letter"],trajectory_id=chosen["trajectory_id"],anchor=chosen["anchor"],
    requested_spatial_class=chosen["h1"]["requested_spatial_class"],desired_category="likely_wall_blocked_proxy",
    desired_category_unavailable=False,fallback_used=False,actual_h1_motion_class=chosen["h1"]["motion_class"],
    actual_h1_rotation_abs_deg=chosen["h1"]["rotation_abs_deg"],actual_h1_contact=chosen["h1"]["contact"],
    actual_likely_wall_blocked_proxy=True,conditions=chosen["conditions"])
selection=dict(rules=dict(cases=[dict(letter=case["letter"],desired_category="likely_wall_blocked_proxy")],
    max_cases=1,max_sheet_rows=1,max_gifs=1,max_consecutive_frames_per_case=8),cases=[case],missing=[],
    selection_uses_model_outputs=False)
output=stage/"report/supplementary-blocked"
result=render_examples(records,manifest["source_root"],output,selection)
write_json(output/"selection-from-frozen-index.json",dict(
    reason="The original fixed M blocked-proxy example was unsupported and explicitly fell back to moving; add one actual proxy case without changing those original files or any evaluation index.",
    rule="Minimum (already frozen extra selection_hash, trajectory_id, anchor) among frozen extra H1 likely_wall_blocked_proxy=true windows.",
    frozen_indices_sha256=sha(indices_path),chosen_window_id=chosen["window_id"],selection_hash=chosen["selection_hash"],
    eligible_frozen_extra_windows=len(available),h1_truth=chosen["h1"],h4_truth=chosen["h4"],source=chosen["source"],
    original_M_blocked_support_missing=True,original_examples_or_indices_modified=False,
    qualification="Near-static actual object-pusher contact, gap<=2 and target toward nearest wall is a wall-blocking proxy; no measured wall force or wall-arbiter contact claim.",
    source_script_sha256=sha(__file__),physics_controls=0,model_calls=0,training_index_draws=0))
(output/"render_supplement.py").write_bytes(Path(__file__).read_bytes())
write_json(output/"files.json",[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.iterdir()) if p.is_file()])
assert sha(indices_path)==read_json(output/"selection-from-frozen-index.json")["frozen_indices_sha256"]
print(json.dumps(dict(window_id=chosen["window_id"],letter=chosen["letter"],frames=result["sequences"][0]["frames"],
    output=str(output),proxy_only=True,physics_controls=0,model_calls=0)))
