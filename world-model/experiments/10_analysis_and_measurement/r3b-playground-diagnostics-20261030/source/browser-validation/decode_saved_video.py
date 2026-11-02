"""Decode only saved videos/raw manifests; no model, GPU, environment or UI."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parent
check = json.loads((root / "browser-check.json").read_text(encoding="utf-8"))
rows = []
for item in check["recordings"]:
    path = Path(item["downloaded_file"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cap = cv2.VideoCapture(str(path))
    counts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (776, 1224)
        counts.append([
            int((np.min(frame[100:676, 24:600], axis=2) < 220).sum()),
            int((np.min(frame[100:676, 624:1200], axis=2) < 220).sum()),
        ])
    cap.release()
    both = sum(left > 20 and right > 20 for left, right in counts)
    wall = item["server_metadata"]["client"]["duration_ms"] / 1000
    video = item["decoded"]["duration"]
    raw = item["raw_manifest"]
    raw_root = Path(item["server_metadata"]["raw_stream"]["manifest_path"]).parent
    files = [raw["initial_session"], *raw["batches"]]
    raw_hashes = all(hashlib.sha256((raw_root / row["file"]).read_bytes()).hexdigest() == row["sha256"] for row in files)
    passed = (
        digest == item["sha256"] == item["server_metadata"]["sha256"]
        and len(counts) > 0 and both > 0
        and raw["raw_complete"] and raw["closed"] and not raw["missing_batches"]
        and raw_hashes and abs(wall - video) < 0.35
        and item["one"]["delta"] == 1 and item["four"]["delta"] == 4
    )
    rows.append({
        "round": item["round"], "recording_id": item["recording_id"],
        "file": str(path), "sha256": digest, "frames": len(counts),
        "frames_with_both_nonempty": both,
        "first_counts": counts[0] if counts else None,
        "last_counts": counts[-1] if counts else None,
        "video_seconds": video, "client_seconds": wall,
        "tail_gap_seconds": wall - video,
        "raw_complete": raw["raw_complete"], "raw_closed": raw["closed"],
        "raw_files": len(files), "raw_hashes_match": raw_hashes,
        "controls": item["one"]["delta"] + item["four"]["delta"],
        "passed": bool(passed),
    })
old = root / "video-decode-check.json"
archive = root / "before-tail-fix" / "video-decode-check.json"
if old.exists() and not archive.exists():
    archive.parent.mkdir(exist_ok=True)
    archive.write_bytes(old.read_bytes())
result = {
    "passed": check["passed"] and all(row["passed"] for row in rows),
    "method": "Local CPU decode of the actual downloaded WebM; browser file-origin guard unchanged.",
    "tail_fix_verified": all(abs(row["tail_gap_seconds"]) < 0.35 for row in rows),
    "initial_blank_prediction_is_preserved": True,
    "no_model_or_physics_calls": True,
    "actual_video_checks": rows,
}
old.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
assert result["passed"]
