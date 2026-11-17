"""Collect eight official-T batches that cover object-relative push directions.

Each batch is 8 episodes of 100 controls. Targets use the object pose and can
lie far from the pusher. This is a disclosed geometric prior, not reward search.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_official_play import provenance
from tdyssey.official_collection import OfficialCollectionConfig
from tdyssey.official_collector import OfficialShapeCollector


def main():
    output = ROOT / "artifacts/official-wm-T-angles-2026-10-23"
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index in range(8):
        root = output / f"batch-{index + 1:03d}"
        if root.exists():
            raise FileExistsError(root)
        config = OfficialCollectionConfig(shape="T", episodes=8, steps=100, seed=230000 + index * 100,
                                          mode="angle-coverage", max_target_delta=200, seconds=180)
        note = provenance(config)
        note["action_prior"] = ("angle-coverage uses object position and orientation; "
                                "16 directions and near/mid/far targets up to 150 pixels; no reward optimization")
        manifest = OfficialShapeCollector(config, ROOT / "third_party/diffusion_policy", workers=2).collect(
            root, note)
        summaries.append({"batch": root.name, "controls": manifest["counts"]["control_steps"],
                          "contact_steps": sum(row["contact_steps"] for row in manifest["episodes"])})
        print(json.dumps(summaries[-1]), flush=True)
    print(json.dumps({"status": "complete", "batches": summaries,
                      "controls": sum(row["controls"] for row in summaries),
                      "contact_steps": sum(row["contact_steps"] for row in summaries)}, indent=2))


if __name__ == "__main__":
    main()
