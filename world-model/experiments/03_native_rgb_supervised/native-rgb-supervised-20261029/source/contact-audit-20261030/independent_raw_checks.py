"""Independent saved-array checks; never imports the new audit implementation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tdyssey.envs.official import official_shape

STAGE = Path(__file__).resolve().parent
SOURCE = STAGE.parent / "r3a"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check():
    manifest_path = SOURCE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    with np.load(STAGE / "core/controls.npz", allow_pickle=False) as z:
        core = {key: z[key] for key in z.files}
    checks = []

    def require(condition, label, detail=None):
        checks.append(dict(check=label, passed=bool(condition), detail=detail))
        if not condition:
            raise AssertionError(label)

    require(len(entries) == 912, "successful trajectory count")
    require(len(core["control"]) == 43776, "successful control count")
    # These six first-letter cases are independent of the measured score. Add a
    # preselected regression and a reproducible uniform random case.
    chosen = [next(i for i, e in enumerate(entries) if e["letter"] == letter)
              for letter in "TLOCXB"]
    chosen.append(next(i for i, e in enumerate(entries)
                       if e["trajectory_id"] == "R3A-seen_val-B-014-a0"))
    chosen.append(int(np.random.default_rng(5478).integers(len(entries))))
    samples = []
    for ti in dict.fromkeys(chosen):
        e = entries[ti]
        source = SOURCE / e["file"]
        require(digest(source) == e["sha256"], e["trajectory_id"] + " NPZ identity")
        with np.load(source, allow_pickle=False) as z:
            raw = {k: z[k] for k in ("states", "contact_flags", "contact_tags",
                                     "contact_values", "control_phase", "control_mode")}
        states = raw["states"]
        # Area-weighted component centers are calculated directly here rather
        # than calling the audit's centroid or its motion helper.
        rectangles = official_shape(e["letter"]).components
        areas = np.array([(r.x1-r.x0)*(r.y1-r.y0) for r in rectangles])
        centers = np.array([[(r.x1+r.x0)/2, (r.y1+r.y0)/2] for r in rectangles])
        local = np.sum(centers * areas[:, None], axis=0) / areas.sum()
        world = []
        for state in states:
            cs, sn = math.cos(float(state[4])), math.sin(float(state[4]))
            world.append([state[2]+cs*local[0]-sn*local[1],
                          state[3]+sn*local[0]+cs*local[1]])
        world = np.asarray(world)
        trans = np.hypot(*np.diff(world, axis=0).T) * (96/512)
        angle = np.diff(states[:, 4]) * (180/math.pi)
        flag = (raw["contact_flags"] & 2) != 0
        counts = flag.reshape(48, 10, 3).sum(axis=1)
        rows = slice(ti*48, (ti+1)*48)
        require(np.allclose(core["translation_image96_px"][rows], trans, rtol=0, atol=1e-12),
                e["trajectory_id"] + " independent rectangle centroid motion")
        require(np.allclose(core["signed_rotation_deg"][rows], angle, rtol=0, atol=1e-12),
                e["trajectory_id"] + " unfurled signed angle")
        require(np.array_equal(core["contact_substeps"][rows], counts[:, 0]),
                e["trajectory_id"] + " 100Hz AO counts")
        require(np.array_equal(core["terminal_contact"][rows], flag[9::10, 0]),
                e["trajectory_id"] + " preceding interval terminal flag")
        for field, column in (("object_wall_substeps", 1), ("agent_wall_substeps", 2)):
            require(np.array_equal(core[field][rows], counts[:, column]),
                    e["trajectory_id"] + " " + field)
        require(np.array_equal(core["phase"][rows], raw["control_phase"]), e["trajectory_id"]+" phase")
        require(np.array_equal(core["mode"][rows], raw["control_mode"]), e["trajectory_id"]+" mode")
        motion = np.where((trans < .1) & (np.abs(angle) < .1), 0,
                          np.where((trans >= 2) | (np.abs(angle) >= 5), 2, 1))
        require(np.array_equal(core["motion_class"][rows], motion), e["trajectory_id"]+" six-class motion")
        tags, values = raw["contact_tags"], raw["contact_values"]
        post_rows = (tags[:, 1] == 0) & (tags[:, 2] == 1)
        valid_rows = post_rows & ((tags[:, 4] & 4) != 0) & np.isfinite(values[:, 7:9]).all(axis=1)
        observed = np.zeros(480, dtype=bool)
        observed[tags[valid_rows, 0]] = True
        require(np.array_equal(core["known_impulse_contact_substeps"][rows],
                               (observed & flag[:, 0]).reshape(48, 10).sum(axis=1)),
                e["trajectory_id"]+" impulse presence instead of point sum")
        epsilon_counts = []
        for epsilon in (0, 1e-8, 1e-6, 1e-4):
            positive = np.zeros(480, dtype=bool)
            selected = valid_rows & (np.hypot(values[:, 7].astype(float), values[:, 8].astype(float)) > epsilon)
            positive[tags[selected, 0]] = True
            epsilon_counts.append((positive & flag[:, 0]).reshape(48, 10).sum(axis=1))
        require(np.array_equal(core["nonzero_impulse_counts"][rows], np.asarray(epsilon_counts).T),
                e["trajectory_id"]+" epsilon sensitivity")
        samples.append(dict(trajectory_id=e["trajectory_id"], manifest_index=ti,
                            postsolve_AO_substeps=int(counts[:, 0].sum()),
                            contact_control_count=int((counts[:, 0] > 0).sum()),
                            terminal_contact_count=int(flag[9::10, 0].sum()),
                            motion_class_counts=np.bincount(motion, minlength=3).tolist()))
        if e["trajectory_id"] == "R3A-seen_val-B-014-a0":
            idx = np.flatnonzero(flag[:, 0])
            breaks = np.flatnonzero(np.diff(idx) > 1)
            runs = [[int(chunk[0]), int(chunk[-1]+1)] for chunk in np.split(idx, breaks+1)]
            require(runs == [[30, 181], [391, 443]], "B014 exact contact runs", runs)
            require(counts[38, 0] == 0 and raw["control_mode"][38] == 2,
                    "B014 control38 intent still has no actual contact")
            require(np.array_equal(counts[39:45, 0], [9,10,10,10,10,3]),
                    "B014 brief boundary contacts are not whole intervals")
            first = int(np.flatnonzero(flag[260:, 0])[0]+260)
            require(first == 391, "B014 first actual recontact global index")
            require([bool(flag[10*j-1, 0]) for j in (12,24,36,48)] == [True,False,False,False],
                    "B014 displayed states use preceding interval endpoint")
    # All legal H1 motions must agree with the pre-existing independent catalogue.
    with np.load(SOURCE / "data-interface/window-catalogue.npz", allow_pickle=False) as z:
        rows = z["h1_trajectory_index"].astype(np.int64)*48 + z["h1_anchor"].astype(np.int64)
        require(len(rows) == 41040, "all legal H1 catalogue count")
        require(np.allclose(core["translation_image96_px"][rows], z["h1_translation_image96_px"], rtol=0, atol=1e-12),
                "all41040 H1 translation matches old independent catalogue")
        require(np.allclose(core["signed_rotation_deg"][rows], z["h1_signed_rotation_deg"], rtol=0, atol=1e-12),
                "all41040 H1 signed angle matches old independent catalogue")
    report = dict(kind="contact-audit-independent-raw-checks.v1", source_manifest_sha256=digest(manifest_path),
                  core_controls_sha256=digest(STAGE/"core/controls.npz"),
                  model_calls=0, physics_steps=0, new_audit_helpers_imported=False,
                  deterministic_seed=5478, samples=samples, checks=checks,
                  passed=sum(c["passed"] for c in checks), total=len(checks))
    (STAGE/"independent-raw-checks.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(passed=report["passed"], total=report["total"], samples=len(samples))))


if __name__ == "__main__":
    check()
