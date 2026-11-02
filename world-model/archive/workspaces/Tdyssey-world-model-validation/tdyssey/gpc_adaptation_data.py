"""Frozen official episode metadata and supervised GPC windows, without torch.

Four real observations and their four aligned controls predict the next real
observation. The default legacy domain maps all five images through fixed T1;
the explicitly selected raw domain preserves all five original RGB frames.
Model-test payloads have no loading path in this preparation API.
"""

from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from .dataset import load_episode
from .training_data import initial_state_hash, read_catalog


KIND = "tdyssey.gpc_adaptation_data.v1"
ENVIRONMENT_FAMILY = "official-pusht-capitals-v1"
T1_SPEC_SHA256 = "800751b7a3e32df4fee5660d1ea22c7ee0fb45b51a7b41632d420189238a7838"
ROLES = {"train", "development", "independent_validation", "held_model_test"}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _relative(parent, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Source catalog paths must be explicit relative paths")
    return (parent / value).resolve()


def readiness(manifest):
    """Describe gaps; reviewed development episodes cannot fill validation gaps."""
    rows = manifest["episodes"]
    shapes, gaps = {}, []
    for shape in ("T", "L"):
        train = [r for r in rows if r["shape"] == shape and r["role"] == "train" and r["windows"] and r["payloads_present"]]
        valid = [r for r in rows if r["shape"] == shape and r["role"] == "independent_validation" and r["windows"] and r["payloads_present"]]
        shapes[shape] = dict(train_episodes=len(train), train_windows=sum(r["windows"] for r in train),
            independent_validation_episodes=len(valid), independent_validation_windows=sum(r["windows"] for r in valid),
            independent_validation_scenes=len({r["initial_state_sha256"] for r in valid}))
        if not train:
            gaps.append(shape + ": missing complete official training episodes with four-control windows")
        if not valid:
            gaps.append(shape + ": missing genuinely independent validation episodes")
    return dict(ready=not gaps, gaps=gaps, shapes=shapes, model_test_payload_access=False,
                normalization="fixed T1 RGB; fixed absolute action/511*2-1; no fitted input normalization")


def read_adaptation_manifest(path, *, require_ready=False):
    """Read declarations/catalogs/manifests only; never open episode payloads.

    Every episode in every source catalog must be declared. Original whole-episode
    splits are preserved. Prior-use declarations supplement source scene hashes;
    their truth needs the reviewed provenance audit, not an inferred clean label.
    """
    path = Path(path).resolve()
    payload = path.read_bytes()
    value = json.loads(payload)
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("kind") != KIND or value.get("environment_family") != ENVIRONMENT_FAMILY
            or value.get("t1_spec_sha256") != T1_SPEC_SHA256):
        raise ValueError("Expected the fixed official/T1 adaptation manifest")
    catalogs, declarations = value.get("catalogs"), value.get("episodes")
    if not isinstance(catalogs, list) or not isinstance(declarations, list):
        raise ValueError("Manifest needs explicit catalogs and episode declarations")
    reviewed = value.get("reviewed_scene_sha256")
    used_ids = value.get("previously_used_episode_ids")
    if (not isinstance(reviewed, list) or not all(_is_sha(s) for s in reviewed)
            or len(set(reviewed)) != len(reviewed) or not isinstance(used_ids, list)
            or not all(isinstance(s, str) and s for s in used_ids) or len(set(used_ids)) != len(used_ids)):
        raise ValueError("Explicit reviewed-scene and previously-used episode exclusions are required")
    source, catalog_paths, scenes, image_splits = {}, set(), {}, {}
    for declared in catalogs:
        if not isinstance(declared, dict) or not _is_sha(declared.get("sha256")):
            raise ValueError("Each source catalog needs a fixed SHA256")
        catalog_path = _relative(path.parent, declared.get("path"))
        if catalog_path in catalog_paths:
            raise ValueError("Duplicate source catalog")
        catalog_paths.add(catalog_path)
        if _sha(catalog_path.read_bytes()) != declared["sha256"]:
            raise ValueError("Source catalog identity changed")
        catalog, roots, manifests = read_catalog(catalog_path)
        if catalog.get("environment_family") != ENVIRONMENT_FAMILY or catalog["rates"] != dict(simulation_hz=100, control_hz=10):
            raise ValueError("Only official 100Hz/10Hz T/L collector data is eligible")
        for run, root, manifest in zip(catalog["runs"], roots, manifests):
            from .envs.official import official_shape
            from .official_collection import VERSION, DEVELOPMENT_SEEDS
            shape = manifest["config"].get("shape")
            if (shape not in ("T", "L") or manifest["provenance"].get("collector_version") != VERSION
                    or manifest["provenance"].get("geometry_hash") != official_shape(shape).geometry_hash
                    or not set(DEVELOPMENT_SEEDS).issubset(manifest["config"].get("excluded_reset_seeds", []))):
                raise ValueError("Only official T/L collector geometry and evaluation exclusions are eligible")
            entries = {e["trajectory_id"]: e for e in manifest["episodes"]}
            for row in run["episodes"]:
                identifier, scene, split = row["trajectory_id"], row["initial_state_sha256"], row["split"]
                if identifier in source or not _is_sha(scene):
                    raise ValueError("Duplicate episode identity or invalid source scene hash")
                if scene in scenes and scenes[scene] != split:
                    raise ValueError("Duplicate initial scene crosses whole-episode splits")
                scenes[scene] = split
                entry = entries[identifier]
                present = True
                for item in entry["files"]:
                    if (type(item.get("bytes")) is not int or item["bytes"] <= 0
                            or not _is_sha(item.get("sha256"))):
                        raise ValueError("Source episode files need positive sizes and SHA256 identities")
                    payload_path = root/item["name"]
                    if payload_path.is_symlink():
                        raise ValueError("Episode payload aliases are forbidden")
                    # Stat only for all splits, especially model_test: no held
                    # record, archive, image or tensor is opened here.
                    present = present and payload_path.is_file() and payload_path.stat().st_size == item["bytes"]
                    if Path(item["name"]).suffix == ".npz":
                        old_split = image_splits.get(item["sha256"])
                        if old_split is not None and old_split != split:
                            raise ValueError("Duplicate image payload crosses data splits")
                        image_splits[item["sha256"]] = split
                source[identifier] = dict(trajectory_id=identifier, shape=shape, split=split,
                    initial_state_sha256=scene, control_steps=entry["control_steps"],
                    contact_steps=entry["contact_steps"], windows=max(0, entry["control_steps"]-3),
                    payloads_present=present,
                    root=str(root), entry=_copy(entry), catalog_path=str(catalog_path),
                    catalog_sha256=declared["sha256"], source_manifest_sha256=run["manifest_sha256"])
    normalized, seen = [], set()
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("Episode declarations must be objects")
        identifier = declaration.get("trajectory_id")
        if not isinstance(identifier, str) or identifier not in source or identifier in seen:
            raise ValueError("Episode declarations are duplicate or outside the frozen catalogs")
        seen.add(identifier)
        role, prior = declaration.get("role"), declaration.get("prior_uses")
        if role not in ROLES or not isinstance(prior, list) or not all(isinstance(s, str) and s for s in prior):
            raise ValueError("Each episode needs an explicit role and prior-use list")
        row = source[identifier]
        permitted = {"train": {"train"}, "validation": {"development", "independent_validation"},
                     "model_test": {"held_model_test"}}[row["split"]]
        if role not in permitted:
            raise ValueError("Original whole-episode split cannot be reassigned, especially held model_test")
        if role == "independent_validation" and (prior or identifier in used_ids or row["initial_state_sha256"] in reviewed):
            raise ValueError("Previously used/reviewed data cannot become independent validation")
        normalized.append(dict(row, role=role, prior_uses=list(prior)))
    if seen != set(source):
        raise ValueError("Declare every source episode, including the inaccessible held model_test split")
    used_scenes = set(reviewed) | {r["initial_state_sha256"] for r in normalized
        if r["prior_uses"] or r["trajectory_id"] in used_ids or r["role"] == "development"}
    if any(r["role"] == "independent_validation" and r["initial_state_sha256"] in used_scenes for r in normalized):
        raise ValueError("A reviewed/development scene alias cannot become independent validation")
    result = dict(path=str(path), sha256=_sha(payload), schema_version=1, kind=KIND,
                  t1_spec_sha256=T1_SPEC_SHA256, episodes=normalized)
    result["readiness"] = readiness(result)
    if require_ready and not result["readiness"]["ready"]:
        raise ValueError("Adaptation data is not ready: " + "; ".join(result["readiness"]["gaps"]))
    return result


class GPCAdaptationDataset:
    """Lazy one-episode cache, no padding, no model-test access and no torch."""

    def __init__(self, manifest_path, split="train", *, require_ready=False,
                 image_domain="legacy_t1"):
        if split not in ("train", "development", "independent_validation"):
            raise ValueError("Choose train, development or independent_validation; ambiguous validation and held model_test are forbidden")
        if image_domain not in ("legacy_t1", "official_raw_rgb"):
            raise ValueError("Choose explicit legacy_t1 or official_raw_rgb image domain")
        if image_domain == "official_raw_rgb" and split not in ("train", "development"):
            raise ValueError("Official raw loading permits train/development only")
        self.image_domain = self._bound_image_domain = image_domain
        self.split = self.role = split
        self.source_split = "train" if split == "train" else "validation"
        self.manifest = read_adaptation_manifest(manifest_path, require_ready=require_ready)
        self.episodes = _copy([r for r in self.manifest["episodes"] if r["role"] == split])
        self._bound_episodes = _copy(self.episodes)
        self.offsets = [0]
        for row in self.episodes:
            self.offsets.append(self.offsets[-1] + row["windows"])
        self._cache_index, self._cache = None, None

    def __len__(self):
        return self.offsets[-1]

    def shape_indices(self, shape):
        """Stable indices for an explicit T:L alternating sampler, without RNG."""
        if shape not in ("T", "L"):
            raise ValueError("Shape must be T or L")
        return tuple(i for n, row in enumerate(self.episodes) if row["shape"] == shape
                     for i in range(self.offsets[n], self.offsets[n+1]))

    def _load(self, episode_index):
        row = self.episodes[episode_index]
        # Defense before any payload read, even if a caller mutates public rows.
        if (row != self._bound_episodes[episode_index] or row["split"] != self.source_split
                or row["role"] != self.role or row["role"] == "held_model_test"):
            raise ValueError("Held model_test or reassigned episode payload access is forbidden")
        if (_sha(Path(self.manifest["path"]).read_bytes()) != self.manifest["sha256"]
                or _sha(Path(row["catalog_path"]).read_bytes()) != row["catalog_sha256"]
                or _sha((Path(row["root"])/"manifest.json").read_bytes()) != row["source_manifest_sha256"]):
            raise ValueError("Frozen adaptation/catalog/source manifest changed")
        if self._cache_index == episode_index:
            return self._cache
        trajectory, record, arrays = load_episode(row["root"], row["entry"])
        from .envs.official import official_shape
        from .official_collection import VERSION, DEVELOPMENT_SEEDS
        initial = trajectory.transitions[0].before
        attempts = record["reset_attempts"]
        if (initial_state_hash(trajectory) != row["initial_state_sha256"]
                or trajectory.collector_version != VERSION or initial.shape != row["shape"]
                or initial.geometry_hash != official_shape(row["shape"]).geometry_hash
                or not math.isclose(trajectory.control_dt, .1, rel_tol=0, abs_tol=1e-12)
                or record["reset_seed"] in DEVELOPMENT_SEEDS
                or sum(d["n_contacts"] > 0 for d in record["step_diagnostics"]) != row["contact_steps"]
                or not attempts or attempts[-1].get("issues") or not attempts[-1].get("simulated")
                or attempts[-1].get("accepted_state") != json.loads(initial.to_json())):
            raise ValueError("Official scene, collector, geometry, timing or evaluation exclusions differ")
        self._cache_index, self._cache = episode_index, (trajectory, record, arrays)
        return self._cache

    def __getitem__(self, index):
        import numpy as np
        if self.image_domain != self._bound_image_domain:
            raise ValueError("The fixed image domain was reassigned")
        if self.image_domain == "legacy_t1":
            from .wm_compat_transform import T1_SHA256, transform_t1
            if T1_SHA256 != T1_SPEC_SHA256:
                raise ValueError("The fixed T1 identity changed")
        if type(index) is not int:
            raise TypeError("Window index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect_right(self.offsets, index)-1
        row = self.episodes[episode_index]
        trajectory, _, arrays = self._load(episode_index)
        anchor = 3 + index-self.offsets[episode_index]
        raw = arrays["images"][anchor-3:anchor+2].astype(np.float32)/np.float32(255)
        images = (np.stack([transform_t1(frame)[0] for frame in raw])
                  if self.image_domain == "legacy_t1" else raw)
        absolute = np.asarray([t.action for t in trajectory.transitions[anchor-3:anchor+1]], dtype=np.float64)
        actions = np.asarray(absolute, dtype=np.float32)
        if actions.shape != (4,2) or not np.isfinite(actions).all() or np.any(actions < 0) or np.any(actions > 511):
            raise ValueError("Four aligned absolute actions must stay in [0,511]")
        # The new domain changes image colour only, never the pilot loss inputs.
        normalized = actions/np.float32(511)*np.float32(2)-np.float32(1)
        result = dict(image=images, action=normalized,
                    episode_id=row["trajectory_id"], shape=row["shape"], split=self.split, role=self.role,
                    source_split=row["split"], anchor=anchor,
                    t1_spec_sha256=T1_SPEC_SHA256 if self.image_domain == "legacy_t1" else None)
        if self.image_domain == "official_raw_rgb":
            transitions = trajectory.transitions[anchor-3:anchor+1]
            result.update(image_domain=self.image_domain, inputs=images[:4], target=images[4],
                absolute_action=absolute.copy(), absolute_actions=absolute.copy(),
                history_states=[json.loads(step.before.to_json()) for step in transitions],
                target_state=json.loads(transitions[-1].after.to_json()))
        return result


def fit_train_stats(dataset):
    """Describe unique train controls; never fit model input normalization."""
    if not isinstance(dataset, GPCAdaptationDataset) or dataset.split != "train":
        raise ValueError("Descriptive statistics may read training episodes only")
    import numpy as np
    actions, translations, rotations, contacts, ids, per_shape = [], [], [], [], [], {"T": 0, "L": 0}
    for n,row in enumerate(dataset.episodes):
        if row["role"] != "train" or row["split"] != "train":
            raise ValueError("Statistics cannot use held or development episodes")
        trajectory, record, _ = dataset._load(n)
        ids.append(row["trajectory_id"])
        for transition, diagnostic in zip(trajectory.transitions, record["step_diagnostics"]):
            a,b = transition.before.object_body.pose, transition.after.object_body.pose
            actions.append(transition.action)
            translations.append(math.dist(a.position,b.position))
            rotations.append(abs(math.atan2(math.sin(b.angle-a.angle),math.cos(b.angle-a.angle))))
            contacts.append(diagnostic["n_contacts"] > 0)
            per_shape[row["shape"]] += 1
    if not actions:
        raise ValueError("Training episodes have no controls")
    a = np.asarray(actions,dtype=np.float64)
    return dict(split="train", descriptive_only=True, episode_ids=ids, unique_controls=len(actions), controls_by_shape=per_shape,
        contact_steps=sum(contacts), contact_fraction=float(np.mean(contacts)), action_mean=a.mean(axis=0).tolist(),
        action_std=a.std(axis=0).tolist(), translation_pixels=dict(mean=float(np.mean(translations)),max=float(np.max(translations))),
        absolute_rotation_radians=dict(mean=float(np.mean(rotations)),max=float(np.max(rotations))),
        input_normalization="unchanged fixed T1 and action/511*2-1; these statistics never rescale inputs")
