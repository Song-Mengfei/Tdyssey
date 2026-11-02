# Stage1B source change and observed limitation

Only the generic recontact implementation changed. This note and the exact unified diff were derived after the completed eight-slot run; no code, data, physics or model execution was changed or added while writing them.

Baseline sampler SHA256: `ae7dce99e4159d17d6684558aa41c5169af90623cbcb9e591028b2865487fa41`.
Executed new sampler SHA256: `57fecfc81b7e6ef025b1154102e7c7a6e7ab370aa495ba14796052c3b42ec900`.
Paired report SHA256: `98c5b7e2a5854fda2e4b9f6727cae74588f81a37eff5f143eabaf4aee073d825`.

## Proven old failure mechanism

The old `_recontact` returned a fixed exterior staging waypoint. In the lost-contact branch, extension beyond that waypoint required the approach direction’s inward-normal dot product to exceed 0.2; otherwise the command stayed at the waypoint. The old policy switched back only when waypoint distance was at most 3px. A tangential approach therefore approached a preparation point slowly without a separate observed-contact state. The following are saved trajectory facts; no target switching cycle is inferred.

| Original X scene | First recovery controls | First/last target distance | Absolute target path | Real contact | Moving controls | Maximum cosine to inward normal |
|---|---|---:|---:|---:|---:|---:|
| center | 11–32 (22) | 17.928 → 3.071 px | 0.000000 px | 0 | 0 | -0.013594 |
| left-bottom | 13–38 (26) | 24.502 → 3.103 px | 0.000000 px | 0 | 0 | -0.117057 |
| right-top | 12–35 (24) | 20.245 → 3.055 px | 0.000000 px | 0 | 0 | -0.139976 |

The observations support fixed-target/arrival-tolerance slowdown and missing contact-completion feedback. They do not establish a waypoint-flipping cycle, physical impossibility, or a shape-specific defect.

## Scope preserved

AST comparison confirms the planning, selection, gap placement, area-centroid, initialization, and departure-route functions unchanged: `_sampling`, `plan_scenes`, `_rotation`, `_count`, `wall_gaps`, `_translation_room`, `_candidate_priority`, `plan_scene`, `initialize_scene`, `_area_center`, `_departure_route`.
The only changed top-level functions/classes are `_free_route`, `ContinuousContactSampler`. `_free_route` additionally handles a zero-length route as a point, avoiding an invalid degenerate-line distance. The new `RECOVERY_LIMITS` constant and recovery-state methods are generic across shapes.
The controller/environment and geometry/source files below retain their exact baseline hashes:

- `source-snapshot/tdyssey/native_rgb_env.py`: `47ee5c53ed3369da7548ea1e2512e5cf1eed315e68b290dcf2fa1ee7dd5500d5` (unchanged).
- `source-snapshot/tdyssey/native_rgb_data.py`: `49e54d8d40a2dec03eb75525b0e10b4c6f8e3372409dc14f1c64aba79ec3aa54` (unchanged).
- `source-snapshot/tdyssey/geometry.py`: `865bece79920bb543eda1f19753fa33a13de5236eb044e32a8d99ff9bc560456` (unchanged).
- `source-snapshot/tdyssey/letters.py`: `b168ee52db573ee5b626d09640ea475f484f0a74056700ec22f9a5205c7f0a28` (unchanged).
- `source-snapshot/tdyssey/envs/official.py`: `f3aaf10786c8eefc9db2462f9a5d6ea782adc06eef6860aecbcc74cd37817fc7` (unchanged).
- `source-snapshot/manifests/official_dp.json`: `6418e2bceac63e22d4db26122d3bf2630452667b18c5383033ba41e9b858adbc` (unchanged).
- `source-snapshot/official-source/diffusion_policy/env/pusht/pusht_env.py`: `759107dc0ea6017b8afe3380ce23817713edf26af725fc4f03c0416ced940bd0` (unchanged).
- `source-snapshot/official-source/diffusion_policy/env/pusht/pymunk_override.py`: `85e7c54b5e9630a9f7e675250ee514ae9b2249d11654ec0811d8e4bcf413fe39` (unchanged).

The executed controller remains `finite-force-pd-v1` / `fd358dba8f26a616ec6c55b3f6d532681bf5036e2cf79fe5a467001e3a77ae96`: same force, speed handling, mass/inertia, walls, 10Hz/100Hz timing, fixed 2px penetration threshold and 96px rendering. Root verified the full initial-state hashes for all six pairs and frozen inputs for both new X scenes. No intermediate reset, pose assignment, model input, model loss or prediction repair was introduced.

## Bounded change

Recovery now persists across the fixed phase schedule: verified free path → staging vicinity → separately validated inward physical contact → collision-confirmed contact → sustained moving contact. Staging arrival never sets actual-contact confirmation. Only a full verified exterior command may use lookahead. The final inward command checks the true clipped command’s exterior prefix up to entity contact; existing tolerated contact-shell geometry does not replace collision feedback.

Frozen numerical limits: `{"alternative_switches": 1, "contact_moving_controls": 2, "episode_controls": 20, "final_contact_controls": 8, "maximum_episodes": 4, "maximum_free_lookahead_world_px": 24.0, "no_progress_controls": 4, "path_controls": 12, "progress_tolerance_world_px": 0.25, "staging_arrival_tolerance_world_px": 4.0, "sustained_contact_controls": 3}`.
No-progress permits at most one alternative; failed arc attempts are disabled for the remainder of that trajectory, with cooldown and at most four recovery episodes. Failure does not label geometry unreachable. An incomplete recovery at control 63 records `trajectory_budget_exhausted`. Terminal episode metadata ends at that control rather than leaking into unrelated later controls. Post-contact moving counts exclude the first actual-contact control.

## Actual outcome: mixed, not a universal success

| Slot | Scene / role | Moving valid H1 windows old → new (denominator 61) | Near-static old → new |
|---|---|---:|---:|
| stage1b-00 | X center / paired | 18 → 35 | 41 → 23 |
| stage1b-01 | X left-bottom / paired | 15 → 17 | 46 → 43 |
| stage1b-02 | X right-top / paired | 20 → 31 | 38 → 26 |
| stage1b-03 | B left-top / paired | 5 → 5 | 49 → 53 |
| stage1b-04 | O left-top / paired | 14 → 14 | 46 → 45 |
| stage1b-05 | T center / paired | 48 → 43 | 11 → 16 |
| stage1b-06 | X center / new | new → 34 | new → 20 |
| stage1b-07 | X right / new | new → 23 | new → 33 |

The first X recovery in each of the three paired cases now reaches real contact and sustained moving contact. Whole-trajectory improvement is uneven: X left-bottom remains low; later X recoveries still hit the path timeout. B and O paired motion counts did not improve, T decreased, and the new boundary X scene remains below the moving/near-static targets. The two new X scenes are independent diagnostics, not matched baseline comparisons. No extra retry slots, filtering, denominator change or post-run tuning was performed.

The eight executed slots consumed 512 controls / 5120 substeps, with 0 invalid transitions; maximum substep penetration was [0.47079731170229877, 0.9831806491762922, 0.0]. Stage1+1B remains 3968 controls across 62 execution slots.
Physics validity is distinct from recovery effectiveness. Formal collection, training, inference, server execution and packaging remain unreleased pending parent review. No further code fix or actual execution is included in this source note.

Evidence: exact diff (external historical reference: `minimal-change.diff`), [paired report](recontact-report/recontact-report.md), physical audit (external historical reference: `physics-audit.json`), preserved baseline identity (external historical reference: `baseline-source-identity.json`).
