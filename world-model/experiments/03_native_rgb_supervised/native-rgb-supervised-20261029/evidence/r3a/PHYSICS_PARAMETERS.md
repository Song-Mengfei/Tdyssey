# R3A actual collection parameters

These values were read from the actual six initial physics identities, rather
than inferred from controller intentions. The selected protocol is unchanged:
`58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6`.

| Parameter | Actual value |
|---|---|
| Velocity half-life under free damping | 0.1 seconds |
| `Space.damping` (one-second retention parameter) | 0.0009765625 |
| Retention over each 0.01-second physics step | `0.0009765625 ** 0.01` |
| Control frequency | 10 Hz |
| Physics frequency | 100 Hz; ten substeps per control |
| Trajectory length | 48 controls, 4.8 simulated seconds |
| Action | Absolute pusher target `(x, y)` in `[0, 511]` |
| PD gains | `kp = 100`, `kv = 20` |
| Pusher force limit | 10000 simulator force units |
| Pusher speed cap | 200 world pixels/second, checked before each physics step |
| Pusher mass / radius | 1 simulator mass unit / 15 world pixels |
| Pusher moment of inertia | Infinite; no pusher spin |
| Object mass | 1 simulator mass unit for every seen letter |
| Shape friction / elasticity | 0 / 0 for object, pusher and walls |
| Gravity | `(0, 0)` |
| Solver iterations / collision slop | 50 / 0.1 world pixels |
| Wall inner coordinates | 7 through 504 world pixels |
| Transition penetration limit | 2 world pixels; a violation stops the attempt |
| RGB | Original 96 × 96 white background, gray object and blue pusher |

The damping multiplier acts over elapsed time. Under free damping the velocity
retains 50% over 0.1 seconds. During pushing, applied forces and collisions also
affect velocity. The recorded post-collision speed can differ from the speed cap
that is applied before the next physics step.

| Letter | Actual moment of inertia | Body-local physical center of gravity |
|---|---:|---|
| T | 3000 | `(0, 45)` |
| L | 1875 | `(0, 0)` |
| O | 2340.2721088435374 | `(0, 0)` |
| C | 2613.977324263039 | `(0, 0)` |
| X | 2316.646658663465 | `(0, 0)` |
| B | 2204.024943310658 | `(0, 0)` |

Masses, moments, geometry, force control and collision settings remain those of
the selected existing simulator. Full measurements are in
`physics-identities/{T,L,O,C,X,B}.json`. The nested `base_control_protocol` retains
its historical description of the upstream base; the actual selected damping
is the measured `physics.space.damping` above.

The new collection scope is 128 independent TRAIN plus 24 independent seen
validation trajectories per letter, with the split fixed before acquisition.
The initial poses cover twelve yaw bins and nine position regions with variation
within each bin. Current-state privileged geometry guides the collector; it does
not provide model features or a future-physics search. Successful trajectories
are retained even when some sections stall. Failed attempts and reserved costs
are retained separately, with at most two pre-frozen replacement candidates.
