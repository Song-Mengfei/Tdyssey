# Fixed-step candidate2 before running

Candidate1 failed control29: adaptive tiny final remainders caused changing native impulse warm-start dt ratios, followed by alternating huge/ordinary dynamic velocities and eventual numerical overflow. No material overlap was hidden. Full failed rows and costs remain in B-candidate-v1-failed.json.gz and B-probe-summary.json. Original first writer failure also remains separate.

Freeze candidate2: same unilateral velocity projection, epsilon.25world, valid overlap1world, solver50 and 100Hz PD/10Hz controls; replace variable microsteps with exactly10 equal .001s microsteps per .01s PD update (1000Hz physics). No tiny remainders and no state/cache reset, object pose projection, artificial damping or speed cap. New explicit protocol versionv2. At most one37-control B replay initially; root owns wider regression. Thresholds are unchanged, not widened based on results.
