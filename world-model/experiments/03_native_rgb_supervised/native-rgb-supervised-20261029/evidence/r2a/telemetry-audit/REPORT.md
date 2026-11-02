# Independent R2A recorded-telemetry audit

Integrity checks passed: **True**.

Recorded 512 charged controls and 5120 completed physics substeps; planned denominators remain 64 slots / 512 controls.

The unclipped force is an exact formula reconstruction from recorded state and action. The clipped force and cap-entry velocity are engine-property measurements. Only post_solve impulses are aggregated; pre_solve events are excluded from impulse totals.

## P0

Recorded substeps: 2560; force clipping: 143; speed clipping: 0.

- measured_clipped_force_maxabs_error: n=2560, median=0.0, max=0.0.
- after_cap_velocity_maxabs_error: n=2560, median=0.0, max=0.0.
- free_agent_actual_damping_velocity_residual_norm: n=1160, median=0.0, max=0.0.
- free_agent_other_damping_counterfactual_residual_norm: n=1160, median=21.980586566505224, max=100.00000000000003.
- free_object_actual_damping_velocity_residual_norm: n=1100, median=0.0, max=0.0.
- agent_speed_post: n=2560, median=26.560000000000016, max=100.00000000000001.
- object_mass_COM_speed_post: n=2560, median=3.9560816897638587, max=50.00000000000001.
- penetration_world_px: n=2560, median=0.026529653064088166, max=0.7726018974535425.
- contact_approach_reconstructed_closing_speed: n=1578, median=9.071471928989716e-05, max=76.70720277752368.
- contact_approach_reconstructed_abs_tangential_speed: n=1578, median=8.698030035354273, max=105.15822409356073.
- contact_point_physical_COM_lever_length_world_px: n=1578, median=61.77483321607076, max=85.32552047567277.
- agent_object_contact_episode_seconds: n=57, median=0.11, max=0.6.

Contact episodes=57; observed contact-loss transitions=33; recontacts after an earlier observed episode=29.

- all_raw_controls: 256 controls / 16 initial families; area-centroid image displacement median=0.09604444743884155, max=0.9353549721969963.
- contact_with_nonzero_impulse: 158 controls / 14 initial families; area-centroid image displacement median=0.2588979504384934, max=0.9353549721969963.
- contact_impulse_and_force_toward_object_normal: 158 controls / 14 initial families; area-centroid image displacement median=0.2588979504384934, max=0.9353549721969963.
- no_agent_object_contact: 75 controls / 16 initial families; area-centroid image displacement median=0.0, max=0.0.
- free_object_coasting_no_agent_or_wall_contact: 75 controls / 16 initial families; area-centroid image displacement median=0.0, max=0.0.
- object_wall_collision: 22 controls / 3 initial families; area-centroid image displacement median=0.324946044466235, max=0.6079024242941585.
- agent_wall_collision: 0 controls / 0 initial families; area-centroid image displacement median=None, max=None.

## P1

Recorded substeps: 2560; force clipping: 42; speed clipping: 932.

- measured_clipped_force_maxabs_error: n=2560, median=0.0, max=0.0.
- after_cap_velocity_maxabs_error: n=2560, median=0.0, max=2.842170943040401e-14.
- free_agent_actual_damping_velocity_residual_norm: n=1682, median=0.0, max=0.0.
- free_agent_other_damping_counterfactual_residual_norm: n=1682, median=199.01220823456782, max=200.00000000000006.
- free_object_actual_damping_velocity_residual_norm: n=1508, median=0.0, max=0.0.
- agent_speed_post: n=2560, median=143.91853364242274, max=247.30188030695243.
- object_mass_COM_speed_post: n=2560, median=78.98372531574736, max=392.8795701982306.
- penetration_world_px: n=2560, median=0.0, max=1.8470413904870213.
- contact_approach_reconstructed_closing_speed: n=1007, median=2.588037799629e-15, max=209.5481341038498.
- contact_approach_reconstructed_abs_tangential_speed: n=1007, median=19.70611154559409, max=372.8037799011166.
- contact_point_physical_COM_lever_length_world_px: n=1007, median=60.04511678734073, max=82.5617613937858.
- agent_object_contact_episode_seconds: n=56, median=0.12, max=0.55.

Contact episodes=56; observed contact-loss transitions=55; recontacts after an earlier observed episode=28.

- all_raw_controls: 256 controls / 16 initial families; area-centroid image displacement median=1.4083305535393151, max=7.3635929211645745.
- contact_with_nonzero_impulse: 116 controls / 14 initial families; area-centroid image displacement median=1.180059519364022, max=7.19898797925072.
- contact_impulse_and_force_toward_object_normal: 116 controls / 14 initial families; area-centroid image displacement median=1.180059519364022, max=7.19898797925072.
- no_agent_object_contact: 130 controls / 16 initial families; area-centroid image displacement median=1.5408855141950837, max=7.3635929211645745.
- free_object_coasting_no_agent_or_wall_contact: 115 controls / 13 initial families; area-centroid image displacement median=1.1683711017773788, max=5.446388004278877.
- object_wall_collision: 50 controls / 4 initial families; area-centroid image displacement median=1.8227439008104118, max=7.3635929211645745.
- agent_wall_collision: 3 controls / 1 initial families; area-centroid image displacement median=0.956006537051691, max=0.956006537051691.

## Limits

The no-contact velocity equation is tested only where neither pre- nor post-solve telemetry records a contact involving that body. The alternative-damping calculation uses the same recorded entry state and force; it is algebra, not an unexecuted trajectory. Approach and tangential speeds use a source-ordered reconstruction: entry velocities plus actual callback points and the same substep post pose. They are not direct callback-local measurements; tangential motion is not proof of a frictional sliding law. Actual contact shape friction and elasticity are zero in both arms; normal impulses still exchange momentum. A nonzero contact impulse together with force toward the object normal is observed pushing evidence, not a claim that every displacement was actively initiated by the pusher. P1 coasting and larger motion do not establish ideal sampling or select a final protocol. Wall impulses and peak speeds remain stability evidence. Raw-control coasting summaries are not H1/H4 window denominators. The main R2A report retains proper history windows and all failed/unexecuted slots.

Source mismatches: 0; telemetry anomalies: 0. Full counts, failures, pair checks and row references are in statistics.json.
