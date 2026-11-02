# Independent finite-damping decay audit

Checks passed: **True**; anomalies=0.

The object half-life analysis is restricted to recorded zero-force/zero-torque/zero-gravity substeps without any object-agent or object-wall event. The agent uses the full forced update, including measured PD force; agent motion is not called free decay.

## P0-frozen

Actual substeps=2560, force clips=143, speed clips=0; expected q(.01s)=0.0, q(.1s)=0.0.

- single_substeps: 1100 records/16 initial families; linear nonzero-entry n=4, q range=0.0..0.0; angular nonzero-entry n=4, q range=0.0..0.0; zero/near-zero entries counted separately in JSON.
- complete_controls: 75 records/16 initial families; linear nonzero-entry n=0, q range=None..None; angular nonzero-entry n=0, q range=None..None; zero/near-zero entries counted separately in JSON.
- free_runs: 59 records/16 initial families; linear nonzero-entry n=4, q range=0.0..0.0; angular nonzero-entry n=4, q range=0.0..0.0; zero/near-zero entries counted separately in JSON.

- no_agent_contact_full_forced_linear_residual: n=1160, median=0.0, max=0.0.
- no_object_contact_full_linear_residual: n=1100, median=0.0, max=0.0.
- no_object_contact_full_angular_residual: n=1100, median=0.0, max=0.0.
- agent_speed_post: n=2560, median=26.560000000000016, max=100.00000000000001.
- object_COM_speed_post: n=2560, median=3.9560816897638587, max=50.00000000000001.
- object_omega_abs_post: n=2560, median=0.008448242225914112, max=0.7392847744938413.
- max_penetration: n=2560, median=0.026529653064088166, max=0.7726018974535425.

AO episodes=57, loss transitions=33, recontacts after an observed episode=29.

## P1-damping1-candidate

Actual substeps=2560, force clips=42, speed clips=932; expected q(.01s)=1.0, q(.1s)=1.0.

- single_substeps: 1508 records/16 initial families; linear nonzero-entry n=1044, q range=1.0..1.0; angular nonzero-entry n=940, q range=1.0..1.0; zero/near-zero entries counted separately in JSON.
- complete_controls: 115 records/13 initial families; linear nonzero-entry n=83, q range=1.0..1.0; angular nonzero-entry n=74, q range=1.0..1.0; zero/near-zero entries counted separately in JSON.
- free_runs: 72 records/16 initial families; linear nonzero-entry n=40, q range=1.0..1.0; angular nonzero-entry n=38, q range=1.0..1.0; zero/near-zero entries counted separately in JSON.

- no_agent_contact_full_forced_linear_residual: n=1682, median=0.0, max=0.0.
- no_object_contact_full_linear_residual: n=1508, median=0.0, max=0.0.
- no_object_contact_full_angular_residual: n=1508, median=0.0, max=0.0.
- agent_speed_post: n=2560, median=143.91853364242274, max=247.30188030695243.
- object_COM_speed_post: n=2560, median=78.98372531574736, max=392.8795701982306.
- object_omega_abs_post: n=2560, median=0.2057118822591586, max=2.3955247901857115.
- max_penetration: n=2560, median=0.0, max=1.8470413904870213.

AO episodes=56, loss transitions=55, recontacts after an observed episode=28.

## D-half-life-0.1s

Actual substeps=2560, force clips=23, speed clips=599; expected q(.01s)=0.9330329915368074, q(.1s)=0.5.

- single_substeps: 1180 records/16 initial families; linear nonzero-entry n=714, q range=0.9330329915368072..0.9330329915368076; angular nonzero-entry n=714, q range=0.9330329915368073..0.9330329915368075; zero/near-zero entries counted separately in JSON.
- complete_controls: 75 records/14 initial families; linear nonzero-entry n=43, q range=0.4999999999999999..0.5; angular nonzero-entry n=43, q range=0.4999999999999999..0.5000000000000001; zero/near-zero entries counted separately in JSON.
- free_runs: 106 records/16 initial families; linear nonzero-entry n=74, q range=0.01674646035212958..0.9330329915368076; angular nonzero-entry n=74, q range=0.01674646035212958..0.9330329915368075; zero/near-zero entries counted separately in JSON.

- no_agent_contact_full_forced_linear_residual: n=1336, median=0.0, max=0.0.
- no_object_contact_full_linear_residual: n=1180, median=0.0, max=0.0.
- no_object_contact_full_angular_residual: n=1180, median=0.0, max=0.0.
- agent_speed_post: n=2560, median=111.50770038108078, max=245.86481950038623.
- object_COM_speed_post: n=2560, median=38.29262451808451, max=244.8134261975191.
- object_omega_abs_post: n=2560, median=0.10469107282010871, max=2.253059596126229.
- max_penetration: n=2560, median=0.018145359678138462, max=1.177760422461894.

AO episodes=94, loss transitions=92, recontacts after an observed episode=66.

## D-half-life-0.3s

Actual substeps=2560, force clips=36, speed clips=812; expected q(.01s)=0.9771599684342459, q(.1s)=0.7937005259840997.

- single_substeps: 1349 records/16 initial families; linear nonzero-entry n=883, q range=0.9771599684342456..0.9771599684342462; angular nonzero-entry n=792, q range=0.9771599684342458..0.977159968434246; zero/near-zero entries counted separately in JSON.
- complete_controls: 82 records/14 initial families; linear nonzero-entry n=50, q range=0.793700525984099..0.7937005259840996; angular nonzero-entry n=44, q range=0.793700525984099..0.7937005259840996; zero/near-zero entries counted separately in JSON.
- free_runs: 119 records/16 initial families; linear nonzero-entry n=87, q range=0.255843472999193..0.977159968434246; angular nonzero-entry n=82, q range=0.255843472999193..0.977159968434246; zero/near-zero entries counted separately in JSON.

- no_agent_contact_full_forced_linear_residual: n=1513, median=0.0, max=0.0.
- no_object_contact_full_linear_residual: n=1349, median=0.0, max=0.0.
- no_object_contact_full_angular_residual: n=1349, median=0.0, max=0.0.
- agent_speed_post: n=2560, median=129.70965875652053, max=243.77998282309858.
- object_COM_speed_post: n=2560, median=54.169772738432656, max=310.40853397684737.
- object_omega_abs_post: n=2560, median=0.17399419256803647, max=2.2245414948683777.
- max_penetration: n=2560, median=0.0, max=1.7870119839703165.

AO episodes=101, loss transitions=100, recontacts after an observed episode=73.

## Interpretation limits

Zero-entry values do not define a retention ratio. A collision ends an episode; no decay estimate spans it. D0 destroys prior velocity at an update; D1 has no finite damping half-life. Finite candidates are tested against their specified native D**elapsed rule, not tuned to these observations. Velocity retention is not a displacement-retention claim: native position integration also includes bias velocity following contacts. Observed coasting, contact retention, wall impulses and velocity peaks must be assessed together; this audit does not select ideal physics or establish learnability or shape preservation. Original R2A files were only read; all outputs are in the new R2A2 telemetry-audit directory.
