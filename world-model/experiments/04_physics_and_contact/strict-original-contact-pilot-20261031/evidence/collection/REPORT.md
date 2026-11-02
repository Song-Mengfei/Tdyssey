# Strict original contact sampling report

Original100Hz / solver10 / KINEMATIC pusher / damping0 / upstream PD100,20. These are saved physical collection results; no model prediction improvement was measured. Training is paused and all data remain pending human approval.

All attempted scenes: 36; observed controls: 2304; legal controls: 2304; actual substeps: 23040; simulated duration: 230.4000s. Simulation duration is not CPU/GPU elapsed.

|Phase|Letter|Attempts|Complete|Valid/observed controls|H1 translation P50 imagepx|H4 P50 imagepx|Positive AO moving time|Wall-only time|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|calibration-v1|T|3|3|192/192|0.000|0.000|0.0%|0.0%|
|calibration-v1|L|3|3|192/192|0.000|0.000|0.0%|0.0%|
|calibration-v1|O|3|3|192/192|0.000|0.000|0.0%|0.0%|
|calibration-v1|C|3|3|192/192|0.000|0.000|13.6%|0.0%|
|calibration-v1|X|3|3|192/192|0.000|0.000|0.0%|0.0%|
|calibration-v1|B|3|3|192/192|0.000|0.000|0.0%|0.0%|
|calibration-v2|T|3|3|192/192|0.324|1.296|64.8%|0.0%|
|calibration-v2|L|3|3|192/192|0.000|0.079|38.6%|0.0%|
|calibration-v2|O|3|3|192/192|0.299|1.346|51.9%|0.0%|
|calibration-v2|C|3|3|192/192|0.451|1.801|64.4%|0.0%|
|calibration-v2|X|3|3|192/192|0.667|2.157|57.8%|0.0%|
|calibration-v2|B|3|3|192/192|0.438|1.631|74.7%|0.0%|

P50 cells preserve each split separately when more than one split exists; full quantiles/bins and denominators are in [summary](report/summary.json).

Positive video roles require actual legal H1 translation >=0.5 imagepx or rotation >=5deg. If no trajectory qualifies, the role is unavailable; random, weak-motion and boundary roles do not imply successful contact or movement.

Original penetration remains a possible environment limitation. An illegal observed substep ends that trajectory; no pose/velocity correction, hidden interpolation, reset, retry or replacement is performed. Legal H1/H4 metrics use exactly0.1/0.4s adjacent saved states and never skip frames.

Actual positive-impulse contact arcs are listed separately from planned contacts. Unassigned arcs remain ring=-1/arc=-1. Arc times are nonexclusive; simultaneous points do not inflate contact seconds.

The formal supplement trigger is diagnostic only and uses every valid formal H1, including early controls without four-frame history. Calibration and supplement do not enter its denominator. No automatic release is granted.

HTML report (external historical reference: `report/index.html`) · all attempts (external historical reference: `report/attempts.csv`) · controls (external historical reference: `report/controls.csv`) · H1/H4 windows (external historical reference: `report/windows.csv`) · coverage (external historical reference: `report/contact-coverage.csv`) · [candidate membership](candidate-manifest.json)
