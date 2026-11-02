# Coverage diagnosis after the frozen stage2B batch

234/234 new initial-selector prefix audits match: actual control-bin counts inherit2A, planned independent attempts inherit2A, and each new legal trajectory increments exactly once. Final396 attempts are22/letter. Reconstructed saved coverage differences=0; closed-hole/reachability conflicts=0.

| Population | Actual H/all | H/K reachable lower bound | Canonical hit/R | Unknown | Closed | Contact traj. | Contact controls |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2A | 827/1408 | 827/987 | 765/925 | 303 | 118 | 162/162 | 7187/10368 |
| 2B | 892/1408 | 892/1002 | 815/925 | 288 | 118 | 234/234 | 10489/14976 |
| merged | 1023/1408 | 1023/1025 | 923/925 | 265 | 118 | 396/396 | 17676/25344 |

Merged actual hits add196 distinct arcs beyond2A. The remaining known lower-bound misses are O55 and P55. Unknown265 arcs are not proven unreachable; closed-hole118 arcs are separately proven inaccessible from the exterior. K1025 is a lower bound across canonical/initial/observed poses, not all physically reachable arcs.

O55 was initially planned4 times in2A, P55 six times; neither was actually hit.2B makes0 new initial attempts at either and0 natural hits at either. O actual coverage grows34→44 with10 other new arcs; P35→48 with13 other new arcs. This avoids repeated initial targeting but does not solve or prove inaccessible either55. Planned initial target hit234/234 in2B is reported separately from actual coverage.

All396 trajectories have actual contact. The actual cube has23887 control/arc incidences representing17676 distinct contact controls:14851 moving,1649 intermediate,1176 near-static. These are raw actual-contact controls, not H1 candidate gate counts. All eight true pre-solve contact-angle bins have point support, concentrated in bins1/2/5/6; counts include repeated100Hz points and do not establish balanced independent approaches.

Four signed wall gaps and continuous area-centroid dx/dy and rotation are retained in contact-controls.csv; initial gaps are in initial-scene-gaps.csv. All-control gap counts preserve6694 negative signed minima without rounding. The fixed2px, ten-substep physics audit remains a separate root result; no actual wall-arbiter points were stored, so proximity is not called a wall-collision count.

2A has9 independent trajectories/letter;2B adds13; merged22. Different scenes, gap quotas and budgets prevent attributing the entire coverage increase to the ranking update.2A source/data and frozen2B source were not changed. No physics controls, environment or renderer construction, model calls, or training-index draws occurred.

Postprocessor sourceSHA256: `8dc02a2d960c46e7a291362f6977a3d7d894d5bf81767c47a9c91fbec7595dc8`.
Full unhit/unknown/closed lists, angles, overlapping feature counts, independent trajectories and prefix evidence are preserved in the sibling JSON/CSV/SVG files.
