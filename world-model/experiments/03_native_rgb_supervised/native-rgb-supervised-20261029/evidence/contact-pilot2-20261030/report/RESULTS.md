# CONTACT-PILOT2 cold report

Fixed60cases; attempted60,unattempted0. Actual charged2880,complete2880,knownsubsteps28800,unknown0,partial0,reservation0.

Original48are repeat diagnostics,not new families. Only the12fresh planned cases can supply new independent support when actually executed. Allunexecuted entries below are N/A,not static observations. A halted prefix does not establish performance of the fullfixedqueue.

Contact means actual known100Hz agent-object post_solve evidence. Nearstatic means area-centroid translation<0.1image96px AND absolute unwrapped rotation<0.1degree per0.1s control. Obvious motion means translation>=2image96px OR absolute rotation>=5degrees; small motion is the remainder. These unchanged CONTACT_AUDIT thresholds are not reset to make this pilot pass.

| Population | Attempted/planned | Charged/full | Contact+obv charged | Contact+obv48slot lower | Emptycharged upper | Empty48slot upper | Strictrecovery observed/tasks | Censored/never-separated/notreached | Observedmedian/P90s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R3A_original_A | 24/24 | 1152/1152 | 23.96% | 23.96% | 40.71% | 40.71% | 16/24 | 6/2/0 | 0.77/2.115 |
| CP1_original_A | 24/24 | 1110/1109 | 32.88% | 31.68% | 22.61% | 25.43% | 17/24 | 4/2/1 | 0.37/0.41000000000000003 |
| CP2_repeat_A | 24/24 | 1152/1152 | 15.10% | 15.10% | 65.54% | 65.54% | 17/24 | 7/0/0 | 0.68/0.78 |
| CP1_original_B | 24/24 | 1152/1152 | 38.89% | 38.89% | 28.82% | 28.82% | 24/24 | 0/0/0 | 0.365/0.9249999999999996 |
| CP2_repeat_B | 24/24 | 1152/1152 | 8.33% | 8.33% | 69.88% | 69.88% | 20/24 | 4/0/0 | 0.3/0.68 |
| CP2_new_families | 12/12 | 576/576 | 6.94% | 6.94% | 73.61% | 73.61% | 8/12 | 4/0/0 | 0.68/0.7170000000000001 |
| CP2_all_planned | 60/60 | 2880/2880 | 10.76% | 10.76% | 68.89% | 68.89% | 45/60 | 15/0/0 | 0.58/0.78 |
| CP2_attempted_prefix | 60/60 | 2880/2880 | 10.76% | 10.76% | 68.89% | 68.89% | 45/60 | 15/0/0 | 0.58/0.78 |

Strictprotocol recovery requires a complete10known-substep noAO control during fixedwithdraw,then realpostsolve contact at/aftertask26. Never-separated does not become0delay success. Rightcensored,never-separated andnotreached remain separate. Anyprotocol observation without priorAO beforewithdraw is separatelyflagged as firstacquisition,not a proven separation/reacquisition; observed_with_prior_AO and itsdelayquantiles give the conservative sensitivity without changing frozenprimarydefinitions. Unrecovered engineeringgoal uses allplannedtasks conservatively. Plannedslots are hypothetical fixedschedule sensitivity,not additional physicaltime.

Engineeringgoals remain>=40%contact+obvious,<=25%empty,observedrecoverymedian<=.5s/P90<=1s,unrecovered<=5%,and zero newphysicalguardfailures. Allcriteria must be assessed on the completed fixedqueue; thresholdmet on a shortprefix is not a pass.

| Population | Letter | Attempted/planned | Contactobv charged | Latecontactobv charged | Wallstall | Push/noAO | Longesthold | Unconfirmedroutecommands |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CP2_repeat_A | T | 4/4 | 13.54% | 5.21% | 0 | 93/138 | 0 | 0 |
| CP2_repeat_A | L | 4/4 | 15.62% | 12.50% | 0 | 92/138 | 0 | 0 |
| CP2_repeat_A | O | 4/4 | 17.71% | 13.54% | 0 | 87/138 | 0 | 0 |
| CP2_repeat_A | C | 4/4 | 26.56% | 13.54% | 0 | 67/138 | 7 | 0 |
| CP2_repeat_A | X | 4/4 | 11.46% | 22.92% | 0 | 92/138 | 7 | 0 |
| CP2_repeat_A | B | 4/4 | 5.73% | 0.00% | 0 | 106/138 | 0 | 0 |
| CP2_repeat_B | T | 4/4 | 15.62% | 3.12% | 0 | 87/138 | 0 | 0 |
| CP2_repeat_B | L | 4/4 | 15.10% | 13.54% | 0 | 92/138 | 0 | 0 |
| CP2_repeat_B | O | 4/4 | 17.71% | 23.96% | 0 | 84/138 | 0 | 0 |
| CP2_repeat_B | C | 4/4 | 0.00% | 0.00% | 0 | 99/138 | 0 | 0 |
| CP2_repeat_B | X | 4/4 | 1.56% | 1.04% | 0 | 102/138 | 0 | 0 |
| CP2_repeat_B | B | 4/4 | 0.00% | 0.00% | 0 | 102/138 | 0 | 0 |
| CP2_new_families | T | 2/2 | 0.00% | 0.00% | 0 | 63/69 | 0 | 0 |
| CP2_new_families | L | 2/2 | 7.29% | 0.00% | 0 | 57/69 | 0 | 0 |
| CP2_new_families | O | 2/2 | 2.08% | 0.00% | 0 | 57/69 | 0 | 0 |
| CP2_new_families | C | 2/2 | 27.08% | 25.00% | 0 | 28/69 | 0 | 0 |
| CP2_new_families | X | 2/2 | 0.00% | 0.00% | 0 | 60/74 | 0 | 0 |
| CP2_new_families | B | 2/2 | 5.21% | 10.42% | 0 | 44/69 | 0 | 0 |

Fixedfault/regression cases retain smallmotion andhold counts besideobviousmotion,so aggregatedimprovement cannot hide B04 or T040 regressions.

| Case | Version | Status | Contactobvious | Contactsmall | No-contactsmall | Nearstatic | Holdcomplete |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CP2-R-A-X-03 | R3A | completed | 3 | 2 | 0 | 43 | 0 |
| CP2-R-A-X-03 | CP1 | failed_physics | 2 | 2 | 0 | 1 | 0 |
| CP2-R-A-X-03 | CP2 | completed | 0 | 4 | 4 | 40 | 0 |
| CP2-R-A-T-03 | R3A | completed | 1 | 5 | 0 | 42 | 0 |
| CP2-R-A-T-03 | CP1 | completed | 2 | 4 | 0 | 42 | 38 |
| CP2-R-A-T-03 | CP2 | completed | 0 | 9 | 16 | 23 | 0 |
| CP2-R-A-B-04 | R3A | completed | 33 | 5 | 5 | 5 | 0 |
| CP2-R-A-B-04 | CP1 | completed | 14 | 16 | 16 | 2 | 8 |
| CP2-R-A-B-04 | CP2 | completed | 0 | 15 | 19 | 14 | 0 |
| CP2-R-A-T-01 | R3A | completed | 6 | 0 | 5 | 37 | 3 |
| CP2-R-A-T-01 | CP1 | completed | 23 | 12 | 5 | 8 | 4 |
| CP2-R-A-T-01 | CP2 | completed | 11 | 6 | 16 | 15 | 0 |

Requestedarc evidence uses exact actualpostsolve arc membership; validimpulse requires a pointrow on thatsamearc,norm>1e-8. Adjacent/incidental contact is no substitute; zeroobserved hits are not proof of unreachability. Completecontaining-control motion is a co-occurrence,not isolated causal motion.

| Case | Version | Status | Requestedarc | Contactsubsteps | Positiveimpulsesubsteps | Longestpositive run s | Motion+positivecontrols |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CP2-R-B-T-04 | CP1 | completed | 1 | 0 | 0 | 0.0 | 0 |
| CP2-R-B-T-04 | CP2 | completed | 1 | 27 | 27 | 0.14 | 1 |
| CP2-R-B-L-04 | CP1 | completed | 28 | 0 | 0 | 0.0 | 0 |
| CP2-R-B-L-04 | CP2 | completed | 28 | 37 | 36 | 0.09 | 0 |
| CP2-R-B-O-04 | CP1 | completed | 42 | 0 | 0 | 0.0 | 0 |
| CP2-R-B-O-04 | CP2 | completed | 42 | 41 | 40 | 0.1 | 0 |
| CP2-R-B-C-04 | CP1 | completed | 73 | 0 | 0 | 0.0 | 0 |
| CP2-R-B-C-04 | CP2 | completed | 73 | 34 | 32 | 0.08 | 0 |
| CP2-R-B-X-04 | CP1 | completed | 19 | 0 | 0 | 0.0 | 0 |
| CP2-R-B-X-04 | CP2 | completed | 19 | 37 | 36 | 0.09 | 0 |
| CP2-R-B-B-04 | CP1 | completed | 44 | 0 | 0 | 0.0 | 0 |
| CP2-R-B-B-04 | CP2 | completed | 44 | 38 | 37 | 0.1 | 0 |
| CP2-N-T-02 | CP2 | completed | 1 | 17 | 14 | 0.13 | 0 |
| CP2-N-L-02 | CP2 | completed | 28 | 19 | 15 | 0.09 | 0 |
| CP2-N-O-02 | CP2 | completed | 42 | 31 | 27 | 0.1 | 0 |
| CP2-N-C-02 | CP2 | completed | 73 | 33 | 32 | 0.08 | 0 |
| CP2-N-X-02 | CP2 | completed | 19 | 24 | 22 | 0.09 | 0 |
| CP2-N-B-02 | CP2 | completed | 56 | 45 | 45 | 0.12 | 0 |

Fullcase/version metrics:cases.csv; percase andletter deltas:paired-deltas.csv/paired-letter-deltas.csv; sixclasses and denominator variants:six-classes.csv; strictrecovery:recontact.csv; weakgoals:requested-arcs.csv; completeoriginalrisk evidence remains in hashed*.telemetry.json.gz referenced by lossless controller-telemetry-index.csv.gz/risk-index.csv.gz. Reporting uses0newphysics/modelcalls; originaltrainingpool and checkpointhashes are unchanged by this script.
