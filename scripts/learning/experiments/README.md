# Run scripts

The shell scripts that ran the policy-learning experiments in `docs/LEARNING_FINDINGS.md` (sections 6-8), kept as a
record of the exact commands, seeds and settings behind each result. Decision numbers (Dxxx) refer to
`docs/DECISIONS.md`.

Each script sources `common.sh`, which makes the repository root the working directory and puts data, checkpoints and
logs in `$TDYSSEY_WORK` (default `/tmp/Tdyssey`, outside Git). They expect the official Diffusion Policy setup
(`docs/OFFICIAL_BASELINE.md`), the official human T demonstrations in `$WORK/data/pusht-official/`, and the outputs of
the earlier stages under `$WORK/runs`. They were written for one 80-93 GB GPU and launch several jobs at once.
Apart from the path changes they are the scripts that ran; after those changes they were checked with `bash -n` only.

## Ten shapes (T, near-T, Y, D, L, E, F, J, P, U)

| Step | Script | Decision |
| --- | --- | --- |
| Five more letters: dynamics data, world model v2, teacher check, baseline | `chain_newletters.sh` | D047 |
| Teacher demonstrations for E, F, J, P, U | `chain_newletter_demos.sh` | D047 |
| Teacher demonstrations that record the intended 16-step plan | `chain_plan_demos2.sh`, `chain_plan_demos3.sh` | D048 |
| Correction rounds with intended-plan labels (from round 16) | `chain_planlabel.sh` -> `chain_dagger_loop10v3.sh` | D048 |
| Correction rounds, 96 px main line and 192 px side line | `chain_dagger_loop10v6.sh` | D048 |
| Final test, 1000 fresh scenes per shape | `chain_fairtest_d049_base.sh`, `chain_fairtest_d049_cand.sh` | D049 |
| 192 px student on the same scenes | `chain_fairtest_d050.sh` | D050 |
| Ten-shape students on the 17 untrained capitals | `chain_unseen17.sh` | D052 |

## All 26 letters and near-T

| Step | Script | Decision |
| --- | --- | --- |
| Dynamics data, world model v3, teacher check, demonstrations for 17 letters | `chain_26prep.sh` | D051 |
| Correction rounds (both lines) | `chain_dagger_loop26c.sh` | D051, D053 |
| 96 px rounds 45-50 | `chain_26m_45to50.sh` | D054 |
| 192 px continuation | `chain_192_continue.sh` | D055 |
| Round 50 against round 44 on fresh scenes | `chain_d054.sh` | D054 |
| Executing 8, 12 or 15 of the 16 planned steps | `chain_d056.sh` | D056 |

`chain_dagger_loop26c.sh` covers every 26-letter round. Round 0 (retraining on all 27 shapes) is
`chain_dagger_loop26c.sh m 96 "<demo dirs>" <ten-shape student> 0 0 58000`. Rounds 1-13 ran an earlier copy that is
identical to this script with `STEPS=20000 NEWSCENES=30 SPACING=50` and `IBASE` 58000 (96 px) or 52100 (192 px);
later rounds use its defaults (D053) with the seed ranges listed in D053-D055.

The final test (D051) ran plain evaluation calls on a node whose launcher script was not kept; the settings are also
stored in every result file (`policy`, `seeds`, `policy_noise_seed`):

```bash
SH="T T_narrow A B C D E F G H I J K L M N O P Q R S U V W X Y Z digit3 digit4 digit7 plus hash arrow"
for k in $(seq 0 11); do      # 12 parts of 25 = 300 fresh scenes per shape; omit --student for the original policy
  .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 10000000 --start $((25*k)) --scenes 25 \
      --execute 8 --shapes $SH --student <checkpoint> --out $WORK/runs/fairtest-d051/<name>-part$k.json
done
.venv/bin/python -I scripts/learning/report_pair.py $WORK/runs/fairtest-d051 base ten96 m26 h26
```

The candidates were picked by `scripts/learning/select_candidates.py` (the D051 rule).
