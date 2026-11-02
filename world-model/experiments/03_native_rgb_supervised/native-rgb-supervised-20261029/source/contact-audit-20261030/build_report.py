"""Build human-readable findings from closed saved-array audit artifacts."""
import csv
import json
from pathlib import Path

STAGE=Path(__file__).resolve().parent
def read(name):return json.loads((STAGE/name).read_text(encoding='utf-8'))
def pct(value):return f'{value*100:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |'for row in rows])
def save(name,text):(STAGE/name).write_text(text.strip()+'\n',encoding='utf-8')


def main():
    s=read('statistics/main_summary.json');d=s['denominators'];extra=read('supplemental-summary.json')
    a=d['A_TRAIN_all'];chosen=d['C_TRAIN_selected'];r=s['recontact']['by_split']
    six=lambda q,k:q['six_mutually_exclusive_classes'][k]
    near=lambda q:sum(six(q,k)['count']for k in('no_contact_nearstatic','contact_nearstatic'))/q['denominator']
    mainrows=[]
    names=(('A_TRAIN_all','全部成功 TRAIN'),('B_TRAIN_legal_H1','合法 TRAIN H1'),
           ('C_TRAIN_selected','选入的 TRAIN 目标'),('VAL_all','全部成功 seen-val'),('VAL_legal_H1','合法 seen-val H1'))
    for key,label in names:
        q=d[key];mainrows.append([label,f"{q['denominator']:,}",pct(q['any_actual_contact_fraction']),
            pct(q['contact_substep_duty']),pct(six(q,'contact_obvious')['fraction']),pct(near(q))])
    main_table=table(['独立统计分母','0.1s 区间数','区间内曾接触','100Hz 接触占比','接触且明显运动','基本静止'],mainrows)
    report=f'''# 采集与真实接触审计（2026-10-30）

**采集已完成；数据中有大批真实推动，但空跑、慢恢复和顶墙停滞仍占明显比例。**
912 条成功轨迹中，TRAIN 768 条、seen-val 144 条，每条 48 个控制步。
本轮只检查已存状态；新增物理步、模型调用、训练更新均为 0。速度半衰期保持 **0.1s**。

{main_table}

TRAIN 全部采集时间为 **3,686.4s**，其中实际 post_solve 接触 **1,826.10s**。
表中“曾接触”只需该区间 10 个物理子步中至少 1 个有接触，不能当成整个 0.1s 都在推。
“基本静止”固定为形心位移 <0.1 个 96px 图像像素且转角绝对值 <0.1°；
“明显运动”固定为位移 ≥2px 或转角绝对值 ≥5°。接触加运动是观测共现，不是严格因果分解。

* **空跑成本：**全部 TRAIN 有 15,101/36,864 区间（40.96%，1,510.1s）推杆移动却没有实际接触；
  其中 12,156 区间（32.98%，1,215.6s）物体还基本静止。按阶段/模式识别的寻路或恢复候选为
  10,045 区间（27.25%，1,004.5s）；这是前述情况的重叠子集，不能相加。撤离和静态负例另列。
* **计划不等于接触：**27,252 个计划 push 区间中，8,842（32.45%）实际无接触。
  3,214 个“接触但基本静止”区间中，3,143 同时出现过物体碰墙。
  TRAIN 前半段接触且明显运动为 43.41%，后半段降至 21.60%。
* **恢复慢：**TRAIN 756 条有计划 recontact 的轨迹中，668 条恢复；已恢复者延迟中位数
  0.73s、P90 1.31s、最大 2.12s。88/756（11.64%）到轨迹结束仍未恢复，按 2.2s 右删失，
  没有混进延迟均值。seen-val 对应未恢复 21/138（15.22%），中位数 0.71s、P90 1.37s。
* **实际训练池的支持：**11,983 个“接触且明显运动”目标来自 735 条独立 TRAIN 轨迹，
  最多的 10 条仅占这些目标的 2.92%。其中 11,799 个还具有有效非零冲量证据，
  11,787 个同时有非零冲量且至少 5/10 子步接触。50% motion 池中只有 9 个明显运动目标完全无接触。
  选入池仍有 31.12% 基本静止，不能用“50% motion”描述全部采集时间。

![四个独立分母的真实接触与运动](presentation/contact-motion-independent-denominators-QA.png)

**处理建议：**保留这批原始数据及负例，不把整库描述成连续推动数据；优先修复恢复路线和顶墙停滞，
针对 X 的边界/反向旋转及实际未覆盖接触弧提出小规模补采。是否恢复 R3B 由规划对话审核本报告后决定。
本轮未自动补采或训练。

[详细表与限制](ANNEX.md) · [补采建议](SAMPLING_PROPOSAL.md) ·
[时间线、原视频和测量样例](presentation/index.html) · [CSV 六类分解](statistics/mutually-exclusive.csv) ·
[完整分层 CSV](statistics/strata.csv) · [机器可读汇总](statistics/main_summary.json)
'''
    save('REPORT.md',report)

    cross=table(['Actual outcome','All TRAIN count / %','Selected TRAIN count / %'],[
        [key,f"{six(a,key)['count']:,} / {pct(six(a,key)['fraction'])}",
             f"{six(chosen,key)['count']:,} / {pct(six(chosen,key)['fraction'])}"]
        for key in a['six_mutually_exclusive_classes']])
    strata=list(csv.DictReader((STAGE/'statistics/strata.csv').open(encoding='utf-8')))
    def strata_table(field):
        return table([field,'Controls','Any contact','Contact+obvious','Nearstatic','Planned push / no contact'],[
            [row['value'],row['count'],pct(float(row['any_contact_fraction'])),
             pct(float(row['contact_obvious_fraction'])),
             pct(float(row['no_contact_nearstatic_fraction'])+float(row['contact_nearstatic_fraction'])),
             row['planned_push_controls']+' / '+row['planned_push_no_contact']]
            for row in strata if row['denominator_name']=='A_TRAIN_all'and row['stratum']==field])
    bins=table(['Contacting physics intervals / 10','All TRAIN count','Selected target count','Seen-val count'],[
        [key,a['contact_substep_bins'][key]['count'],chosen['contact_substep_bins'][key]['count'],
         d['VAL_all']['contact_substep_bins'][key]['count']]for key in a['contact_substep_bins']])
    eps=table(['Impulse norm epsilon','TRAIN presence substeps / all substeps','TRAIN duty','Selected duty','Seen-val duty'],[
        [key,f"{value['substep_presence_count']} / {value['substep_denominator']}",pct(value['duty']),
         pct(chosen['nonzero_impulse_presence_sensitivity'][key]['duty']),pct(d['VAL_all']['nonzero_impulse_presence_sensitivity'][key]['duty'])]
        for key,value in a['nonzero_impulse_presence_sensitivity'].items()])
    durations=table(['Split','Contact runs','Run duration median / P90 / max (s)','Positive-impulse seconds'],[
        [split,q['postsolve_contact_run_seconds']['count'],
         ' / '.join(f"{q['postsolve_contact_run_seconds'][v]:.3f}"for v in('median','p90','max')),
         f"{q['positive_impulse_contact_seconds']:.2f}"]for split,q in extra['by_split'].items()])
    recovery=table(['Split','Recontact trajectories','Observed','No recovery / share','Observed median / P90 / max (s)'],[
        [split,q['planned_recontact_trajectories'],q['observed_recovery_trajectories'],
         f"{q['unrecovered_right_censored_trajectories']} / {pct(q['unrecovered_fraction'])}",
         ' / '.join(f"{q['observed_delay_seconds'][v]:.2f}"for v in('median','p90','maximum'))]for split,q in r.items()])
    failure=read('core/failed-attempts.json')
    failures=table(['Attempt','Charged / complete','Known substeps','Partial substeps','AO contact seconds'],[
        [q['trajectory_id'],f"{q['charged_controls']} / {q['completed_controls']}",q['known_substeps'],q['partial_substeps'],
         f"{q['contact_seconds']:.2f}"]for q in failure['attempts']])
    zeros=table(['Population','Exactly zero translation and rotation','Nearstatic fraction'],[
        [label,f"{d[key]['exact_zero_motion']['both_exact_zero']['count']} / {pct(d[key]['exact_zero_motion']['both_exact_zero']['fraction'])}",pct(near(d[key]))]
        for key,label in names])
    save('ANNEX.md',f'''# Saved-array contact audit: technical tables and limits

The main findings are in REPORT.md. Definitions were frozen before extraction in
definitions.json. All source trajectory and metadata hashes were checked. No new
physics, checkpoint loading, Torch execution or model calls were needed.

## Denominators and six exclusive outcomes

{main_table}

{cross}

Each six-class column sums exactly to its denominator and 100%. The 2,304 early
TRAIN controls excluded from H1 are retained in all-acquisition statistics.
Selected targets are a deterministic, biased subset, not an unbiased estimate of
all time. Distinct successful controls total 43,776; no failed prefix enters this
successful-data table. Legal H1 targets total 41,040 across the separate splits.

## Contact duration, endpoint and impulse evidence

{bins}

TRAIN terminal contact is {pct(a['terminal_contact_fraction'])}; selected terminal
contact is {pct(chosen['terminal_contact_fraction'])}. Any contact and terminal
contact differ. A one-substep brush contributes 0.01s, not 0.1s of contact.

{durations}

{eps}

All flagged AO contact substeps have a valid saved impulse row (unknown count 0).
TRAIN has 3,638 known exactly-zero impulse substeps. The recorder repeats event
total_impulse for each point and lacks an event/arbiter ID; neither point impulses
nor arc impulses are summed into force, work or total physical impulse.
core/impulse-provenance.json records source lines and raw examples.

In the 11,983 selected contact+obvious targets: 11,799 have at least one valid
norm>1e-8 substep; 11,789 have >=5/10 contact substeps; 10,962 have 10/10; 11,787
satisfy both nonzero-impulse presence and >=5 contact. Five need not be consecutive.
Observed support is 735 trajectories for any-contact+obvious and 734 for the
impulse-supported form. This does not establish strict pusher causality.

Nearstatic TRAIN controls contain 307.29s of AO contact, 301.11s concurrent with
object-wall flags and 6.18s without a simultaneous object-wall flag. At control
resolution, contact+nearstatic is 3,214 intervals: 3,143 with some object-wall
contact and 71 without. These are observations, not proof that the wall caused
every stalled interval. Agent-wall data remain separate in strata.csv.

## Time, intent and motion

{strata_table('phase')}

{strata_table('mode')}

{strata_table('half')}

Planned push has 6,632 free_route intervals (6,593 no-contact), 2,240 hold
intervals (1,311 no-contact), and 18,380 contact_push_intent intervals (938
no-contact). No-contact is an actual outcome, not a name for intended withdrawal.
operation-time.csv separates moving pusher with no contact, route/recovery
candidates, withdrawal and static negatives. These operational sets overlap;
their durations must not be added together. Pusher moving means endpoint motion
>=0.1 image96px, not accumulated path length. Each classified interval lasts0.1s;
physical contact seconds instead use the measured100Hz flags.

{zeros}

TRAIN translation median/P90/max is0.1444/2.2799/3.1003 image96px; signed rotation
min/median/max is -13.4233/0/15.7218degrees; absolute rotation median/P90 is
0.1194/5.8078degrees. Full continuous and intermediate-threshold sensitivity
summaries are in main_summary.json and strata.csv. Exact zero is distinct from
nearstatic. Motion uses area centroid, not body origin or symmetry-folded yaw.

## Recovery and spatial support

{recovery}

All unrecovered cases have2.2s right-censor windows; they are not zero-delay
recoveries. Static trajectories with no planned recontact are N/A. Per-letter,
scenario, initial region and space delays are in recontact.csv/main_summary.json.

{strata_table('letter')}

{strata_table('regiontype')}

Letter-by-region controls, both rotation signs and independent-trajectory support
are in letter-region-support.csv. X TRAIN corner contact+obvious support is726
controls from54/56 trajectories; positive rotation has261 controls/29 trajectories,
negative79/19. X TRAIN center/edge/corner contact+obvious fractions are34.03%,
33.00%,27.01%. X seen-val corner support is163controls/8trajectories and must
not be added to TRAIN. Initial region describes the reset, not position throughout.

Actual dominant contact arc is assigned once per control by distinct-substep
votes, with lowest-ID ties and an explicit unmapped class. Concentration is in
main_summary.json; multi-arc marginals are never summed as independent time.
Previously measured outer-arc gaps and untested O/B closed inner rings remain;
zero coverage is not proof of physical unreachability.

## Selected-pool exposures and batch distribution

All23,984 targets appear once in the frozen1,499x16 first-epoch plan. Original
window-ID order SHA256 is800655f0dc54234f92c9784dc4b0ff94e44910d488d39f32439ff625a2ee2335.
No short/repeated/dropped target or repeated family within a batch. Actual model
draws/updates remain0. Target contact P10/median/P90 is8/11/13 of16, worst3/16;
contact+obvious is5/8/11 of16, worst1/16. Neither statistic has a zero batch.

Four-action prefixes create95,936 repeated input control exposures over34,836
distinct TRAIN controls;2,028 raw controls have zero prefix exposure. Weights
min/median/max including zeros are0/3/4. Weighted any-contact is63.50%, weighted
contact+obvious46.00%; these are exposures, not additional independent physical
time. input-exposure.npz preserves every weight; batches.csv preserves all batches.
Selected contact+obvious top1/top10 trajectory concentration is0.309%/2.921%.

## Failures, verification and inspection

{failures}

The six failures contribute115charged,110complete controls,1,129known substeps,
29partial substeps,zero unknown advances and one unstarted reservation. Combined
successful+failed costs are43,891charged/43,886complete/438,889known substeps.
Failure rows retain actual contact, complete-control motion and partial endpoint
motion; partial durations are not promoted into full controls. Original failure
videos remain in ../r3a/report/failure-videos/.

Six core CPU fixtures and six summary fixtures passed. Independent root checks
passed106/106:8original NPZs with direct rectangle-weighted centers and flag/impulse
checks plus all41,040H1 motions compared to the pre-existing catalogue. The
presentation retains all70original MP4/PNG SHA values, aligns35old examples to
core, and adds13deduplicated choices (six fixed-seed5478 random letter cases plus
measured extremes). Fifty newly rendered saved frames match collection RGB hashes.
Root visually reviewed the cohort and recorded-example overview. No annotations
enter model inputs or ground-truth RGB. B014 shows event391 at interval[3.91,3.92),
delay1.31s from planned2.60s; its saved poststate is3.92s/delay1.32s. Control38 is
contact_push_intent with zero contact;39 has9/10,44 has3/10. Stateframe0 isN/A.

The pre-existing repository publication-evidence check still reports three hash
mismatches (custom-L-summary,pretrained-T-summary,palette-T-summary); those files
were not modified for this audit. No model-quality conclusion follows from this
data audit. Shared source export/server jobs and the eight held-out letters remain
unchanged. final-verification.json and final-storage.json record final bindings.
''')
    save('SAMPLING_PROPOSAL.md','''# Targeted next-collection proposal — not executed

The library contains11,983 contact+obvious TRAIN targets across735 independent
trajectories and11,787 with both valid nonzero impulse and>=5contact substeps.
Do not discard it wholesale or replace its real negatives with duplicated good
windows. Withhold the claim that all acquisition is contact-rich:40.96% of TRAIN
intervals are moving-pusher/no-contact, and late-half contact+obvious is21.60%.
R3B remains paused for the planning chat's decision.

1. Before collecting more, amend the controller proposal so phases follow actual
   contact observations. Use current/past post_solve AO flags and valid impulse
   presence for contact acquisition/loss; keep planned phase/mode separately.
   A route towards an outdated staging point must not silently consume the rest
   of the push phase. Recompute feasible routes from current geometry, cap a
   recovery attempt, log failure/censoring and move to a new independent family.
   Diagnose wall-contact+nearstatic with tangential or alternate exterior contact
   choices. Preserve those failures as negatives rather than deleting them.
2. Keep all physics unchanged:half-life0.1s,10Hz control/100Hz physics,force PD,
   speed/force caps,mass/inertia,walls and penetration guard. No teleporting,
   midtrajectory resets,future-physics lookahead or reduced motion thresholds.
   Report both any-contact and sustained/impulse-supported movement; contact
   alone cannot establish useful motion.
3. Suggested first pilot, pending release:at most48 fresh TRAIN families,8 per
   letter,with a2,304charged-control cap including failures. Per letter,allocate
   four recovery cases,two boundary rotation cases and two low-support exterior
   arc cases; distribute initial yaws/regions and retain independent seeds.
   This is a targeted pilot,not an unbiased replacement for the existing library.
   Prioritize X corner negative rotation (79controls/19families versus positive
   261/29),its known exterior-arc gaps,and T's low actual contact duty44.70%.
   O/B closed inner rings need a geometry-feasibility study; do not call them
   reachable or unreachable from zero coverage,or force the pusher through walls.
4. Freeze proposed acceptance targets before running:contact+obvious>=40% of all
   pilot controls,moving-pusher/no-contact<=25%,observed recontact median<=0.5s,
   P90<=1.0s,and right-censored recovery<=5%. These are proposed engineering
   targets,not retrospectively relaxed gates. Always include all attempted
   controls and failures; report per-letter/family support so pooled means cannot
   hide weak shapes. Compare to existing matched strata descriptively; targeted
   nonrandom changes do not establish a causal controller improvement.
5. Freeze current TRAIN/validation identities and original natural-validation
   sampling. Do not filter or reassign validation using these motion/contact
   outcomes. Keep natural validation separate from targeted diagnostics and keep
   all eight held-out letters sealed. Any training release should use paired
   original6000-versus-new results on the same natural/stress windows and copylast
   baseline; this audit itself makes no model-improvement claim.

No new collection or model work is authorized by this proposal. Send this audit
and proposal to the planning chat under the standing callback authorization,
preserve its actual receipt,then stop in waiting_parent without polling.
''')
    result=dict(kind='contact-audit-stage-result.v1',audit_complete=True,training_resumed=False,
        recommendation='Retain useful independent pushing support; disclose route/stall costs and request parent decision before training or a targeted pilot.',
        all_TRAIN_controls=36864,all_TRAIN_seconds=3686.4,all_TRAIN_contact_seconds=1826.10,
        all_TRAIN_contact_and_obvious=11983,all_TRAIN_contact_and_obvious_fraction=six(a,'contact_obvious')['fraction'],
        all_TRAIN_nearstatic_fraction=near(a),selected_targets=23984,
        selected_contact_obvious_trajectory_support=735,selected_impulse_and_5of10_contact_obvious=11787,
        moving_pusher_no_contact_controls=15101,moving_pusher_no_contact_fraction=15101/36864,
        recontact_by_split=r,physical_half_life_seconds=.1,model_calls=0,physics_steps=0,
        report='REPORT.md',annex='ANNEX.md',proposal='SAMPLING_PROPOSAL.md',callback_required=True)
    save('result.json',json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    print(json.dumps(dict(report_written=True,files=['REPORT.md','ANNEX.md','SAMPLING_PROPOSAL.md','result.json'])))


if __name__=='__main__':main()
