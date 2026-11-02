"""One-page CP2 findings from closed, independently checked saved data."""
import json
from pathlib import Path

STAGE=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pct(value):return f'{value*100:.2f}%'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |'for row in rows])


def main():
    summary=read(STAGE/'report/summary.json');groups=summary['groups'];q=groups['CP2_all_planned']
    independent=read(STAGE/'independent-check.json')
    assert independent['passed']and not independent['mismatches']
    rows=[]
    for key,label in(('R3A_original_A','原 24 场景：R3A'),('CP1_original_A','原 24 场景：CP1'),
                      ('CP2_repeat_A','原 24 场景：CP2'),('CP1_original_B','CP1 另 24 场景'),
                      ('CP2_repeat_B','同 24 场景：CP2'),('CP2_new_families','12 个全新场景：CP2')):
        r=groups[key]
        rows.append([label,r['charged_controls'],pct(r['contact_obvious_fraction_charged']),
            pct(r['moving_pusher_noAO_fraction_charged_upper']),pct(r['late_contact_obvious_fraction_charged'])])
    outcomes=table(['实际接触 × 运动','区间数','占全部 2,880 区间'],[
        [key,value,pct(value/2880)]for key,value in q['six_classes'].items()])
    text=f'''# CONTACT-PILOT2 结果（2026-10-30）

**60 例均完成，已验证零穿透保护失败；但采集效率大幅退化，本版本不推广。**
这轮修复了可指认的集成问题并保存了真实回归证据，没有根据物理结果继续调参或重跑。
60 例由 **48 个 CP1 已见场景重复验证 + 12 个新独立 TRAIN 场景**组成，不能称为 60 个新独立场景。
物理费用 **2,880 charged / 2,880 完整控制区间、28,800 已知子步（288.0s）**；
失败、部分区间、未知推进、未开始 reservation 和未执行案例均为 0。模型调用、checkpoint 加载和训练更新为 0。

{table(['比较组','charged 分母','接触且明显运动','推杆空跑','后半段接触且明显运动'],rows)}

全部 CP2 的“接触且明显运动”为 **310/2,880（10.76%）**，低于 40% 目标；
空跑 **1,984/2,880（68.89%）**，高于 25% 目标。真实接触为 **5,608/28,800 子步（19.47%）**。
所有案例完整执行，因此实际 charged、完整区间和原定 60×48 时间槽分母相同，敏感性结果没有被早停抬高。
CP1 的原 24 场景有一条失败前缀，保留了它的 1,110 charged 分母；完整三方逐案例表另列。

* **修复有证据：**13 项生产路径静态回归通过。T03 的目标与旧 departure bridge 一并提交，当前几何逐步复验；
  B04 的不合法 point/normal/arc 组合不再强行拼接；真实接触观测与可执行目标分开。
  X03 的近墙风险在旧轨迹 control3 即触发，覆盖旋转角点；所有实际动作均经过风险检查，模式标签不能绕过。
  原 X03 本轮完成 48 步，穿墙失败没有复现。物理和 2px guard 没有改动。
* **实际任务仍退化：**B04 的明显运动区间为 R3A **33 → CP1 14 → CP2 0**；
  原有正常推动回归 T040 为 **6 → 23 → 11**，后半段 CP1 的 13 个降为 CP2 的 5 个。
  T03 和 X03 本轮各 0 个明显运动区间，虽然完整运行且没有顶墙停滞。
  全部顶墙停滞为 0，不能据此宣称推动能力改善。
* **恢复未达标：**60 例都在撤离期间验证到至少一个完整、已知且无 AO 的控制区间。
  45 例随后恢复接触，15 例在 2.2s 观察窗结束时右删失（25%）；从未分离和未到阶段均为 0。
  已恢复者中位数 **0.58s** 超过 0.5s 目标，P90 **0.78s** 达到 1s 目标。
  45 例也都有撤离前真实接触，不把从未接触过的首次取得接触说成分离恢复。
* **弱弧命中不等于有效推动：**六个旧弱弧和六个新弱弧都有真实 requested-arc 命中及同一接触点行的正冲量，
  没有用邻弧、min(arc_ids) 或宽松位置容差替代。有效冲量连续段最长仅 **0.08–0.14s**；
  只有旧 T 弱弧有 **1 个**目标接触且明显运动区间，其余 11 个为 0。

全部计划 push 区间中 **1,412/2,075（68.05%）**无实际接触，寻路/恢复且无 AO 的区间为 1,399。
risk gate 在 390 个命令触发；完整 hold 只有 17 个（最长连续 7 个），未确认路线命令为 0。
这表明退化主要表现为小幅运动与大量已验证路线上的无接触移动，不能简单归咎为一直 hold。
持续跟随命令仅 337 个，其中 286 个伴随明显运动；法向接近 630 个、事务路线 1,167 个。
B04 的风险触发和非法 pair 拒绝均为 0，指定弧虽有正冲量命中，却未满足 5/10 且末端命中的建立条件，
并因 2worldpx 法向对齐限制反复返回 staging。它不是单独由近墙阈值造成的退步。
多个机制同时改变，[只读诊断](DIAGNOSIS.md) 记录命令索引和限制，不声称隔离出唯一原因。

{outcomes}

接触与运动定义保持 CONTACT_AUDIT 不变：真实 100Hz AO post_solve；明显运动为形心位移 ≥2 image96px 或绝对转角 ≥5°，
基本静止为 <0.1px 且 <0.1°。接触与位移是观测共现，不是严格因果推动分解。
本轮恢复定义按父要求加严：完整无 AO 撤离后才接受首次真实接触；旧基线也按此重新计算供比较，未改写其原报告。

**建议：**保留原 R3A 已验证有效的数据及本轮所有证据；CP2 不加入训练池、不替代默认采集控制器、不继续扩采。
具体集成修复和零 guard 已有回归与实际验证，但本版本不能作为高效推物数据采集器。
规划对话可只读核对并向用户整理“原 R3A 有效库上的模型验证”和“采集器后续研究”两条建议。后续任何实施均需用户明确审批。
本轮未自动恢复 R3B，也不会再循环调参到通过。

[机制与冻结规则](CONTROLLER.md) · [近墙规则及限制](SAFETY.md) · [13 项实际回归回执](root-regression-check.json) ·
[60 案例冻结计划](plan/cases.json) · [所有版本与逐字母完整表](report/RESULTS.md) ·
[固定 X03/T03/B04 视频和面板](report/index.html) · [12 弱弧逐项 CSV](report/requested-arcs.csv) ·
[原始数据独立核对](independent-check.json) · [机器汇总](report/summary.json)
'''
    (STAGE/'REPORT.md').write_text(text,encoding='utf-8')
    result=dict(kind='CONTACT-PILOT2-final-result.v1',execution_complete=True,attempted_cases=60,complete_cases=60,
        repeated_seen_cases=48,new_independent_TRAIN_families=12,charged_controls=2880,known_substeps=28800,
        guard_failures=0,unknown_advance_substeps=0,all_engineering_targets_passed=False,controller_promoted=False,
        contact_obvious_fraction=310/2880,empty_fraction=1984/2880,strict_recovery_observed=45,
        strict_recovery_censored=15,requested_weak_arc_hits=12,weak_requested_contact_obvious_controls=1,
        recommendation='Do not promote or extend CP2. Keep valid R3A library and pilot evidence; parent must separately decide model validation versus later collector research.',
        model_calls=0,checkpoint_loads=0,training_updates=0,training_pool_modified=False,report='REPORT.md')
    (STAGE/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(report_written=True,targets_passed=False,all60completed=True)))


if __name__=='__main__':main()
