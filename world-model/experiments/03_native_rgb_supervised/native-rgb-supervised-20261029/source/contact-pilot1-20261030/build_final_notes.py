"""Summarize the closed pilot without changing controller or collected data."""
import json
from pathlib import Path
import sys
import numpy as np

STAGE=Path(__file__).resolve().parent
ROOT=STAGE.parents[2]
sys.path.insert(0,str(ROOT))
from tdyssey.native_rgb_contact_audit import actual_signals,motion,classify_motion

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def pct(value):return f'{value*100:.2f}%'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |'for row in rows])


def main():
    summary=read(STAGE/'report/summary.json');plan=read(STAGE/'plan/cases.json');manifest=read(STAGE/'manifest.json')
    groups=summary['groups'];old=groups['A_old_baseline'];new=groups['A_new_controller'];fresh=groups['B_new_fresh']
    records={r['case_id']:r for r in summary['attempts']}
    sensitivities=dict(A_scheduled_control_slots=1152,A_unexecuted_control_slots=1152-new['charged_controls'],
        A_new_contact_obvious_fraction_scheduled=365/1152,
        A_new_empty_upper_scheduled=(250+new['unclassified_charged_controls']+1152-new['charged_controls'])/1152,
        note='Scheduled-slot sensitivity is not additional charged physics or observed time. Unexecuted slots receive no successful movement credit and pessimistic empty credit.')
    verified={}
    for group in('A_matched','B_fresh'):
        rows=[r for r in summary['attempts']if r['group']==group and r['withdrawal_release_observed']]
        delays=[r['recontact_delay_seconds']for r in rows if r['recontact_observed']]
        verified[group]=dict(verified_release_cases=len(rows),observed_recovery_after_verified_release=len(delays),
            unrecovered_after_verified_release=sum(not r['recontact_observed']for r in rows),
            observed_median=None if not delays else float(np.median(delays)),
            observed_p90=None if not delays else float(np.quantile(delays,.9)))
    weak=[]
    for case in plan['cases']:
        if case['group']!='B_fresh'or case['role']!='fresh_weak_exterior':continue
        record=records[case['case_id']];target=case['candidate']['selected_contact']['arc_id']
        entry=next(e for e in manifest['entries']if e['case_id']==case['case_id'])
        with np.load(STAGE/entry['file'],allow_pickle=False)as z:
            arrays={k:z[k]for k in('states','contact_flags','contact_tags','contact_values','contact_arc_offsets','contact_arc_ids')}
        signals=actual_signals(arrays);t,r,_=motion(arrays['states'],case['letter']);classes,_=classify_motion(t,r)
        contacted=[];useful=[]
        for control in range(len(t)):
            rows=signals['arcs'][control*10:(control+1)*10]
            if any(target in arcs for arcs in rows):
                contacted.append(control)
                if classes[control]==2 and signals['nonzero'][control*10:(control+1)*10,1].any():useful.append(control)
        weak.append(dict(case_id=case['case_id'],letter=case['letter'],frozen_initial_target_arc=target,
            actual_target_contact_controls=contacted,actual_target_contact_and_obvious_with_impulse_controls=useful,
            actual_outer_arc_ids=record['actual_outer_arc_ids']))
    # Diagnose the physical guard from the actual final partial poststate.
    bad=next(e for e in manifest['entries']if e['status']!='completed')
    with np.load(STAGE/bad['file'],allow_pickle=False)as z:
        state=z['substep_states'][-1];penetration=z['substep_penetrations'][-1].tolist()
    import cv2
    from tdyssey.native_rgb_r3a_data import RecordedStateRenderer
    renderer=RecordedStateRenderer(ROOT.parent/'Tdyssey/third_party/diffusion_policy',maximum_frame_cache_bytes=0)
    try:frame=renderer.render('X',state)
    finally:renderer.close()
    picture=STAGE/'report/CP1-A-X-03-last-known-partial.png'
    cv2.imwrite(str(picture),cv2.cvtColor(np.moveaxis(frame,0,-1),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,9])
    notes=dict(kind='CONTACT-PILOT1-root-interpretation.v1',recommendation='do_not_promote_or_expand',
        engineering_targets_fully_passed=False,scheduled_slot_sensitivity=sensitivities,
        recovery_after_verified_withdrawal=verified,fresh_weak_arc_outcomes=weak,
        guard_failure=dict(case_id=bad['case_id'],charged_controls=6,complete_controls=5,known_substeps=58,
            partial_substeps=8,last_poststate_time_seconds=.58,last_substep_penetrations=penetration,
            penetration_columns=['object_wall','agent_object','agent_wall'],last_known_partial_picture=str(picture),
            picture_note='Original renderer of known actual partial poststate; no collection RGB was returned for the aborted partial control, so no collector RGB hash exists for this one state.'),
        model_calls=0,new_physics_steps=0,training_pool_modified=False)
    write(STAGE/'mechanism-notes.json',notes)
    rows=[]
    for key,label in(('A_old_baseline','A 旧控制器（已存基线）'),('A_new_controller','A 新控制器（同初始场景）'),('B_new_fresh','B 新独立场景')):
        q=groups[key];rec=q['recovery'];delays=rec['observed_delay_seconds']
        rows.append([label,f"{q['successful_48control_trajectories']}/{q['trajectories']}",q['charged_controls'],
            pct(q['contact_obvious_fraction_charged']),pct(q['moving_pusher_noAO_fraction_charged_upper']),
            f"{delays['median']:.3f}/{delays['p90']:.3f}",
            f"{rec['right_censored']}+{rec['not_reached']} / {rec['planned_tasks']}"])
    report=f'''# CONTACT-PILOT1 结果（2026-10-30）

**本轮已完成；恢复接触的速度改善，但整体未通过预先冻结的工程目标，暂不建议推广。**
48 个案例全部尝试，47 个完成，1 个触发原有物体穿墙保护。没有补跑或按结果调参。
实际费用为 **2,262 charged controls、2,261 完整区间、22,618 已知物理子步（226.18s）**；
失败的 8 个部分子步保留，未知推进和未开始 reservation 均为 0。预算余下 42 步未使用。
模型调用、checkpoint 加载和训练更新均为 0；原 R3A 库和训练池保持冻结。

{table(['组别','完成/尝试','charged','接触且明显运动','空跑上界','恢复中位/P90 秒','未恢复+未到阶段/任务数'],rows)}

接触/运动阈值与 CONTACT_AUDIT 相同。空跑为推杆端点移动≥0.1 image96px 且整区间没有 AO 接触；
上界把未分类 charged 区间作最不利处理。恢复延迟从固定任务 control26 开始，以真实 post_solve 为准。
右删失与未到恢复阶段分开：A 新控制器为 4 条末尾未恢复、1 条提前安全失败，不能合并成成功样本。

* **确有改善：**A 配对组接触且明显运动从 276/1,152 提高到 365/1,110；后半段从
  88/576（15.28%）提高到 155/552（28.08%）。已恢复者延迟中位/P90 从 0.680/2.113s
  降到 0.310/0.410s。原基线直接复用，未消耗新物理重新跑旧控制器。
* **仍未达标：**全部新试验接触且明显运动 813/2,262（35.94%）低于 40%；空跑保守上界
  25.77% 高于 25%；未恢复或未到阶段 5/48（10.42%）高于 5%。B 新场景的 38.89%、28.82%
  也未达前两项，尽管 24/24 都在真实撤离后恢复了接触。
* **不能掩盖退步：**A 的 B 字母明显运动从 61/192 降到 42/192，顶墙停滞从 29 升到 36；
  T 的顶墙停滞从 32 升到 46。X 的顶墙案例在第 6 个控制区间、第 8 个物理子步触发保护，
  物体碰墙穿透达到 {penetration[0]:.6f} worldpx，超过保持不变的 2px 限制。它造成的截断不能解释成已解决停滞。
* **分母敏感性：**A 新轨迹比旧基线少 42 个实际 charged 区间。按原定 1,152 个时间槽给未执行部分
  零运动信用，A 的运动比例为 31.68%，空跑最不利上界为 25.43%；这些是假设时间槽统计，不能算新增物理时间。
  另有两条 A 案例在撤离阶段未验证到完全脱离接触，恢复表中的 0 延迟不能当作成功完成撤离。
* **弱弧目标未实现：**B 中六条弱支持外缘案例都没有在其冻结初始目标弧上记录到实际接触。
  它们在其他弧上的数据仍保留，但不能用“选择了弱弧目标”宣称填补了这六个覆盖缺口。

控制器在新文件中实现了：跟随当前实际接触点/法向；对失联防抖；按路径进展和距离估计恢复时间；
停滞时按冻结条件换可达外缘点；连续 AO+OW 且基本静止时尝试沿墙或换边。
原定撤离区间完整保留，保存的 planned phase 没有改成有利标签。速度半衰期仍为 **0.1s**，
10Hz/100Hz、PD、限速、质量惯量、墙、渲染和穿透保护均保持不变。

只读诊断定位到具体缺口：T 的换点候选通过检查时借用了旧接触点的离开桥接路径，换点后却丢失这段路径，
导致重复保持；X 的穿墙发生在持续推动中，尚未达到“连续静止”逃逸触发条件；B 的退步集中于一个案例的
法向跟随、重复换点和小幅运动。[完整诊断与原始索引](DIAGNOSIS.md) 保留了证据和推断限制，未据此重跑本轮。

**处理建议：**保留原 R3A 有效库和本轮所有成功/失败证据；本 pilot 不自动加入训练池。
暂不扩大新控制器采集，也不恢复 R3B。下一步若获授权，应先处理运动中接近墙的安全性、
B 的接触点/法向跟随退步及少数仍无法恢复的路线；本轮不继续迭代到“通过”为止。
A 是按困难 TRAIN 案例选出的配对机制检验，B 也是定向新场景，均不代表总体自然分布。

[控制器规则与实现说明](CONTROLLER.md) · [48 案例冻结计划](plan/cases.json) ·
[逐案例、逐字母完整结果](report/RESULTS.md) · [全部六类 CSV](report/six-classes.csv) ·
[配对差值 CSV](report/paired-deltas.csv) · [机器汇总](report/summary.json) ·
[同场景新旧视频及时间线](report/index.html) · [原始数据独立核对](independent-check.json) ·
[分母、撤离和弱弧补充](mechanism-notes.json) · [失败的最后已知部分状态](report/CP1-A-X-03-last-known-partial.png)
'''
    (STAGE/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(report_written=True,recovery_verified=verified,weak_arc_outcomes=[dict(case_id=r['case_id'],target=r['frozen_initial_target_arc'],contact_controls=len(r['actual_target_contact_controls']),moving_impulse_controls=len(r['actual_target_contact_and_obvious_with_impulse_controls']))for r in weak])))


if __name__=='__main__':main()
