[父对话阶段2A审查：仅放行阶段2B剩余正式TRAIN采集；验证与训练继续禁止]

我已读取2A报告/诊断，并独立对162个NPZ逐文件SHA、面积质心位移、角度与原始候选分母复算：10368controls、9882H1、moving5983、near_static3444、0invalid全部匹配，每字母9空间都存在。M=267moving/253near_static/549，46.08%静止确实未通过，结论保持。我们现在完成原计划22条/字母的余13条，而非删掉M或再加无限救场探针；2A单批未达记录不得被覆盖，合并22条后仍按原门槛审查。

我另查了2A原始scene：O在center/left/right/left-top反复选staging_segment55（4条）；P在center/left/top/left-top/right-top/left-bottom选55（6条）。计划目标miss正集中于这些反复选择。当前_candidate_priority只优先“没碰到过”，未考虑“已独立尝试多次仍未碰到”，会反复追同一段，妨碍实际接触覆盖。以下仅允许初始化选点的明确小修正，连续1B推移/恢复策略保持冻结。

一、唯一放行范围
阶段2B只采TRAIN：18训练字母 ABCEFHIJLMOPSTUVXY ×13新独立轨迹×64步，最多234执行槽/14976controls，包含本阶段所有实际执行、失败、部分和重放。不得额外开物理探针/补采/替代槽。
阶段2A的162条原样保留并加入最终TRAIN目录/清单视图；阶段1/1B和旧native轨迹均不并入正式TRAIN。
若全完成，正式TRAIN396条/25344controls，每字母22条；新监督实验采集累计3968+25344=29312，旧native+新累计63872。验证的162条/10368controls仍未放行；未来全预算39680<=40000不变。
本阶段0梯度、0模型推理、0训练索引消费、0未见预测、0seen_val采集、0服务器、0打包。
阶段2C独立验证、阶段3及以后须等下一次父指令。

二、空间和间隙配额——执行前固定，不能只补易场景
每字母新增9基础场景，9空间各1条，再加下面固定4个额外空间。以下18字母额外空间表全局每类恰好8条，合并2A后9空间全体各44条；M额外保留左上与左下，不能只用中心降低静止率：
{
  "A": [
    "center",
    "left",
    "right",
    "top"
  ],
  "B": [
    "bottom",
    "left-top",
    "right-top",
    "left-bottom"
  ],
  "C": [
    "right-bottom",
    "center",
    "left",
    "right"
  ],
  "E": [
    "top",
    "bottom",
    "left-top",
    "right-top"
  ],
  "F": [
    "left-bottom",
    "right-bottom",
    "center",
    "left"
  ],
  "H": [
    "right",
    "top",
    "bottom",
    "center"
  ],
  "I": [
    "right-top",
    "left-bottom",
    "right-bottom",
    "center"
  ],
  "J": [
    "left",
    "right",
    "top",
    "bottom"
  ],
  "L": [
    "left-top",
    "right-top",
    "left-bottom",
    "right-bottom"
  ],
  "M": [
    "left-top",
    "left",
    "left-bottom",
    "top"
  ],
  "O": [
    "bottom",
    "left-top",
    "right-top",
    "left-bottom"
  ],
  "P": [
    "right-bottom",
    "center",
    "left",
    "right"
  ],
  "S": [
    "top",
    "bottom",
    "left-top",
    "right-top"
  ],
  "T": [
    "left-bottom",
    "right-bottom",
    "center",
    "left"
  ],
  "U": [
    "right",
    "top",
    "bottom",
    "left-top"
  ],
  "V": [
    "right-top",
    "right",
    "right-bottom",
    "center"
  ],
  "X": [
    "left",
    "right",
    "top",
    "bottom"
  ],
  "Y": [
    "left-top",
    "right-top",
    "left-bottom",
    "right-bottom"
  ]
}
LOCATIONS固定次序center,left,right,top,bottom,left-top,right-top,left-bottom,right-bottom。
新增9基础场景的边界间隙取与该字母该空间2A已有4/18/48不同的下一档（循环+1）；额外4场景如果边界则取该字母该空间尚未覆盖的第三档（循环+2）。中央不造gap标签。每类总体计数与每字母缺少的空间×gap组合都明确报告，不承诺22条覆盖全部24边界组合。
seed/letter/location/gap/执行顺序与选点规则先冻结，与全部历史scene/seed去重；不看结果换seed，不重复原初态造新轨迹。每条合法初始化后连续物理步进，不瞬移或拼接。
M、O/P、V/X/J/T的不足只作为报告关注项，不准对应字母专用改控制力、动作、物理、时步或阈值。

三、唯一允许的选点更新：泛化的“尝试轮转”规则
- 维护与真实hit计数完全分开的planned_attempts[letter][arc]，每个已实际开始的独立TRAIN轨迹最多对其初始目标arc计1次；用2A真实记录初始化，2B仅更新自身已发生的TRAIN记录。不能把planned_attempt算成hit或抬高覆盖率。
- 几何合法性/外部可达证据和现有motion-room tier仍先约束；在同一tier内优先独立planned_attempt次数少的arc，之后沿用实际未命中/低hit、旧TRAIN先验和冻结随机tie-break等原排序因素。这使55号等反复未命中的段不一直压住其他合法候选。规则适用于全部18字母，不能硬编码55或O/P。
- planned_attempts只来自正式2A+此前2B；真实coverage也累计正式2A+此前2B，旧TRAIN prior单列；不重置2A已有coverage，不把probe/val命中加入正式训练覆盖。
- 可在plan_scene新增明确参数/版本字段或外层候选排序，保存2A源码快照，逐轨迹记录排序输入与实际choice。不要用伪造coverage_counts实现惩罚，也不要把尝试失败arc标成不可达。自然碰撞到任何arc仍照实计hit。
- 此版本命名和hash需与2A区别，合并数据显式保留两种初始化选点版本。1B ContinuousContactSampler、RECOVERY_LIMITS、PHASES、动作距离、原6side_sign及其余既定映射、模型/损失/物理fd358都保持不变。
- 只做相关纯CPU排序/预算/去重/合并检查；不扩大成新的采样器研究或悄改重接触timeout。

四、完成时验收与可用数据
分别输出2A原批、2B新批、2A+2B合并统计，三者原始分母不能混淆。
- 每字母合并原始有效H1目标仍moving>=40%、near_static<=45%；H1 moving定义平移>=2worldpx或转角>=1度，near_static<=.5px且<=.005rad。不达则如实回传，不能删难帧、弃字母、降阈值或自动继续采。
- 独立报告所有真实controls和有效H1/H4；完整22×64时每字母1408controls/1342H1，18字母共24156H1；H4每字母1276、全22968。异常/不足按实际分母，不复制帧补数。
- 实际接触点覆盖热图、全轮廓/规范可达/已证实下界/未确认/封闭洞分母，独立contact轨迹、角度/真实运动量、实际四墙与双墙gap交叉统计。
- 重点对比O/P目标55独立计划尝试次数、实际命中、其它arc新增命中和仍未确认段；不能仅报告planned-hit百分比提高。M保留2A困难角落并列新角落分布，不用容易场景平均抹掉缺口。
- 四层moving/intermediate/static-contact/static-no-contact的独特窗口、独立轨迹、空间支持与70/10/10/10有放回计划。重复抽样合法，但不能称新物理证据；不要为了各10%故意制造大量静止。空层/过度集中如实列出，0实际训练采样。
- 固定物理100Hz子步审计、2px阈值不变；无效转移绝不进入可用TRAIN。失败尝试仍记账并保留。资源/程序/物理异常立即安全停并回传；不自行替代失败槽。
- 提供可消费的最终TRAIN清单或只读合并视图，引用2A原始文件与2B新文件/hash，不复制整套旧数据、不覆盖2A manifest。验证身份与训练完全分离，尚未创建验证数据。
- 少量新真实连续帧以事先规则选取覆盖M/O/P及其它字母，不需要重新大量渲染所有旧样例。
- 新旧native全部持久产物<=3GiB，保留历史数据/权重/账本，不重复下载或装环境。

五、主动回传与停下
先把本次父审查原文保存到stage2b/PARENT_REVIEW.md，更新SUPERVISION/stage-state/STATUS为“只授权2B剩余TRAIN；2C验证与3–6未授权”。保存冻结计划/新选点版本身份。
完成后调用send_message_to_thread回父chat __LOCAL_CHAT__(host local)，提供逐字母合并门槛表、M角落/O-P覆盖变化、四层支持、来源版本与合并manifest、物理审计及累计成本的绝对证据路径，然后写waiting_parent并结束。不轮询，不要求人类转发，不自行采验证或训练。
