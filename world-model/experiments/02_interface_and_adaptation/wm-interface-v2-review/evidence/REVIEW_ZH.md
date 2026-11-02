# 保存图像接口 v2：离线修复与复核

接口修复已完成，但目前仍不能用该模型可靠地计算停止一致的想象奖励。
本次只复用旧 RGB/NPZ/JSON；环境创建、reset、restore、真实 render、控制步、
世界模型/策略推理、采样和参数更新全部为 **0**，无安装、下载或预算重开。

## 结果

计划 1304 行，可用 1284 行。旧 L01/hold 失败分支的 20 个缺失行全部保留。
噪声重复嵌套在同一物理分支内，不能当作独立样本；所有数据均是已看过的开发数据。

- **真实图像**：24 校准 + 252 历史 + 252 实际未来，共 528 行全部可测。
  校准覆盖率 MAE：L .008360→.006988，T .005699→.004651；部分误差尾部变差，
  不是全面改善。±.03 开发假设区间覆盖这 528 行，不构成独立校准保证。
- **近 .95 的 8 个校准状态**：点估计从 TP0/FN4 变为 TP2/FN2；区间判定为
  6 个 unknown、2 个 non_success。四个真实成功全部仍未被确认；成功判定未验证。
  严格 `coverage > .95` 不变，无预测成功时 precision 不可定义。
- **预测图像**：pusher 可测 754/756；object 仅 104/756（L 12/564，T 92/192），
  少于 v1 的 126/756。H1 连 T 也是 0/48；H4 T 为 34/48。因此问题不限于 L
  形状迁移。不能将 gate 失败自动归因于模型动力学。
- **评分**：所有 16 个 anchor 停止一致选择均弃权。H4 原始端点仅 T12/T13
  能完整选择，实际可评分的 3 对全部来自 T13，选择 hold；同一子集 hold regret
  也为 0，不能据此宣称超过 hold。L 的 12 对和 T 的另外 3 对均弃权。
- **区间局限**：预测中只有 100/104 可测行的区间覆盖真值；另外 4 行失败。
  三个局部候选拟合并不是完整多解分析，明确标 `ambiguity_assessed=False`。

接口保留亚像素/抗锯齿证据，目标仅在姿态拟合完成后用于覆盖率；刚体形状、
物体姿态、pusher、覆盖率和终止各自有有效性及原因。优化 ROI 来自图像，最终
形状检查覆盖整图与完整模板，不缩放物体，不用未来真值修复答案。未知的早期
终止会阻断序列评分，但不会删除后续原始端点测量。

T10/T11 在 v1 的最大二值物体分量相同，但原 RGB 仍有不同边界像素。
这证明 v1 丢失信息，不能证明 96×96 RGB 本身不可区分；v2 的不同点估计也
尚不能确认严格成功。L00 H1 红色小团、L06 H1 缺失竖臂是具体形状保持问题；
T13 H1 则仍混合外观、形变与接口拒绝。L06 toward/5480/H3 的正确数字是
coverage .262077 对 truth .265205，误差 .003128；其位置误差 7.223px、角误差
4.461°，说明小 reward 误差会隐藏姿态误差。

## 可复核文件

- [完整英文结果与解释](../../../../archive/workspaces/Tdyssey-world-model-validation/docs/WM_INTERFACE_V2_RESULTS.md)
- 原逐图数值与完整拟合记录 (external historical reference: `../wm-interface-v2-development/per-image.jsonl`)
- 修正元数据后的逐图表 (external historical reference: `per-image-reviewed.csv`)
- [修正后的终止混淆表](termination-confusion-reviewed.csv)
- [描述性误差分布、区间覆盖与共同可测子集](review-metrics.json)
- 原始端点/序列评分 (external historical reference: `../wm-interface-v2-development/sequence-scores.csv`)
- 排序和分母 (external historical reference: `../wm-interface-v2-development/rankings.csv`)
- 独立物理基线 (external historical reference: `../wm-interface-v2-development/baselines.csv`)
- 保存 RGB 对照图 (external historical reference: `saved-rgb-evidence.png`)
- 旧资产后置审计 (external historical reference: `../wm-interface-v2-audit/after.json`)

原完整输出保持不动；复核导出只去掉 v1 不适用的 88 个“区间”重复统计行，
把未评估歧义改为 unknown，并明确缺失输入原因。任何姿态、有效性、覆盖率、
序列评分、排名数值均未改。源码同步修复报告元数据，未再执行解码。

主测量/聚合耗时 CPU **273.594s**、wall **282.641s**，原输出 **10,999,946 bytes**。
这些耗时不含预览、序列化和后续轻量报告整理；未声称是整个人工审阅耗时。
30 项针对性测试和 source/link、message、diff 检查通过。595 个旧文件合计
186,686,009 bytes 均保持哈希，其中 B/C 150 文件为 97,364,187 bytes；旧 v1
decoder/config 和两个权重均未改变。

下一步仅建议把官方输入视觉域/早期形状保持、外观与估计器、接触/旋转动力学
分层验证，再独立检查停止与动作排序。证据尚不能确定输入颜色修改、表示变更、
数据适配或某种权重微调何者足够。本次未开始任何新推理、采样或适配实验。
