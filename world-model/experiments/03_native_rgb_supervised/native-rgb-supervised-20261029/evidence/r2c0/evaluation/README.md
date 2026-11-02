# R2C0 评估准备（尚未运行模型）

固定TRAIN可视12、val natural12、val motion12窗。缺层0项。

TRAIN可视窗均在首480实际draw出现且完整H4合法；同letter先平移后旋转，排除重复窗。natural每个已冻结val轨迹仅hash抽anchor，不筛运动；val轨迹本身是刻意含运动的诊断切分。motion与natural按trajectory+anchor去重，仍可能共享同轨迹历史/未来，统计分开。

4历史RGB来自存储原帧；每步模型只收4动作，H4七动作前缀仅审计。原GPC/训练后同窗同batch/seed；H1复用H4第一步。有限越界原值回灌并保留硬失败，非finite/shape错误硬停；无未来truth/state/pose/重渲染/修复。

本次只有真RGB预览和CPU接口检查，无预测、训练或300/1200执行许可。原物理及模型账本未动。

预览入口index.html链接全部7页。额外TRAINother页不是评估cohort，6标签均由首480已曝光、H4合法other窗minhash并逐项去重，独立只读复核完全一致；主indices未变。8项NumPy/评分/索引CPU检查通过。精确静止缺层为0，近静止/小运动仍保留。
