# C/D 最终审阅：完整完成，整体验收未通过

本轮 C、D 各完成 30,272 个唯一目标、7,568 段、1,313 个训练家族、1,892 次成功更新，AMP 跳步均为 0；C 达到三项旧 A 运动门槛，但 O 未学会正确顺时针响应、B 仍丢孔，D 三项运动门槛也均未达到，因此两组整体均未通过，等待用户审阅。

当前唯一 run：RWCD-20261101-81bdec2c68934570a46dac1db04b3a72。配置 SHA256：6ae814d58c7af48b6c2cc6e63181a6f254281453b2f1d7a1c63f387725142c46。真实完成于 2026-11-01 21:10:42（新加坡时间）。此前未启动的 9693 配置快照没有发生模型、优化器或物理调用，不计为实验。

## 具体变化与可比范围

共同起点为原 OC epoch2 完整 checkpoint（c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187），model/Adam/scaler/global 与 conditioning RNG 均恢复。C 使用真实历史及原残差损失的 0.50 全图 + 0.25 区域 + 0.25 边缘权重；D 使用同一损失，每批两段真实历史、两段生成历史，62.5% 目标仍为全真实上下文。50/50 clean/weak 策略保持。无新数据、物理、结构、动作接口、服务器或 UI 模型变更。

C 对旧 A 是本轮损失变化的主要对照；D 对 C 是本轮历史比例变化的主要对照。D 对旧 B 同时改变损失和历史比例，仅作描述。所有结果属于已暴露的 seen-val/训练家族开发比较，未宣称独立最终测试或普遍因果结论。

## 冻结运动门槛：完整 16 个 seen-val 运动家族

| 指标 | 旧 A 门槛 | C | D |
|---|---:|---:|---:|
| 真历史单步 H1 IoU，越大越好 | ≥0.9129967823 | 0.9161933805，通过 | 0.9012529929，失败 |
| 真历史单步 H1 位置误差 px，越小越好 | ≤0.2703910734 | 0.2563001892，通过 | 0.2845351190，失败 |
| 原始 AR16 位置误差 px，越小越好 | ≤1.1391641914 | 1.0950019292，通过 | 1.2530504033，失败 |

每项实际分母均 16 available / 0 unknown，没有用部分样本放宽门槛。C 的 H1 位置比旧 A 下降约 5.2%，AR16 均值下降约 3.9%；但 C H1 仍劣于共同 epoch2 的 0.2526206551 px。

AR16 seen-val 位置均值 / 中位数 / P90：A 1.139164 / 0.915873 / 2.233085；C 1.095002 / 0.966589 / 2.113335；D 1.253050 / 1.090330 / 2.440018 px。C 对 A 9 例改善、7 例退步，中位数稍差；D 对 C 仅 4 例改善、12 例退步。H1 位置 C 对 A 8 改善 / 5 平 / 3 退步，D 对 A 4 / 1 / 11。

32 个运动诊断家族的 train16、seen-val16 已分开保存在完整报告与审计。训练组 AR16 C 对 A 位置 14 改善 / 2 退步；D 训练均值 3.38853 px 略好于 C 3.50162 px，但 seen-val 更差。这个固定小队列不足以判定普遍过拟合。

真历史 t+16 使用该时刻之前四帧真值，属于独立单步归因；AR16 仅起点四帧真值，后续递推原始预测，无未来真值回灌或形状修复。

## O 旋转、B 孔洞与 C 棱角

O 固定例 AC-O-01-08:a3 真值 H16 顺时针约 +50.08°。独立可见灰轴 / 孔轴累计响应：C −0.501° / +0.012°；D −4.643° / −4.801°，均落在“无可辨旋转”，并未实现正确顺时针运动。D 在 h11 仍有一次可判的累计反向。C 主要是不转，不能把反向幅度减小称为修复。全部 8 个 O 的 H16：C/D 均为 7 unknown + 1 无可辨旋转，0 个可判正确。局部 128 个目标各有 126 unknown + 2 无可辨旋转。小真值转角、轴校准、对称性和可见性等限制继续保留，原正式角度 unknown 未被独立可见轴附录改写。

B 固定例 AC-B-00-08:a19：真值末帧 2 孔；共同起点 2、旧 A 0、旧 B 1、C 1、D 1。C/D 从 h13（1.3s）起持续只有 1 孔，比旧 A/B 的 h14 更早失孔。其余 7 个 B 的可见孔数均为 2、无新增孔洞失败或未知；展示例仍失败，所以整体孔洞条件不通过。孔数相同也不证明刚性或位置正确。

正式 8 个 C 的 H16：旧 A / C / D IoU 为 0.546138 / 0.534743 / 0.573444；轮廓误差为 1.438118 / 1.456445 / 1.303225 px。C 的区域权重没有带来整体角部改善；D 的重叠与轮廓比 C 好，但不能据此宣称开口和刚性已修好。强运动 C 四例中，D 对 C IoU 实际仅 1 改善 / 3 退步，均值提升受单例影响；轮廓四例改善、位置三例改善一例退步，属于混合结果。O/C 固定板仍明显运动不足，C 的尖角逐渐变圆，B 形状持续变形。

## 六字母、静止、蓝推杆和逐例退步

正式 48 例、H1…H16 全部保留。H16 IoU：

| 组别 | 旧 A | 旧 B | C | D |
|---|---:|---:|---:|---:|
| 全部48 | 0.641489 | 0.686780 | 0.640077 | 0.671314 |
| O 强运动4 | 0.494329 | 0.495957 | 0.489462 | 0.511805 |
| C 强运动4 | 0.208444 | 0.289388 | 0.188277 | 0.245873 |
| B 全8 | 0.569973 | 0.624794 | 0.561055 | 0.604443 |
| T 全8 | 0.788248 | 0.842133 | 0.798268 | 0.831236 |
| L 全8 | 0.762111 | 0.810750 | 0.753779 | 0.772305 |
| X 全8 | 0.547548 | 0.572575 | 0.562220 | 0.570485 |
| 固定静止6 | 0.948978 | 0.962464 | 0.942275 | 0.961385 |

C 对 A 全48 IoU 为 20 改善 / 2 平 / 26 退步，静止6例全部退步。D 对 C 为 37 / 1 / 10，T 为6改善2退步，L为6改善1平1退步，X为5改善3退步；D 仍低于旧 B 全48平均。正式48 H16位置均值 A/B/C/D 为 1.708269 / 1.396755 / 1.712375 / 1.491181 px。

蓝推杆 H16 位置均值 A/B/C/D 为 0.385906 / 0.265619 / 0.417778 / 0.327738 px；C 劣于 A，D 改善 C 但劣于旧 B。现有高亮没有蓝推杆缺失，不代表穿透或接触物理已经解决。C/D 正式 H16 形状 unknown 均37/48，角度 unknown 均48/48；B 形状 unknown8/8。X 旧 B 的一个伪孔未在 C/D 末帧出现，但不能替代 O/B 未达标条件。

平均 IoU、位置、轮廓同时混合形状与位姿，不支持保形、正确力矩、无穿透或容量失败的结论。未额外做动作干预或敏感性实验。

## 实际费用、曝光与证据

合计 60,544 次监督曝光、3,784 次优化器成功更新，AMP0，没有重抽样凑更新。两组均独立从 lineage11367 到13259。D 生成3,784段、11,352张历史图、34,056次图像级去噪前向，额外计算已计入，不能称等计算量比较。5次完整恢复/续接和4次保护性RNG核验通过。C/D 监督branch、目标噪声、weak噪声、scaler和目标区域统计配对。

前置诊断实际768帧、2,304次图像级前向、0更新；GPU工程预检24次图像级前向单列。接口检查通过表示未发现本次核查范围内的保存XY、时序、归一化或输入损坏；模型反向预测本身不能证明原物理或接口代码错误。

全worker墙钟 4,450.929s（74分10.9秒），engine 4,133.043s。训练loop C1,178.230s、D1,895.643s；新C/D评测合680.423s；前置诊断240.078s已含启动时间；报告尾段150.295s。checkpoint总71.935s中65.773s已含loop；CPU区域工作与GPU重叠，不能重复相加。这些是混合墙钟，没有独占GPU时长或金额账单。首32批外推后，实际剩余全墙钟66.1分钟，落在此前54–124分钟范围。

峰值CUDA allocated1.673GiB、reserved1.887GiB。worker最终资源快照共享5,594,232,187 bytes，加256MiB原子预留及guards合5,933,970,811 bytes，低于10,000,000,000上限。旧缓存58.884s仅历史，本轮新缓存构建/物理积分/新采样均0。旧 R3B 失败33.047s保留为历史费用，不混入本轮。旧A/B/起点评测复用，新增模型调用0，不重复收费计时。

终态审计仅读保存结果、NumPy重评分和媒体解码，无Torch/模型/GPU/物理/新渲染。3,457个冻结绑定、560个正式/运动预测raw、48个前置诊断raw、清单48输出/659输入核对通过；2,240个高亮目标重评分与原指标一致。28张PNG可读，14段视频均16帧、10fps。C/D实际epsilon配对且与原B4种子重放一致；历史旧AR没有保存epsilon张量，原种子/槽位/来源和输入证据保留，首帧48/48 bitwise一致，不冒称恢复了缺失历史张量。

## 可查看文件

六列顺序均为：真值 / 共同epoch2起点 / 旧A / 旧B / C / D。H1/4/8/16对应0.1/0.4/0.8/1.6秒；10fps视频正常播放。这些固定O/C/B例在旧A/B时期就是失败例，继续保留供前后核查，未按新输出挑选，也不代表总体平均。

- 完整HTML报告 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/index.html`)、原始完整REPORT (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/REPORT.md`)、逐项门槛 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/support-signals.json`)
- O：图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-O-H1-H4-H8-H16.png`)、全部16帧 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-O-all-H1-H16.png`)、10fps视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-O-10fps.mp4`)
- C：图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-C-H1-H4-H8-H16.png`)、全部16帧 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-C-all-H1-H16.png`)、10fps视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-C-10fps.mp4`)
- B：图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-B-H1-H4-H8-H16.png`)、全部16帧 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-B-all-H1-H16.png`)、10fps视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/formal-B-10fps.mp4`)
- O/B前置同案归因报告 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/diagnosis/REPORT.md`)，完整HTML还提供train/seen-val真实历史单步与AR的O/C图和视频、T/L/X图和视频。
- 训练费用审计 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/postrun-training-audit.json`)、正式结果媒体审计 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/postrun-formal-audit.json`)、运动与方向孔洞审计 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/postrun-diagnostic-audit.json`)、原输出清单 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/region-weighted-cd-20261101/report/final-manifest.json`)

整体未通过。报告完成后只执行已授权实际回传和已结束任务注册清理，状态等待用户审阅；没有第二轮、8步、补采、改模型/配置、服务器或UI模型切换授权。
