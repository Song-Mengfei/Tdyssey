# 高运动拟合最终审阅：有明显改善，完整目标未达到

这轮已经跑完：2026-11-01 23:53:45（新加坡时间）结束，完整3000次成功更新、48000次曝光，AMP跳步0。没有继续训练。48段实际参加训练的轨迹单步和闭环指标明显改善，但H4/H8/H16仍未达到全部冻结门槛，原seen-val的C退步，训练拟合和验证均不能判为完成。

唯一run：HMF-20261101-107372d515664c92a26e3e480899efe0。
配置SHA256：872013d0b17e7d3e52933312a1b74993beb0ed6fa2fd04ee4054e0c1f7220cc1。
原批准PLAN SHA256：138d641c88cafe9a82e3c805aa2bf539219684135615e551fa8d862562b9e0d0。

## 实际做了什么

O/C/B各16个独立训练家族，共48段、768个唯一H1目标，其中444个强运动目标。24段原seen-val及原3失败展示例未参加本轮训练。每批12strong+4full，实际48000曝光中42790次是强运动；O/C/B各16000次，48家族各999–1001次。重复曝光47232次，首次全目标覆盖在第97批。两个池有重叠，172批存在共180个批内重复槽，已计入重复曝光；不当成新增数据或重跑。

只有一个分支从指定epoch2完整恢复model/Adam/scaler/global与conditioning RNG，谱系11367→14367。沿用原MSE、LR1e-5、micro4/effective16、真实历史50/50clean/weak，实际micro clean5958、weak6042。没有生成历史、区域损失、新物理、新数据、结构、服务器或UI模型切换。

起点和最终完整评估48训练+24验证+3旧失败例，中间250/500/1000/2000只评预先固定12训练例。最终12例另有两个预冻结噪声，起点同种子配对。所有图保留相同案例、动作、批槽和实际epsilon；无输出修复。

## 单步改善明显，C仍严格低于0.95

以下是所有强H1目标各自给四帧真实历史的独立单步，不是闭环16步。像素单位均为96×96图像。

| 训练字母 | 强H1数 | 起点IoU→最终 | 起点位置误差→最终px | IoU≥0.95 |
|---|---:|---:|---:|---|
| O | 152 | .902704→.956125 | .223622→.140389 | 通过 |
| C | 152 | .902829→.9498276392 | .264735→.182026 | 未通过，不能四舍五入放行 |
| B | 140 | .924309→.962920 | .214034→.120617 | 通过 |

平移主导目标最终位置均值O .145775/C .187570/B .120621px，均达到≤.3px；相对复制前帧的位置误差下降约84–90%，旋转主导目标的轮廓误差也下降超过50%。这些支持单步局部预测已学到部分变化，不能替代方向、刚性和闭环验收。

## 连续预测仍不够准确

AR只在起点给四帧真图，之后使用原始float预测反馈，没有未来真值回灌。最终各字母H4/H8/H16平均IoU均未达到0.90：

| 字母 | H4 IoU / 位置px | H8 IoU / 位置px | H16 IoU / 位置px |
|---|---|---|---|
| O | .852857 / .468646 | .772095 / .868091 | .662086 / 1.613411 |
| C | .825599 / .489257 | .738944 / .957657 | .553690 / 2.020358 |
| B | .886628 / .300906 | .822656 / .504492 | .716530 / .994098 |

H16起点→最终IoU为O .299594→.662086、C .266712→.553690、B .438732→.716530；位置误差分别4.915337→1.613411、6.358006→2.020358、2.887701→.994098px。H16 IoU逐例：O15改善1退步，C14改善2退步，B15改善1退步，改善没有掩盖坏例。

全48段达到IoU≥.85的比例H1/H4/H8/H16为100%/58.33%/37.5%/14.58%，后3个时距远低于90%。H16位置P90分别O4.386862、C5.054510、B1.638454px。最差训练O为HMF-AC-O-02-27-a3，IoU .216667/位置5.590012px；最差训练C为HMF-AC-C-04-10-a3，IoU .162857/位置6.500723px，均保留原图和视频。

## 棱角、方向、孔洞和蓝推杆

训练H16原始轮廓综合误差O1.776242→.660962、C2.984491→.881087、B1.226500→.577309px，约下降62.8%/70.5%/52.9%。这个量混合位置、朝向与形状，不能把它称作“棱角保真提升了70.5%”。固定O/C图仍能看到欠转及角部偏差。C的开口孔数/连通性通过也不证明开口形状和尖角保持正确。

本轮图像轴量测的可靠分母很小。训练GT→GT可判局部旋转仅O3/94、C0/99、B1/120，累计仅O1/211、C0/183、B11/213；其余是量测unknown，不能直接算模型反向或没有能力。最终可判累计B有9个时刻方向正确，其中6个角误差≤5°；这不是9条完整轨迹通过。多数方向与刚性条件仍不可验收。原O失败例另按此前已冻结的O专用灰色轴+真正封闭孔轴对保存RGB复核，最终H16约−12.000°/−14.077°，真值+50.077°，仍反向、首次可判在h7。该补充没有改本轮generic-axis unknown或验收门槛，也没有新增推理。

训练B的16×16目标以及seen-val B的8×16目标，在RH/AR中都没有同帧可见孔数不匹配或孔洞未知。训练B仍有一个RH时刻新增1个tiny灰像素（HMF-AC-B-02-25-a3 h10），两个孔和主要连通组件仍在，不能误称为丢孔。原B失败例闭环h14/15/16仍为1孔，而真值2孔；RH16个独立目标孔数正常，不能串成闭环已修复的证据。蓝推杆可见性和位置较好，不能据此证明无穿透或接触物理正确；本轮没有新增物理验证。

固定12例额外两个噪声均未通过完整条件，全部输出保留，未择优。所有unknown不算通过。本轮3000步结果只说明当前配方下仍未完整拟合，不能据此归因架构无能力。

## 验证轨迹和旧失败例单列

| 原seen-val字母，各8段 | 强H1 IoU 起点→最终 | H16 IoU 起点→最终 | H16位置px 起点→最终 |
|---|---:|---:|---:|
| O | .945364→.945029 | .456442→.584041 | 3.042651→1.879591 |
| C | .915408→.897708 | .253279→.232526 | 3.880715→4.851748 |
| B | .930029→.939388 | .514797→.561241 | 2.108555→1.584618 |

O/B部分指标改善，C退步；不能用ALL均值掩盖。C验证H16 IoU中位仅.105764、位置P90为7.559669px。逐例H16 IoU改善/退步：O6/2、C3/5、B6/2。

原3失败例H16 IoU全部退步：O .480000→.325758、C .129353→.089974、B .371849→.281938。最终位置分别3.662547/6.498226/4.374717px。RH同一末时刻IoU却为.981685/.946602/.959502，说明给真实历史的局部结果不能证明闭环成功。

## 实际耗时、资源与核验

后台worker墙钟4123.711s（68分43.7秒），其中训练loop2273.484s（37分53.5秒）、6阶段评测函数1507.111s（25分7.1秒）、完整checkpoint保存172.677s、轻量保存3.532s、saved-only报告进程56.735s，另有启动、导入、验证、IPC等开销。此墙钟不含前期代码/选片工程准备，也不等于纯GPU计费时间；没有金额或电费账单。

首32的原外推过于保守：首批40.828s使剩训练估6751.78s，实际2200.69s，原估约3.07倍；首32到完成实际55分49.5秒，低于原86.77–202.79分范围。后续23:16快照给的30–75分估计，实际余37分16秒，落在范围内。ETA没有被当成时间上限或扩训授权。

峰CUDA allocated1.676GiB、reserved1.912GiB。最终worker资源快照共享6,334,008,614 bytes，加256MiB原子预留及guards为6,673,747,238 bytes，低于10GB；本root当时725,032,967 bytes。旧R3B失败33.047s仅历史，不混本轮；零新采样/积分/生成历史训练。

模型调用合计训练12000 batch/48000 image forward，评测5952去噪batch/23616 image forward，合71616图像级前向；评测7872预测帧，含额外噪声1536帧。完整恢复和6次评测RNG/progress保护通过，所有loss和reservation精确对应冻结3000批。final/latest字节SHA为eef20649a528d896a7721f810dd5ad77e37924a02a1da2764379dee5d9364641。best-fit也是update3000，保存早于finished元数据最终化，文件SHA不同，未声称序列化文件完全相同。

终态审阅仅用已保存数据，无Torch/模型/GPU/物理/新渲染。75原始来源、492预测archive、124输入证明、1984组实际epsilon记录及188个报告文件已核，175PNG、8段报告视频和48段GT视频可读且10fps。所有75例和额外噪声都在索引内。原support.stats.worst字段对IoU实际取max，不能当最差；本审阅采用P10/min和已核实的真实最差例表，不修改原结果。

## 查看结果

固定首例是训练前预定的每字母第一例，未按新输出换图。列依次为GT / copy / 起点0 / 250 / 500 / 1000 / 2000 / final3000；行H1/H4/H8/H16为0.1/0.4/0.8/1.6秒。全部75例在索引中，中间未评的格子明确留空。

- 完整图视频索引 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/index.html`)、原完整报告 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/REPORT.md`)、逐项标准结果 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/support-signals.json`)
- O训练固定图：AR16 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/AC-O-01-25-15aa8880-autoregressive-H1-H4-H8-H16.png`)、真实历史单步 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/AC-O-01-25-15aa8880-real_history-H1-H4-H8-H16.png`)
- C训练固定图：AR16 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/AC-C-03-01-e2438774-autoregressive-H1-H4-H8-H16.png`)、真实历史单步 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/AC-C-03-01-e2438774-real_history-H1-H4-H8-H16.png`)
- B训练固定图：AR16 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/AC-B-03-27-1e8aecaa-autoregressive-H1-H4-H8-H16.png`)、真实历史单步 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/AC-B-03-27-1e8aecaa-real_history-H1-H4-H8-H16.png`)
- 训练闭环10fps视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/train-autoregressive-10fps.mp4`)：O首例0–1.6s、C25.6–27.2s、B51.2–52.8s；训练单步视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/train-real_history-10fps.mp4`)
- 验证闭环视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/seen_val-autoregressive-10fps.mp4`)、旧失败闭环视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/original_failure-autoregressive-10fps.mp4`)、固定12例全部噪声视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/report/media/fixed12-three-noises-autoregressive-10fps.mp4`)
- 执行费用审计 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/postrun-training-audit.json`)、正式媒体审计 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/postrun-formal-media-audit.json`)、拟合诊断与量测限制 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/high-motion-fit-20261101/postrun-fit-diagnostic-audit.json`)

完成实际回传后状态为waiting_user_review。仅清理已结束的本轮单次任务注册和删除负责人检查；没有继续训练、扩采、改配方、服务器或UI切换授权。
