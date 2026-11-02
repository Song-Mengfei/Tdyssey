# O/C两轮续训：结果、归因和限制

两轮完整完成，新增61,870次样本曝光、3868次成功更新、AMP跳步0，谱系9433→11367→13301，零新物理/新样本。更多同配方训练仍有学习收益，但O/C强运动第3轮退步，递推误差仍明显；不能归结为“只要继续训就能解决”，也没有证明容量失败。

完成时间：2026-11-01 07:47:36（新加坡）。实际全流程8小时35分12秒；此前2–3小时估计明显偏低。两轮训练循环分别4小时12分26秒、3小时50分59.9秒，包含加载、校验、保存和CUDA；现有日志不能进一步确定变慢原因，不能把墙钟当GPU核时。

## 正式48家族结果

同一48个seen-val家族、同动作/噪声、正式3次去噪，原第1轮预测直接复用。下表均为H16；位置是96px图像可见灰质心误差，不是真实位姿误差。

|组别|N|IoU 第1→2→3轮|位置误差px 第1→2→3轮|
|---|---:|---|---|
|全体|48|.5972→.6306→.6226|2.132→1.819→1.768|
|O全部|8|.5716→.641→.624|2.056→1.659→1.812|
|C全部|8|.558→.586→.525|2.576→2.061→2.262|
|O强运动|4|.4625→.5318→.4757|2.777→2.386→2.955|
|C强运动|4|.2319→.2761→.1875|4.881→3.863→4.197|
|B全部|8|.5088→.5744→.5305|2.934→2.480→2.714|

O/C强运动合计8例IoU .3472→.4039→.3316。第2轮在这些开发评测上较好，但不是所有指标/未来场景的已证最优；不自动更换用户模型或继续训练。O示例仍少转，C仍有长程形变；B有一例第3轮预测孔洞0、真值2，不能只报质心改善。

**原生成报告的角度表头“known n”仅指有数值，并非角度可辨识或形状合格。** 第3轮H16形状未知为37/48，O为7/8、B为8/8。原始角度诊断受遮挡、对称性和变形影响，不能据其均值下降声称欠转修复；未知不算通过。轮廓误差混合位置、朝向和形状，不是独立棱角保真分数。没有物理穿透修复或策略提升结论。

## 真实历史单步与连续递推

固定的运动诊断组为训练O/C各8、seen-val O/C各8，按真值接触和旋转选例，不按模型质量，偏向运动；训练和验证分开，非新自然测试集。两种方法按同一目标时刻、动作和噪声配对。

|组别/方法|N|t+16目标图IoU 第1→3轮|位置误差px 第1→3轮|
|---|---:|---|---|
|训练/真实历史单步|16|.978→.981|.095→.105|
|训练/AR16|16|.393→.425|4.335→3.356|
|seen-val/真实历史单步|16|.969→.976|.109→.079|
|seen-val/AR16|16|.570→.626|1.711→1.193|

这里“t+16真实历史单步”每次都得到该目标之前4帧真实历史，**不是仅起点历史的H16闭环预测**。AR16仅起点4帧为真实，随后原始预测回灌，无未来真值或修形。相同目标下的差距支持递推历史误差是因素；后段有些真值运动已变小，不能把后段高单步分数当所有高速接触都解决。训练C诊断组的AR16 IoU仍有退步，不能只看合并均值。

固定样本/固定扩散噪声残差loss：训练 .00139060→.00133907→.00128330；seen-val .00076707→.00065084→.00062763。支持当前选定样本上继续学习；这组不支持简单“训练好、验证持续恶化”的过拟合模式，但不能排除局部退化或证明全局收敛。不能以两组绝对loss高低直接比较泛化难度。

以下补图仅在既有32个冻结运动诊断例中，按各split/字母的真值H16绝对转角最大选择，不按预测表现，未增加模型调用。seen-val O真值转角约-32.04°、C约-56.16°，真实历史较能跟随，AR16仍有方向或角部变形。

O明显旋转的真实历史/AR补图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/postrun-media/diagnostic-seen_val-O-max-truth-H16-rotation.png`) · C明显旋转的真实历史/AR补图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/postrun-media/diagnostic-seen_val-C-max-truth-H16-rotation.png`)

## 3次与16次去噪

仅第1轮已评测的O4/C4强运动例：H16 IoU .3472→.2839，位置误差3.829→4.676px，16次没有救回这8例，反而总体更差。它是推理因素对照，不是训练收益，正式默认仍3次。原3步输出直接复用；按冻结seed/原B4 shape/slot重建并注入初始噪声，仅8例增加模型推理。旧初始epsilon没有单独保存，B1/B4数值舍入也可能不同，不能把所有差异都唯一归因于去噪步数，更不能外推所有场景。

## 已保存的图和视频

每张板按H1/H4/H8/H16显示同案例。模型预测没有形状后处理，显示转换不回灌。

|内容|O|C|
|---|---|---|
|第1/2/3轮正式对照|O图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/formal-O-epoch1-2-3.png`)|C图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/formal-C-epoch1-2-3.png`)|
|正式对照视频|O 10fps (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/formal-O-epoch1-2-3-10fps.mp4`)|C 10fps (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/formal-C-epoch1-2-3-10fps.mp4`)|
|seen-val真实历史/AR|O图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/diagnostic-seen_val-O-true-vs-ar.png`)|C图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/diagnostic-seen_val-C-true-vs-ar.png`)|
|训练集真实历史/AR|O图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/diagnostic-train-O-true-vs-ar.png`)|C图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/diagnostic-train-C-true-vs-ar.png`)|
|第1轮3/16去噪|O图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/inference-only-O-epoch1-denoise3-vs16.png`)|C图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/inference-only-C-epoch1-denoise3-vs16.png`)|

B孔洞失败对照 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/formal-B-epoch1-2-3.png`) · B视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/formal-B-epoch1-2-3-10fps.mp4`) · 全部六字母、诊断和视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/index.html`) · 原完整机器结果表（角度known n须按上文解释） (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/REPORT.md`)

## 完整性、恢复和资源

第2、第3轮各30935唯一窗口/1934批、真实成功更新1934、AMP跳步0；末批7样本，没有重复凑满或凑成功数。每轮重排不同，5次评测的RNG前后哈希一致、模式恢复、优化器应用0、训练cursor未改。起点完整model/Adam/scaler/global+conditioning RNG相等恢复，旧AR1 progress侧标志未用来重跑。

336个冻结绑定均匹配，144份正式预测、128份诊断、8份16步预测哈希匹配。12张板和7段视频均存在；视频实际解码16帧、10fps。未加载模型或新增GPU评测来做这次结果核对。

最终共享占用2,387,405,424bytes，仍在10,000,000,000bytes内并保留256MiB原子余量。CUDA峰值allocated1.664GiB/reserved1.883GiB。新物理费用0；旧R3B失败33.047s保留为历史，不混入本轮。

第2轮完整checkpoint (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/training/epoch2.pt`) SHA256=c3ba4099b42f054b4cc60a675e93265d8aee8649e9a7825b3ad70a269e1fb187。

第3轮完整checkpoint (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/training/epoch3.pt`) SHA256=a25f8dfbee80d7b724baaf22c09a38c8f6650e3345048121cf9c7630d79c8b10；last与第3轮一致。

依据：曝光/恢复/RNG/开销核对 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/postrun-training-review.json`) · 正式评测、未知项及文件核对 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/postrun-formal-review.json`) · 真实历史/递推/固定loss/16去噪核对 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/postrun-diagnostic-review.json`) · 实际RNG恢复证明 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/training/rng-preservation.json`) · 原结果清单 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/report/final-manifest.json`) · 阶段终态资源收据 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/oc-continuation-20261031/background/reporting-storage.json`)。

本任务停止在结果审阅，不自动第4轮、补采、生成历史训练、新loss/LR/结构或服务器操作。完成实际回传后状态waiting_user_review，并关闭负责人完成检查及已终止单次任务注册。
