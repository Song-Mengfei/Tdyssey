# Four-step generated-history A/B: final reviewed result

2026-11-01 Singapore. Run GHAB4-20261101-a1e93e676f804ff68b9aaf5d99b24383. Config SHA256 56927d96995d2582263ba730ab227ab923554610128d7ce32a75f8dbf9ededef.

本轮完整完成：A/B 合计60,544次监督目标曝光、3,784次成功更新，AMP跳步0，零新物理/采样。B改善总体递推和部分C形状，但O强运动没有超过共同起点，运动验证组的单步精度及部分落点反而变差，不能视为棱角、旋转或孔洞问题已经解决。

A在每个目标使用真实历史；B在线从独立纯噪声生成P1/P2/P3，detach后回灌。两组都完整恢复同一个epoch2 model/Adam/scaler/global+conditioning RNG，原物理、split、六字母、96px、原残差损失、LR1e-5、AMP、有效目标batch16和3次去噪保持。每组7568段/1313家族/30272唯一目标，各1892批；首32批计入总数，没有重跑或补成功数，663短尾H1窗口同时排除。B额外生成22704历史帧、68112图像级denoiser前向，因此监督量相等不代表计算量相等。

## 正式48家族：共同起点 / A / B

下表是H16原始自递推。IoU越高越好，可见质心误差px和轮廓距离越低越好。它们混合位置、朝向和形状，不能单独量化刚性或棱角保持。

|组别|n|IoU 起点/A/B|质心误差px 起点/A/B|
|---|---:|---|---|
|全部|48|.63063 / .64149 / .68678|1.81880 / 1.70827 / 1.39676|
|O强运动|4|.53178 / .49433 / .49596|2.38583 / 2.91748 / 2.89173|
|C强运动|4|.27607 / .20844 / .28939|3.86279 / 4.22466 / 3.59699|
|B字母|8|.57442 / .56997 / .62479|2.48026 / 2.55794 / 2.17845|
|T|8|.75352 / .78825 / .84213|1.21261 / .87679 / .72981|
|L|8|.69577 / .76211 / .81075|1.28003 / .85460 / .54720|
|X|8|.53343 / .54755 / .57257|2.22056 / 1.91866 / 1.34290|
|冻结静止子集|6|.94716 / .94898 / .96246|.13523 / .14117 / .12088|

总体B对A的H16 IoU增加.04529，质心误差下降18.2%，轮廓距离.86929→.74578。48例中IoU37改善/1平/10退，质心31改善/1平/16退。IoU中位数.70829→.75234，P10 .29169→.35009；质心中位数.87463→.78382，P90 3.90809→3.70507，但最大误差7.56156→8.63538，尾部最坏例仍退步。完整H1..H16逐例和均值/中位数/尾部在原报告JSON，不能只展示均值。

总体H1/H4/H8/H16 IoU分别为：起点.98118/.85030/.75116/.63063，A .98273/.86307/.76441/.64149，B .98591/.89357/.80338/.68678。正式48例的起步运动偏小；B的H1 IoU仍略低于复制上一帧.98596，位置.06819px也略差于复制.06609px，不能据此证明明显接触运动单步已解决。静止六例虽B优于A，复制上一帧仍为IoU1和零位置误差。

O强4的H16 B仅比A高.00163，仍低于共同起点；轮廓距离A .99688→B1.02546反而更差，逐例IoU2好/2坏。O示例H8/H16欠转仍明显。C强4的IoU较A增加.08094，质心误差下降14.9%，轮廓距离2.70392→2.31625；3好/1坏，最差IoU仍只有.08625。当前图可见B减轻了A的L形塌陷，但不足以称为刚性保形或正确旋转。

预先固定的正式O/C/B三张展示例，恰好都属于B对A的H16 IoU退例，不能拿来冒充总体改善代表，也不能删除。O01-08起点/A/B IoU为.4973/.3671/.2436，位置2.097/3.883/4.860px；C01-24 IoU为.1290/.1300/.0895，位置5.376/5.146/4.465px，属于落点较近但重叠更差的混合结果；B00-08 IoU为.3640/.2318/.1935，位置4.435/7.562/8.635px。相反，正式C05-00:a35的IoU .5035/.2883/.5861，完整逐例表保留这一改善例及所有其余结果，未另挑有利预测生成新图。

B字母7/8例H16 IoU较A更好；默认B00-08例A预测0孔/真值2孔，B保留1孔/真值2孔，仍不合格。B组最大位置误差8.63538px比A7.56156px更坏。B还在X01-16例凭空产生1个可见孔洞（真值0）。所有正式H1/4/8/16没有完整蓝推杆消失，但仍有错位、接触遮挡，不能推论穿透修复或动作响应已正确。

蓝推杆H1平均位置误差A .14612→B .15783px，7改善/29平/12退；H16 .38591→.26562px，30改善/8平/10退。即使总体灰物体H1改善，也不能把蓝推杆或明显运动单步的代价隐去。

## 32个既有运动诊断家族：代价不能被总体均值掩盖

固定训练O/C各8、seen-val O/C各8，按真实运动选例，偏向运动、不是新的自然分布测试。真实历史每次给当前目标之前四帧；t+16这一列仍是单步归因，不能称为只给起点历史的H16预测。AR16只有起点历史为真，之后原始float预测递推，无未来真值、显示裁剪或修形回灌。

|组别/方法|n|IoU 起点/A/B|质心误差px 起点/A/B|
|---|---:|---|---|
|训练 AR16|16|.39069 / .41175 / .45246|3.89455 / 3.74937 / 3.47337|
|seen-val AR16|16|.59034 / .64663 / .66764|1.59255 / 1.13916 / 1.31863|
|seen-val O AR16|8|.61861 / .70263 / .71280|1.61275 / .96825 / 1.03605|
|seen-val C AR16|8|.56207 / .59063 / .62249|1.57236 / 1.31007 / 1.60120|
|seen-val 真历史、t+16目标的单步|16|.97753 / .97829 / .97140|.08681 / .06680 / .08727|
|seen-val 真历史、起点H1运动目标|16|.90839 / .91300 / .89310|.25262 / .27039 / .31074|

seen-val AR16虽然B的IoU比A好，但位置误差更差15.8%；逐例位置仅5/16改善、11/16退。起点明显运动H1单步B也比A差：IoU下降.01990，质心误差增加14.9%。因此本轮没有完整实现“改善递推同时保持运动单步”的预期支持信号，不应自动推广。训练组改善不能替代验证结果，也不能从真实历史/递推差距断言唯一原因。

所有正式48例角度仍按未校准处理，H16形状未知37/48，B字母8/8未知；数字角度、角点数、孔洞差为0都不能作为刚性、穿透或欠转修复的独立证明。O/C强运动每组仅4例，共同起点epoch2依据开发验证挑选；这里只是一轮已见字母开发A/B，不是未见字母泛化或收敛证明。四步B的第4步历史仍留G(t)一帧真值，不能保证16步稳定。

## 实际费用与核验

后台整轮实际4194.488s，即69分54.5秒，15:43:40至16:53:35 Singapore完成。A/B训练循环分别1291.207s（21分31秒）/1593.091s（26分33秒），含I/O、检查和循环内周期保存；不能再把累计checkpoint计时全量相加。原32批训练外推约39分钟偏低，实际训练循环合约48.1分钟。三次受保护的正式/诊断评测区间合879.707s（14分40秒），混合GPU推理和CPU保存评分；其余为启动、恢复、循环外保存、完整文件/空间核验、saved-only报告和进程阶段交接。这里没有纯GPU内核/CPU占用 profiler，不编造硬件原因或按模型墙钟冒称纯GPU费用。

GPU峰值分配约1.671GiB、预留1.889GiB。最终原报告资源收据共享4,793,507,095bytes，通过10,000,000,000bytes上限（含256MiB原子预留）。1.674GB uint8缓存计入其中，无全库float复制。缓存建立58.884s和CPU预检属于工程准备；实际GPU原生采样等价预检1.048s、8预测图像/24图像级denoiser前向、0更新另计。旧R3B失败33.047s仅作历史费用保留，不混入本轮。

3107/3107冻结绑定匹配，两组日志各1892行batch0..1891无重放；五次完整共同起点/32批恢复通过，四次预检/评测RNG保护通过，逐损失监督branch及最终RNG相同。P未来标签扰动隔离/每损失梯度检查是CPU toy调用实际生产函数及原native数学，实际完整网络GPU检查验证原生采样等价；不冒称另做了完整网络GPU未来标签扰动实验。原始数据、旧epoch2/3、检查点和证据均保留，未安装/替换环境或切换用户UI模型。

## 可直接查看的同案例原始预测对照

每张图/视频四列依次是真值、共同epoch2、A、B；图突出H1/4/8/16，同时保留完整16帧图和正常10fps视频。默认正式O/C按冻结强运动ID，B按原首ID；诊断按原32例中真实H16转角最大选，不看模型预测挑有利例。

- O正式对照 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-O-H1-H4-H8-H16.png`) · 完整16帧 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-O-all-H1-H16.png`) · 10fps视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-O-10fps.mp4`)
- C正式对照 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-C-H1-H4-H8-H16.png`) · 完整16帧 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-C-all-H1-H16.png`) · 10fps视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-C-10fps.mp4`)
- B孔洞/缺块对照 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-B-H1-H4-H8-H16.png`) · 完整16帧 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-B-all-H1-H16.png`) · 10fps视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/formal-B-10fps.mp4`)
- seen-val O明显运动AR对照 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/diagnostic-seen_val-O-autoregressive-H1-H4-H8-H16.png`) · O真实历史单步 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/diagnostic-seen_val-O-real_history-H1-H4-H8-H16.png`)
- seen-val C明显运动AR对照 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/diagnostic-seen_val-C-autoregressive-H1-H4-H8-H16.png`) · C真实历史单步 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/diagnostic-seen_val-C-real_history-H1-H4-H8-H16.png`)

原完整报告，全部T/L/X/O/C/B及训练/验证图视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/index.html`) · 所有H1..16统计 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/summary.json`) · 每例改善/持平/退步 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/case-comparisons.json`)

训练独立审查 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/postrun-training-audit.json`) · 正式结果/媒体独立审查 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/postrun-formal-audit.json`) · 运动诊断独立审查 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/postrun-diagnostic-audit.json`) · 原报告manifest (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/generated-history-ab-20261101/report/final-manifest.json`)

完成后只做实际授权回传和等待审批的状态/已结束任务清理；不自动再训、8步扩展、采样、改配置、服务器或UI模型操作。最终实际发送状态以background/final-callback-result.json为准，原worker completion的callback_sent=false是发送前快照。
