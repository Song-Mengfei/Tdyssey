# 本轮计算完整；存储验收失败，等待用户审阅

run_id: HID-20261102-a82470c47c4d4cef825fbccae6f7da40  
runtime-config SHA256: 4851e0154a319765398c3c2fb70df077e26e0899bfcaa97ccec9df108de22fd6

授权的A+B推理已全部完成，整体运行仍为failed=true：最终存储检查发现新目录501,952,724 bytes，超过500,000,000 bytes；随后保存失败completion后达到516,100,792 bytes。报告/完成元数据的写入保护没有覆盖全部增长。本轮没有重跑、提高限额或修改冻结源/配置。停止后只把一个21,603,513-byte派生逐例JSON附录无损gzip为3,269,256 bytes并验证解压字节一致；目录恢复到约497.8MB。原始预测raw和原失败记录保留，压缩不追溯改变失败结论。详见存储失败与无损归档证明 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/background/STORAGE_LIMIT_FAILURE.json`)。旧执行manifest保留；A-per-case-regressions.json现在是指向完整gzip和原SHA的引用说明。

计算结束时间2026-11-02 01:34:29（新加坡）。整个worker墙钟246.059s，推理阶段222.629s，报告17.301s。原生采样与噪声证据计时114.913s、评分42.262s、raw保存/hash6.870s；这些分项并非纯GPU费用。worker自身CPU user143.516s/system29.328s，单进程BelowNormal，Torch intra/inter、OpenCV和数值库线程均1，没有CPU模型回退或其他项目进程操作。旧HMF/R3B33.047s是历史费用，不计作本次推理。

实际A=3456/B=216，共3672/3672预测；11016/11016图像级去噪前向；950/950采样调用、2850/2850批前向，active_call=null。构建模型1次、加载6份state-only权重；没有额外失败模型调用、优化器更新、backward、新物理或新数据。准备阶段仅对6张旧保存图做统计一致性检查，模型调用0。原配置/原3去噪/配对噪声/原始浮点递推不变。0/final和中期train12复用原证据，未重推。

A补齐原seen-val24（O/C/B各8家族）和旧失败3例的250/500/1000/2000中期RH/AR全部16时刻。RH逐时刻读取真实历史，不能称16步闭环。下表是seen-val原始AR16平均gray IoU，按0/250/500/1000/2000/final排列：

|字母|0|250|500|1000|2000|final|
|---|---:|---:|---:|---:|---:|---:|
|O|.45644|.50239|.50584|.54534|.55028|.58404|
|C|.25328|.20060|.19555|.17253|.21195|.23253|
|B|.51480|.55965|.52740|.56556|.51868|.56124|

O平均改善但8例中2例退步；B也是6例改善、2例退步。C在首个250检查点已低于起点，最终5/8例IoU退步；其AR16可见质心误差3.88072→4.85175 imagepx，训练固定4例却明显改善，不能把训练拟合称验证通过。O/B对应质心误差3.04265→1.87959、2.10856→1.58462。原始完整H1/4/8/16、全16时刻和逐例证据均保留，不靠挑选平均数放行。

旧失败O/C/B的AR16最终IoU分别.32576/.08997/.28194，均低于各自起点.48000/.12935/.37185；中期起伏不能称修复。旧B最终可见孔洞/拓扑失败；C长期滚动仍漂移。seen-val最终AR16的形状状态O=8unknown，C=2fail+6unknown，B=1pass+1fail+6unknown；旋转O有7unknown、C/B各8unknown。可见质心/轴与IoU不证明刚体棱角、旋转方向、接触或穿透已修复。没有测T/L/X，不对这些字母作新结论。

B为12个真实强运动历史（每字母3训练家族+1旧失败seen-val家族）、2模型、3动作、3冻结噪声；216张预测。过去3动作和4帧RGB不变，仅当前第4动作改变，B4槽位/噪声配对。recorded与hold的24个模型-案例对，灰物体可见质心/轮廓/灰区RGB变化均大于本次同动作换噪声平均变化；不是只有蓝杆在变化。E2灰质心action/noise比范围3.21–28.55；final有11个有限比4.16–14.52，另1例噪声为0、动作变化为正，比值未定义，不能记无限或自动通过。final灰质心动作差绝对量.135–1.023 imagepx。蓝杆单独记录，不以蓝杆变化证明物体响应。

仅recorded有同状态GT。每字母训练分母3家族×3噪声=9图，seen-val为1家族×3噪声=3图，下表IoU/可见质心误差imagepx为这批保存指标的算术汇总，噪声图不是独立家族：

|分母|字母|E2|final|
|---|---|---|---|
|train3家族|O|.85702/.41386|.95102/.10365|
|train3家族|C|.87094/.42156|.94314/.17269|
|train3家族|B|.88020/.49753|.94091/.20566|
|seen-val1家族|O|.82782/.58179|.86425/.46437|
|seen-val1家族|C|.84627/.20218|.85430/.33283|
|seen-val1家族|B|.92959/.25830|.94302/.19510|

C旧失败例单步IoU略升但质心误差变差，不能称全面改善。B记录动作的72图形状均unknown；所需旋转均unknown，其余not_required。hold/alternative没有同状态反事实真值，因此动作敏感性不证明物理方向/旋转正确，更不证明新的训练收益或泛化。三种噪声只覆盖这12例，不代表统计充分。原unknown和验收门槛不放宽，没有选最好seed。

完整生成报告 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/REPORT.md`)、HTML图索引 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/report/index.html`)、全16步指标 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/report/A-all16-curves.json`)、动作/噪声对照 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/report/B-action-noise-comparisons.json`)、完整逐例附录gzip (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/report/A-per-case-regressions.json.gz`)、执行账本 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/inference/result.json`)、原失败completion (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/hmf-inference-diagnostics-20261102/background/completion.json`)。静态曲线和同案PNG已生成，没有新视频编码；仅引用原有视频。工程一致性检查不等同模型改善。

本轮仅回传既有结果，等待用户审阅；不续训练，不新增推理、物理、数据或UI/模型切换。最终回传和结束任务注册/监控的真实回执另存background，整体失败记录不改。
