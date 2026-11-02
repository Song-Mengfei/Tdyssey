直接目标确实产生了明显运动，但本轮120次仅46条完整，74条因原观察标准“穿透>1世界像素”终止；模型未训练、未测，不能声称预测或棱角改善。

已结束并保留全部失败/合法前缀，没有补采、替换、放宽阈值或改原物理。采样的旧12/2/24/24限制已移除，启动包装器Job问题已修复；本轮没有继续修改采样条件。

运行：DTA-20261031-7b4e4819a73b41a78dba24947fd6579b；配置SHA256=205d313d2feced75db088c9120ede0f9cd9738a059e1af14d584b07f95be643c。2026-10-31 15:31:46新加坡时间完成。

全120尝试、96独立家族；3446实际控制/3372合法控制/33992原生子步，初始化0。正冲量AO接触132.32/339.92模拟秒=38.93%；24条完全无正冲量AO。低阈值接触且推动48.2秒(482控制)，接触但受阻/小运动87.5秒，静止170.3秒，推杆空移30.5秒，物墙且无正AO0.7秒，非法末控制2.72秒。不能把38.93%接触或低阈值推动当成明显运动达标率。

H1/H4/H8/H16完整合法窗口分别3012/2808/2566/2117，对应69/61/57/45独立家族。共同H16锚点2117个，四个H均用这一锚点集合另报配对结果；不是2117个独立场景。按各H自己的窗口分母，明显平移/旋转计数分别112/135、227/265、163/219、83/119；展示档保持H1=1px/3度、H4=2px/5度、H8=4px/10度、H16=8px/20度，平移旋转分开，数据未被按此过滤。

|字母|完整/20|AO正冲量接触比例|H16配对窗口/家族|H16净位移px P10/P50/P90|H16净转角度 P10/P50/P90|
|---|---:|---:|---:|---|---|
|T|10|43.79%|460/9|0.000/0.000/3.538|0.000/0.000/8.238|
|L|6|22.19%|277/7|0.000/0.000/0.664|0.000/0.000/6.327|
|O|7|38.74%|322/7|0.000/0.000/4.130|0.000/0.000/7.162|
|C|8|52.05%|368/7|0.000/0.000/6.926|0.000/0.000/14.029|
|X|7|27.94%|322/7|0.000/0.000/1.714|0.000/0.000/11.739|
|B|8|43.84%|368/8|0.000/0.000/4.738|0.000/0.000/13.524|

P50约0说明总体仍被静止/小运动窗口主导；确有强运动例子，不能把全部轨迹说成没动，也不能把精选片段当总体改善。本轮场景与旧cal2不配对，不给伪造的改善百分比。

每字母按真实运动固定规则选出的代表视频/图如下（均为真值，不是预测）：
- T：平移H16=16.04px视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-T-00-1-b0.mp4`)；同锚点真值图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/truth-boards/DT-T-00-1-b0-t4.png`)；顺转H16=28.67度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-T-01-1-b0.mp4`) / 逆转H16=-36.57度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-T-02-1-b0.mp4`)
- L：平移H8=6.34px视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-L-04-1-b0.mp4`)；此短前缀无完整H16逐帧图；顺转H16=52.26度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-L-01-1-b0.mp4`) / 逆转H16=-28.30度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-L-02-1-b0.mp4`)；顺转真值图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/truth-boards/DT-L-01-1-b0-t3.png`)
- O：平移H16=10.62px视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-O-01-1-b0.mp4`)；同锚点真值图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/truth-boards/DT-O-01-1-b0-t3.png`)；顺转H16=50.01度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-O-01-1-b0.mp4`) / 逆转H16=-37.86度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-O-02-1-b0.mp4`)
- C：平移H16=22.45px视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-C-00-1-b0.mp4`)；同锚点真值图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/truth-boards/DT-C-00-1-b0-t4.png`)；顺转H16=30.55度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-C-01-1-b0.mp4`) / 逆转H16=-61.48度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-C-02-1-b0.mp4`)
- X：平移H16=8.98px视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-X-01-1-b0.mp4`)；同锚点真值图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/truth-boards/DT-X-01-1-b0-t4.png`)；顺转H16=66.58度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-X-01-1-b0.mp4`) / 逆转H16=-30.72度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-X-02-1-b0.mp4`)
- B：平移H16=10.02px视频 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-B-01-1-b0.mp4`)；同锚点真值图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/truth-boards/DT-B-01-1-b0-t3.png`)；顺转H16=64.38度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-B-01-1-b0.mp4`) / 逆转H16=-42.63度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/videos/DT-B-02-1-b0.mp4`)

关键缺项：49/74非法发生在首个主段控制；非法对为推杆-物体64、推杆-墙9、物体-墙1。首例DT-T-00-0-b0在第4控制/第5子步(0基索引3/4)，AO穿透1.101771世界像素，累计真实模拟0.35秒即停止。原生动力学下这些直接目标触发既定观察门槛；这不是模型失败，也没有用动作限幅掩盖。

24组以continue为参照的分支公共4帧状态/RGB和3历史动作全部一致，但这24对没有双侧合法H16。补充枚举每个三分支家族的全部36个无序配对后，T和C各有1个turn-vs-withdraw合法H16配对：T两结果面积质心相距6.233图像像素/转角相差15.398度，C相距5.204像素/0.556度。只有2个独立初态家族的结果分离证据，三种计划均合法的家族仍为0；不能把原报告的0个baseline配对扩大解释为所有分支都不能比较。全部源/动作记录正确不等于采样覆盖通过。

真实OW接触3.38秒，其中正冲量2.32秒；各字母四边覆盖不完整，只有C有右下角的实际双墙接触，X存在0.03秒OW但没有完整合法展示窗口。再接触仅X有1条事件，H16位移3.16px/转角16.12度，未达8px/20度明显运动档，只能称再接触事件诊断；其余五字母无合格事件例子。每字母尚有3–8个外弧段未观测到接触，O/B封闭孔洞不列作缺失要求。L没有H16>=8px平移例，最大4.62px；它的平移展示是H8=6.34px短合法前缀。

CPU费用：collector墙钟875.545秒，整阶段1112.381秒；逐轨迹内部计时合计79.264秒，额外时间包含预约/目录扫描、持久化、报告和编排，未作精确CPU耗时分解。339.92是模拟秒。模型/GPU调用/优化器更新均0。旧R3B失败33.047秒仍在原报告，不能混入本轮。报告空间收据：新目录27037419字节、共享1670946485字节，随后仅增加审查/回传凭据，预算远低于256MiB及共享3GiB+256MiB预留。

完整文件：总报告 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/REPORT.md`) · 全部真值视频/图/热图 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/index.html`) · 按场景/H完整分布 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/summary.json`) · 全尝试索引 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/attempt-index.csv`) · 共同H16锚点表 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/paired-common-H16.csv`) · 全部原始数据manifest (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/raw-manifest.csv`) · 报告manifest (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/report/final-manifest.json`) · 补充全部分支配对 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/background/all-sibling-branch-pairs.json`) · 独立终态核验 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/direct-target-acceptance-20261031/background/terminal-audit.json`)

120原始文件哈希/遥测校验、42源绑定、56报告文件以及所有动作=实际传入=冻结计划均核验通过。O/C真值逐帧图已人工视觉核查。未再生成报告、未物理重放。

当前结论：有可用的明显真值运动和合法短前缀，但74/120非法、静止占比高、分支H16与边界/再接触不足；不自动放行扩大采集或训练。请父聊天仅只读整理给人类审阅。实际回传后设waiting_user_review并移除已完成的本轮检查任务，后续任何采样/配置/模型/服务器工作仍须人类明确批准。
