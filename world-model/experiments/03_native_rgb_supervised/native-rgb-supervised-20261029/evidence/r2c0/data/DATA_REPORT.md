# R2C0引用数据与抽样准备

R2B只通过小规模学习诊断数据准入。53轨迹按完整family切为TRAIN41/seen-val12，只有新选定0.1秒协议；没有RGB副本、模型或物理调用。seen-val刻意含运动，不是无偏自然测试或未见字母。

TRAIN861 H1/738 H4，val252 H1/216 H4。明显运动定义为H1面积质心≥2图像px或abs角≥5°，其它合法窗保留。Dataset五帧是四历史输入和一个H1真值，四动作是三个历史动作和当前动作；仅image/action允许进入网络。

480周期每字母64明显+16其它；每6draw随机字母轮转，再letter/group family均衡，再family互斥运动类别/window队列。两运动条件重叠只归both一次。family类别不齐时如实列支持，不能借别的字母补类。后续RNG延续，4800不是机械复制首周期。所有重复只计exposure。

|字母|明显unique/family|其它unique/family|
|---|---|---|
|T|62/7|85/7|
|L|64/7|83/7|
|O|41/6|106/7|
|C|29/5|118/7|
|X|37/6|89/6|
|B|50/6|97/7|

首480使用372个不同窗；拟4800使用861个不同窗、最大单窗曝光64。逐16draw batch、各cycle的真实phase/contact/幅度/方向/arc/family/repeat见draw-audit.json。

split-plan.json记录seed/hash选点、TRAIN保留平移/旋转/可用正负、初态near检查与完整signature重复披露；原始source duplicate审计另见父目录source-duplicate-audit.json。H1/H4catalogue保持全部源窗口，抽样不改自然分母或原值。

300×16/4800draw只是冻结提案；300尝试/1200模型秒尚未授权，closed250/868.484旧账本不动。
