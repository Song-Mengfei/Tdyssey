# Tdyssey 原始物理接触修复与候选采样验收

物理回归已通过；正式采集已尝试 0/360 条，尚未完成交付。

浏览完整HTML报告 (external historical reference: `index.html`) · 采集运动／接触／覆盖／视频 (external historical reference: `collection/report/index.html`) · [清理报告](cleanup/REPORT.md) · [机器摘要](summary.json)

生成时间：2026-10-31 00:18:02 SGT。这是已保存证据的只读汇总，不执行物理、推理或训练。

当前最新回归：v4 / contact-constrained-original-kinematic-rgb.v6 / passed；各版本失败与费用分别保留。

**主接触证据已纠正：** 旧B-v6控制31／32右图没有AO，仅是轨迹分离，不能作接触修复通过图。现改用B左上角t=.320/.330秒精确子步：v6同帧AO正冲量＋OW正冲量、杆中心在实体外，AO重叠约.009世界像素、OW约.032；整控制累计接触不能替代当帧判断，.340秒无AO也保留。

精确接触前后图 (external historical reference: `physics-design/contact-proof/B-TL-exact-contact-proof.png`) · 同子步事件／冲量／signed gap证明 (external historical reference: `physics-design/contact-proof/exact-frame-proof.json`) · [所选帧CSV](physics-design/contact-proof/selected-frame-metrics.csv)

另有B自由推动(.30,.60]的600子步／0.3秒连续正AO冲量，实际H1位移2.1–3.4图像像素，不是以不接触冒充修复。连续接触与真实运动图 (external historical reference: `physics-design/contact-proof/B-free-sustained-contact-motion.png`)。这些是定向例证，不是总体接触率。

B从完整构造＋3hold＋37请求动作重放：原始分支对用户370子步最大状态差0，精确再现OW18.898176/AO20.641872世界像素重叠；保存的v6分支同请求37/37完整，最大OW/AO/AW=0.000000/0.050000/0.000000，非法中心子步0。轨迹从第一控起改变，不是同中间状态原位修复。

所示为保存的v6子实验；与最新回归协议相同。 旧请求前缀所选31／32图无AO，保留为轨迹分离证据，不能当接触修复通过图。

历史请求前缀实际10Hz视频 (external historical reference: `physics-design/B-v6-before-after-10Hz.mp4`) · 控制32轨迹分离图（非接触通过） (external historical reference: `physics-design/B-v6-before-after-control32.png`)

真实IAB录制／保存／下载／打开播放已通过，raw完整性及hash均以实际回执为准。本次不变录像包含 5 次神经推理（B H1=1、C H4=4，5控制／0.5秒物理时间）；同期服务计数为 85，其中另 80 次不归本录像测试，检测到用户同时操作／重置，不能把85次全部计为测试费用。录像约94.6秒是含暂停的真实UI墙钟，不是采集10Hz轨迹时长；完整模型秒数未暴露，不编造。 存档验收时服务PID16784，模型与v6协议身份记录在after-browser-health；这是保存时快照，不实时推断用户当前场景。

正式目标360条；已尝试0条，完整0条。校准／补采与正式分列；计划不等于执行，完整不等于可训练。

实际清理7个退役checkpoint，释放596,831,938B（569.18MiB）；7499/6000/唯一初始化及关键证据保留。旧TRAIN仍依赖loader/audit，本轮未删。

物理保留KINEMATIC+damping0与原几何／质量／惯量／材料，改变积分／solver／不可行动作执行并版本化。7499模型仍是旧物理训练，属于跨物理预览；请求目标和实际端点／速度干预分别保存。公开控制状态含上次microstep ticks，内部solver缓存不序列化。

**训练更新0。** 实际回传父聊天后等待用户审批，不自动训练、扩大采样或启动远端任务。
