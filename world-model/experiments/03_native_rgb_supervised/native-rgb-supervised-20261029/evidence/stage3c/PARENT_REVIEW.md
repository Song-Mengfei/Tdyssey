【父审查：3B训练链路通过，但运动/形状目标未达；仅放行3C同18样本的有界续训诊断，正式TRAIN仍未放行】

先保存本条到stage3c/PARENT_REVIEW.md并更新SUPERVISION.md和stage-state.json。用户授权的主动回传规则不变：实际发送父对话__LOCAL_CHAT__（local），保存成功、waiting_parent，然后结束。我派发后停下等待，不轮询，用户不用转发。

父审查：
我已直接从原RGB重新算mask IoU：TRAIN hold误差0.23258317101512865、fitted0.22807596210836506、9/18wins；DEV0.1960324594762658→0.27435196499414033、4/36wins，和你的报告一致。129条更新/尝试日志、2064draws/18unique、296参数变化、last SHA e94f9349e1affb8372aacc4613bd57ade496704a50b63d257f77cb4c5ad3626a已核验。已查看TRAIN原图，棱角/窄槽圆化明显，不能称保持原形。
附加CPU诊断：可见灰mask质心TRAIN误差约0.651→0.497图像px，13/18改善；DEV约0.636→0.870。这个受形变/遮挡影响的辅助量不能替代原评分、不能证明动作因果，更不能改判通过；但也不能把结果简单称作“完全复制上一帧”。目前是部分位置学习+轮廓误差+验证回退混合。

决定：
不扩大训练集、不进入阶段4。仅把同一18样本的学习曲线延长到总250个完整更新，检查129之后继续优化是否实质改善。明确这是父审查新增的一次诊断授权，原3+3B的200/600边界和实际444.625秒记录永久保留，不把它重写成“原来就允许延长”。新3C最多121次优化器尝试、最多600秒模型工作，全轮3000/5400和3GiB上限不变；不是自动追加未用预算。

依次执行：
1. 全部输入/标准冻结：沿用原TRAIN18、DEV36、truth、batch成员与seed、calibration、score protocol、原H1loss、原sampler。仍image/action→image，原全8,729,827参数，micro4/effective16、AMP、clip1、AdamW lr初始1e-4/wd.01、原scheduler T_max3000/eta_min1e-5。不能重新初始化优化器/学习率/采样RNG、换样本、换噪声策略、挑最好seed或新增任何运动/几何模块/loss。
2. 从3B update129的完整last.pt精确续接，不从原发布权重重训。保留3B文件不可覆盖。恢复model/optimizer/scaler/scheduler/RNG/WindowOrder/progress；开始时应updates129、order cursor2064、optimizer296states，下一完整更新130，累计目标250。模型构造若须先加载原发布权重再载last，全部计时且明确这是实现加载步骤，不是训练重启。
3. 原restore_checkpoint严格比较config/identity，会因新输出/阶段预算/源码版本变化而拒绝。允许最小、显式审计的continuation入口：固定parent_checkpoint SHA与原payload身份，逐项验证原语义，再只放行本指令造成的阶段名、输出位置、局部预算、续训来源/新增入口源码身份差异。模型/原loss/sampler、data/split/18和36indices、optimizer/normalization/effective batch、scheduler、随机状态语义必须一致。禁止删除通用resume验证、伪造旧identity或宽泛忽略source/config差异。记录迁移前后身份、允许的精确差异和父checkpoint来源；新checkpoint保存当前真实identity。必要的恢复适配不改变训练数学。
4. CPU/mock只做本次必要的恢复身份/预算边界/失败记账检查，防止错误恢复被放行；不另跑GPU smoke。真实恢复语义在这一次计费运行内核对，不能以ZIP字符串检查冒充实际恢复。若存在未获授权的实质差异，停止回传，不猜测放宽。
5. 使用现有全局native-rgb-cost-ledger.json，新stage条目必须从closed累计129attempts/129updates、444.6250000000073秒继续。3C local最多121尝试/600秒；model构造、checkpoint load、训练、loss/原生推理、保存/语义检查/cleanup均计时。每次尝试先预扣；失败不退费，不额外重启。旧native1899/5352.516继续单列。
6. 沿用ModelWorkBudget的<=420秒不启动新梯度及>=180秒尾部预留；预计121更新可在此内完成，但以实际预算先到为准。target updates=250，不能误把121当累计目标；local attempts<=121独立守卫，累计progress不因stage切换归零。全轮余额不是后续授权。
7. 不重测原发布baseline，也不重新预测update129。直接引用3B完整frozen和129拟合数组及hash作为比较，原首批复算已确认字节一致，无需再耗费推理。3C只在最终完整checkpoint上生成一次原18+36 H1，原14批/seed全部不变；不评H4/全319/未见8。
8. 保存本阶段last完整checkpoint。每50个累计完整更新可同名原子保存本阶段last；最终点不足250也如实报告，不另选denoising-best。只需保留3B旧129和本阶段最终完整点，避免重复大checkpoint。不复制全部数据。
9. finite越界继续按3B策略保存/status/原评分处罚但不abort；nonfinite/结构/OOM/训练错误仍停。禁止clip/affine映射/重新校准。全部原始RGB留存。
10. 评分继续原20%灰1-IoU改善+12/18wins、原失败/完整分母与形状规则，不因129结果改阈值。列出hold/frozen/update129/update最终的TRAIN18逐窗与汇总、DEV36及31moving/4static/1intermediate。轮廓/RGB误差与对象/推杆分别报告；接触导致旧shape unknown继续unknown，不能用“unknown很多”宣布形状通过。
11. 在相同18训练样本展示129和新最终预测对照，优先让所有18可复核，明确看棱角/洞/窄槽是否恢复及motion误差变化。不需要新的图像修补或诊断阈值。18和36的实际native输出才是主证据；loss下降/参数更新只证明训练发生。
12. 3C只做这一次续行。即使原目标通过也不自行进入正式TRAIN；如果未改善，诊断“在同18样本续到本次步数仍未达”并停，不能宣称整个GPC路线无效，也不自行继续500/1000步。

交付实际命令、源码/恢复身份diff与聚焦检查、完整last及SHA、129→最终更新/尝试/样本暴露/梯度/学习率/显存和时间、全部原始54 H1/逐窗指标/对照图、3/3B/3C及全轮账本。新模型总工作如满额最多1044.625秒，不含旧5352.516；尝试最多全轮250，资源仍留给后续父决策。
物理新增0，监督39680/全native74240不变；新旧全部产物+源码/cache+原子reserve<=3GiB，无安装下载、删旧权重、服务器或ZIP。
完成/遇阻必须主动回传父对话并结束。我审查后才决定是否需要进一步诊断或放行正式数据训练。
