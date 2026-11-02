# R2C1R1 父明确批准一次纠正后重启（2026-10-29）

来源：制定 Tdyssey 世界模型实施计划，对话__LOCAL_CHAT__。原失败全部保留；本次仅花原1200秒的剩余额度，不另开完整预算，不需再次请求启动。

父已确认R2C1是启动日志错误，非数据或学习失败。现在明确批准一次纠正后重启，阶段R2C1R1；应用最小修复和必要的新运行身份/剩余预算绑定，CPU集成检查通过后直接完成原本那轮本地训练/评估。无需再请求启动许可，不另开完整20分钟预算。

独立父审：
- 我重新对实际last.pt流式内容计算SHA/大小，与b0442673691e1d87ae9aa266040d3d949bd3c59dd14ccd41e63c548d21bfa14d /35,126,889B一致，没有torch.load。
- 从实际trainer AST仅提取safe_json/event，在CPU复现了path重复绑定。应用所提两个位置改名的内存副本后，以含真实path/bytes/sha256/updates的metadata形状执行initial_t0/final_complete_boundary/error_save三个日志调用，实际JSON完整保留checkpoint path，均通过；非event AST完全一样，补丁后source SHA b88dc658aa7b5d8b88939598943eed4286b7105b6e54c8f93d0975904e9cee3e。父未修改live trainer。证据r2c1/parent-review/logger-review.json及synthetic-logger-probe.jsonl。
- 当前global active_stage=null，250attempt/250complete，895.8430000000008秒；SHA517e9a6e46c3d56d33ea3899037f910090ff749020da892cb2ea4fff9200422e。本次父授权机器记录r2c1/parent-review/retry-authorization.json。

本轮剩余预算精确为：
原上限1200秒 − 已花27.35899999999674秒 = 本次保守封顶1172.641秒；
attempt尚未使用，仍最多300；本次最多到872.641模型秒就停止新attempt，至少300秒尾部，baseline实际耗时要求更多时只能更早停。
失败启动+此次重启合计模型时间不得超过原1200秒。全局3000/5400总上限不变，不退款、不重置，失败stage仍完整留账。所有重新构造/加载、baseline、训练、保存/重载/评估/错误清理继续计费。本次运行不是免费resume。

允许修改范围：
1. 应用已审logger最小修复。保留旧r2c1目录、onceclaim、原配置/快照、错误/缺测报告及t0权重，不覆盖或删除它们。新R2C1R1目录/新配置/新一次性claim、冻结修复后的源码与原始输入身份，避免直接重复旧命令。
2. 我读到了仍硬编码旧值的入口/预算守卫，不能只改配置然后再次撞这些守卫：
   - trainer的Exactly300/1200检查；
   - run.py的300/1200/900/300字典及旧prior ledger检查；
   - budget.py的GLOBAL_SECONDS_BEFORE/PRIOR_LEDGER_SHA、make_budget精确caps及__init__旧250/868.484假设。
   允许把这些变成显式传入并严格核验本次父授权的“250/895.843 prior +300/1172.641/872.641/300 caps”和当前ledgerSHA/历史前缀；或者新建等价有界入口。原失败快照不改。不能用修改私有globals、删claim或放松全局总额绕过。ModelWorkBudget本来支持更小子上限，应保留其尾部与全局约束。新的diff除日志及运行身份/预算链接外，训练/预测/评分数学、数据/抽样、模型设置保持不变。
3. 补上真正漏掉的CPU集成边界：stub checkpoint writer返回实际metadata字段→实际save-success logger写入；检查本次所有现有展开metadata事件；用纯fake时钟/临时fakeledger检查新prior身份、1172.641上限、872.641 cutoff、>=300尾部、错误preflight不允许模型开始。不要只测函数存在/AST。不得为这些检查构造模型/加载权重/启动CUDA。保持相关测试有限，不扩成无关重构。
4. 原失败0draw/0optimizer/0baseline，直接从原始4084...权重及fresh optimizer/scaler/scheduler/RNG、draw cursor0开始最清楚；不接旧250，旧失败t0只保留证据。新轮仍需要正常initial_t0保存/日志的真实执行，之后按原顺序baseline→有界训练→最后完整checkpoint/重载验证/final同窗评估。所有真实操作记在本次剩余秒数里。
5. 一旦上述必要修复/CPU检查及存储/GPU前检通过，直接执行一次新claim；别停在“补丁准备完毕，是否允许运行”。若再次发生实质工程错误或OOM/nonfinite/wrongshape，保存已发生的真实证据和费用，停止回传，不自动再起第二轮。有限OOR继续沿原协议标记并回灌，不误当初始化/数值崩溃。
6. 原R2C0 split/4800 draws/36 eval窗、80/20策略、image/action-only全参数原GPC、effective16/micro4、AMP/optimizer、3步sampler、全部原RGB及用户选定0.1s半衰期保持不变。没有新采样/physics/unseen/server/install/download。成功完成仍按原授权交同窗copy-last/original/trained H1/H4与四步raw、实际draw/loss/grad/更新/shape unknown和checkpoint，完成主动回传停止，不扩训。
7. 新重启持久产物<=256MiB，共享所有旧数据/权重/失败/父审+256MiB原子预留<=3GiB；旧失败r2c1约49MB也必须算入不可删除部分。所有baseline/final原输出与检查点均在这个总限内。
8. 若又在任何推理前就停止，交一个简短故障报告+真实stack/费用/状态/已有checkpoint即可；无需再生成整套24页NaN占位图库。缺测用小表明确列出，不能伪装成预测。这不是删除既有失败材料，而是避免新一轮无预测时做无用展示工作。

应用修复、完成这一次剩余额度内的真实训练并主动回传。父本轮派发后结束，不轮询。
