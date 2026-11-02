# R2C1父派发原文（2026-10-29）

来源：制定 Tdyssey 世界模型实施计划，对话__LOCAL_CHAT__。当前人类监督计划已明确授权本阶段实现并执行一次，不需再次询问启动。

R2C0父审通过。现在明确放行R2C1：完成真实native predictor/trainer入口，并执行一次有界3060Ti本地训练和同窗评估。不是仅写下一份提案；必要接口完成、CPU契约检查和资源检查通过后即可运行，不用再请求本轮启动许可。完成后主动回传并停止。

父审证据：
1) 父从53原NPZ独立复算全部2067个H1/H4行，幅度/旋转/80%分类一致；逐项调用实际Dataset并与原数组比对1113窗的5RGB帧/4动作bitwise一致。
2) 4800个draw逐个映射正确，全来自TRAIN861窗；10周期均每字母64明显+16other。TRAIN41/val12，36评估窗cohort/历史/动作/target对齐无交叉，natural/motion anchor去重。4800类计数1676translation_only/2121rotation_only/43both/960other，与回传一致。父看了train、val-motion、other实际输入预览。
3) 数据/评估桥源已读。父还直接核对原GPC phase-two train.py使用utils.TrainDataset：其RGB是[0,1]、action min0/max511，与当前一致；通用data.Episode.load的[-1,1]不是该训练入口，不能据那个类另改本路径归一化。原denoiser输出wrapper[-1,1]语义不修改。
父审在r2c0/parent-review/independent-audit.json及stage-decision.json。draws4800 SHA e4ee1035144ec053245f900c5b03fd2afe56365ec69db079647f69bcc1a0a0fc；split0b05ad6ff820e34c0f4c6e3a0c961bee96da70d0adbd3f8c35e068d5861c79c0；eval indices48829714f3739276a79f4f9740600c3e70d3831a5a6f4b8ea7bc96fda517bcb5均冻结。

本次执行上限/固定设置：
- 最多300次新的optimizer ATTEMPTS、最多1200秒全部模型相关工作，任一先到停止。加载/模型构造/原始GPC H1H4推理/所有训练microbatch/最终H1H4/保存/必要重载/失败清理均计时。最晚900模型秒停止新attempt，至少预留300秒尾部；按baseline实际耗时估算如尾部需要更多，只能更早停训练。不要承诺完成300更新，不超时“补完评估”。
- 这是剩余全局3000/5400预算内的新增子预算，当前global closed250attempt/868.484000000004秒不清零；新阶段单独链接。历史另一native1899/5352.516仍另列。
- 从原本地GPC denoiser SHA4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892严格加载全部8,729,827参数，重新初始化optimizer/scheduler/scaler/RNG；不用旧250权重/优化器。原架构、原H1 residual objective、3步native sampler不变，无LoRA替代、无冻结主干、无motion/pose头/几何loss、无输出修形改色。
- 采用已提案effective16/micro4×4、AMPfp16/FP32loss、AdamW1e-4/betas(.9,.999)/wd.01/eps1e-8、GradScaler128、clip1、cosineTmax3000 eta1e-5、seed5478。所抽的是已冻结4800顺序前缀，实际少跑则报告实际前缀分布。别因时间不足换更容易样本或跳过other。

依次完成：
A. 创建R2C1新配置/目录和显式runner，不修改冻结R2C0提案、indices、source-snapshot或旧实验结果。实现already-normalized predictor桥和新训练入口；不调用旧18/36、250/500限制的入口然后动态改私有globals/FunctionType绕过。原模型源码及数学不改。源闭包、配置、初始权重、采样/评估身份在首个真实模型操作前冻结。
B. CPU接口检查聚焦真实风险：新bridge动作恰归一一次（接口已是[-1,1]，直接交native，不进旧absolute wrapper，不逆变换再归一）；history4/actions4形状与时序；新trainer只传image/action且micro4正确累积成effective16；预算在每次真实操作前检查/attempt在draw前预扣；完整检查点身份/状态。只做必要检查，不额外运行GPUbenchmark/模型预热/真实forward做“测试”；任何真实模型调用都从1200秒算起。保存此时GPU占用与存储前检；依提案CUDA reserved<=6GiB。OOM/非finite/错误shape/工程异常时保留原证据与费用并停止，不自动改batch/重跑。
C. 实际运行：
  1. 开始子预算，严格加载原始权重，记录所有初始tensor摘要和trainable参数/数量。用同一原始模型做冻结36窗的original GPC H1/H4；copy-last只用原输入，无模型或额外物理。H1复用H4第一步，baseline/final相同3步sampler、精度、batch4、cohort/order/配对seed。eval在no_grad/eval模式，保存/恢复训练用RNG，别让baseline调用次数改掉训练RNG/draw序列。
  2. 原GPC输出质量差、颜色不对或有限OOR是评估结果，按既有协议记录失败指标但继续本轮，不重演早先finite OOR导致整轮停训的问题。有限OOR原值照样回灌；真正nonfinite、错误shape、OOM/工程异常或预算不足才中止对应模型工作并保存缺测分母。原baseline不得被修色/裁剪成“合格”。
  3. 进行授权内训练，使用R2C0 PreparedInputs的纯TRAIN image/action。记录每attempt及complete/skipped状态、实际draw ids/运动幅度/contact/letter、loss、grad norm、LR、scaler、显存和累计模型秒；microbatch loss按effective batch正确加权。全模型train模式与完整参数梯度/更新需审计，不把denoising loss下降当运动预测成功。检查预算必须预留安全保存与最终评估时间；没到300也可因时间正确结束。
  4. 到最后完整optimizer边界保存一个完整checkpoint，包含model/optimizer/scaler/scheduler/CPU+CUDA及其它训练RNG/dataorder queues+cursor/PreparedInputs完整状态/配置源数据评估SHA/attempt与completed及费用链接。midmicro数据恢复不是梯度恢复，不把半步当完整检查点；异常时partial状态与费用另存。只保留本阶段一个完整payload，旧实验payload不动；不用训练loss挑所谓best模型。
  5. 用训练后最后完整权重做同36窗H1/H4评估，在预算内完成必要一次checkpoint重载验证，不能因重载/评估再开无限预算。baseline已有结果复用，不重复生成。记录最终实际操作顺序和是否验证完；若预算不足，缺项明列并停止，不能伪称已做。
D. 四步生成均须保留原输出（包括H2/H3的中间原RGB/范围/硬失败），可以在predict callback旁路持久化，不改冻结rollout数学。H4只能滑入原预测RGB，不喂未来真帧/state/pose/模板/重渲染，不加形状修复。端点指标沿用冻结协议；另列中间step失败，不能用H4最后一帧恢复正常掩盖途中失败。
E. 交付以“是否学会灰色物体运动”为主：
  - TRAINvisual、seen-val natural、seen-val motion三个冻结组，H1/H4分开，copy-last/original GPC/trained三法同窗。灰物体前景RGB误差/IoU/轮廓、逐窗胜负与relative improvement、near-static/small drift、失败和unknown数量齐；蓝推杆单列，不用蓝推杆改善替代灰物体。
  - 不删失败样本或只报有效子集均值。报告冻结分母、可评/失败/缺失/unknown，说明val都是6seen字母且motion-enriched轨迹、部分窗口重叠，非8未见泛化。
  - 主图用清楚中文展示真值/复制上一帧/训练后预测，H1和H4清楚标出；原始GPC放明确标“原始T任务GPC”的独立baseline页，避免红色原baseline混成当前训练输入。完整36窗及失败都提供，别只展示好例子。轮廓磨圆/增减块/断连/遮挡未知要如实呈现，fail=False不能写成刚性保形通过。
  - 本轮是有界学习诊断，没有自动扩训许可。即使训练loss下降或某组胜过copy，也必须回传等父审；若仍不如copy，报告实际draw运动分布和逐窗结果，不能再笼统归因“大多数数据不动”。

资源与结束：
- 0新physics/采集/渲染、0未见字母访问、0服务器/安装下载/上传包。已选0.1s不再验证。物理累计41976/76536及旧余24/320不动。
- 新R2C1持久产物<=256MiB（含一个完整checkpoint、所有方法四步原预测、图表/日志/源码）；全部旧新native/源/cache/父审+256MiB原子写入预留<=3GiB，操作前/保存前复核。不要删任何旧数据/权重/失败。
- 本次“实现必要入口→运行一次授权内训练→评估保存→主动回传”已全部放行，不要停在actual_model_entry_ready=false再次问是否可以启动；若有实质无法在授权范围解决的错误，则保存并报告具体错误。
- 完成后报告实际attempt/completed/费用/实际数据分布/同窗指标/全部预测页面/完整checkpoint及可复现命令，主动回传父并保存receipt，waiting_parent STOP。不得自动续训、重新起跑、扩采18字母或开server任务。父本轮派发后停止，不轮询。
