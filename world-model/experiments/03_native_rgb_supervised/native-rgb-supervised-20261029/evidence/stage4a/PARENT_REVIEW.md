【父审查3C：小样本诊断通过；仅放行阶段4A正式训练准备，0模型/0优化器。下一条审查后才执行首批500】

保存本条stage4a/PARENT_REVIEW.md，更新SUPERVISION.md和stage-state.json。沿用已核验用户授权：完成或遇阻主动send_message_to_thread回传父对话__LOCAL_CHAT__（hostId=local），记录成功、waiting_parent后结束；我派发后停止等待，不轮询，不需要用户转发。

父独立审查：
从原RGB重算TRAIN hold/129/250误差为.2325831710/.2280759621/.1851714990，250比hold改善20.3848248%、15/18胜、全部18比129改善；DEV .1960324595/.2743519650/.2594243475、仅6/36胜hold且2空推杆，完全吻合。日志是121新更新130..250、1936新draws，最终SHA7a3c3fed60f444536afe8999d6f6346d1ce41258f8df7d87dc78f18bcb4ba28f匹配；全轮账本closed250/868.484000000004秒。已查看129/250原图，轮廓圆化仍明显。这个单固定样本/噪声协议刚过门槛，不是稳健泛化、刚性形状或服务器训练验收。

阶段4A只做CPU代码/数据准备和聚焦检查：
准备下一阶段完整正式TRAIN的首500步实验。不要在本阶段加载任何权重、调用denoiser/native sampler、构造CUDA模型、开始优化器或物理步进。不重采数据、不评未见、不打包。合格后回传让我检查，不能自动执行4B。

一、正式训练实验不沿用微拟合权重
1. 预注册下一次4B从原发布SHA4084ed...1892重新初始化，完整原GPC image/action→image、全参数、原H1残差MSE。3C250只作可拟合性证据，不能拿它当正式TRAIN初始权重。
2. 正式TRAIN仍只有joint-data内396条/24156有效H1；独立seen_val162轨迹不参与训练。1/1B探针、旧native不混入。18训练字母ABCEFHIJLMOPSTUVXY，8未见DGKNQRWZ保持封存。训练时只送RGB/actions，运动标签/字母/状态只用于离线抽样审计。
3. 所有原始数据、manifest、历史checkpoint、报告和source-snapshot不覆盖。允许正常修改活动训练接口来实现本阶段功能，记录真实新hash/diff并保留旧快照；“历史冻结”不意味着活动训练代码永远不能迭代。不要为了保持活动文件旧hash继续扩大FunctionType/private-globals注入方案。正式入口使用明确的sampler工厂/参数、预算接口和验证callback，保留旧默认/API兼容。原模型、loss和sampler数学及冻结物理仍不得改变。

二、实现可保存/恢复的70/10/10/10训练抽样
1. 依据冻结H1定义构建互斥四层pools：moving14692，intermediate1017，near_static_contact1176，near_static_no_contact7271。候选总24156，必须与原数据独立统计一致。不能用stage3“>=1图像px或3°”筛选来替代正式moving定义，也不能删除静止原始数据。
2. 明确实现每180个draw为一周期：18字母每字母10个配额，7moving+1intermediate+1静止接触+1静止无接触。周期内180个(letter,stratum)槽用固定独立seed5478的抽样RNG洗牌。每batch16从流连续取，可跨周期，不要求每个batch恰好70%；累计完整周期精确配比，尾部差额据实报告。
3. 每个(letter,stratum)内按独立trajectory均衡轮转/随机排列，再从该trajectory该层的window无放回随机轮转，耗尽再洗牌。这是有放回训练暴露，不冒充新增物理证据。阶段2B已披露稀少支持和空间缺层，不能为了凑比例新增静止数据/换字母。若某letter-stratum全空则停止回传，禁止隐性借别层补齐。
4. 状态保存/恢复包括180槽顺序和cursor、各层trajectory/window队列和cursor、独立sampler RNG以及policy/pool/data hash。不得借用denoising或评测RNG。只选择TRAIN dataset index；batch tensor keys仍仅image/action。
5. 本阶段允许小型CPU dry-run检查周期比例/恢复连续性，不算实际训练draws。用固定小样本或最多2个真实180周期即可，不生成或缓存几万float图像；实际训练暴露日志以后按已预扣attempt记账。报告candidate unique/trajectory支持和dry-run与未来actual draw区别。
6. 保留18字母、9初始空间、实际边界/接触/运动和低支持层计数。不要再把“轮廓轨迹hit多”当作训练运动比例证据。500次有效更新预期8000draws只作预估，actual attempts/completed/unique/repeats须分别统计。

三、准备原生H1/H4开发验证，索引不得变
1. 严格使用stage2c已冻结development36的顺序、轨迹/anchor/hash及H1/H4真值标签，不重抽，不用旧linspace。完整319与8未见本阶段及计划首500均不评。自然与额外DEV组成及32独立轨迹/时间相关性继续披露。
2. 未来4B计划在正式更新250和500（若时间提前停则最终完整点）各验证一次。每次36条进行4步纯原生自回归，一次rollout同时留H1与H4及中间raw帧；已生成H1不再重复预测。原生batch4、原3B DEV36分batch/step1 seed固定，step k seed=(原batch_seed+k-1)%2^32，所有模型一致。
3. 输入history=s[a-3:a+1]，连续7个绝对动作u[a-3:a+4]。第k次sampler用actions[k-1:k+3]；预测图像直接shift到history。绝不喂未来真值图像/状态，hold在所有horizon都固定原s[a]。真值仅独立scorer持有s[a+1]/s[a+4]。
4. 特别处理原native范围：上游量化允许[-1,1]。最初4张真实RGB必须[0,1]；自生成的finite RGB按原数值原样反馈，不能让我们包装层的[0,1]检查再次截断H4，也不能clip、(x+1)/2映射或改上游sampler。标识generated-history来源，保留shape/finite校验。超出显示[0,1]仍按既有评分规则记该horizon失败，不能重标ok；但有限越界本身不终止后续原生rollout。真正nonfinite/shape/OOM应停止本次模型工作并保存完整边界/失败分母，不能缩batch或换seed掩盖。
5. CPU stub验证H4：捕获每步history/action，逐值证明第2..4步收到前一步原始预测（含一个预设有限负像素），未来真值改变不影响rollout输入；生成后颜色/轮廓函数不能参与feedback。检查a=3/60，step1/4动作窗口与目标对齐，最后一条不越界。此检查不需要实际模型。
6. baseline：原发布H1 DEV36已有3B原始输出可引用，hash/模型/输入/配对seed一致；未来新增baselineH4可由既有H1原输出继续剩3步，记录未重测H1。若实现选择完整重跑4步，必须事前在执行计划列明这36重复H1的成本并验证原H1一致；不能隐含重复或挑有利baseline。本阶段不执行baseline。
7. H1/H4沿用原stage3评分数值/失败处罚/颜色校准，H4对象真值改为a+4，hold仍a，报告两horizon各36分母。moving分组各用各自冻结真值标签；H4端点moving与路径any-moving/全static分开，不能套用H1静止标签。形状/推杆失败、unknown、漂移单列，mask配准仅评分不修图。
8. 选择正式checkpoint的规则在预测前固定：仅比较本次正式TRAIN250/500或时间早停最终点，不与微拟合250混淆。完整72项H1+H4均有记录才可参加best选择；失败仍保留。排序先按72项hard_generation_failure总数少，再按0.5×H1-moving平均灰1-IoU+0.5×H4-endpoint-moving平均灰1-IoU低，最后同分选较早更新。分组空缺必须预检发现并回传，不能运行后换指标。denoising_loss只是诊断，不能再作为best依据。best只是有限候选中的开发选择，绝不表示模型验收通过；形状圆化/unknown仍完整报告。

四、计划下一阶段4B的边界（本条不授权执行）
- 原发布权重重新开始formal_updates0；最多500优化器尝试/最多1800秒模型工作，原全轮3000/5400/3GiB总限制不变。目前closed全轮250/868.484，若4B用满则750attempts/2668.484秒，剩余不是自动授权。所有加载、原生验证、固定loss、保存、异常/重放计入。
- micro4/effective16、AMP、seed5478、AdamW1e-4/wd.01、clip1、scheduler T_max3000/eta_min1e-5保持。原模型数学不变；仅数据抽样换成上面的正式策略。
- 以实际计时保留最后H1/H4评测/保存预算，暂设尾部>=180秒，最迟局部1620秒不启动新更新，模型工作1800秒先到即停；不得因为未到500补跑。异常一次停回传，不自动重启。
- 保留last完整checkpoint和最多一个不同的best完整点。若best就是last，best.json引用同一文件即可，别复制相同105MiB大文件。包含sampler状态/所有RNG/optimizer/scaler/scheduler/真实来源/账本，便于后续在父明确放行时继续。
- 实验3系列既有数据和权重全部保留。对所有新旧产物/活动源码/cache/原子临时空间统一计数。0新物理，39680/74240不变。服务器只由用户手动运行，当前不准备新ZIP。

五、4A交付后主动回传
本阶段交付新正式入口/明确接口diff，冻结config与draw政策，4层候选及180周期CPU核对/恢复记录，36×H1/H4对齐/无未来图像泄漏/finite负像素原样反馈检查，selection score规则和实际分母，计划4B命令/预算/存储估计。聚焦检查，不重跑几百项无关检查；旧3个published-media问题保持披露。
明确本阶段0模型加载/0原生推理/0优化器/0物理/0未见/0服务器。完成即实际回传父对话，我复核采样与H4接口后才另发4B500步执行指令。不要越过这一步。
