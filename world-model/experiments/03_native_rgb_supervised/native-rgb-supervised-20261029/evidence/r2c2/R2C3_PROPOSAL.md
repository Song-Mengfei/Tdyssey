# R2C3提案：用纵向训练曲线区分训练不足与采样因素

**仅提案，尚无C3训练runner，也未执行C3模型、loss、优化器或数据抽样。** 配置为项目 `configs/native_rgb_r2c3_proposed.json`。先由父对话验收已完成的R2C2，再下达下一阶段实施指令；不从本提案自动启动，不再次向用户索取常规训练批准。用户未设置模型工作时间上限，record_only_v2持续有效；无1200/900/300/5400秒截止、剩余时间理由、伪巨大时间帽或暂停计时。

300更新只能证明早期学习与训练链路，不能证明训练充分、收敛、容量上限或保形无望。R2C2只检验固定300权重的采样次数影响；不因增加采样次数未修复棱角，就排除训练不足。1000/3000/6000也是观察点，不是人类终身训练上限；曲线仍改善时父对话可决定10000以上的下一阶段。

## 恢复与不变量

唯一起点为 `r2c1r1/run/last.pt`：105,300,718B，SHA256 `3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0`，同一谱系已完整更新300次。完整恢复model、AdamW、GradScaler、cosine scheduler、Python/NumPy/Torch/CUDA/独立noise-generator RNG、PreparedInputs全部order/slot/queue RNG及cursor、progress。构造承载这些state的对象后立即加载保存状态，不重新初始化试验、重置warmup或从模型权重单独起跑。旧另一试验250更新不计入该谱系；全局费用账本此前550complete继续保留，C2仅新增86.907秒、global1469.125秒统计而未增更新。

R1保存payload包含原config/identity，既有restore helper会严格要求其相等；新阶段必须用明确的后继identity验证原payload与本配置的允许差异，不偷偷改旧config或绕过身份比较。恢复后核对模型、optimizer/scaler/scheduler/RNG/PreparedInputs摘要与payload相同，特别是delivered_draws=4800、pending_indices=null、micro_cursor=0、scheduler.last_epoch=300。模型仍为原GPC的8,729,827个参数全部可训练。

只用6个seen字母T/L/O/C/X/B、41 TRAIN family/861个H1窗及原12条seen-val；固定0.1秒半衰期、原RGB/action、96分辨率、4history+H1target、一次动作归一化、原H1 phase-two residual MSE、effective16/micro4×4、AMPfp16/FP32 loss、AdamW lr_base1e-4/betas(.9,.999)/eps1e-8/wd.01、clip1。不同时改抽样比、数据、模型、loss、形状修复或几何/位姿模块。

|同一谱系累计complete|新增complete|本段实际draw（无异常时）|累计draw|
|---:|---:|---:|---:|
|300（已有）|0|0|4800|
|1000|700|11200|16000|
|3000|2000|32000|48000|
|6000|3000|48000|96000|

因此本阶段共5700新增complete、91200新增draw，而非再训300。80/20采样数学与队列不变；每480draw周期每字母80次，其中64 moving/16 other。若无异常，新190周期的预期moving72960/other18240。实际顺序、窗口/letter/family/category/phase/contact曝光与repeat分布全部记录，重复曝光不称新增独立数据。

**两个真实工程门槛必须显式迁移。** 原R2C0WindowOrder的take和load_state_dict硬限4800，恢复4800之后继续take会失败。新版本仅扩展流范围至96000，保留相同slot、family/category/window选择数学与所有RNG/queue状态。CPU证明旧4800索引和完整state逐项相同，4800断点迁移后4801开始不断流、恢复再继续等于一次生成、每480周期仍80/20；不修改冻结旧源码，不重新洗牌或把cursor置零。PreparedInputs身份继承原数据/采样数学，后继流范围与迁移来源另记显式identity。

旧范围不只global3000：`RecordOnlyTimeLedger.change`另硬绑每stage reserved_updates≥300，`R2C1R1ModelWorkBudget`的max_attempts/reserve及R1 CLI config_contract/CAPS也固定300。这些与旧全局3000attempt守卫均只是先前阶段安排，不是用户终身上限。新公开后继实现必须同时替换stage300、CLI CAPS300、global3000和stream4800，而不编辑冻结旧源码；CPU明确证明新stage第301次attempt继续、5700新增complete到达谱系6000/global6250，不会训练到300又意外停。新账本scope明确5700新增complete和三个观察点，原550 attempts/complete与所有历史stage原样保留。每次尝试在draw/forward前持久计费；异常尝试不会退款。AMP真实skip不得计为complete或前进scheduler：记录scale前后、optimizer是否实际应用、draw与费用，保留最近安全完整payload及失败现场，停止交父审，不能静默重试、伪造5700成功或自动多跑。结构/nonfinite/OOM等亦按真实失败报告。

## 学习率与loss曲线

令u为本谱系已成功完成的更新数。保存状态恢复后，不重新创建从零开始的学习率序列。到3000为止维持原CosineAnnealingLR(T_max=3000, eta_min=1e-5)：

`L(u) = 1e-5 + (1e-4-1e-5) × [1+cos(π×min(u,3000)/3000)]/2`。

第u次optimizer更新使用 `L(u−1)`，成功后scheduler前进至 `L(u)`，且仅当u≤3000时调用scheduler.step。3000完成后的optimizer lr为1e-5；3001至6000均hold该值，不再调用会反弹的cosine step。保存原scheduler state（last_epoch停在3000）及单独的hold_phase/真实complete counter；不伪造last_epoch等于6000，不重建warmup。每次记录实际使用lr和step/hold后的lr，不把日志中的post-step lr误说成刚执行更新的lr。

|已完成u|L(u)，供下次更新使用|
|---:|---:|
|300|9.779754323328192e-5|
|301|9.778295768677619e-5|
|1000|7.75e-5|
|3000/3001/6000|1e-5|

记录所有attempt的真实四个micro loss、加权effective-batch loss、gradient norm、AMP scale/skip、应用更新数、draw曝光和lr；不得用前后训练batch loss的偶然变化代替固定窗曲线。固定loss在300/1000/3000/6000测同一36窗（TRAINvisual、natural、motion各12），用原H1 residual objective、batch4、固定独立noise-plan seeds/hashes及一致fp16/FP32口径，无backward。每点9个forward批、四点计划36批；300点在C3获放行后才测，不追加到已完成C2。噪声seed域固定为 `[R2C3-fixed-loss-v1,5478,cohort,batch_index,window_ids]` 的canonical hash，不能包含训练观察点u。评估前后完整保存/恢复训练RNG、generator、PreparedInputs与mode，固定loss不得消耗训练噪声或抽样顺序。冻结noise seeds/hash，避免存大份重复noise tensors。

## 横向×纵向评估矩阵

在1000/3000/6000每点，固定原生去噪3和16；300点直接引用R1 trained3及R2C2 trained16，不再生成。300点的8只作中间参照，以后不加8或挑更漂亮的设置。sigma_min .002/max5/rho7/order1/churn0/cond0，以及原denoiser内部精度、量化、归一化完全保持。

每setting复用冻结36窗、原顺序、batch4、cohort/batch/windowID/step配对seed和4history+4actions接口，四步raw反馈不裁剪。未来truth/state/label不进预测。单setting实际计划36 sampler batch、144float32帧、72个H1/H4端点评分；三点×两setting合计216 sampler批、864raw帧、432端点评分；order1预期denoiser forward共3×36×(3+16)=2052，必须用真实计数而非填预期值。固定loss forward另列，不能混成采样调用。所有有限越界和中间失败保留。

三组分别给gray IoU、gray ROI RGB MAE、原轮廓/质心误差、相对LAST漂移，blue另列。natural的8个非明显运动窗与4个严格近静止辅助窗另列全分母及逐窗胜负。记录H1/H4 shape failure/unknown、predicted/truth components与holes、extra_fragments、missing_components、holes_difference、tiny_fragment_pixels；遮挡/边界unknown不作pass。每观察点全部36窗H1/H4展示truth/copy-last/trained3/trained16，可引用300对照，避免重复原红T基线图库。原float评分/回灌和纯PNG显示编码分开。

横向3→16改善支持采样因素；固定采样次数下纵向300→1000→3000→6000改善支持300训练不足；两者可同时存在。两者均未改善只限制此数据/训练方案，不证明唯一病因、模型上限或更多训练无意义。差异小或遮挡未知多则证据不足，由父安排最小后续检查。TRAIN loss/metric改善而seen-val、静止或shape恶化应显示过拟合/保持能力代价。natural与motion同12轨迹且部分窗重叠，不宣称独立24轨迹或8个未见字母泛化。到6000仍改善可由父决定10000以上，不自动续跑。

## 最坏字节规划与原子写入

所有已交付旧weights、data、失败证据原地保留。只新增一份完整 `r2c3/run/last.pt`，在1000/3000/6000安全边界原子覆盖；6000仍保留该full，不另存重复6000模型。首次新保存前引用现有R1 full300，不造新t0副本。1000与3000各保存一个model-only轻量milestone及progress/identity，不复制Adam/scaler/scheduler/RNG。原模型全部state tensor逻辑字节34,920,332B；原完整payload观测105,300,718B，仅作参考，预算对其增长另留余量。

raw逻辑精确 `864×3×96×96×4=95,551,488B`，与压缩率无关。最多216个batch/step容器，92MiB配额在raw之外留917,504B容器/冻结ID头部空间。H1/H4永久保存raw文件+batch row+step0/3引用，不重复写float64端点图像；可节省另一份95,551,488B。评分在内存构造同值视图，公开reference loader在上线前证明与原float64端点容器的metrics_input_identity/评分输入逐值相同，原评分数学不变。压缩可使用lossless，但预算按未压缩raw计算。

|新增持久类别|硬规划MiB|字节|
|---|---:|---:|
|一份完整last（含有界metadata增长）|112|117440512|
|1000/3000 model-only，各36MiB|72|75497472|
|全部864 raw及容器|92|96468992|
|评分JSON/CSV与机器报告|24|25165824|
|可读图库|32|33554432|
|紧凑attempt/micro日志、draw/exposure审计|16|16777216|
|源码快照、配置、身份metadata|8|8388608|
|额外增长余量|44|46137344|
|**新stage合计**|**400**|**419430400**|

R1 events300次约3,792,456B；若逐字照搬全16窗metadata到5700次，会超紧凑日志配额。C3只记16个索引/必要ID与数字字段，引用冻结861窗catalogue，不反复复制整窗metadata。96000个int32 draw仅384000B，曝光汇总有界；无全量RGB训练样本复制。完整checkpoint只带当前PreparedInputs状态与有界progress，长日志/全部draw数组外置引用，不能每次打包历史大数组。

根与独立审计当前实测既有shared=2,456,891,776B。按**整个400MiB stage**、256MiB原子余量和另1MiB非stage账本/status增长保守预留：

`2,456,891,776 + 419,430,400 + 268,435,456 + 1,048,576 = 3,145,806,208B`，低于3GiB=3,221,225,472B，余 **75,419,264B**。这是全stage最坏预留，不依赖压缩率或旧文件删除。C3启动前必须在最终C2报告/父回传后重新扫描实际shared并冻结；既有shared须≤2,532,311,040B才能保持上述400MiB+256MiB+1MiB全预留。如最终超出，报告准确缺口，不删旧产物、不缩成再训300。

原子余量不是永久第二份full：每观察点先在完整optimizer边界safe-save（旧full保留直到临时full完整校验与replace成功），再做该点新eval；大文件临时写串行，图库在模型关闭后冷态生成。持久stage配额与物理峰值分别核对，临时替换亦计入共享实际占用；256MiB足够覆盖≤112MiB的单次full替换与有界I/O metadata，不可并发产生多份大临时payload。旧R1 last始终不改不删。最终full做一次真实重载，核验model/optimizer/scaler/scheduler/RNG/PreparedInputs/progress，并从该完整状态完成最后的固定矩阵；各轻量点明确不可作完整续训checkpoint。

本次只交付提案/config。后续父放行后先实现以上最小后继restore/order/ledger/lr/endpoint引用支持和针对性CPU检查、冻结身份及存储预留，再唯一执行实际C3。不采新数据、不做physics、不重验阻尼、不接触8个未见字母、不启动服务器；服务器完整训练始终由用户手动执行。完整C2报告与本提案一同主动回传父对话后，实施端停止等待。
