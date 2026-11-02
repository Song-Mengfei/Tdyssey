【父审查：阶段2C通过；仅放行阶段3，一次有界的原生GPC小样本拟合】

继续遵守已核验的人类授权：完成本阶段必须实际主动回传父对话__LOCAL_CHAT__（hostId=local），记录成功、waiting_parent，然后结束。我派发后停下，不轮询；不用用户转发。先将本指令完整保存stage3/PARENT_REVIEW.md，并更新SUPERVISION.md、stage-state.json当前授权；阶段4–6仍未放行。

父审查证据：我独立检查了联合目录558份NPZ及metadata全部SHA，558个trajectory_id、seed、fullscene均不同；直接从状态重算TRAIN24156H1中moving14692/static8447，seen_val9882中moving6022/static3408；所有子步最大穿透1.051749px。319唯一索引全部属于seen_val、H4历史/未来有效、moving标签匹配，natural与extra互斥且合并正确，dev36是子集；162个natural重新按冻结规范计算全部候选hash排序，选择与hash全一致。已查看真值连续图。5个blocked空槽与同轨时间重叠保留，离线coverage两行路径修复没有改SHA比较/物理，接受该披露。它们不需要进一步补采或修饰。

目标：在真正发生可见运动的少量TRAIN样本上，验证原始GPC image/action→image能否生成更接近下一帧的原始RGB。训练loss下降只是辅助；不能把这次微拟合称作泛化验收。以下依次执行，不能跳到全量训练。

1. 冻结本阶段样本和评测协议（任何模型推理/更新前）
- TRAIN只使用已冻结joint-data里396正式训练轨迹。每个18训练字母各选1个独立窗口，共18个，范围符合此前16–32小样本计划。不得使用seen_val或旧native/1/1B探针作为训练。
- 候选条件：a在3..60且a-3..a+3全部有效；当前步真实contact>0；面积质心位移*96/512>=1图像px，或者绝对转角>=3°。这些是本次“运动可见”微拟合子集条件，不改总体moving/static定义。我已从TRAIN检查每字母有36–327个满足可见运动+接触的H1候选，H4合法约束仍由你严格筛选；若某字母真无支持立即回传，不降阈值或替换字母。
- 每字母取最小SHA256排序，hash输入固定UTF-8 canonical JSON数组 ["stage3-visible-microfit-v1",2026102904,letter,trajectory_id,npz_sha256,metadata_sha256,anchor]，ensure_ascii=true、separators=(',',':')；不按模型结果选、不取最大的运动量。保存18行精确dataset index/轨迹/anchor/hash/运动/接触/实际边界标签。
- 独立开发验证严格使用stage2c/report/indices/indices.json已冻结development的36个窗口（31moving/4static/1intermediate），不重新抽样、不用旧linspace val_indices。319全验证和未见8字母本阶段不跑。
- 本阶段只评H1，冻结输出比较为：复制最后观测帧hold、原发布权重frozen_gpc、最终微拟合checkpoint。TRAIN18与dev36分别汇总，不能混淆拟合与泛化。
- 原生推理batch_size=4，固定两组窗口顺序及分batch方式；同一窗口组/批次成员/形状与seed在frozen和fitted一致，明确是批次配对噪声，不冒称逐窗口顺序推理噪声相等。使用原发布sampler全部配置、原始RGB输出，无后处理。只做更新前与最终两套预测，不每几步生成一次图库。

2. 准备最小必要工程改动，CPU检查先行
- 复用原native_rgb_training、原始supervised_denoising_loss、原生sampler和stage2c引用dataset。允许添加本阶段入口、显式固定train/val indices、本阶段预算上限，以及只保留本阶段最终checkpoint的控制；不重写模型，不新增运动/姿态头、模板移动、输出修补、mask/几何loss、字母ID输入。
- 原训练器当前val_indices仍是np.linspace且默认用denoising_loss选best，不能让这个旧默认潜入。显式传入已冻结36索引做诊断；本阶段用最终checkpoint，不做“denoising最佳”挑选。将来阶段4的原生预测选best会另行放行。
- 新全轮训练账本设aggregate上限3000尝试/5400秒，本阶段另设<=200优化器尝试、<=600秒模型工作墙钟硬限，所有模型加载、基线/最终原生推理、固定噪声loss、训练、调试/失败/重放都计入本阶段600秒及全轮账本；CPU源代码检查/纯离线评分另列墙钟，不谎称GPU耗时。旧native1899尝试/5352.516秒单列历史且不重置。
- 不要为了阶段3把全局上限改成200/600从而破坏后续账本和scheduler；使用独立阶段cap。本阶段开始第一次模型工作即计时。根据首次基线耗时预留最终推理/保存预算，最迟阶段已用420秒就不再启动新梯度更新；若所需尾部预算更大则更早停止。600秒耗尽时保存可用完整边界checkpoint、报告缺测，不越限补跑。
- 现有check()只disk_bytes(artifact_root)可能只数新目录，必须确保新旧native产物、相关源码/cache和原子保存临时文件共同受3GiB限制。当前约1.5907GiB；沿用现有环境/权重，无下载/安装，不删旧实验。阶段3只保留完整last.pt及必要小记录，避免重复best/last两份大checkpoint；周期保存可同名原子替换本阶段last，不能碰旧权重。
- 聚焦CPU/mock检查预算200尝试/600秒、全局累计、显式indices/时序、模型batch仅image/action、恢复/保存完整性和失败计账。不要另开不计费GPU smoke train。无需重复全量物理采集审计或扩展修3个既有published-media问题。

3. 一次实际3060Ti微拟合
- 从原发布GPC权重严格初始化，不从旧1500/1668权重继续；记录hash、8,729,827参数总数及全部requires_grad。原H1单步残差去噪MSE，原AdamW lr=1e-4、weight_decay=.01，AMP、clip_grad_norm=1，seed5478。
- microbatch4/effective16（旧本机已实测可用，峰值约1.93GiB），每步4次累积；现有按字母/轨迹平衡WindowOrder只喂那18个固定样本，循环重复是有意的微拟合，实际unique=18、draws和各窗口暴露次数分别记账。此阶段不使用70/10/10/10，因为只验证可见运动样本能否被拟合；该分层留给正式TRAIN阶段。
- scheduler继续T_max=3000、eta_min=1e-5，不能微拟合200步就把学习率衰减到底。
- 在更新前取得原发布模型18+36 H1原生输出，保存只读原始数组和方法/seed/hash。可以用同一模型更新前eval得到frozen基线，恢复train及训练RNG，避免常驻两套GPU模型；不得让采样改变训练噪声序列或requires_grad。
- 然后最多200尝试或阶段时间阈值先到即止；每50次完整更新可原子保存last，保存最终完整model/optimizer/scaler/scheduler/RNG/WindowOrder/账本身份。所有尝试先预扣，AMP跳步/异常/失败不退账。OOM/非有限/缺失梯度/协议异常立即保存最近完整点并回传，不自行换学习率、AMP方式或重启跑第二轮。
- 记录所有参数梯度可达、实际变化张量数、attempted/completed区别、loss轨迹、学习率、样本draws、时间、峰值显存和停止原因。用最终可用完整checkpoint生成同样18+36 H1输出；若预算不足，明确哪些没完成，不能只报loss当完成。

4. 原生输出验收与报告（纯评分不训练）
- 每个样本保留 last observed / true next / hold / frozen / fitted 的原始RGB。TRAIN18必须全部可复核，不只展示最好的一两个；dev36亦保留全部数组，图库可按事前固定规则缩成若干页。没有绿色目标/红十字；不修补形状。
- 主要直接指标：未对齐的灰色物体mask IoU（误差1-IoU）、原始轮廓误差、前景RGB误差。必须把物体与推杆指标分开，不能让白背景或蓝推杆的改善掩盖灰色字母没动。若添加前景RGB评分，只是离线评分，固定ROI/公式在预测前并披露，不是训练新loss。
- 复用既有渲染/颜色校准先验证本次renderer/protocol身份，必要的校准只使用18已见形状离线渲染且0物理step，不能看预测改阈值。原score_rgb的配准运动量是辅助，需披露网格/栅格精度，不能用粗角度配准宣称1°内精确。
- 空物体/空推杆/碎块/出界/模型推理失败必须留在原18或36分母；例如空物体raw IoU=0/误差1并另记失败，不能因score_rgb抛异常把坏样本删掉。报告非有限/无法推理时不能从有限样本均值声称达标。
- 形状报告组件/洞数、面积误差、配准后形状差与额外碎片、原评测的deformation_failure及assessable分母；接触遮挡/裁切无法判定时明确unknown，不能当形状通过。字母出现断裂、凭空增块或趋同T都必须展示。对齐只用来评分，原始输出始终保留。
- 预注册微拟合动态诊断目标：TRAIN18上fitted平均(1-raw_mask_IoU)较hold降低>=20%，且至少12/18个逐窗优于hold；并报告相对frozen的变化。它与loss下降分开记录。生成失败使该诊断不通过，形状异常单列交父审查，不因总体IoU漂亮掩盖。这个目标不等于完整验收；200次内没达到只能说明本次有限拟合未证明成功，不能直接断言GPC路线不行。
- dev36独立报告整体及31moving/4static/1intermediate、逐字母/实际边界、frozen/hold/fitted配对结果；只对moving比较胜过hold，static检查漂移/形状，不要求击败零误差。微拟合未泛化不与TRAIN拟合失败混为一谈。
- 结论分别写：训练链路是否正确、18训练样本是否实际拟合运动、独立36验证怎样、形状缺陷是什么。不以“39/46项测试通过”代替模型证据。

5. 结束条件与主动回传
不论诊断是否达标，完成一次后都停止，不启动正式500/1500步、不评全319/H4、不跑8未见、不追加采集、不打包/上服务器。本次新增物理控制始终0，监督轮39680/历史合计74240不变。
交付REPORT、冻结18训练索引/36引用、完整checkpoint与hash、原始预测、全部逐窗指标和图库、命令/源码身份/聚焦检查、累计账本（旧+新及剩余）和已知缺陷。实际send_message_to_thread回传父对话，请求我审查后再放行下一步；成功记录callback及waiting_parent，结束。
