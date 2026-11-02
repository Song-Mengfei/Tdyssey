【父审查：阶段3是入口提前中止，尚未训练；放行一次阶段3B纠正续行，沿用原阶段3剩余预算】

先将本指令存stage3b/PARENT_REVIEW.md，更新SUPERVISION.md和stage-state.json。每步主动回传的已核验人类授权不变。父对话__LOCAL_CHAT__、hostId=local。我派发后结束，你完成/遇阻实际send_message_to_thread回传、保存成功、waiting_parent后结束。不让用户转发，不轮询，阶段4–6未放行。

我的独立审查：
1. 已读取原始frozen-train18.npz，确认A共6个负channel值、min=-0.0196077823638916，其余3张有限；后14张确实未尝试。t0 checkpoint SHA eb7b8d...c7cc匹配，账本28.3440000000046秒/0尝试/0更新已closed。
2. 已读取并核验本地原版源码，diffusion_sampler.py SHA d2995d0150e03f3115995b726786b213fa76167d8a589c21722fb797d0d06dd4，denoiser.py SHA e4ce1c17a26b97fcee1d863e56f49b3c32c247900d08d06586941ef337463c41，均吻合固定release。wrap_model_output本身clamp(-1,1)并量化回[-1,1]；sampler最后直接返回x。因此有限负值可以由原生采样器产生，不是缺梯度/NaN或环境故障。输入训练RGB仍为[0,1]，这不是授权转换整个模型输入输出到[-1,1]。
3. 你的预注册abort太严格，阻断了要验证的训练。接受已提出的小diff：finite out-of-range继续生成失败记录但不abort；nonfinite/wrong-shape/OOM及训练非有限仍立即停。没有必要因基线图像本来不好而阻止训练。

此次授权范围：
A. 保留stage3全部原目录、t0、失败输出/评分/费用/源码快照，不改写为成功，也不再称这次为训练失败。stage3b新目录记录修改、复用身份和实际结果。旧权重/其他实验保留。
B. 唯一预测行为改变为去掉finite out-of-range分支的bad=True；仍保存原始浮点数组、out_of_range_native_RGB status及范围计数。不要clip、加epsilon放宽、重标定颜色或做(x+1)/2映射；原GPC结构/源代码/原loss/原sampler/AMP/AdamW与全部训练配置不变。
C. 原阶段3评分协议和20%+12/18诊断目标不变。finite range失败仍按现有失败惩罚/全分母保留，不改成“ok”来通过。报告把这些惩罚值与实际可评分RGB测量分开，未生成/未训练不能写成实测模型性能。附每张finite range的min/max/负值及>1计数即可，不增加另一套挑优评分。
D. 仅必要工程适配：新3B入口/配置/源码身份、对既有closed账本的精确续行检查、残余阶段预算与新输出路径。不要去掉身份或预算守卫，而是验证原stage3失败已closed、charged_updates=0且charged_seconds等于保存28.3440000000046、原18/36索引/paired-batches/真值/calibration哈希一致。给本次建立独立执行标志，重复启动仍拒绝。新旧代码版本分别冻结，原stage3 frozen源码快照不能覆盖。
E. 18TRAIN与36DEV继续使用原已冻结文件和顺序，不重新选择、不重渲染物理、不改变batch成员/seed。优先引用原truth/索引/配对配置；如现有接口需要少量副本，验证逐字节相同并计入存储，不复制完整物理数据。

执行与预算：
- 仅授权再启动一次实际3B模型执行。从原发布权重严格重新初始化一次，不从旧1500/1668或t0假称恢复训练。
- 允许完整重测frozen18+36 H1，保持原批次/seed。首4张是显式重复、计入本次模型时间；无需为省这4张另造复杂的续批机制。将重复首4张与旧原数组做CPU对照并报告；若存在差异须核查输入/源码/随机性身份，不能隐瞒或挑更好的一版。此次统一使用3B完整baseline与3B最终fitted配对；旧输出永久保留。
- 同一个native-rgb-cost-ledger.json，aggregate仍3000attempts/5400秒；旧native1899/5352.516照旧历史。stage3+3B合计仍<=200尝试、<=600秒，原28.344秒不得退费或重置。
- 当前3Bmax_seconds必须从原账本精确计算600-28.3440000000046=571.6559999999954；3Bmax_attempts=200-原stage3尝试=200。最晚启动新梯度时间相对于3B为420-28.344=391.656秒，且基线测得的尾部reserve可以要求更早停止。现有ModelWorkBudget已接受较小max_seconds/update_cutoff，直接使用这些参数，不能又用默认600/420。
- 计时仍包括加载、重复baseline、训练、fixed loss、最终原生预测、保存、torch checkpoint语义检查、失败清理。CPU准备/离线评分另列。全旧新产物+源码/cache+原子预留继续<=3GiB。当前existing约1.85GB；不下载/安装/删旧数据。
- 原生H1，18固定窗口循环训练，micro4/effective16、seed5478、lr1e-4、wd.01、全8,729,827参数、AMP、clip1、scheduler T_max3000/eta_min1e-5，最终last而非denoise-best。禁止新运动模块/几何loss/模板等。
- 需要真正进入backward/optimizer，记录all-gradient可达/参数变化/attempted-completed/采样次数/峰值显存；CPU检查不能替代。若finite范围问题再次出现在baseline或fitted，按已修规则记录并继续本轮，其本身不再触发整轮中止。
- 其他真正异常仍安全停、保存最近完整边界并回传；不得第三次启动、额外GPU探针或临时改lr/AMP/网络。最多200尝试或剩余时间先到即停，预留最终54H1与保存时间。缺测如实报告，不越预算。
- 仅做必要CPU/mock检查：有限越界保存且继续、nonfinite/shape仍abort、既有账本28.344不丢、3+3B合计600/200不越、冻结18/36/配对不变。不要为一行修复扩大成新测试工程。

交付/回传：
在stage3b提供原始54窗口frozen/fitted、完整last、全部逐窗评分/原18训练图库与dev分组、loss/梯度/实际变化、耗时/显存/次数、原3+3B合计账本和源码diff。结论分别说明链路、18训练拟合、36独立验证和形状缺陷。
不评H4/全319/未见8，不新增物理控制，不扩大正式训练，不上服务器/打包。监督控制仍39680、历史总74240。
完成这一次纠正续行后实际主动回传父对话审核并停止；无论达标与否，下一阶段都由我再下指令。
