# R2C0 已完成：冻结小规模学习诊断的数据与接口

已完成R2B53条引用、TRAIN41/seen-val12切分、80/20固定抽样序列、H1/H4评估索引、真实输入预览及纯CPU接口。**0新控制、0模型加载/推理、0GPU/训练。300次尝试/1200秒只是拟下一步上限，当前未获执行许可。** R2B仅通过小规模WM学习诊断的数据准入，完整触点/方向/边界覆盖仍未通过。

## 数据与切分

源仅R2B已验收53×24控制/25帧，引用原NPZ/metadata/遥测哈希，不复制整套RGB，不混旧D0/D1/R2A/R2A2。选定0.1s协议SHA `58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6`，无阻尼比较。X-left-top静态blocked仍在R2B覆盖报告，不是训练样本。6字母都是既有TRAIN18成员，8保留字母继续封存。

seed5478、domain-separated SHA排序，从满足约束的不同family对中确定val一平移/一旋转；TRAIN保留平移、旋转及可用正负方向和完整H4可视支持。root独立重算选择与可行候选一致。完整轨迹/family分离，不让同轨迹/重叠窗跨TRAIN与val。

|字母|TRAIN轨迹|seen-val平移family|seen-val旋转family|TRAIN明显/其余合法窗|
|---|---:|---|---|---:|
|T|7|R2B-new-02-T-right|R2B-new-07-T-left-bottom|62/85|
|L|7|R2B-new-11-L-right|R2B-new-09-L-center|64/83|
|O|7|R2B-new-25-O-left-bottom|R2B-new-19-O-left|41/106|
|C|7|R2B-new-29-C-right|R2B-new-28-C-left|29/118|
|X|6|R2B-new-38-X-right|R2B-new-43-X-left-bottom|37/89|
|B|7|R2B-new-45-B-center|R2B-new-48-B-top|50/97|

TRAIN H1/H4=861/738，seen-val=252/216，合计1113/954。H1 anchor3..23、H4 anchor3..20，历史3个转移+4帧正确对齐。53条全状态/动作/RGB及初始状态完全重复0，跨轨迹完全相同原RGB帧或state帧0。另明确“同letter、推杆距离≤15worldpx、物体origin≤12worldpx、wrapped角≤5°”的近重复规则，初始及任意保存帧对均0；不因此声称无几何相似性或无偏抽样。所有6letter为seen，val轨迹刻意含运动，本地诊断不能冒充未见字母/无偏自然测试。

## 真正进入训练draw的分布

本次明显运动严格定义为H1面积质心平移≥2个96图像px **OR** abs角≥5°；旧2worldpx定义不用于本池。源TRAIN283明显+578其余窗全部保留。每480周期每letter80次=64明显+16其余，总384/96；先字母、letter/group内独立family均衡，再互斥translation_only/rotation_only/both类别及window队列。重叠仅归both一次；源both仅C的2窗/1family，其它字母此联合类缺层保留。

首480实际category=169平移only+211旋转only+4both+96other，故173次≥2px、215次≥5°、明显union384；覆盖41family、372个unique窗，108次重复曝光，单窗max7。平移median/P95/max=1.42528/2.34867/2.42424px；abs角median/P95/max=3.11795/9.18783/12.6668°。这是OR定义，不要求两个边际中位数分别超过门槛。

拟4800暴露=1676平移only+2121旋转only+43both+960other；覆盖861个unique TRAIN窗/41family，3939次重复曝光，单窗max64，不是新增4800独立数据。每letter640明显+160other。10周期使用延续RNG/独立队列，不机械重复首周期；冻结480/4800 indices、state480/state4800及逐16draw共300batch/逐cycle幅度、phase、contact、实际letter:arc、正负lever、family和重复次数见data/draw-audit.json。

other支持不伪造静止：严格零平移且零角的合法H1为0。辅助near-static仅审计定义(<.1图像px且<.5°)，TRAIN中接触23窗、无AO322窗，较小运动233窗；release242、实际AO loss43、wallstall22（辅助标签重叠）。首480确抽到nearstatic接触3/无AO46、小运动47、release35、AO loss9、wallstall3；逐字母真实arc/独立family及缺类详见supplemental-draw-support.json。它们不改变冻结80/20策略或源数据。

## 冻结评估与真实输入预览

|冻结组|N|H1平移≥2px|H1旋转≥5°|H4平移median/max|H4绝对角median/max|
|---|---:|---:|---:|---|---|
|train_visual|12|6|6|8.32288/9.3955|9.60063/36.4477|
|val_motion|12|6|6|5.94465/9.39806|9.15569/31.9577|
|val_natural|12|3|1|1.65708/9.38768|0.516482/28.0159|

TRAIN可视12=每letter平移/旋转各1，均来自首480真实暴露且H4合法；没有按模型输出挑图。val natural每轨迹hash取1个H4合法anchor，未按运动筛anchor（但val轨迹自身预选含运动）；12中4明显、8其余。val motion每轨迹1个明显H1且H4合法窗，共12，与natural按trajectory+anchor去重。两val组仍可能共享历史/未来帧与轨迹，重叠披露于preparation-audit.json；分别评分，不合成24条独立轨迹。

中文原RGB预览来自保存96像素：4历史帧、H1/H4 truth、图外绝对动作数字/箭头、真实幅度、分类、split、family与首周期曝光。所有输入保持白/灰/蓝，root检查1325原帧绿色和红色像素0；标注不进入网络。训练12主可视、自然val和motion val预览位于evaluation/truth-*.png；另有首480 other训练示例页（仅展示，不增评估cohort）。没有生成任何模型预测或环境重渲染。

## 可复用接口与CPU证据

`R2C0Dataset`直接读源，五帧=[4历史输入+H1 target]，四动作=[3历史+当前]；`PreparedInputs`输出的model_inputs仅image/action，不传state、pose、letter、moving标签、触点或模板。float32 RGB原uint8/255，动作原式/511*2-1归一化；全部1113窗与原帧/动作bitwise一致。单轨迹cache≤1MiB，没有RGB磁盘副本。

`R2C0WindowOrder`有显式池身份、RNG、队列、游标；完整4800流和state480后的4320步逐项与冻结序列一致。PreparedInputs支持数据层midmicro精确恢复及校验，**这不表示未保存梯度的真实优化器可中途恢复**；未来训练完整checkpoint边界须另保存model/optimizer/scheduler/scaler/RNG、draw状态和预算。

`rollout_raw`仅接真实初始history4与7动作审计前缀；每次predict callback只收history4/actions4。H1复用H4第一步，此后只滑入模型生成的原RGB。动作在此接口恰归一一次，不能把normalized动作交旧absolute-normalizing wrapper，也不能逆变换后再归一化。真实模型bridge尚需下一阶段显式接入，当前没有借旧FunctionType/private globals覆盖。

root联合实际数据+mock审查中，修改未来truth不改变callback输入或输出，false执行guard阻止任何callback；有限OOR原值精确保留/回灌，非finite/wrong-shape硬停并保留原失败与缺测分母。15项聚焦unit tests、数据真实加载、接口合成/实际联测及root独立对齐审查均通过，未导入Torch或构造环境/模型。源码与依赖闭包见source-identity.json/source-snapshot。

后续同窗比较copy-last、原始GPC、训练后模型，按TRAIN展示/val natural/val motion分开报H1/H4、静止/小运动漂移、灰物体前景RGB/IoU/轮廓、蓝推杆指标及硬失败。保持原始数值，有限越界评分硬失败不裁色；遮挡或viewport不完整保留shape unknown，assessable_no_detected_failure不能写成刚性保形通过。灰物体改善不能被蓝推杆改善替代，训练改善不等于未见泛化。评分桥已用既有固定calibration做纯数组检查，无阈值再拟合/渲染。

## 拟下一步配置与当前限制

`configs/native_rgb_r2c0_proposed.json`：原始本地GPC phase-two denoiser SHA `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`（35,043,916B），只做流式哈希未反序列化；不接旧250更新权重。原8,729,827参数image/action→image架构、全参数训练、原residual objective，无motion/pose head、几何loss或颜色/形状修复。沿用已实际跑过的native3060Ti设置：effective batch16、micro4×4、AMP fp16/float32 loss、AdamW lr1e-4/betas(.9,.999)/weight_decay.01、gradclip1、GradScaler128、原cosine3000 horizon/eta_min1e-5；来源和原文件SHA见source-settings-audit.json。

拟上限300 optimizer attempts与1200秒总模型计费先到停；包括model/load、原始baseline H1/H4、训练、最终H1/H4、save/必要reload、失败/清理。最晚900秒停止新尝试并留300秒尾部；实际所需headroom增加只允许更早停，不保证满300。OOM/失败预扣不退、不自动试microbatch/重跑。原全局ledger仍closed250/868.484000000004秒，余2750attempt/4531.516秒；拟300/1200只是余量内提案，未reserve或reset。历史另一native1899/5352.516单列不冲销。

必要数据/抽样/评估数组接口已准备并CPU联测；**真实模型执行入口仍未接入/未获准启动**。旧train的18TRAIN/36dev或旧formal36/250/500硬限制不直接复用，本阶段未调用它；下一阶段需显式新runner、already-normalized predictor bridge和model/time/storage/checkpoint guard。没有用临时私有全局覆盖来假装已可训练。

本阶段0新physics/env/model/GPU/optimizer/training/扩采/服务器/安装下载/上传包；累计物理41976监督/76536历史，旧余24/320不动。原R2B及父审2254个输入身份逐项不变，旧账本与权重/失败记录保留。新增产物连新源码≤32MiB，共享所有旧新source/cache/父审+256MiB原子预留≤3GiB，确数见storage-final.json。完成后实际主动回传并保存receipt，waiting_parent停止，等下一阶段明确指令。
