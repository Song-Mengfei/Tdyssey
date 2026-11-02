# R2C3：真实完整恢复至6000更新的训练长度学习曲线

已从实际300更新checkpoint恢复model/optimizer/scaler/scheduler/全部RNG与PreparedInputs顺序，完成新增5700次尝试、5700次真正optimizer更新、0次AMPskip、91200次新draw；本谱系6000更新、累计96000draw。旧另一谱系250只留历史账本，不计这条训练曲线。无300新推理/固定loss，复用全部旧输出；不是再训300就下结论。

|组|去噪次数|H4 IoU：300→1000→3000→6000|
|---|---:|---|
|train_visual|3|0.562509 → 0.714488 → 0.851869 → 0.866609|
|train_visual|16|0.529868 → 0.691525 → 0.870844 → 0.886459|
|val_motion|3|0.531987 → 0.595447 → 0.616894 → 0.635105|
|val_motion|16|0.489620 → 0.597446 → 0.610955 → 0.637674|
|val_natural|3|0.658888 → 0.726143 → 0.725701 → 0.727948|
|val_natural|16|0.608107 → 0.704770 → 0.726951 → 0.723843|

全部同窗逐组、逐窗横向3/16和纵向训练长度比较，非明显8窗、近静止4窗、轮廓/形状/孔洞碎片、RGB、质心、漂移与蓝推杆单列见[完整科学报告](report/REPORT.md)、实际曲线 (external historical reference: `report/index.html`)、[科学摘要](report/SCIENCE_SUMMARY.md)。TRAIN增益和验证变慢/停滞必须并列；不能用训练loss、均值或条件可判定集合代替所有窗结论。unknown只是既定遮挡/近邻代理或边界门控，不能当pass，也不能据此遮掩可见畸变。1000的24unknown均为1px灰mask膨胀与蓝mask相交代理，非24个真实遮挡证明。

## 用户指定的O/C棱角对照

固定4窗主图及O/C细节 (external historical reference: `report/focus-gallery/index.html`)严格四列：真实未来/训练1000/训练3000/训练6000，全部3次去噪、同动作/配对seed，H1(.1s)与H4(.4s)分开。O/B/C/X固定窗口，未挑好看样本。O/C同row所有列使用同坐标裁切、同整数nearest，不旋转平移对齐、不锐化平滑修形，不保存重复大数组。具体外直边/转角/O孔/C开口和笔画目视判断在report/focus-visual-review.json；IoU提高或阈值未触发不能代替棱角。旧parent-view-1000/3000均保留并计入stage。

## 实际工程与资源事实

原GPC图像+动作→图像、原phase-two residual loss、全部参数、固定6seen/41TRAIN+12seen-val、80/20、batch16/micro4与物理0.1s协议不变。无新数据/physics/env/阻尼验证/未见8字母/服务器/下载/几何或运动模块。旧stream4800、stage300和global3000计划守卫显式迁移；所有slot/queue RNG连续，不重复前4800。实际draw索引全存并冷重建核对。正常GradScaler溢出降scale继续，skip消费draw/费用而不计complete或LR；本次实际skip数见上。学习率原cos至3000后保持1e-5，原scheduler.last_epoch停3000、真实complete计数独立保存，没有反弹或warmup重置。

新6组生成评测共216 sampler批、2052实际sampling denoiser forward、864原始float32四步帧、432新端点评分；300与copy复用216行，合计648端点。固定noise计划loss只在1000/3000/6000，27批；核对seed/IDs/shape与generator前后hash，未独立逐字节核对未保存的sigma/epsilon tensors。所有新评分在实际model clock内完成，冷报告0模型/0load/0评分。

每100success保存同一个完整last，1000/3000另留model-only，6000完整点实际重载核验。最终checkpoint `__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r2c3\run\last.pt`，105331438B，SHA `c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148`。只有2轻量+1完整新权重，无重复t0；旧权重/数据/失败/费用全保留。初始完整恢复7项实际核验全true，最终参数相对300变化296个张量，完整reload=true。13必要CPU合同通过，141源码/快照与3222旧保护文件及raw/评分对应独立冷审通过，详cold-integrity-review.json；旧通用仓库CRLF/LF三差异保持已知，未声称全绿。

本次实际全model work 6876.985000秒、全局8346.110000秒仅统计，用户未设模型时间上限，所有旧硬截止不再生效。stage<=400MiB/shared<=3GiB含256MiBatomic，最终storage-final.json；所有旧交付、两套prelaunch source freeze历史、父图都保留。

6000是当前学习曲线观察点，不是收敛、能力上限、保形无望或用户总训练额度。固定采样下训练更久的收益支持300不足是因素；同权重采样效应及其交互独立呈现，不强行二选一。是否继续10000+依实际曲线交父审，不自动执行。

1000/3000已实际主动回传且不中断训练。现在按长期直接人类授权主动回传完整6000结果、绝对图片路径与图页，保存真实receipt后waiting_parent STOP；不轮询、不自动更多模型工作。
