# R2C3 实际运行与费用（时间只统计）

真实新阶段 attempts=5700、complete=5700、AMP skips=0，模型血缘累计6000更新（从300恢复），新实际draw=91200、血缘samples=96000。另一个旧实验250更新不混入这条6000血缘。

整个模型阶段6876.985s，record_only_v2，没有时间硬帽。全局账本累计8346.110s、6250尝试/6250完整更新；历史费用不退不重置，active_stage=None。

|累计更新|去噪|sampler尝试/完成|denoiser尝试/完成|raw帧|端点评分|sample纯耗时s|CPU评分s|setting总s|peak reserved B|
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
|1000|3|36/36|108/108|144|72|4.424|5.937|19.797|2025848832|
|1000|16|36/36|576/576|144|72|20.578|5.813|39.672|2025848832|
|3000|3|36/36|108/108|144|72|4.345|5.641|15.969|2025848832|
|3000|16|36/36|576/576|144|72|19.660|5.436|31.437|2025848832|
|6000|3|36/36|108/108|144|72|4.171|5.407|15.109|2025848832|
|6000|16|36/36|576/576|144|72|19.673|5.374|30.891|2025848832|

300_3的forward未在R1单独实际计数，不用36×3理论值冒充观测；300_16的实际记录来自C2。旧R1 final总评估耗时与本表纯sample耗时不是同口径，不能直接给异口径加速比。

训练forward尝试/完成=22800/22800，backward microbatches=22800，实际optimizer应用=5700；固定loss forwards=27。完整恢复加载次数=1；最终真实full reload verified=True。参数相对300变化tensor数=296。

GPU全阶段peak reserved=2025848832B、allocated=1794471936B。普通finite AMP overflow降scale继续，保留已消费draw/噪声/队列，attempt收费但不涨complete/LR；连续16才按失稳停，真实skip事件与scaler见AMP-skips.csv。

## 持久检查点

|种类|更新|字节|SHA256|路径|
|---|---:|---:|---|---|
|最终完整可恢复|6000|105331438|c5bfc0c7119ca5f4852e5eb6192f8c015730999ed5ea26d081e5b4912441a148|__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r2c3\run\last.pt|
|model-only，不是resume payload|1000|35021646|1b02decc9ece3550d7b5c57517a3801d66354442f74d4293b5984fb87051db0a|__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r2c3\run\milestones\model-001000.pt|
|model-only，不是resume payload|3000|35021646|dc30e9c9cb16cf94ac139611851a139848ab81d9cba900b3b5208eee28618c28|__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r2c3\run\milestones\model-003000.pt|

1000/3000观察marker中的full_payload_at_observation是当时rolling last.pt的历史SHA引用，路径后来继续原子更新；不能把旧SHA误当最终文件应保持不变。独立model-only 1000/3000保留，6000保留最终完整状态。每100成功完整边界保存；不另存重复300完整权重。

LR使用真实恢复的原CosineAnnealingLR(T_max3000,eta_min1e-5)；更新u使用L(u−1)，成功后设置L(u)。到3000后保持eta_min，不再调用原cosine.step导致反弹；truecompleted/phase另持久。实际每步LR在training-updates.csv，失败/AMP skip不推进。

实际新曝光91200、血缘总曝光96000，独立TRAIN families=41，unique窗=861，本阶段单窗最高重复=1216。这是保存TRAIN的重复曝光，非新独立数据；完整80/20每字母/类别计数及draw archive SHA见actual-exposure-summary.json。

完整原生RGB训练与四步raw-feedback接口不变；0新physics/采样数据/未见字母/阻尼比较/服务器。耗时无论超过任何旧试验数字都只是统计，不要求用户补时间。6000仅观察点，下一阶段由父对话安排，服务器完整训练永远由用户手动执行。
