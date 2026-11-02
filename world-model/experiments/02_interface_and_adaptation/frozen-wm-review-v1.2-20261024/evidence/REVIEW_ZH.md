# 冻结世界模型验证：审阅报告

2026-10-24，新加坡时间 17:34:30–17:50:22。独立分支
`codex/frozen-world-model-validation`，基于 `e0842d7`。

**资产问题已解决；当前冻结 GPC 尚不能提供可用的动作排序。**
本轮没有训练策略或世界模型，也未联系另一个 DPPO 离线分析对话或修改它的工作文件。
这是带几何先验的 development 诊断，不代表全 reset 成功率、泛化或长期 DPPO 回报。

## 资产

原 csgo 权重是 upstream Git 自带完整文件，并未损坏。它与 HF phase-two 权重有
相同 298 个 keys，但 **296 个 tensor 不同**，不是单纯序列化差异。原文件和
manifest 均保留。已获取精确版本：35,043,916 bytes，SHA256
`4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`，
revision `c16a467ec176af19bba5136564a81d8d61d72af9`。

GPC source commit、20 个源文件哈希与精选报告 7 文件均通过核查。首次 DNS 失败
与零传输记录保留，没有下载数据集或安装环境。搜索范围内未找到旧自研
best.pt/catalog，该历史比较不可运行，没有重训或随机初始化替代。

证据：`../frozen-wm-assets/resolved-v2/asset-audit.json`、
`../frozen-wm-assets/download-events.json`。

## 校准通过，但成功精度不足

解码器在首次真实渲染前冻结，只使用 RGB、已知 L/T 几何与固定 goal，
不使用未来真实 pose/contact/ROI 修补预测。本轮没有调参或增加 palette 变体。

| 24 个独立校准状态 | L | T |
| --- | ---: | ---: |
| 可测 | 12/12 | 12/12 |
| 位置 P90 | 1.039 px | 1.060 px |
| 角度 P90 | 0.354° | 0.668° |
| 推手位置 P90 | 0.821 px | 1.175 px |
| coverage MAE | 0.00836 | 0.00570 |
| 连续误差门槛 | 通过 | 通过 |

但近 .95 的 8 状态中，**4 个真实成功全部漏判**。没有预测成功，precision 无法
计算。两个 T 状态真实 coverage 为 .961905/.923810，却解码出相同 pose 与
coverage .939254。96 px 下的连续误差通过，不能推广为严格成功判定精度。

## 模型结果与门槛

16 固定锚点、4 固定候选、3 固定噪声。每个真实分支只运行一次，全部噪声复用
该分支。保留 raw RGB 反馈，只在测量/预览中 clip。GPC 没有直接 pose/contact/
reward head，geometry 接近量仅是 contact proxy。

| H4 数量 | L | T |
| --- | ---: | ---: |
| 原始图像可解码 | 4/141 | 34/48 |
| 考虑此前预测停止历史后的 score | 0/141 | 6/48 |
| 三噪声均可用的候选 | 0/48 计划候选 | 0/16 |
| 锚点排序弃权 | 12/12 | 4/4 |

H1 可测只有 L 0/141、T 6/48。此前缺失时不能用 future truth 判断预测是否已
成功，因而部分 H4 pose 可测仍不能恢复完整 score。**不是所有 H4 图像都不可测。**
完整候选 coverage MAE、实际 selected regret 不可得；保守 pair accuracy 0 和
regret 上界 1 是弃权罚值，不是实际选了一个很差的动作。

| 工程筛查 | L | T |
| --- | --- | --- |
| 非接触漂移 | 失败，测量缺失 | 失败；已测子集 2.683 px>2，且有缺失 |
| 接触运动 | 失败，0/9 分支完整测量 | 失败，仅 2/3 完整 |
| 明显旋转 | 失败，0/3 完整 | 样本不足，1 锚点<2 |
| 终点 coverage | 物理缺1分支；完整评分也缺失 | 失败，完整评分缺失 |
| 排序 | 样本不足，4 可区分锚点<6；已知 eligible pairs=12 | 失败，2锚点/6对但全部弃权 |
| 直接 contact 概率 | 不可测，无 head | 不可测，无 head |

L 接触物理覆盖 9 锚点/9 分支、明显旋转 3/3，数量已达最低要求。
运行时旧汇总让全局缺失的一条物理分支覆盖了这些计数；离线 v1.2 已修正为
测量失败，未知物理行仍单列，原报告不改。持久不动的 state 基线也独立于模型
解码计算，明确为使用最后真实 pose 的离线特权参考。汇总修正没有重跑任何
解码、渲染、模型或物理过程。

图像代理有正有负：接触 object-mask IoU 优于 persistence（L .510/.352、
T .757/.376），非接触则退化（L .490/1.000、T .817/1.000）。RGB MSE 包含
灰色转红色及目标外观丢失，不能单独判断动力学。T14/toward 单个旋转分支 H4
角误差 .814°，zero-rotation 为 11.455°，但仍不足以通过 T 旋转门槛。

在完整的 11 个 L、4 个 T 锚点上，hold endpoint coverage 为 .42309/.47851，
均匀候选期望 .39173/.44470，真实最佳 .42935/.48858；hold regret 为
.00626/.01007，均匀 regret .03762/.04388。这些是离线参考，不是模型选动作成功。

## 失败、预算与资源

首次执行因 Windows ledger 原子替换 WinError 5 中止，当时尚未加载 GPC。
已完成的 24 校准和 L00 四分支直接复用。L01/hold 在 reset 预留时失败，永久保留
失败，不重试、不补替；未执行的 reset/settle 也保守扣费。恢复沿用同一账户及
原截止时间，修复停机时间计入 30 分钟。

| 费用 | 实际/保守消费 | 上限 |
| --- | ---: | ---: |
| 分支槽 | 63完成+1失败 | 64 |
| 校准 | 24 | 24 |
| reset / restore | 87实际、88计费 / 87 | 88 / 88 |
| 控制 | 189 history+252 future=441 | 448 |
| 控制物理子步 | 4410 | 4480 |
| reset settle | 87实际、88计费 | 88 |
| GPC图像 / Euler | 756 / 2268 | 768 / 2304 |
| optimizer updates | 0 | 0 |
| 含停机 wall | 951.88秒，约15分52秒 | 1800秒 |
| 运行结果 / telemetry及账本 | 约10.50 / 1.90 MiB | 256 / 64 MiB |

原 worker PID 22072，恢复 worker 3516，均用实际 PID/nonce/Job Object；恢复
exit 0，无 guard stop。最低 RAM/commit/GPU 为 3.030/10.211/6.281 GiB；worker
working set 峰值1.210 GiB，process commit峰值2.341 GiB；CUDA allocated/
reserved 峰值约105.17/130 MiB。权重哈希、全网络fingerprint、eval/freeze及
Python/NumPy/CPU/CUDA RNG 检查通过，网络已释放。监控不能保证采样点间瞬时内存。

独立 reconciliation 28项全通过：原 B/C 的150个工件（97,364,187 bytes）、原校准/
分支、失败证据及两个checkpoint均不变。1789 events/893 operations与账本一致，
无pending、无退款。63个完成分支均未严格成功，全部用完四步候选；不是策略成功率评估。

## 审阅入口与下一步

- `raw_results.csv`：768固定行，保留失败分支12个缺失行。
- `branch_summary.csv`：256个分支/H1–H4行。
- `contact_rotation.csv`、`coverage.csv`、`ranking.csv`、`gates.csv`：分层表。
- `resources.csv`、`costs.csv`、`reviewed-summary.json`、`manifest.json`：资源和来源。
- `../frozen-wm-validation-v1-20261024/previews/`：16锚点、2校准、1误差曲线图。
- `../frozen-wm-assets/final-independent-reconciliation.json`：独立核对。

原报告SHA256：`c7b1e3257468874940ad074d3ffc00c3d3d88f9f6de16531aba6d2c3cc6e3795`。
focused tests 39通过；source-only全量217通过、160跳过。

**下一步先修生成图像测量接口与近 .95 终止精度，再用新冻结协议确定动力学适配需求。**
继续检查形变、非接触漂移和接触/旋转；GPC还缺可靠直接reward/contact输出。
当前没有进入 imagined policy 训练的证据，本轮没有自动训练或扩大预算。
