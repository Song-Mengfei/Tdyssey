# R2C0：原始 GPC 学习诊断训练提案

**这是待父审的配置，不是训练许可。** R2C0 实际只完成保存数组、抽样和评估接口的 CPU 准备；0 Torch 导入、0 权重反序列化、0 模型/优化器/CUDA、0 环境/物理。后续最多 300 optimizer attempts / 1200 秒总模型计费尚未预扣或执行。配置为 `configs/native_rgb_r2c0_proposed.json`，`is_proposal=true`、`execution_authorized=false`、`actual_model_entry_ready=false`。

## 初始化与实际设置依据

只允许本地原始 GPC phase-two denoiser：`../Tdyssey-world-model-validation/artifacts/frozen-wm-assets/denoiser.pth`，35,043,916 字节，SHA256 `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`。本阶段仅流式读取文件字节核 SHA，没有 `torch.load`/pickle。其固定来源为 `han2019/gpc_checkpoints`、revision `c16a467ec176af19bba5136564a81d8d61d72af9`、`world_model_checkpoint/phase_two_checkpoint/denoiser.pth`；不是 GPC action policy。源码 commit `a96146e568826ef46134f73c22fdf59a675e7264` 的本地 20 个清单文件全部匹配。

旧 stage3C 的 250-update `last.pt`（SHA `7a3c3fed60f444536afe8999d6f6346d1ce41258f8df7d87dc78f18bcb4ba28f`）明确拒绝用于本轮初始化或 resume。未来若获批，应从原始权重新建 optimizer/scaler/scheduler/model RNG 和新数据队列；旧结果和账本保留。

模型保持原始 96px、四历史 RGB 与四动作条件、预测第五帧的原 GPC image/action→image 全网络。训练仍是原 H1 phase-two residual MSE；不添加 pose/motion head、几何 loss、前景加权或轮廓修复。原生采样器保持三步 Euler、sigma_min .002、sigma_max 5、rho 7、order 1、s_churn/s_cond 0。几何/运动标签只决定离线抽样和审计，不进入网络输入。

| 拟参数 | 值 | 依据与范围 |
|---|---|---|
| seed | 5478 | 本机 native 固定 seed；不是声称照抄上游脚本的 seed1 |
| effective batch / microbatch | 16 / 4，累计 4 次 | 原 native 工程真实 batch1/2/4 backward 选择4，3B/3C 实际沿用 |
| AMP / loss | FP16 autocast，loss FP32，GradScaler 初值128 | 既有 native 训练实现；不是上游训练脚本的完全同精度复刻 |
| optimizer | AdamW，lr1e-4，betas .9/.999，eps1e-8，weight_decay .01 | 原 native 常量；上游 phase-two 也明确 lr .0001 |
| gradient norm cap | 1.0 | 原 native 训练实现 |
| scheduler | CosineAnnealingLR，T_max3000，eta_min1e-5 | 保留原总训练时程；本阶段300次上限不暗改为300步余弦时程 |
| CUDA reserved 上限提案 | 6GiB | 沿用本机限制；本阶段未查询或试跑 GPU |
| 数据缓存 | 1MiB 保存数组 LRU，无 renderer worker | 新数据直接读取原 RGB，禁止新建环境重渲染 |

既有硬件记录为 RTX3060 Ti，约8GiB、Torch2.5.1+cu124。工程 microbatch4 实测 peak reserved 2,055,208,960B；3C 含恢复/推理/训练的记录为 2,059,403,264B。它们是历史测量，不保证新的 H4 评估在未来同样用量/速度，不借本阶段新增 benchmark 或自动 microbatch 重试。路径、SHA、版本和来源层次见 `source-settings-audit.json`。

## 本次真正送入模型的数据与抽样

只引用已选 0.1 秒半衰期协议 `58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6` 下的 R2B53 条完整轨迹。原始 NPZ/RGB 不复制、不重标签；旧 D0/D1/R2A/R2A2 不混入。完整 family 切分为 TRAIN41 条/861 H1/738 H4，seen-val12 条/252 H1/216 H4；静态受阻 X-left-top 只留覆盖缺口，没有伪造样本。

保存数据、split、H1/H4 catalogue、pools、480/4800 draws 及恢复状态的**实际文件 SHA 已写入提案配置**。明显运动定义为每0.1s面积质心平移≥2个96px图像像素或 abs角≥5°；80%明显/20%其余，先均衡字母，再独立 family，再类别/窗口队列。480周期每字母80（64/16）；4800只是最多300×16的暴露序列，不是新增独立样本，也不机械重复第一周期。

网络输入只返回 `image float32[B,5,3,96,96]` 和 `action float32[B,4,2]`。RGB 为保存的 uint8/255，动作仅一次归一到[-1,1]。字母、family、pose/state、几何与类别在审计侧，不进入模型 payload。

## 明确的新接口，而非改写旧 4A

`tdyssey.native_rgb_r2c0_data.R2C0Dataset` 与 `metadata_only_sampler_factory` 提供新 manifest/catalogue 和 80/20 队列。新 `tdyssey.native_rgb_r2c0_interfaces.PreparedInputs` 通过显式 data/sampling/evaluation identity 接入：`begin_update()` 给16个索引，`next_microbatch()` 给4个数组样本，`state_dict/load_state_dict` 保存队列/RNG、pending indices、micro cursor、draw count 和总 checksum。篡改 pending/cursor/draws 会在修改 sampler 之前拒绝。

CPU 中途 microbatch 恢复检查**仅证明数据状态恢复**，不恢复在途梯度。未来真实训练只能在完成 optimizer update 的干净边界保存/恢复完整 model、optimizer、scaler、scheduler、模型 RNG 与数据状态，不能用这个数据 snapshot 冒充训练 exact-resume。

`evaluate_prepared_cohort` 显式接收 rollout callback、predict callback、cohort 名、冻结 identity 和 execution guard；guard 拒绝时不调用 rollout。它只转交保存历史、七动作和窗口 ID，不转交 future truth/state/geometry。新评估 callback 接收的是已归一动作；不能直接塞给旧“接收绝对动作并内部归一”的 wrapper，避免双重归一。未来真实 native predictor 与费用 guard 必须另行明确接线。

原 4A 的旧文件/快照完全保留。其 sampler 绑定旧18字母/396轨迹/70-10-10-10，IO 绑定旧2C/DEV36，旧 train 的 `last_only` 强制18/36，formal 强制36与250/500。那些入口不能直接运行本提案，不能用 `FunctionType` 或 private globals 绕开。旧 formal_io 当时无测试或实际 plan/actions7 准备，root 最后一个恢复校验修改也未在暂停后补测；不改写为“4A已完成”。本轮由新数据/评估/数组接口替代必要准备，实际 model/training entry 仍未实现或获准。

最终联合 CPU 证据为 `interface-joint-cpu-check.json`：真实数据首16索引与冻结 draws 一致，4样本 payload 正确，数据中途恢复一致，三种 checksum 篡改被拒，三个实际12窗 cohort 与保存数组 identity/对齐相符。36次 callback 是 copy-last **CPU stub**，不是 native 模型推理；原 closed ledger 字节未变。root 另有 `root-joint-interface-audit.json` 的完整4800序列及队列恢复检查。

## 后续评估与费用提案

固定三组：TRAIN可视化12、val-natural12、val-motion12；三组分开报告，motion筛选组不称自然测试，seen-shape不称未见泛化。新 `evaluation/indices.json`、`protocol.json`、`preparation-audit.json` 的实际 SHA 与全部 cohort identities 已绑定。copy-last不需模型；原始 GPC 和最终训练模型使用相同窗、batch4与配对种子。H1共享四步 H4 rollout 的第一步，每个模型最多36个 native batch calls，共最多72，不重复推 H1，也不复用旧3B数据上的 baseline。

未来仅拟“原始 baseline＋最后一个完整训练边界”两次评估，不另加250中间评估。保留原始 float RGB；有限越界按原 native 语义保留、原值回灌并计 horizon 失败，不裁剪修色；nonfinite/shape/OOM 停整段模型工作，缺测保分母。报告分别列明显运动 H1/H4、静止漂移、灰物体前景 RGB/IoU/轮廓、蓝推杆和硬失败。形状/洞/刚性不可判断时保留 unknown，不能靠推杆变好或 denoising loss 下降声称物体运动/保形通过。

拟上限是300个 optimizer attempts或1200秒**全部模型工作**，先到停，最多4800训练暴露，不保证完成300更新。计费包括原始加载、baseline H1/H4、训练、最终评估、完整保存与必要 reload/核验、失败和清理。建议最晚900秒停止启动新更新，至少300秒留尾部；实际 baseline/最终评估/保存估计较慢时只能更早截止。开始每个模型操作与更新前检查费用、存储及剩余 headroom；不能把加载或保存移出计费时钟。

原全局账本仍 active_stage=null、250尝试/250完成/868.484秒，SHA `2729b06c8e565618624c65e1835701f2a37eabbcc9da0722c5bccad8bd7c40e6`，本阶段前后逐字节相同。全局3000/5400余2750尝试/4531.516秒，300/1200仅其中子集且未 reserve。旧 native1899尝试/5352.516秒单列保留。未来 checkpoint 拟只保一个完整 last payload及原子预留，不靠 denoising loss 选 best；是否可保存/reload仍须未来运行前按实际余量检查。

R2C0自身产物≤32MiB，本负责人≤3MiB；全旧新资源含256MiB原子预留≤3GiB，旧证据一律保留。当前没有模型 checkpoint 或预测数组。本阶段完成后停止，等待父对数据/抽样/评估索引与提案的审查；不能据此启动300/1200、扩采或训练。
