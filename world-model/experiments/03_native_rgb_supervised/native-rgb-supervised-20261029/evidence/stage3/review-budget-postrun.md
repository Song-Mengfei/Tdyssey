# 阶段3单次执行后预算审查

结论：预算与冻结身份证据一致。本轮是**训练前原生基线输出范围触发早停**，尚未尝试梯度更新，不能据此判断GPC拟合失败。唯一实际执行session 67280以exit 1结束；本审查只读元数据、源码、原始预测数组及文件hash，未加载权重、调用模型/GPU或新增物理步。

| 项目 | 实际 | 固定上限/余额 |
|---|---:|---:|
| 新轮优化器尝试/完成 | 0 / 0 | 全轮3000次余额3000 |
| 全模型工作秒数 | 28.344 | 全轮5400秒余额5371.656 |
| 本阶段预算 | 0次 / 28.344秒 | 200次 / 600秒 |
| 新增物理控制 | 0 | 无新增授权 |

账本仅1个已关闭失败阶段，active_stage=null，最终model-work.json与持久账本一致。错误快照28.328秒早于finally；最终28.344秒包含清理和结账，最后一次心跳27.875秒到结账的0.469秒未遗漏。t0进度中的26.406秒是当时保存的进度快照，不能替代最终全模型计时。设备查询已在时钟内，首个基线批次实测0.641秒；完整基线未完成，measured_baseline_seconds=null。此停机由输出范围异常触发，未碰到时间上限或420秒新梯度截止。余额只是未耗用预算，已关闭的单次失败阶段不自动获得重试许可。旧native的1899尝试/5352.516秒单列保留；历史加本轮为1899尝试/5380.860秒，未重置历史成本。

首次冻结TRAIN批次只生成4张原生RGB。A窗train-2b-018:11的27648个值中6个小于0，最小−0.0196077823638916、最大1；其余3张有限且在[0,1]。原数组原样保存，阈值未改、无裁剪修补。18行仍保留：1范围失败、3正常、14未尝试；dev36及全部拟合输出均未生成。组合文件保留TRAIN18/DEV36的原分母，缺测为NaN，不能用3张正常输出替代整体成功率或拟合比较。

原生before_updates回调在固定loss、reserve_update、WindowOrder.take、AMP反传及optimizer/scaler训练step之前。运行progress为updates=0、samples=0、validation=[]，没有固定训练loss、窗口曝光或训练/采样日志。296项初始参数hash与基线回调hash一致。AdamW、GradScaler、scheduler已为t0初始化；它们未进入训练更新，异常清理zero_grad不产生梯度。拟合loss下降、梯度范数、参数更新和预测改善均**未测量**。

初始完整last.pt在全部训练状态对象建立后、基线前保存，回调异常路径明确保留该完整t0边界。现文件35096361字节，SHA256=eb7b8dfc17b1ae157ccc5f477cdcacc400891f1db54709c4ac428234d090c7cc，与run/error.json和归档审计一致；归档CRC通过且10个必需字典key齐全，没有best.pt或第二权重文件。此审查未torch.load；计划中的完整CPU恢复核验未被执行，因此不把归档key/CRC检查声称为逐tensor或exact-resume恢复证明。

冻结清单56个现源/快照、运行identity的15个现源/快照全部hash匹配；启动marker绑定相同source-identity，显式18 TRAIN/36 DEV索引未改变。原发布权重本次审查未读取。root确认基线范围风险提示在本轮启动后才到达，不构成采前已拒绝的警告；本轮仍按冻结严格协议早停。

证据：model-work.json、model-work-error.json、native-rgb-cost-ledger.json（上级目录）、run/error.json、checkpoint-archive-audit.json、predictions/frozen-train18.json/npz、comparison-arrays.json、source-identity.json及run/source-snapshot/identity.json。逐文件SHA与审查边界见review-budget-postrun.json。无源码更改、重跑、阈值变更或下一阶段执行。
