# R2C1：初始检查点保存后的日志错误，训练/评估未开始

本轮已按授权启动唯一一次真实运行，严格加载原始GPC全部8,729,827参数，并保存t0检查点。随后日志参数重名触发TypeError，按“工程异常即停、不得自动重跑”规则关闭。**本轮0优化尝试/更新、0draw、0训练forward/backward、0原生采样；尚不能判断学习效果。** 不把未执行预测的NaN占位误称模型数值失败，也不归因于数据运动不足。

## 实际顺序和原因

运行前13项聚焦CPU合同/mock检查通过，2341个历史输入与原始4084...权重哈希核对通过。125个源码/配置文件冻结；GPU前检为3060Ti8GiB、used1711/free6314MiB、driver610.88。原始模型加载成功，296个parameter tensors全部可训练，state_dict每tensor的shape/dtype/numel/SHA/finite/min/max/mean保存在run/initial-tensors.json。没有推理预热或GPU benchmark。

save_training_checkpoint先torch.save→fsync→os.replace，返回{path,bytes,sha256,updates}；随后save_checkpoint调用 `event(events,"checkpoint",label=...,**info)`。日志helper定义 `event(path,kind,**values)`，info中同名path与第一个位置参数冲突，进入函数体前抛 `TypeError: event() got multiple values for argument 'path'`。因此t0保存成功、baseline尚未开始；cleanup_errors为空，CUDA资源清理后全局费用stage关闭。

这是实现错误。先前CPU检查漏了“真实检查点返回metadata→日志调用”的集成边界，不能凭这些检查宣称真实训练入口已验证。本阶段没有修源后重跑、没有改batch/模型/数据/物理参数。

## 费用、数据和检查点

| 项目 | 实际 |
|---|---:|
| 唯一执行次数 | 1 |
| 新optimizer attempts / completed | 0 / 0 |
| 实际draw / training forward / backward | 0 / 0 / 0 |
| native sampler batch attempts / completed | 0 / 0 |
| 全部模型相关秒（含加载、保存、失败和清理） | 27.359 |
| 全局attempts / completed | 250 / 250 |
| 全局模型秒 | 895.843 |
| 本轮CUDA allocator peak reserved | 35651584 B |
| 新physics / environment / rendering | 0 / 0 / 0 |

原global250/868.484000000004秒原样保留并追加一个R2C1费用stage；新27.359秒不退不清零。全局剩2750attempt/4504.157秒。历史另一native1899/5352.516单列；物理41976监督/76536历史以及旧余24/320不动。原300/1200子预算名义未花部分不能自动变成重新启动许可，本次once-claim已经使用。

本轮实际抽样分布为空（0draw），loss/梯度/LR更新/80%运动比例均未测量；不能把R2C0冻结4800计划冒充本轮实际暴露。41TRAIN/12seen-val及其861/252H1、738/216H4、80/20计划和全部原数据仍完整，源/物理/采样规则没有改变。

唯一检查点：`__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r2c1\run\last.pt`，updates0，35,126,889B，SHA256 `b0442673691e1d87ae9aa266040d3d949bd3c59dd14ccd41e63c548d21bfa14d`。保存接口包含model/新optimizer/scaler/scheduler/CPU+CUDA+Python+NumPy+noise RNG/PreparedInputs队列游标/配置源数据评估身份/进度及保存前费用链接。它是未训练的t0，不是训练后模型，**尚未完成实际重载验证**；冷态仅核大小和SHA，没有再次torch.load或构造模型。旧250权重未使用，旧payload未删。

## 完整缺测而非学习结论

| 冻结cohort | 窗数 | H1三方法评分 | H4三方法评分 |
|---|---:|---|---|
| TRAINvisual | 12 | 全部未执行 | 全部未执行 |
| seen-val natural | 12 | 全部未执行 | 全部未执行 |
| seen-val motion | 12 | 全部未执行 | 全部未执行 |

三方法为复制上一帧、原始T任务GPC、训练后模型，共216个计划endpoint评分行，全标未执行/缺失，灰物体RGB/IoU/轮廓、蓝推杆、逐窗胜负/relative improvement、静止漂移和shape unknown均不编造数值。baseline/final共288个四步原预测槽位实际生成0；初始化NPZ中的NaN仅是未尝试占位，不是模型输出，H2/H3/H4亦无实际范围或数值失败可测。复制上一帧图列只是原保存输入的视觉参照，本轮其CPU指标也未在计费stage执行，冷态不补算。

report/index.html提供所有36窗H1/H4的24页可读图：主列真值/视觉复制上一帧/未执行训练后预测，原始T任务GPC另页，全部失败/缺测保留。原报告器把NaN占位显示为nonfinite并有标题拥挤，原24页和正确JSON保留；独立artifact-only展示修复明确“未执行预测（工程中止；NaN仅占位）/复制未评分”，调整图头间距，未改变数组、模型、冻结score数学或评分。missing-raw-predictions.csv及abort-missingness.json列完整288缺测，display-repair-record.json留证。

这不是“模型不如copy-last”或“物体没有动”的实验结果；本轮没有产生可比较的预测/训练结果。R2B仅小规模学习数据准入通过、正式触点/方向/边界覆盖不足的既有限制仍保留，6seen字母也不能称8未见泛化。

## 可审核的最小修复，未应用/未重跑

offline-repair/proposed-logger-parameter-fix.patch只把helper第一参数path改为log_path及其一处引用，保留JSON metadata.path。纯CPU复现原错误完全一致，独立函数用真实metadata形状验证修复后绑定/日志内容；所有非event函数AST相同、建议源码syntax通过。独立审查全部8个event调用，仅检查点info.path有已知同类冲突。它只是日志修复建议，不是完整训练运行验证。

冻结trainer SHA `e882799be46ee02f7f1c0ffa5d69c2d3c826f2a06b92ab8cbb546b5feb9c683c`未改；原run/claim/费用/失败均保留。patch SHA `107bb4940821e7cbf4ae56e1e400c753157812684753110cb0f5a5c4f17d8ada`。后续是否应用、使用哪一新stage/费用链接并重试，由父下一指令决定；本轮不删除claim绕过限制，也不重新加载权重。

原执行命令：

```powershell
& '../Tdyssey/.venv/Scripts/python.exe' -X utf8 -B scripts/native_rgb_r2c1_run.py --config configs/native_rgb_r2c1.json --execute
```

该目录once-claim已用，不可直接重复。冷态可复核文件哈希与故障复现CPU脚本；任何新真实模型操作需下一阶段指令。stderr另含HF hub cache写权限提示，但原权重严格加载和t0保存已成功，明确停止原因是上述TypeError；没有下载/安装或清理cache。

## 证据与状态

model-work.json是实际关闭费用/错误/计数证据，execution-command-result.json保留原stdout/stderr；cold-integrity-review.json的20项检查通过，2341历史输入及125源/快照全部未变。source-identity/config/initial-tensors及原权重身份齐。新增产物≤256MiB、共享旧新/source/cache/父审+256MiB原子预留≤3GiB，具体storage-final.json；没有删旧数据/权重/失败。

本轮执行链已关闭、训练和评估未完成。完成故障报告/所有缺测页面/补丁建议的实际主动回传后保存receipt并waiting_parent STOP，不自动重试、续训、扩采或开server。
