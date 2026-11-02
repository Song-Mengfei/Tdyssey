# 阶段1：监督式重新采样探针（2026-10-29）

**只完成阶段1。六字母原始有效H1候选的运动门槛与物理审计通过；尚未训练模型，后续阶段均未放行。**

## 实际执行和预算

固定六字母T/L/O/C/X/B，各中央、四墙、四角9类，共54条独立连续轨迹、每条64步。
本阶段实际3,456次控制，全部完成。保存34,560个100Hz子步的float64穿透检查；
0非法转移、0部分步、0失败轨迹、0初始化拒绝。固定2px阈值未放宽；
物体-墙/物体-推杆/推杆-墙最大穿透为0.463885/1.007387/0世界px。
采集wall time约103.438秒，离线统计/展示另记。剩余544控制不自动使用；54个场景名额已用完。

finite-force-pd-v1全部物理参数、96px渲染和绝对XY动作语义保持不变。初始化后，
策略只提议动作；runner逐步检查完整状态连续性并独占step。没有瞬移、轨迹中重置、
拼接历史或8px动作限幅。近/中/远目标32/96/224px，实际目标距离连续分布完整报告。

新增优化器更新、模型推理、GPU工作、未见字母预测和服务器作业均为0。
旧native账本仍为34,560控制、1,899训练尝试、5,352.516训练秒；新旧合计38,016控制。
旧实验完整保留。新旧产物及native源文件约1.37GiB，仍低于共同3GiB上限；
精确字节数和实际可用磁盘见resource-ledger.json与stage1-result.json。

## 原始分母与运动结果

moving：单个0.1秒控制步面积质心平移≥2世界px，或原始unwrapped转角绝对值≥1°。
near_static：平移≤0.5世界px且转角≤0.005rad；其余intermediate。
几何/状态只用于采样、审计和离线分层，不进入网络。

| 数据口径 | 总数 | moving | near_static | intermediate |
| --- | --- | --- | --- | --- |
| 全部真实转移 | 3,456 | 2,243（64.90%） | 1,001（28.96%） | 212（6.13%） |
| 全部原始有效H1候选 | 3,294 | 2,228（67.64%） | 947（28.75%） | 119（3.61%） |

每条前3步不足四帧历史，结构性排除共162；0无效历史排除。所有静止片段保留，
没有删静止窗口后算门槛。每字母全部真实转移及候选分布 (external historical reference: `statistics/motion-statistics.json`)
和逐控制CSV (external historical reference: `statistics/real-transition-statistics.csv`)均保留。

| 字母 | 真实控制 | H1候选 | 候选moving | 候选near_static | 40%/45%目标 |
| --- | --- | --- | --- | --- | --- |
| T | 576 | 549 | 399（72.68%） | 131（23.86%） | 通过 |
| L | 576 | 549 | 412（75.05%） | 124（22.59%） | 通过 |
| O | 576 | 549 | 359（65.39%） | 167（30.42%） | 通过 |
| C | 576 | 549 | 400（72.86%） | 128（23.32%） | 通过 |
| X | 576 | 549 | 304（55.37%） | 229（41.71%） | 通过 |
| B | 576 | 549 | 354（64.48%） | 168（30.60%） | 通过 |

候选平移中位数/P95为3.412/4.981世界px，即96px图像0.640/0.934px；
转角中位数/P95为0.917/3.599°。moving不代表图像上大幅可辨运动，不能代替
阶段3原生采样器的真实拟合检查。统计包含完整连续分位数、零值、旋转弧长与图像尺度。

| 实际接触（H1候选） | moving | intermediate | near_static |
| --- | --- | --- | --- |
| 有 | 2,228 | 119 | 109 |
| 无 | 0 | 0 | 838 |

拟训练70/10/10/10仅分析支持量，未实际抽样或训练。六字母四层都存在；按全量3290索引，
moving/intermediate/静止接触约需1.03/2.76/3.02倍有放回重复。C静止接触约4.91倍，
L intermediate约4.15倍；不能声称是独立新增样本。不放回的总体精确配比只支持1090索引。

## 空间、墙角和未解决问题

九类初始空间各6条轨迹、384真实控制、366候选。48个边界初始化的真实间隙4/18/48px
各16条，中央6条均>64px。四角同时保存两面墙的间隙，最近一墙不冒充双墙都靠近。

| 初始空间类 | moving/366 | near_static/366 |
| --- | --- | --- |
| center | 242 | 112 |
| left | 270 | 83 |
| right | 260 | 91 |
| top | 265 | 89 |
| bottom | 254 | 99 |
| left-top | 191 | 161 |
| right-top | 244 | 103 |
| left-bottom | 231 | 121 |
| right-bottom | 271 | 88 |

**不能用平均掩盖薄弱场景：B左上角仅5/61 moving，O左上角14/61；
X左下/中央/右上分别15/18/20（各61）。** X恢复路由122个候选中121近静止、
只有2个接触，主要是失去接触后恢复不足，不能全归因于墙阻挡。
详见54轨迹的运动和四墙间隙 (external historical reference: `statistics/trajectory-motion-and-gaps.csv`)及[诊断](STAGE1_DIAGNOSIS.md)。

初始四角不等于整段持续在四角：1464个四角来源候选中，922个窗口的两墙最大间隙已>64px。
实际位置、四面有符号间隙、双墙最大间隙和边界行为均另报。轻微负间隙单列，
没有假装严格零穿透或放宽2px阈值。

全部真实控制的互斥边界诊断：沿墙moving611、朝墙414、边界释放442、近静止接触109
（墙阻挡代理43、其他66）、其他边界710、内部1170。没有墙接触仲裁计数；墙阻挡
只是结合对象接触、几何间隙和动作方向的明确代理。

## 真实接触覆盖和可达缺口

54/54轨迹发生真实对象接触，共2,564/3,456接触控制。初始预定弧段实际命中52/54，
其中O为7/9，其余9/9。固定8px弧段的碰撞点、接触控制步和独立轨迹分开计数；
历史覆盖只影响优先级，没有混入新数据命中。

| 字母 | 实际命中/全部轮廓段 | 命中/已证实可达下界 | 未确认可达段 | 封闭孔段 |
| --- | --- | --- | --- | --- |
| T | 51/60 | 51/55 | 5 | 0 |
| L | 44/53 | 44/50 | 3 | 0 |
| O | 33/93 | 33/42 | 14 | 37 |
| C | 43/75 | 43/54 | 21 | 0 |
| X | 42/84 | 42/46 | 38 | 0 |
| B | 41/98 | 41/47 | 13 | 38 |

已证实下界=规范合法可达∪九初始姿态合法可达∪实际命中，不代表全部可达轮廓。
未确认不能叫不可达；封闭孔没有通过瞬移进入。轮廓特征、未命中/未确认ID、角度、
独立轨迹和逐姿态间隙见[覆盖说明](coverage/stage1-reachability-summary.md)、[完整JSON](coverage/stage1-reachability-summary.json)。
接触控制明细 (external historical reference: `coverage/contact-controls.csv`)有3428个control/bin关联，对应2564个真实控制；
同一步多段接触不会多算真实步。

## 实际连续帧，非模型输出

!五帧连续对照 (external historical reference: `examples/motion-vs-near-static-contact.png`)

连续真实推动GIF (external historical reference: `examples/continuous-real-moving.gif`)；
连续近静止接触/墙间隙代理GIF (external historical reference: `examples/continuous-real-near-static-contact-wall-proxy.gif`)。
均从保存的连续真实状态离线渲染，0额外物理步，帧号/数据SHA见examples/gif-run.json。

## 验证、命令与下一阶段请求

23项采样/统计/计账CPU检查通过。独立复核54 NPZ、54元数据哈希和16项启动源码/
配置/上游/历史覆盖输入；旧核心物理、几何和历史训练账本不变。启动前冻结，未按结果改动。

实际命令（既有环境，无安装）：

```powershell
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_supervised_stage1.py collect --execute --config configs/native_rgb_supervised_stage1.json --output artifacts/native-rgb-supervised-20261029 --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_supervised_stage1.py audit --config configs/native_rgb_supervised_stage1.json --output artifacts/native-rgb-supervised-20261029 --physics-root ../Tdyssey/third_party/diffusion_policy
```

检查命令/哈希见pre-collection-checks.json；日志collection-stdout.txt、statistics-stdout.txt；
冻结配置source-snapshot/frozen-config.json；源身份source-identity.json；账本resource-ledger.json；
失败明细physical-failures.json；独立审查post-run-independent-review.json。
已有manifest拒绝重复采集，统计也拒绝静默覆盖。

**请求父对话审查**：[阶段2–5精确计划](PLANNED_STAGES_2_TO_5.md)已提交，但未执行。
建议先确认B/O左上角、X重接触和少数层独立支持量，再决定是否放行阶段2。
拟名义配额训练25,344+验证10,368+本阶段3,456=39,168控制，仍需父对话批准。
余量不自动使用。采样门槛通过不等于自动获得训练许可。

[监督备忘录](SUPERVISION.md)与stage-state.json记录父对话和回传规则：
完成后必须实际主动回传，再标记成功并停下，等待下一条阶段指令。
