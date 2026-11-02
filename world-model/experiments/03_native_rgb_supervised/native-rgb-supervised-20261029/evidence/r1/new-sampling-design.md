# R1 新采样设计与R2待审请求

当前只交方案，**R2未获执行授权**。本轮0新物理、0模型/优化器。4A IO已停在`stage4a/io-paused-state.json`：新IO源码保留但未完成、无测试/数据准备执行；原采样器及360次CPU模拟证据保留。

## 依据与能下的结论

- 原24156个H1的面积质心位移中位0.4446、P95 0.9313、最大0.9623 **96px图像像素/0.1s**，>=1px为0；H4中位1.8722、P95 3.7083。18个微拟合窗全由旋转条件入选，并无>=1px平移支持。70%重抽样不会创造缺失幅度。
- 已确认`damping=0`每100Hz积分抹去旧动态速度；质量1、力上限10000、dt.01产生自由pusher新力项100worldpx/s，理想等质量正向接触约50worldpx/s，即0.9375图像px/0.1s。**这是条件量级，不是全局上界**：B已有5.132worldpx反例。角接触、惯量、碰撞缓存/位置bias仍需配对实验分离。
- 保留kp100/kv20、agent **pre-step speed cap200**、10×.01s、两body质量1、原形状惯量/CoG/碰撞规则。远目标中位122.096worldpx、推导首子步约52.11%力饱和，不保证再加target距离增加物体幅度。原力字段积分后清零，不是实际10子步力迹；有效Shape摩擦/弹性未保存，不能声称µ=1是原因。

详见`motion-amplitude-diagnosis.md`、`motion-amplitude-source-evidence.json`及`recording-alignment-audit.json`。面积质心用于位移；T的物理质量中心用于力臂，两者分开。不能靠跳帧、加速视频、改moving门槛或多训练宣称解决。

## 最小协议比较

| 协议提案 | 唯一物理差异 | 其余固定 | 状态 |
|---|---|---|---|
| P0-frozen | 无，space.damping=0 | kp100/kv20/F10000/mass1/pre-step200/100Hz×10/旧惯量及材料 | 冻结基线 |
| P1-damping1-candidate | **仅space.damping 0→1** | 与P0完全相同，不能同时改force/速度/质量/dt | 新独立协议候选，未实现/未验证 |

先看原P0下持续正常接触和偏心推转能否改善实际接触持续率，再用同scene/同绝对action的P1配对检验速度记忆。每次_setup/reset后、第一步前都应实测并记录damping和完整参数；P1必须新source/protocol hash、独立目录。当前JSON的proposal-spec hash不是已运行协议身份，实际材料值尚待获批日志记录。不能把新P1数据混入旧P0 TRAIN/val或重写历史结果。

2–4px平移对应每0.1s 10.667–21.333worldpx，4–8px对应21.333–42.667worldpx，5–15°也是待验证目标。同200 pre-step设置是否足够尚未证实；它也不能当object全局速度界。4–8px只是挑战缺口，本R2不自动升force/speed或改时间尺度。若安全目标不可达，直接回传证据，再另审单因素协议或明确H4时间任务；不降名为“明显大平移”。

## 两阶段R2：512 → 父审 → 最多1296

| 阶段 | 独立初始场景families | 执行槽 | 每槽cap | 最多新控制 |
|---|---:|---:|---:|---:|
| R2A小gate | 16：6中心、4墙+4角、2负例 | 16×2协议×2动作=64 | 8 | 512 |
| R2B（尚未选择协议） | T/L/O/C/X/B各9空间=54 | **父从gate明确接受的单一协议**×54 | 24 | 1296 |
| 合计 | 70 | 118 | — | **1808** |

R2A结束无论成功与否都停止、父独立审核；B不自动采用P1，父可选P0/P1或都不放行。24步是2.4秒cap，不代表接触/出墙/重新接触已发生。总提案上限<=2000，余192未分配、默认重放0、不得自动补seed/槽。旧余320不动；A的512本身已超旧余320，必须新R2明确授权的账本，不能重置旧40000限制。若全1808获批并完成，累计监督41488/历史总76048；现在仍39680/74240。

每个控制尝试先预扣；失败、半步、重放不退。任一实际子步穿透>2worldpx、非finite、physics/IO异常立即停**整个phase**，保留部分轨迹，无替代槽。planned/初始化拒绝/attempted/actual/completed/charged分列。初始化拒绝保留计划分母；192和旧320不用于“补齐”。

## 请求表与动作可复核内容

`r2-probe-requests.json`含70组完整场景spec与118槽，CSV是摘要，预算为`r2-probe-budget.json`。全部execution_enabled=false。只用6个既有TRAIN代表形状，不读8未见。

场景从冻结2A+2B的合法object pose/yaw与reachability metadata中，按letter/初始space/静态约束确定性选择，不依据旧运动成功或未来R2结果。对象整体平移到明确4/18/48worldpx初始墙gap；四墙、角落双墙都逐值保存。pusher只在**初始**设到已知外部可达8worldpx弧段的合法staging/contact路径；移位后的pusher-wall静态余量、物理CoG/面积质心、零初始v/force/torque、reset参数和source NPZ/meta SHA全存。此为静态几何计划，不是动态安全通过。

R2A同family两个protocol的初始参数及8个绝对actions逐值相同；配对action SHA已核相同。A0是持续法向，A1是有符号45°斜推或75°切向偏置，两个负例则hold对free-pusher退离；不是把新动作冒称原旧selector。前2控接近，4控持续推，1控撤回初始staging，1控重新接触。命令为真实PD目标，不移动状态。为保证target位于[0,511]，仅预先做**共同径向缩放**并记录实际向量/scale，未逐坐标clip；不能把缩小的命令或最终小位移称为达成大幅度目标。

B的24cap使用JSON中的事件驱动B24规则：实际接近(<=3)→接触持续(<=5)→滑动(<=3)→撤离(<=2)→重接触(<=5)→再推(<=4)，共22控，最多共享2控恢复；不因尚未完成继续加控制。接触确认要求连续3个真实arbiter-contact控，route/staging到达另记；丢失接触走合法路径，不能传送或按时间表写成功。目标用当前body pose变换冻结localarc，只有采样策略可读状态，未来模型仍只收RGB/action。非法目标/不安全路径留失败/缺层，不clip/换seed来伪通过。

法向/斜向/切向偏置、左右符号、关于**物理CoG**的有符号力臂独立保存；位移仍按面积质心。持续压同一段不增加unique arc，技术配对也不增加独立scene family。计划并非真实接触命中；封闭洞/未确认路径不用，未到达保留缺口。

### 静态已发现的缺口（不替换）

- T/right-top、L/right-bottom、X/right-bottom：所选合法arc未达到请求的力臂bin。
- X/left-top：所选合法arc的计划方向未达到请求process方向对齐bin。

这4个B槽已在JSON/CSV标missing，未解决前不能当目标覆盖或自动dispatch；由父在B放行时明确处理，R1不换seed。初始scene参数hash只是计划JSON指纹；实际初始化后需记录并核对各paired arm真实state完全一致，不能用render-state接口继续物理轨迹。

## 新正式数据的幅度/支持建议（仍不是采集授权）

建议以二维平移/转角连续分布为主，以下只是**目标场景/条件结果配额**，不能删除接近/静止/失败来强凑raw比例：25%平移2–4px、10%平移4–8px挑战、20%旋转5–15°、15%明显混合、10%小运动、10%无接触、10%真实受阻/明确标proxy负例。未支持的10%挑战格直接报缺失，不能借小运动改名填入；新比例也不能替代视频验收。

命名预案：平移主导目标同时角度<1°；旋转主导目标同时平移<1px；混合目标平移>=2px且角度>=5°。这是待审目标分层，不改原2worldpx/1° moving定义。所有raw连续值、signed角、路径/端点和大小运动交叉表仍完整保留。

后续18 TRAIN字母可先按**每字母至少12个独立初始family**作为设计下限（并非足够训练/统计证明）：至少2平移、2旋转、1混合、1小运动、1无接触，以及进入/沿墙/离墙/受阻各1，整体仍需>=12不同family；每字母至少4个**实际命中**8px弧段、normal/oblique/tangent与力臂不同符号有独立支持。train/val整轨迹/scene/seed分离，验证预算和配额需R2验收后另审。R2只有6代表，不能宣称18字母已满足。

边界按过程的四个signed gap、角落两墙各自gap，报告0–2/2–8/8–32/32–64/>64worldpx与进入/沿墙/离墙/受阻实际次数；初始9空间不能代替过程覆盖。若无实际wall arbiter/impulse，near-static+contact+gap只能叫墙挡proxy，不算真实墙阻挡已证。

## R2必须补的因果日志与验收

每100Hz记录pre/post位置、速度/角速度、赋给body的PD force（限幅前后）、agent200速度截断前后；真实mass/I/physicalCoG与Shape friction/elasticity；agent-object/object-wall/agent-wall接触/impulse分类；10子步penetration与四gap。原control-boundary的force=0不能代替这些字段。

逐0.1s报告原始分母与contact条件分布的2–4/4–8px、5–15°支持；也保留小运动、no-contact、stalled contact和failed recovery，不用moving比例替代幅度。视频完全采用既有`video-acceptance.md/json`：每槽连续原96px+明示6x、10Hz/1×、全部失败与逐控曲线，不在R1渲染/生成视频。

R2产物建议候选cap128MiB；最大1926张原96RGB未压缩约50.78MiB，另留100Hz compact logs/视频/曲线。全部新旧产物继续3GiB，另留>=256MiB原子空间，不删旧权重/失败证据；执行前/逐轨迹按当时真实余量复核，超限停回传。

下一步仅是R1回传父审；父明确释放R2A后才实现/冻结诊断记录与候选协议并执行有界gate。R2B仍再审，全部样本幅度与连续视频验收通过后，才另审正式TRAIN/独立验证和训练。不能保证只换动作、只改damping或多训练就修好运动/圆化形状。

计划独立families/计划unique arcs（非actual hit）：{"T": {"families": 12, "planned_unique_arcs": 12}, "L": {"families": 13, "planned_unique_arcs": 12}, "O": {"families": 12, "planned_unique_arcs": 12}, "C": {"families": 11, "planned_unique_arcs": 11}, "X": {"families": 11, "planned_unique_arcs": 11}, "B": {"families": 11, "planned_unique_arcs": 11}}。
