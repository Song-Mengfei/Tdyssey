# R1：质心、单位与记录时间对齐只读核查

结论：在既有 TRAIN18 与 DEV36 的 54 个冻结窗口、50 条独立轨迹中，没有发现把小位移人为缩小的状态压缩、动作错位或世界/图像缩放错误。当前运动指标用**面积质心**，不是 body origin。此处只核查记录链路，不重复全量 TRAIN 运动分布，也不证明新采样方案能产生大位移。

本次 0 新物理控制、0 模型/权重/Torch、0 渲染、0 新测试。逐窗数值、输入 SHA256 和源码 SHA256 在同目录 `recording-alignment-audit.json`。4A 的已做/未做边界另存 `../stage4a/validation-suspended-boundary.json`；不再开发或测试 H4 接口。

## 面积质心、物理质量中心与 body origin

状态列 2:4 保存 body origin，列 4 保存未取模角度，不直接代表视觉物体质心。状态写出 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:153`)；世界几何变换 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:172`)。

统计使用 `c_area(s)=origin+R(theta)*local_area_centroid`，相邻帧的平移为 `norm(c_area(s[t+1])-c_area(s[t]))`，转角使用保存的未取模角度之差。面积质心公式 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_motion_statistics.py:47`)；位移和角度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_motion_statistics.py:86`)；面积加权局部质心 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/geometry.py:126`)。

T 保留原几何原点：局部面积质心为 `(0,40.7142857143)`，原物理质量中心 CoG 为 `(0,45)`。其余 17 个训练字母的面积质心在原点（浮点误差约 1e-14）。因此不能把“面积质心”和物理 COM/CoG 写成完全相同的量。T 几何与质量中心 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/envs/official.py:13`)。本审计保留两者，不改变旧统计定义。

实际 T 窗口 `train-2b-175:16`：

| 同一 0.1 秒转移 | 位移 world px | 换算 96px 图像 px |
|---|---:|---:|
| body origin | 2.155766924 | 0.404206298 |
| 面积质心（原统计） | 3.004925517 | 0.563423534 |
| 物理质量中心 `(0,45)` | 3.175907420 | 0.595482641 |

该步转角为 −3.184266290°。原记录 `observed.area_centroid_translation_world_px` 和冻结窗口的位移均精确等于面积质心复算值，误差为 0。用 body origin 反而会低估此例，但原统计没有这样做。该区别不能解释其余 17 字母也出现的小幅 H1 平移。

## 世界坐标到 96px 图像

真实几何在 512×512 世界/栅格中绘制，再用 OpenCV `INTER_AREA` 缩小到 96×96；位移比例是 `96/512=0.1875`，即 1 图像 px = 5.333333333 world px。原始栅格和缩放 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:219`)。2 world px 的旧 moving 平移阈值仅为 0.375 图像 px；它不等于肉眼明确的大平移。

54 个冻结窗口中的原位移与该缩放复算误差为 0。读取既有 full-object-mask 的面积加权栅格质心后，掩码位移与连续几何位移的最大差为 0.253184475 图像 px（`seen-val-2c-039:5`）；T 上述例差 0.028406651 px。这里披露差异，不把几何质心当逐像素测量完全相同：整数顶点栅格化、面积重采样、像素中心约定均会影响掩码质心；可见灰色掩码还受到蓝推杆遮挡。此轮没有新渲染来逐项归因，也未改变校准或阈值。绝对像素中心的固定偏移不能拿来改变位移比例。

## 动作、状态与模型时间

采集循环先保存 `s[0]`，每次仅执行一次 `env.step(action[t])`，随后依次保存 `s[t+1]`、`action[t]`、接触与 10 个子步穿透审计；规划前后及 observe 后都检查物理状态未被修改。原采集循环 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/scripts/native_rgb_stage2a.py:150`)。每个控制步固定 10×0.01 秒，因此 H1=0.1 秒，H4 端点=0.4 秒。固定步进 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:188`)。2B/2C 复用该采集路径。

| 以 anchor=a 为例 | 真实索引 |
|---|---|
| H1 四帧历史 | `s[a-3],s[a-2],s[a-1],s[a]` |
| H1 四动作条件 | `action[a-3:a+1]`，末动作实际把 `s[a]` 推到 `s[a+1]` |
| H1 真值 | `s[a+1]` |
| H4 所需七动作 | `action[a-3:a+4]`；第 4 次调用使用 `action[a:a+4]` |
| H4 端点真值 | `s[a+4]`，不能把路径累计位移当端点平移 |

Dataset 五帧/四动作窗口 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_data.py:647`)与原始 H1 loss 第五帧目标 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/raw_world_model_training.py:203`)一致。旧缓存生成索引 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/scripts/native_rgb_stage3_prepare.py:97`)也一致。微拟合选择条件是“平移至少 1 图像 px **或**转角至少 3°”并要求真实接触，不是纯大平移条件。固定选择条件 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/scripts/native_rgb_stage3_prepare.py:43`)。全量幅度/旋转条件核查由独立运动报告提供。

54 个窗口均通过：原 NPZ/metadata SHA 与冻结引用相同；每条 65 个 float64 状态、64 个动作、64×10 子步；metadata step 为 0..63；初始状态等于已保存 scene；无中途 reset 声明；接触计数等于该 control 的 offsets 差；子步最大穿透等于保存的 control 穿透。缓存 `actions_absolute` 与对应原动作切片逐值相同；`last==histories[:,3]`，`hold==last`，窗口 ID 顺序相同，7 转移 H4 合法标志均真。

## 压缩与证据边界

NPZ 使用 ZIP_DEFLATED，对已存 NPY 数值无损。状态保存为 float64，没有压缩成像素整数或少存时间帧；动作在 ZIP 前明确保存成 float32，接触点也为 float32。保存 dtype (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/scripts/native_rgb_stage2a.py:193`)。54 窗口的保存动作重算目标距离与原 float64 planning metadata 的最大差仅 `1.36161012e-5 world px`；这是 dtype 舍入，不是约 5 world px 运动被压小的证据。

本次仅有限核查 54 个原冻结窗口，不把它宣称全 396 条轨迹的重新物理验收；原全量 physics-audit 仍独立保留。没有逐子步 wall-clock 时间戳，也没有 100Hz 施加力/速度完整轨迹，不能从控制边界常为零的 force 推断“整段没施力”。现有代码时间链路和缓存没有支持跳帧/错位/缩放制造小位移的解释；物理更新约束、采样策略和可达幅度的原因应结合独立物理审计判断，实际 R2 探针尚未授权。
