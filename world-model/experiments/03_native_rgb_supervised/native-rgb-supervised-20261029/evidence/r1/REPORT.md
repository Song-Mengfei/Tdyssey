# R1：小位移根因、重新采样方案与验收门槛

**当前停止训练。** 已保存未完成的4A工作，不执行4B、R2或任何模型/物理试跑。结论是：现有H1平移确实全部亚像素；记录链路没有发现把它人为缩小的错误；当前“有限力动态推杆＋每子步清除旧速度”的物理组合与观测量级高度一致。只读检查不能把这个解释升级成唯一根因或严格全局上界，下一步应先审核小型成对探针，再验收真实幅度和连续视频。70% moving重抽样及继续训练都不能代替这一步。

## 1. 实际幅度缺口

全部正式TRAIN为396条轨迹、25344个原始控制；有效H1为24156，H4端点为22968。未混入1/1B探针、旧native或seen_val。面积质心与转角均从既有状态独立复算，未渲染或步进。

| 单个预测间隔 | N | 面积质心平移中位 / P95 / 最大，96px图像像素 | 转角中位 / P95 / 最大 |
|---|---:|---|---|
| H1，0.1秒 | 24156 | 0.4446 / 0.9313 / 0.9623 | 0.7648° / 3.9281° / 5.6372° |
| H4端点，0.4秒 | 22968 | 1.8722 / 3.7083 / 3.7884 | 3.4563° / 14.9516° / 21.4483° |

H1中 **≥1、≥2、≥3图像px的平移均为0**。原moving14692/24156=60.82%，near_static8447=34.97%，intermediate1017=4.21%；旧moving平移阈值2worldpx只有0.375图像px，不能把比例称为明显大平移。moving子集的中位平移也只有0.7985px。H1≥3°旋转3572窗、≥5°仅20窗；原18微拟合窗全因旋转条件入选，平移仅0.4817–0.8041px、转角3.0521–4.6706°，没有≥1px平移。

按旧2worldpx/1°门槛，H1分解为3645个仅过平移阈值、1841个仅过旋转阈值、9206个两者均过，以及1017中间/8447近静止；“仅过一种阈值”不等于另一分量为零。实际接触H1有16883窗，中位0.7438px；无接触7273窗，中位0、最大0.0520px。真正push阶段13011窗的中位也仅0.8256px、最大0.9623px，说明不是只把接近/静止帧混入后才显得小。

完整18字母×4层、实际接触/无接触、实际墙/角/间隙、动作阶段与连续分布在统计JSON (external historical reference: `motion-amplitude-statistics.json`)，解释在[独立幅度报告](motion-amplitude-diagnosis.md)。H4分组中anchor层/阶段与四步端点/路径含义分开。现有540个likely-wall-blocked只是代理；没有object-wall arbiter/impulse记录，不能冒称都已证实被墙挡住。

## 2. 确定事实、条件推导与未知

确定的本地控制为动态pusher质量1、物体质量1、原形状惯量，`kp=100/kv=20`、力上限10000、agent的**积分前速度限制200**，每控制步10×0.01秒。T仍保留原moment3000和物理CoG(0,45)。代码先计算并限幅PD力、限制旧pusher速度，再物理步进：参数 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:19`)、动态推杆 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:100`)、更新顺序 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/tdyssey/native_rgb_env.py:188`)。

继承的`space.damping=0`不是“无阻尼”。匹配已安装Pymunk6.9.0/Chipmunk7.0.3-dfc2fb8的引擎源码使用`damping**dt`，动态速度更新为旧速度乘该因子，再加`F/m*dt`；位置还包含独立bias项，力/力矩积分后清零。[确切引擎步进](https://github.com/vibloteket/munk2d/blob/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpSpaceStep.c)、[Body积分](https://github.com/vibloteket/munk2d/blob/dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970/src/cpBody.c)。原运动学pusher不经过同样的动态速度积分，而新动态pusher保留了这个设置。

| 证据级别 | 结论 |
|---|---|
| 源码确定 | damping0在每个100Hz动态积分清除旧速度；质量/力/时间步限制新的速度项。 |
| 既有实测 | 控制边界pusher速度最大100worldpx/s；物体物理CoG速度最大50.0098。所有25740条边界状态的force/torque为0，不能解释成整段未施力。 |
| 只读重构 | 用state[t]和已存action[t]重构首个子步，13208/25344控制达到力限幅，接触H1中12775/16883，微拟合17/18。不是已记录的全部10子步施力曲线。 |
| 条件量级 | `10000/1×0.01=100worldpx/s`；等质量、近似非弹性、正向无旋转简化接触约50worldpx/s，对应0.1秒5worldpx≈0.9375图像px。 |
| 不能声称 | 5worldpx是全局上界或唯一原因。B的既有`train-2b-163:15`实际达5.132211worldpx；未记录的子步速度、碰撞缓存/位置bias与接触惯量不能忽略。 |

原25344控制的目标距离中位122.0958worldpx、P90为224；H1合法子集另为145.7221中位，分母不混。远目标不等于物体大位移。目标更远但PD力已饱和时不能突破力上限；提高200速度限制也不是当前记录中已被证实的主瓶颈。`body.friction=1`不能直接证明实际接触Shape摩擦为1，真实材料值未保存，应在获批R2记录，不能把“高摩擦”当已证根因。源码与证据分级 (external historical reference: `motion-amplitude-source-evidence.json`)。

1/2/3图像px每0.1秒分别要求平均面积质心速度53.33/106.67/160worldpx/s，现有支持均为0。这不是动作可达性的数学不可能证明；也没有证据允许承诺只调动作就安全、稳定达到这些幅度。

## 3. 排除记录/单位误解，保留真实不确定性

[独立记录审计](recording-alignment-audit.md)核查54个原冻结窗口/50条轨迹：NPZ与metadata哈希、65状态/64动作、每控制10子步、动作与下一状态、H1/H4索引、缓存history/hold均一致。状态float64、ZIP无损；用保存float32动作重算target-distance，相对precast planning metadata的最大差仅1.36e-5worldpx，未把该标量差冒称原action XY逐坐标比较。

面积质心为`origin+R(theta)*local_area_centroid`，不是body origin，也不与物理质量中心混称。T局部面积质心(0,40.7143)，物理CoG(0,45)。同一T转移的origin/面积质心/物理CoG位移分别为2.1558/3.0049/3.1759worldpx，旧统计使用的是正确面积质心。世界到96px比例确为96/512；既有栅格mask与连续几何质心位移可差到0.2532图像px，不能当逐像素完全一致，也不足以解释整个数据的幅度分布。

未发现用错origin、缩放、状态压缩、动作错位或跳帧制造小位移的证据。本次没有新渲染来逐项分解栅格差异，也没有新物理干预来量化各因素贡献。

## 4. 最小方案与明确不做的改变

优先在原P0下设计持续实际接触、正常/偏心/斜推/切向、合法失联与重新接触，减少空转并测实际幅度；这能改善接触过程，不能保证补出缺失的大平移。最小机制对照是**单独新版本P1仅将damping0→1**，其余force10000、pre-step200、mass/inertia、kp/kv、dt、碰撞与2px质量阈值全部保留。P1尚未实现或运行，必须有新source/protocol hash、独立目录，不能混入旧TRAIN/val。

更高force、speed或不同预测时间尺度都不在本次R2请求。提高force cap也受PD目标范围和速度项限制，不能承诺实际达到新力或幅度。把H1改成0.4秒虽可使用现有更大H4位移，却改变任务时间与动作语义，不能靠跳帧/拼接/改播放速度冒充当前0.1秒H1。

新的实际采样目标是2–4px平移、4–8px挑战和5–15°旋转，均以**原始96px、单个0.1秒**计且尚待验证。建议条件配额为25%平移2–4、10%较大平移挑战、20%旋转、15%混合、10%小运动、10%无接触、10%真实受阻/明确proxy负例。没有支持的层直接列缺口，不改名字、阈值或用重抽样填满。具体定义、未来每字母至少12独立families/至少4实际弧段的初步支持下限见[新采样设计](new-sampling-design.md)；这不是足够训练的证明，也不是正式采集授权。

实际接触仍按8worldpx弧长分段，分开记独立场景/真实命中、法向/斜推/切向、力臂符号、滑动/失联/重接触；持续压同一点不增加unique覆盖。封闭洞、窄槽和未确认可达不混称。边界用全过程四个signed gaps、角落两墙各自gap，分进入/沿墙/离墙/受阻；初始location和planned arc不是实际覆盖。R2需补100Hz pre/post状态、限幅前后力/速度、实际Shape材料及对象-墙/推杆接触冲量分类。

## 5. 待父审的R2请求，不执行

| 步骤 | 场景/槽 | 控制上限 | 放行条件 |
|---|---|---:|---|
| R2A小gate | 16初始families×2协议×2动作=64槽，每槽8控 | 512 | 父明确批准新预算与仅damping1的候选协议；完成即停回传 |
| R2B覆盖检查 | 6代表字母T/L/O/C/X/B×9空间=54槽，每槽24控 | 1296 | 父先审核gate，再选择单一P0/P1或拒绝全部；不自动续行 |
| 全提案 | 70计划families、118计划槽 | 1808 | ≤2000；192未分配、重放默认0，旧余320不支取 |

完整请求JSON (external historical reference: `r2-probe-requests.json`)含来源NPZ/meta SHA、body origin/面积质心/物理CoG、零初始速度/力、reset参数、4/18/48实际初始gap、合法arc/staging与逐值action。32组协议配对的初始参数和动作完全相同；P0/P1唯一拟变参数确为damping。[根审查](root-proposal-review.json)、请求CSV (external historical reference: `r2-probe-requests.csv`)、预算 (external historical reference: `r2-probe-budget.json`)。这些是静态计划，**全部execution_enabled=false**，不是已执行的物理身份/安全证明。

4个B槽存在静态目标缺口：T右上/L右下/X右下力臂，X左上方向。原样保留计划分母，不派发、不换seed或替代槽；是否解决须在父放行B时明确。实际初始化与配对状态仍须获批后核验。任何子步穿透>2px、非finite、physics/IO异常立即停整个phase，失败/半步/重放计账，计划/启动/实际/完成/预扣分别报告。

幅度验收还须分别给全部原始控制与真实历史可用的H1/H4窗口分母。短gate中最初几个控制的运动，不自动变成有完整四帧历史的训练样本；R2探针本身也不会自动混入未来正式TRAIN或独立验证。

若R2A获批并全完成，监督/历史累计将为40192/74752；若全部1808最终获批并完成则41488/76048。**当前仍39680/74240，新增0**。512本身已超过旧余320，必须新R2明确授权，不重置旧40000账本。

## 6. 视频验收在训练之前

[视频验收规格](video-acceptance.md)要求全部槽及所有失败的真实连续视频：10Hz/1×、原96px与明确标注的6×显示放大、N控制对应N+1原始帧、原始状态/action/帧SHA关联、逐步面积质心位移/带符号转角/实际接触弧段/四墙和双墙gap/穿透曲线。终帧停留与物理时间分开，不插帧、修图、重绘、变速或挑最好片段。类别必须包括平移、旋转、混合、进入/沿墙/退出、受阻/失败和小运动/无接触；缺失如实报告。

R1没有新采样视频，R2尚未执行。后续R2只展示真实物理过程，不生成模型预测。未来模型主图严格分列“输入图像 / 真实下一帧 / 当前微调模型预测”，原错误颜色GPC只在单独明确标注的基线页保留，不混称训练数据。

3C仅说明同18样本的轮廓误差有所改善；圆化仍明显、DEV仍差于hold。**不能保证多训练、只换动作或只改damping就解决保形。** 次序固定为R1父审→明确放行R2A→实际幅度/视频/物理质量审查→另审R2B→验收新采样→重新设计冻结正式TRAIN及独立验证→最终另行授权训练。

## 保存与资源

4A的未完成接口、采样器和CPU检查均保留，边界见root快照 (external historical reference: `root-4a-recoverable-boundary.json`)、IO边界 (external historical reference: `../stage4a/io-paused-state.json`)、验证边界 (external historical reference: `../stage4a/validation-suspended-boundary.json`)。没有完成正式入口/IO集成验收；不能把旧4A方案直接恢复成4B执行。

R1模型/权重加载、原生推理、GPU/优化器、新物理控制、渲染、服务器和打包均为0。全轮模型账本仍closed250尝试/868.484000000004秒；旧native1899/5352.516单列不改。旧权重/数据/失败证据不删，新旧源/缓存/报告继续共享3GiB及256MiB原子预留。R2证据暂提128MiB上限，原96RGB的1926帧未压缩约50.78MiB；执行前须按实际余量复核，不能为腾空间删除旧证据。资源核对 (external historical reference: `storage-and-resource-audit.json`)。

本次诊断和待审方案已保存。前两次自动审批拒绝与当时waiting_user_callback_approval事实原样留档。随后当前对话用户直接授权各阶段报告/实验摘要/本机路径持续主动回传，见持续人类授权 (external historical reference: `../standing-callback-authorization.json`)；第三次实际回传成功，成功receipt (external historical reference: `callback-result.json`)。父已先行直接读取本地报告并下发仅R2A64槽/512控制指令，见[父R2A审查](../r2a/PARENT_REVIEW.md)，与callback成功分开记录。新阶段不含R2B或训练授权。
