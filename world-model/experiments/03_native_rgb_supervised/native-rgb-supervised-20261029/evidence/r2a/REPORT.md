# R2A：阻尼配对探针完成，采样与最终协议仍待父验收

**64槽全部完成512次控制/5120个100Hz子步，0物理失败。P1出现更大平移和旋转，但失联滑行、墙接触与字母缺层明显，不能直接作为正式采样验收通过或选定最终环境。训练继续暂停。**

全部32对连续视频与64原96px无损视频 (external historical reference: `media/index.html`) · 逐槽/逐窗统计 (external historical reference: `report/r2a-report.json`) · [实际条件与覆盖诊断](report/DIAGNOSIS.md) · [独立遥测机制审计](telemetry-audit/REPORT.md)

## 执行范围和实际配对

只执行父批准的原请求中R2A64槽：16独立初始family×2动作×P0/P1，每槽8控制。先跑冻结首T-center A0的16控制并独立检查，再执行剩余496；没有预热、调试控制、补seed、替代槽或重放。只有每槽开头一次reset，没有中途摆放/拼接。两臂实际state15逐值相同、初始96RGB相同，全部非damping物理参数、实际Shape材料、源几何/质量/惯量/CoG、求解参数和绝对动作均严格配对，共32对。

|配置|实际阻尼|协议SHA256|
|---|---:|---|
|P0-frozen|0|fd358dba8f26a616ec6c55b3f6d532681bf5036e2cf79fe5a467001e3a77ae96|
|P1-damping1-candidate|1|f2bf3e796b4d9ef69a5f5b7b0e0131e272c9b0a1cc483dbbf401202af983f406|

P0复用原配置；P1仅space.damping0→1，新候选版本/独立槽目录，原协议、旧数据集与训练索引未覆盖或混入。其余仍mass1、kp100/kv20、力限10000、推杆步前速度限200、10×dt.01/控制.1s。**两臂所有agent/object/wall碰撞Shape实测摩擦=0、弹性=0**，原body.friction属性不能代替Shape材料。

原始请求保留proposal的execution_enabled=false不改，具体R2A执行授权由[PARENT_REVIEW](PARENT_REVIEW.md)、[独立冻结配置](frozen-config.json)和当前stage-state显式给出；原118槽总提案中属于R2B的54槽从未调度。原余320不挪用。

## 实际幅度：raw、H1、H4分开

面积质心=body origin+R×局部面积质心，world→96px比例96/512。T面积局部质心(0,40.7142857)与物理CoG(0,45)分开。H1=.1s；H4是.4s端点，不伪作单步。8控制/9图像仅产生5个完整四帧历史H1、2个H4；全体512raw/320H1/128H4均有效，数据读取错误0。窗口重叠，不冒称独立轨迹。

|总体/时长|配置|N|平移中位/P95/max（96px图像像素）|绝对转角中位/P95/max（度）|
|---|---|---:|---|---|
|raw .1s|P0-frozen|256|0.0960/0.7274/0.9354|0.1472/2.8351/3.9331|
|raw .1s|P1-damping1-candidate|256|1.4083/4.2696/7.3636|1.6655/11.3759/13.7253|
|H1 .1s|P0-frozen|160|0.1463/0.9166/0.9354|0.2827/3.0105/3.9331|
|H1 .1s|P1-damping1-candidate|160|1.8616/4.7469/7.3636|1.6655/12.0499/13.7253|
|H4 .4s端点|P0-frozen|64|1.1380/2.6638/2.6708|1.7487/9.3751/11.0755|
|H4 .4s端点|P1-damping1-candidate|64|7.4463/20.3160/27.6427|6.6250/47.5349/54.9014|

P0的160H1中>=1/2/3/4px均0，>=5°为0。P1的160H1中>=1/2/3/4px分别112/77/43/31，>=5°57、>=15°0；2–4px46、4–8px31。P1raw的对应目标为54/256和31/256，>=5°81/256。以上均为原始分布，不筛掉静止/接近/失联来凑比例；观察到的max不是全部合法动作的数学上界。

按R1联合目标条件，P1完整H1进一步拆解如下。括号内是该控制有agent-object仲裁接触的数量，不保证整步持续接触：

|联合条件|H1数量/160|有推杆接触|无推杆接触|独立初态family|
|---|---:|---:|---:|---:|
|平移2–4px且角<1°|15|10|5|6|
|平移4–8px且角<1°|16|3|13|3|
|旋转5–15°且平移<1px|18|1|17|4|
|混合：平移>=2px且角>=5°|18|7|11|3|

P1>=2px的77个H1中47个无推杆接触；>=5°的57个中43个无接触。P1仅45/160个H1有AO接触，P0为125/160。因此不能把较大幅度直接叫作持续推移或理想采样。联合目标全部raw/H1/H4支持 (external historical reference: `joint-target-support.json`)。

## 位移机制与滑行/碰撞代价

5120个子步的限幅后Body.force实测值与原PD精确公式重构相同；限幅前force是重构，非回调局部变量实测。P0/P1的无agent接触子步1160/1682次，其实测后速度均精确满足v_post=(damping**.01)×v_限速后+F×.01/m，残差0；无物体接触子步1100/1508次的物体速度残差也为0。与同一记录的另一damping代数计算明显不同。结合唯一D差异的配对干预，**D0每100Hz清除旧动态速度这一机制得到本次日志验证**；它是重要原因，不证明动作不足/接触/力限/时间尺度等其它因素不重要，也不证明全部合法动作有固定5worldpx上界。

P0无AO接触的75raw控制面积位移全0；P1无AO接触130raw的中位1.5409/max7.3636px，其中连object-wall也无的严格滑行115raw中位1.1684/max5.4464px。非零AO冲量+向物体法向施力的控制P0/P1为158/116，位移中位.2589/1.1801px。这些接触推动证据与滑行分列，不把后者当主动推移成功。

P0/P1实际AO接触raw181/126、object-wall接触22/50、agent-wall0/3。100Hz接触丢失转换33/55，重接触29/28；连续接触段57/56并非独立初态，也不自动代表计划撤回/恢复成功。P1发生932次推杆步前速度裁剪（P0为0），其子步后pusher速度max247.302、物理CoG速度max392.880worldpx/s；200只是推杆步前限速，不能据此声称物体或子步后速度受200全局约束。

本轮最大子步穿透依次object-wall1.847041、object-pusher1.594444、pusher-wall.505842worldpx，均<=原2px阈值，非有限/物理/IO失败0。仅支持已运行每槽.8秒的质量结果，不能保证延长到24控制仍安全；P1不是已选最终物理环境。

## 实际接触、方向与边界支持

接触点来自真实仲裁器pre/post回调，实际轮廓8worldpx弧段由当时物体局部接触点映射；冲量只聚合post_solve，避免pre/post双计。接近角、切向速度使用已披露的源更新顺序重构，非直接保存的velocity_at_world_point局部变量。P0/P1切向相对速度绝对值中位8.698/19.706worldpx/s；材料摩擦0，不把该运动叫作已证实的摩擦黏着/滑移定律。有符号实际法向力臂为cross(接触点−物理CoG,单位向内normal)，P0正/负/近零712/740/126点，P1为466/451/90点；是相关点样本，不是独立初态。C两臂均无负力臂支持，X的P0也无负支持，因此总体正负齐不代表逐字母平衡。方向的closing/separating/normal-stationary在P0为843/614/121、P1为463/445/99点（epsilon1e−8world/s；法向静止仍可能切向运动）。连续角分布及明确声明为事后探索的方向bins见signed-contact-support (external historical reference: `telemetry-audit/signed-contact-support.json`)，不会用plannedarc/45°/75°动作名代替实际触点或作为预注册验收。

|字母|独立初态family|两臂合计实际命中弧段|P1raw >=2px|P1raw >=5°|
|---|---:|---:|---:|---:|
|T|3|16|34|0|
|L|4|15|14|12|
|O|3|8|22|14|
|C|2|4|0|28|
|X|2|7|6|15|
|B|2|17|9|12|

这里只有16独立初态family、6代表字母，每字母2–4family；不是64独立初态，也未满足未来每字母>=12family的设计建议。多数弧段只有1family支持；连续样本不增加独立覆盖。C在P1没有>=2px平移、T没有>=5°转角；P1的4–8px支持仅T/L/O，B/X没有>=4px。旧4个R2B静态缺口继续未派发。

四墙四角初始槽全部执行，但覆盖按实际四个signed gap、相邻双墙gap和真实墙仲裁分组。raw端点过程有70趋近边界、42沿边、52离边；7控满足近静止+AO/墙接触组合，另1控仅近墙静止proxy。这个分组不是单独无墙反事实因果证明。字母间过程不齐：C未观察到沿边/离边，X仅1个离边控制；不能用初始location标签或总体均值宣称每字母各过程覆盖。全部逐子步gap/冲量/点/状态保留。

## 连续视频与可复核材料

576个原始96×96RGB帧、64个FFV1无损MKV、32个P0/P1并排10fps审阅MP4全部保留。每槽8控制/9帧、物理跨度.8s；编码.9s中的末端.1s显示停留单列，不算运动。64个原视频解码逐像素等于记录RGB/帧SHA；32对视频9帧/10fps均核验。原PNG/MKV是无损依据，H264仅显示编码。6×nearest仅显示放大，不更改幅度定义；无插帧、改速、裁静态帧、修物体、拼轨或额外物理渲染。首T与首角落X的实际第4帧已视觉核查。

数值/实际点坐标/弧段/四gap/穿透/动作与逐步曲线在原96区域外展示。画面“实际接触”是每控制中100Hz接触子步命中数(0–10)，不是独立事件数；点样本数也不是独立弧段数。真实物理失败/partial本轮0，保留全部64槽分母；partial兼容只有明确标注的纯合成工程检查，不冒称真实失败实验。

证据：[独立全数据连续性复核](root-integrity-review.json)、[first pair gate](root-gate-review.json)、[全媒体SHA/时序](media/manifest.json)、[codec/可视核查](media/codec-and-visual-review.json)、source snapshot (external historical reference: `source-identity.json`)、安装物理runtime身份 (external historical reference: `runtime-identity.json`)、冻结请求 (external historical reference: `frozen-requests.json`)、[原数据引用未变](old-data-reference-review.json)。

## 资源、命令与停止边界

使用既有venv/Pymunk6.9.0/Chipmunk7.0.3-dfc2fb8，0安装/下载。CPU上的14个纯mock/source预检没有构造真实环境或额外控制；实际采集只有以下gate/remaining两个一次性执行（独占claim防并发/重跑）。工程元数据opencv包别名读取错误和HTML隐式body插入定位错误均为0物理的本地记录处理错误，保留说明并未重跑任何仿真。

```powershell
& '../Tdyssey/.venv/Scripts/python.exe' -X utf8 -B scripts/native_rgb_r2a_run.py prepare
& '../Tdyssey/.venv/Scripts/python.exe' -X utf8 -B scripts/native_rgb_r2a_run.py gate
# root审核已保存首对后，且仅一次：
& '../Tdyssey/.venv/Scripts/python.exe' -X utf8 -B scripts/native_rgb_r2a_run.py remaining
# cold后仅离线：native_rgb_r2a_report.py / native_rgb_r2a_media.py / telemetry-audit/helper.py
```

实际命令/退出和输出见[gate](gate-command-result.json)、[remaining](remaining-command-result.json)、媒体 (external historical reference: `media/command.log`)。新账本512/512已关闭，原39680/74240账本不改或重置；累计40192监督控制/74752历史总控制。**R2A模型、权重加载、原生推理、GPU、优化器均0**，live模型账本仍closed250尝试/868.484000000004秒，旧native1899/5352.516单列不变。8保留字母封存，旧数据/权重/失败记录、未完成4A边界均保留；没有服务器、上传包或新增运动预测模块/学习目标。

最终存储metadata审计见storage-and-resource-audit (external historical reference: `storage-and-resource-audit.json`)：候选R2A产物<=128MiB，新旧native/source/cache/既有外部资源+256MiB原子预留<=3GiB，另预留1MiB收尾metadata/callback；逐槽均检查。具体byte是注明时点快照，非全电脑占用。

## 父审核前不推进

R1两次自动审批拒绝和当时waiting_user_callback_approval历史保留。随后当前人类明确长期授权阶段报告/实验摘要/本机路径主动回传同父对话，R1第三次已实际成功，见持续授权 (external historical reference: `../standing-callback-authorization.json`)及R1 receipt (external historical reference: `../r1/callback-result.json`)。父先前直接读本地报告与callback成功仍分开。

R2A已按当前用户持续授权实际主动回传成功，真实成功receipt (external historical reference: `callback-result.json`)已保存；当前waiting_parent并停止。父必须实际审查幅度、滑行/接触、边界/物理质量和全部视频；本实施未选择P1、未执行R2B/正式重采样或训练。本轮没有模型输出，因此也没有证实世界模型的刚性保形、运动预测或后续多训能修复圆化。
