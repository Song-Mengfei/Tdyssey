# LOSS-DEFINITION-20261102 最终复核

三臂各128尝试/128成功更新、2048正式原生H1曝光，累计384更新/6144曝光，AMP跳步0；3744帧评测完整，saved-only审计实际退出0。三臂均未通过综合方向性条件和原绝对验收；仅NATIVE_G的C通过短时方向性子条件，C的H16仍退步。

本轮已实际完成于2026-11-02 22:06:15新加坡。原用户要求改变“什么定义为预测错误”：因此这次测试RGB派生的占据几何与空间输运，和上一轮SNR/GDL/FFL分开。以下不把工程通过当模型通过。

## 实际方案与比较边界

- G_ONLY = cG × 平方分母Dice几何损失，明确替换native；没有暗加RGB/残差锚点。NATIVE_G = native+cG×G；NATIVE_T = native+cT×(16方向空间投影W1+相对面积差)。灰物体与蓝杆等权，白背景不作为第三类。
- 每臂独立从完整E2恢复model/Adam/scaler/global与conditioning RNG。训练使用原HMF前128批、effective16/micro4、LR1e-5、clip1、原FP16 forward/FP32loss、50/50真实历史clean/weak。物理、数据、split、动作、曝光顺序、原3去噪raw推理和原评测资格/门槛未变。
- D=c_skip*z+c_out*F来自同一次native forward，其中z含加噪GT。新目标优化的是该单次连续重建；不能说直接优化了纯噪声AR16采样轨迹。无STE/输出warp/模板头/privileged未来状态或隐藏轮廓。
- E2和原BASE128只读复用；BASE的旧R1A128原raw SHA和主指标exact配对证据被审计保留，未为本轮换baseline或重训BASE。
- 每臂正式曝光2048、原目标唯一数768；不是2048条独立轨迹。评测train12/seen-val24/旧失败3共39家族，每臂RH与AR各16时刻，共1248帧。RH给每时刻真实历史后独立预测一步；AR16只回灌自身原始输出，无未来真值。train12不是全部48训练片段验收。
- T/L/X与其它字母、其它seed、其它loss系数/方向数、长训练收敛、独立泛化均未新测。文献中的PH/GW/RAFT/PDE等没有在本轮运行，不能报成实验负结果。

## 结果与已见失败

NATIVE_G的C：seen-val AR4 IoU由BASE 0.651441到0.659662（+0.822个百分点），位置0.454861→0.386860像素（减少14.95%）；强H1两项不差且拓扑fail未增加，所以该字母短时条件通过。但C AR16 IoU 0.249992→0.244821、位置3.702187→3.831008，6/8例IoU退步。不能称C整体修复。
NATIVE_G的O：AR4仅小增，RH强运动位置不达不差条件，H16新增验证例HMF-AC-O-05-08-a3的2孔/GT1孔，拓扑fail由0/8变1/8。O/C棱角、方向或刚性没有因新几何loss得到可靠修复。
G_ONLY：训练面板O AR16 IoU由0.335674到0.412462，但训练面板C/B均下降；验证C AR16由0.249992到0.204634（-4.536个百分点），6/8例下降，位置3.702187→4.019568。原旧O例0.511111→0.398396，形状pass→fail，预测仍不随GT正确转动。纯几何没有在这128步内导致全白/空蓝杆崩溃，但明显没有解决运动，不能据此推荐替换score目标。
NATIVE_T：大部分变化小。验证B AR16 IoU 0.523932→0.525916（+0.198个百分点，6/8例提高），但位置1.667576→1.695223变差；O/C AR16均值仍略降。它未通过任一字母完整方向性条件。新loss数值或梯度非零不等于实际输运/动力学学会。
旧B H16：GT2孔；BASE/NATIVE_G/NATIVE_T各1孔，G_ONLY2孔。但G_ONLY IoU 0.259494、位置6.774768px，较BASE 0.296137/5.671804更差，且明显变形，shape unknown。NATIVE_G/T IoU均0.306843仍低于E2 0.371849；不能把末帧孔数或IoU局部收益当运动/刚体正确。
训练与旧失败同案板已实际目视6张AR板：O仍有不转/反向趋势与孔框变圆；旧C的侧边及角点逐步变成折线/弯角，运动方向不跟GT；旧B后期孔洞与边界明显失真。目视不替代有资格的角度/形状统计。
H16验证：G_ONLY/NATIVE_T的O均8/8 shape unknown；NATIVE_G为1fail+7unknown。所有三臂C/B均2fail+6unknown；O旋转7unknown+1not_required，C/B旋转均8unknown。不能给“棱角或角度改善百分比”，也不能把unknown当通过。所有候选各split原绝对all_passed均false。

### 验证集AR16：IoU / 物体位置误差px（每字母8例）

|字母|E2|BASE128|G_ONLY|NATIVE_G|NATIVE_T|
|---|---|---|---|---|---|
|O|0.456442 / 3.042651|0.466823 / 2.919192|0.466318 / 2.856111|0.458797 / 2.937549|0.461089 / 2.936837|
|C|0.253279 / 3.880715|0.249992 / 3.702187|0.204634 / 4.019568|0.244821 / 3.831008|0.249656 / 3.757154|
|B|0.514797 / 2.108555|0.523932 / 1.667576|0.506821 / 1.637825|0.522678 / 1.681418|0.525916 / 1.695223|

### 逐字母方向性条件

实际通过1/9个候选×字母子条件；各候选须三字母共同满足才算整体验证支持。

|候选|字母|AR4 IoU提高|AR4位置不差|RH强H1 IoU不差|RH强H1位置不差|拓扑fail不增|子条件通过|
|---|---|---|---|---|---|---|---|
|G_ONLY|O|False|False|False|False|True|False|
|G_ONLY|C|False|False|False|True|True|False|
|G_ONLY|B|False|False|False|True|True|False|
|NATIVE_G|O|True|True|True|False|False|False|
|NATIVE_G|C|True|True|True|True|True|True|
|NATIVE_G|B|False|False|True|False|True|False|
|NATIVE_T|O|False|True|False|False|True|False|
|NATIVE_T|C|False|True|False|True|True|False|
|NATIVE_T|B|False|True|False|True|True|False|

### H16逐例IoU得失，相对BASE

|候选|分组|字母|改善|退步|相同|未知|
|---|---|---|---|---|---|---|
|G_ONLY|train|O|2|2|0|0|
|G_ONLY|train|C|0|4|0|0|
|G_ONLY|train|B|0|4|0|0|
|G_ONLY|seen_val|O|3|5|0|0|
|G_ONLY|seen_val|C|2|6|0|0|
|G_ONLY|seen_val|B|2|6|0|0|
|G_ONLY|original_failure|O|0|1|0|0|
|G_ONLY|original_failure|C|1|0|0|0|
|G_ONLY|original_failure|B|0|1|0|0|
|NATIVE_G|train|O|2|2|0|0|
|NATIVE_G|train|C|2|2|0|0|
|NATIVE_G|train|B|2|2|0|0|
|NATIVE_G|seen_val|O|3|5|0|0|
|NATIVE_G|seen_val|C|2|6|0|0|
|NATIVE_G|seen_val|B|3|5|0|0|
|NATIVE_G|original_failure|O|0|1|0|0|
|NATIVE_G|original_failure|C|0|1|0|0|
|NATIVE_G|original_failure|B|1|0|0|0|
|NATIVE_T|train|O|0|4|0|0|
|NATIVE_T|train|C|1|3|0|0|
|NATIVE_T|train|B|1|3|0|0|
|NATIVE_T|seen_val|O|3|5|0|0|
|NATIVE_T|seen_val|C|5|2|1|0|
|NATIVE_T|seen_val|B|6|2|0|0|
|NATIVE_T|original_failure|O|0|1|0|0|
|NATIVE_T|original_failure|C|1|0|0|0|
|NATIVE_T|original_failure|B|1|0|0|0|

## 定义响应与适配限制

GT-only诊断实际12个原训练样本、276张构造、11种定义/分量，tensor-only梯度51次；无新模型调用，不是物理counterfactual、训练augmentation或新模型预测。来源RGB固定三色已从renderer核实，GT重建残差最大0.006946、原hard灰mask与新阈值mask每图差0–4像素；新mask没有取代旧评价mask。
全部24个有资格x/y平移2/4/8px三元组中T严格递增；首个O样本x/y各三点的连续位移梯度为正，梯度下降方向指回GT（6次导数，不能冒充所有样本/任意误差的方向保证）。首例T约0.004723/0.009447/0.018893。
相同无重叠率的近/远错位：G均1.0，T为0.085019/0.170038，说明输运提供Dice所缺的距离响应。一个同质心异形例的质心误差恰0，而G=0.058824、T=0.112939；质心不等于形状或刚体位姿。
全白p=0时G=1、T=1.151614，梯度均有限非零；但原W1/T RGB梯度范数约9.432百万，显示极小质量归一化的强敏感性，不能只凭有限/非零说数值理想。clDice_background该例loss为0却梯度不为0，暴露空结构/soft骨架代理的边界行为，不是拓扑保证。
孔填充只有8例有可见孔、4例无孔unknown；未用蓝杆遮挡区域冒充可见孔。本组没有命中蓝杆触孔的合格填孔例。切断连接12例实际分量数增加；复制块11例仍相连、1例裁切，全部保留unknown，没有谎称12个独立增生正例。
颜色盲区的解析反例（未新增tensor实验）：三色差向量的法向量n=(-11730,15300,-2210)，与灰/蓝方向的整数点积均恰0。故P*n=0；在灰色像素沿n作足够小RGB扰动可不越界，m(I)不变而RGB变，G/T无法识别该色差。这不是实测额外构造，也没有把276报成288。
Kervadec式signed boundary使用RGB二值GT距离场，保存原signed值与相对同GT参考的常数偏移；与软抗锯齿目标不完全同义。soft-clDice骨架10轮、有限16方向W1、moments均各有盲区，没有PH/Betti、刚体、可见完整轮廓或2D最优输运保证。

## 校准、clamp和采样风险

只在首16训练样本、四个B4上一次定单位：cG=0.0233782366559374，cT=3.85030905733752e-05。四个norm(native)/norm(G或T)的普通中位数固定到结束，没有按val/结果调系数。原native loss和逐参数梯度bitwise一致；三个最终组合参数梯度都有限非零。
初始四批native-G cosine约0.147–0.193；native-T约-0.0444–0.0671；G-T约0.580–0.928。它们在初始方向上并不等同，单位匹配不保证训练中持续平衡，更不构成根因证明。
clamp严格越界梯度为0，精确0/1边界PyTorch梯度为1，实测probe [0,1,1,1,0]。预检连续D的u严格越界约45–55%，精确边界0；其中可有白背景附近微小RGB偏差，不能直接说物体有一半错误。推理精确白背景会产生大量u=0边界，不能把u<=0或>=1比例解读成零梯度比例。
每臂全部1248输出均无灰物体/蓝杆面积<1像素的empty，也无RGB越界；这是保存输出检查，不是推杆接触/穿透验证。纯G_ONLY本次未发生全空采样崩溃，仍不证明可在长训练中维持原扩散score或色彩/背景正确。GT白背景上的误差也可能来自运动错位物体，不宜直接解释成背景色污染。

|候选|输出数|灰empty|蓝empty|平均GT背景RGB MSE|平均u严格越界|平均u精确边界|
|---|---|---|---|---|---|---|
|G_ONLY|1248|0|0|0.000489|0.027670|0.958556|
|NATIVE_G|1248|0|0|0.000527|0.379519|0.052768|
|NATIVE_T|1248|0|0|0.000538|0.449614|0.038276|

## 全部分组H1/4/8/16，同案原指标

每格为IoU / 物体位置误差px；其它轮廓、蓝杆、拓扑、shape/rotation资格、全部16时刻及逐例数值在原JSON完整保留。

### train / real_history

|字母/时距|E2|BASE|G_ONLY|NATIVE_G|NATIVE_T|
|---|---|---|---|---|---|
|O H1|0.978573 / 0.064641|0.975802 / 0.075841|0.969390 / 0.100143|0.975789 / 0.085400|0.973987 / 0.097646|
|O H4|0.915156 / 0.183242|0.926897 / 0.186960|0.917981 / 0.171535|0.931390 / 0.172348|0.926897 / 0.186960|
|O H8|0.924522 / 0.179459|0.939054 / 0.109628|0.931823 / 0.135266|0.931228 / 0.130751|0.939054 / 0.109628|
|O H16|0.944227 / 0.181871|0.943901 / 0.147780|0.948252 / 0.136415|0.943003 / 0.150575|0.943901 / 0.147780|
|C H1|0.949082 / 0.187111|0.959630 / 0.113636|0.956321 / 0.125315|0.961861 / 0.114479|0.959630 / 0.113636|
|C H4|0.944073 / 0.174869|0.945083 / 0.202948|0.938799 / 0.172243|0.947348 / 0.206706|0.945083 / 0.202948|
|C H8|0.960445 / 0.185338|0.955508 / 0.213624|0.954161 / 0.203092|0.954181 / 0.230376|0.955508 / 0.213624|
|C H16|0.885714 / 0.224963|0.900976 / 0.247848|0.890377 / 0.300060|0.899884 / 0.237338|0.898629 / 0.266688|
|B H1|0.978981 / 0.081795|0.978073 / 0.067811|0.975700 / 0.052471|0.978075 / 0.074597|0.978073 / 0.067811|
|B H4|0.940578 / 0.135644|0.942367 / 0.134547|0.939111 / 0.128077|0.942310 / 0.145364|0.943096 / 0.140871|
|B H8|0.911736 / 0.243170|0.923146 / 0.172456|0.915459 / 0.210147|0.926159 / 0.208633|0.923146 / 0.172456|
|B H16|0.986691 / 0.054238|0.984297 / 0.059855|0.981820 / 0.078149|0.984268 / 0.072609|0.984297 / 0.059855|

### train / autoregressive

|字母/时距|E2|BASE|G_ONLY|NATIVE_G|NATIVE_T|
|---|---|---|---|---|---|
|O H1|0.978573 / 0.064641|0.975802 / 0.075841|0.969390 / 0.100143|0.975789 / 0.085400|0.973987 / 0.097646|
|O H4|0.733397 / 1.003403|0.756992 / 0.846202|0.724992 / 1.060728|0.755728 / 0.863538|0.754835 / 0.835843|
|O H8|0.547632 / 2.075703|0.579605 / 1.936457|0.567363 / 2.004210|0.583364 / 1.908123|0.579707 / 1.921767|
|O H16|0.379033 / 3.657807|0.335674 / 4.120006|0.412462 / 3.580993|0.336323 / 4.052040|0.329718 / 4.178181|
|C H1|0.949082 / 0.187111|0.959630 / 0.113636|0.956321 / 0.125315|0.961861 / 0.114479|0.959630 / 0.113636|
|C H4|0.793980 / 0.533689|0.821679 / 0.524340|0.807141 / 0.403818|0.814596 / 0.558553|0.819190 / 0.527207|
|C H8|0.673184 / 1.209356|0.677624 / 1.244469|0.663077 / 1.307695|0.678590 / 1.225577|0.673846 / 1.280254|
|C H16|0.195370 / 6.392374|0.264316 / 4.430930|0.179878 / 6.182505|0.270887 / 4.491641|0.260451 / 4.606508|
|B H1|0.978981 / 0.081795|0.978073 / 0.067811|0.975700 / 0.052471|0.978075 / 0.074597|0.978073 / 0.067811|
|B H4|0.801101 / 0.561115|0.832826 / 0.494750|0.814647 / 0.541139|0.833926 / 0.494385|0.830803 / 0.509515|
|B H8|0.548302 / 2.099368|0.594680 / 1.693062|0.577767 / 1.642207|0.590944 / 1.750981|0.591874 / 1.705706|
|B H16|0.406972 / 3.117852|0.493603 / 2.153573|0.454145 / 2.493119|0.489690 / 2.190232|0.488957 / 2.183417|

### seen_val / real_history

|字母/时距|E2|BASE|G_ONLY|NATIVE_G|NATIVE_T|
|---|---|---|---|---|---|
|O H1|0.995746 / 0.039797|0.990113 / 0.072272|0.983983 / 0.075535|0.989163 / 0.080230|0.990110 / 0.072226|
|O H4|0.954097 / 0.116063|0.957628 / 0.096586|0.958345 / 0.101374|0.959432 / 0.084010|0.956727 / 0.095034|
|O H8|0.943622 / 0.155753|0.943543 / 0.165225|0.944011 / 0.171210|0.947953 / 0.161435|0.943108 / 0.165815|
|O H16|0.942950 / 0.147911|0.948747 / 0.078513|0.945622 / 0.134621|0.946896 / 0.078661|0.948338 / 0.082030|
|C H1|0.984084 / 0.090155|0.984674 / 0.075202|0.983433 / 0.060558|0.985257 / 0.074539|0.984067 / 0.068682|
|C H4|0.924478 / 0.199006|0.911609 / 0.231456|0.910470 / 0.255696|0.910942 / 0.209602|0.910554 / 0.231883|
|C H8|0.926570 / 0.293565|0.928995 / 0.248618|0.924093 / 0.243076|0.930117 / 0.267012|0.928995 / 0.248618|
|C H16|0.940310 / 0.194744|0.941596 / 0.180930|0.935453 / 0.174083|0.940945 / 0.193204|0.940981 / 0.188041|
|B H1|0.989752 / 0.050183|0.985776 / 0.056402|0.983020 / 0.071259|0.985772 / 0.049121|0.985776 / 0.056402|
|B H4|0.944056 / 0.163407|0.944191 / 0.165767|0.944145 / 0.166087|0.946034 / 0.164054|0.944573 / 0.166625|
|B H8|0.937412 / 0.135610|0.938617 / 0.150495|0.936298 / 0.154066|0.938284 / 0.154336|0.938995 / 0.148099|
|B H16|0.946402 / 0.131656|0.950757 / 0.174922|0.947238 / 0.152058|0.951455 / 0.169488|0.950387 / 0.173848|

### seen_val / autoregressive

|字母/时距|E2|BASE|G_ONLY|NATIVE_G|NATIVE_T|
|---|---|---|---|---|---|
|O H1|0.995746 / 0.039797|0.990113 / 0.072272|0.983983 / 0.075535|0.989163 / 0.080230|0.990110 / 0.072226|
|O H4|0.852754 / 0.500406|0.854977 / 0.478686|0.840828 / 0.521879|0.855229 / 0.477358|0.853484 / 0.469461|
|O H8|0.675611 / 1.322679|0.700833 / 1.202369|0.674561 / 1.311662|0.700074 / 1.189474|0.699240 / 1.215513|
|O H16|0.456442 / 3.042651|0.466823 / 2.919192|0.466318 / 2.856111|0.458797 / 2.937549|0.461089 / 2.936837|
|C H1|0.984084 / 0.090155|0.984674 / 0.075202|0.983433 / 0.060558|0.985257 / 0.074539|0.984067 / 0.068682|
|C H4|0.630272 / 0.518774|0.651441 / 0.454861|0.631599 / 0.601389|0.659662 / 0.386860|0.648449 / 0.430401|
|C H8|0.442016 / 1.045470|0.445381 / 1.160152|0.432701 / 1.337377|0.449273 / 1.181710|0.446736 / 1.170320|
|C H16|0.253279 / 3.880715|0.249992 / 3.702187|0.204634 / 4.019568|0.244821 / 3.831008|0.249656 / 3.757154|
|B H1|0.989752 / 0.050183|0.985776 / 0.056402|0.983020 / 0.071259|0.985772 / 0.049121|0.985776 / 0.056402|
|B H4|0.758849 / 0.923860|0.778506 / 0.826559|0.774643 / 0.861308|0.777010 / 0.827366|0.776972 / 0.825014|
|B H8|0.633382 / 1.508409|0.647624 / 1.295695|0.635634 / 1.262619|0.649356 / 1.277986|0.644204 / 1.347760|
|B H16|0.514797 / 2.108555|0.523932 / 1.667576|0.506821 / 1.637825|0.522678 / 1.681418|0.525916 / 1.695223|

### original_failure / real_history

|字母/时距|E2|BASE|G_ONLY|NATIVE_G|NATIVE_T|
|---|---|---|---|---|---|
|O H1|0.977099 / 0.039751|0.980916 / 0.055183|0.977099 / 0.039751|0.980916 / 0.055183|0.980916 / 0.055183|
|O H4|0.883803 / 0.107626|0.856643 / 0.299523|0.869718 / 0.152472|0.857143 / 0.378863|0.856643 / 0.299523|
|O H8|0.897527 / 0.325401|0.900356 / 0.318848|0.897163 / 0.351217|0.896797 / 0.350609|0.900356 / 0.318848|
|O H16|0.992674 / 0.052639|0.992674 / 0.052639|0.992647 / 0.036776|0.992674 / 0.052639|0.992674 / 0.052639|
|C H1|0.995050 / 0.051169|0.995050 / 0.051169|0.990099 / 0.040581|0.995050 / 0.051169|0.995050 / 0.051169|
|C H4|0.933014 / 0.285312|0.942029 / 0.268554|0.931707 / 0.068562|0.946602 / 0.298306|0.937198 / 0.236547|
|C H8|0.950980 / 0.087514|0.935961 / 0.137226|0.931373 / 0.117192|0.950249 / 0.107698|0.945545 / 0.149398|
|C H16|0.970588 / 0.100122|0.970443 / 0.121463|0.960591 / 0.170362|0.970443 / 0.121463|0.970443 / 0.121463|
|B H1|0.956923 / 0.027051|0.929231 / 0.237957|0.935185 / 0.214773|0.932308 / 0.233641|0.929231 / 0.237957|
|B H4|0.924699 / 0.248629|0.935583 / 0.148595|0.926380 / 0.209954|0.938650 / 0.189556|0.935583 / 0.148595|
|B H8|0.936170 / 0.157299|0.935780 / 0.173680|0.932722 / 0.149654|0.938650 / 0.171990|0.935780 / 0.173680|
|B H16|0.938272 / 0.137386|0.953271 / 0.124199|0.950000 / 0.117549|0.950156 / 0.156914|0.953271 / 0.124199|

### original_failure / autoregressive

|字母/时距|E2|BASE|G_ONLY|NATIVE_G|NATIVE_T|
|---|---|---|---|---|---|
|O H1|0.977099 / 0.039751|0.980916 / 0.055183|0.977099 / 0.039751|0.980916 / 0.055183|0.980916 / 0.055183|
|O H4|0.639241 / 1.408446|0.670968 / 1.054002|0.651613 / 1.404133|0.667742 / 1.074804|0.661290 / 1.017956|
|O H8|0.504178 / 2.489634|0.559078 / 1.769312|0.492877 / 2.274086|0.577259 / 1.637162|0.561960 / 1.779573|
|O H16|0.480000 / 2.547891|0.511111 / 2.166518|0.398396 / 2.955066|0.509804 / 2.050632|0.490358 / 2.153639|
|C H1|0.995050 / 0.051169|0.995050 / 0.051169|0.990099 / 0.040581|0.995050 / 0.051169|0.995050 / 0.051169|
|C H4|0.546154 / 0.605854|0.610442 / 0.221385|0.631148 / 0.505808|0.615385 / 0.327049|0.604000 / 0.332848|
|C H8|0.310145 / 0.703053|0.384375 / 0.520576|0.422581 / 0.799810|0.388715 / 0.498754|0.391850 / 0.527899|
|C H16|0.129353 / 5.517083|0.139594 / 4.544727|0.145833 / 4.158340|0.137755 / 4.725456|0.140306 / 4.547933|
|B H1|0.956923 / 0.027051|0.929231 / 0.237957|0.935185 / 0.214773|0.932308 / 0.233641|0.929231 / 0.237957|
|B H4|0.834783 / 0.238603|0.844575 / 0.127882|0.868263 / 0.081977|0.838235 / 0.040036|0.844118 / 0.172006|
|B H8|0.692913 / 0.516525|0.680108 / 1.148926|0.682796 / 1.089380|0.690217 / 0.913230|0.677419 / 1.136874|
|B H16|0.371849 / 4.304287|0.296137 / 5.671804|0.259494 / 6.774768|0.306843 / 4.402497|0.306843 / 5.980946|

## 实际费用、恢复和保存核验

- 实际17408图像forward / 上限17504（训练6144、评测11232、预检32），4424batch forward /4544。正式反传1536个microbatch，预检参数梯度16次，独立tensor梯度51次；不把反向、tensor构造或审计误当模型预测。
- 仅1个模型构建、1次预检E2 model-only load、3次正式完整状态恢复，384尝试/384成功，AMP0；0新物理/新数据，0额外seed/步数/模型输出修补。预检32图不是额外正式训练曝光。
- worker活跃混合墙钟593.230s（约9分53秒），不是纯GPU账单；CPU user321.359s/system101.391s。训练G_ONLY/NATIVE_G/NATIVE_T约93.355/121.554/80.850s，评测80.868/89.462/57.705s；定义诊断7.177s、预检3.771s另列，剩余为源核验/IO/报告等。
- GPU峰值allocated1938977792B、reserved2118123520B。单数值进程BelowNormal；Torch intra/inter、OpenCV和数值库线程1；无CPU模型回退、其它进程或系统环境修改。实际IO重试0。
- saved-only审计一次实际exit0：234个FP32 raw SHA（414117288B）、3轻量weights SHA、24PNG头/尺寸/安全路径、12845不可变计账事件，以及更新/AMP/完整恢复/conditioning RNG与E2/BASE评测噪声配对。审计0模型/物理/新渲染；没有重跑审计。
- 上轮loss-screen23220计账/23216完成、旧R1失败33/正式33101、更旧R3B失败33.047s仅历史费用，未混入本轮。
- 新root审计时574.55MB<650MB；共享9.173GB，加256MiB原子与64MiB增长预留后仍<10GB。24MiB终态报告预留保留；最终发送回执后再记录实际体积。
- 本轮无Windows系统任务注册，原active exec中worker与一次审计均实际退出0。不操作任何任务或进程。新30分钟监控先被拒，用户后续直接授权后才实际创建；这一授权与真实工具回执保留。最终真实回传后删除监控，waiting_user_review，不自动续训。

## 可点击完整证据与同案图

- 原完整报告 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/REPORT.txt`)
- 完整HTML索引 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/index.html`)
- 全部16步分组 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/grouped-all16.json`)
- 逐例改善/退步 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/per-case-comparisons.json`)
- 原方向性条件 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/comparison-checks.json`)
- RGB/空物体/蓝杆风险 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/RGB-sampling-risks.json`)
- 定义诊断与公式 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/diagnosis/summary.json`)
- 每构造完整结果 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/diagnosis/measurements.json`)
- 实际完成记录 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/completion.json`)
- 真实saved-only审计 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/saved-audit.json`)
- 审计工具回执 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/audit-result.json`)
- 系数与参数梯度 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/preflight.json`)
- 配对与恢复 (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/pairing-proof.json`)
- train O real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-HMF-AC-O-01-25-a3-4699e73600c89832bad99262.png`)
- train O autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-HMF-AC-O-01-25-a3-b8a6733f613549eb046ae7a2.png`)
- train C real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-HMF-AC-C-03-01-a19-7facef5673a2fda468921f03.png`)
- train C autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-HMF-AC-C-03-01-a19-641b8e0ad014670d3f3c74bb.png`)
- train B real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-HMF-AC-B-03-27-a3-3682133356bc536e08ea8858.png`)
- train B autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-HMF-AC-B-03-27-a3-996ca00b28c7b5329e485902.png`)
- original_failure O real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-AC-O-01-08_AC-O-01-08_a3_H16-35da4809ea674ade07eafe0c.png`)
- original_failure O autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-AC-O-01-08_AC-O-01-08_a3_H16-850b4e866cd5f6bad2a37307.png`)
- original_failure C real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-AC-C-01-24_AC-C-01-24_a3_H16-c909eede4df35a7e974e43e0.png`)
- original_failure C autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-AC-C-01-24_AC-C-01-24_a3_H16-9c023fe61a33f0d98d5ef545.png`)
- original_failure B real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-AC-B-00-08_AC-B-00-08_a19_H1-efbc80fee185c247d2ee46c8.png`)
- original_failure B autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-definition-screen-20261102/report/p-AC-B-00-08_AC-B-00-08_a19_H1-481b8bfa2cd7b6491a86a6b5.png`)
