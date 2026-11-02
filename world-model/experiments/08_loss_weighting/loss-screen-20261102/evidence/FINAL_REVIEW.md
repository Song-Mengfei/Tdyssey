# 文献损失筛选：全部计算完成，三种候选均未通过综合条件

四臂各128次尝试、128次成功更新、2048个原生H1曝光，AMP跳步0；4992个独立评测预测、312份raw和12张同案图均已完整保存、核验。此前Windows账本替换错误按用户明确要求修复并恢复，没有重训BASE/SNR。恢复最终complete=true，原始failure及费用保留。

SNR、GDL、FFL都能在既定GPC接口中计算loss、反传并训练，但三者都未通过预先规定的逐字母方向性筛选，原绝对验收也全部未通过。结论是当前128步筛选没有支持直接替换BASE的全面证据，不是论文方法无效或架构无能力的证明。

本次实际运行的是三项文献方法的GPC适配：SNR采用有效噪声rho=.25/(s²+.1²)、gamma5、min(rho,5)/rho；GDL比较未量化重建D/.5与真图y/.5的相邻梯度幅值；FFL比较保留复数相位的ortho FFT、每样本每通道归一化并detach权重。GDL/FFL系数均1。原BASE直接保留原F.mse_loss的运算和全局mean顺序。没有新数据、AR4训练分支、物理、动作接口、模型结构或推理采样变化。广泛综述由独立分析对话维护；此处仅交付已完成的三候选实验，未把其他文献类别也算作已测试。

BASE128的78份raw字节SHA以及已比对primary/shape/topology字段与R1 A128一致，未静默换基线。原生训练order与每步conditioning RNG/branch在四臂一致，四次完整E2恢复有实际证明；共同E2只复用旧结果，零新推理。

## 主要结果

验证集每字母8个家族。所有候选的AR4平均IoU都略低于BASE；部分位置指标改善，但没有满足“AR4 IoU更高、位置不差、强H1真实历史IoU/位置不差、拓扑失败不增”的完整条件。

GDL对字母B长时有局部收益：验证AR16 IoU .523932→.552540，增加2.86个百分点；位置1.667576→1.622913px。8/8例IoU比BASE提高。但C的AR16 IoU .249992→.234253，位置3.702187→4.230793px，6/8例IoU退步，不能以字母B或ALL平均掩盖C。O的GDL AR16平均小幅改善，逐例4改善/4退步。

SNR整体偏弱：验证AR16相对BASE，O有2改善/6退步、C3/5，字母B5改善/2退步/1平；字母B H4出现一个原BASE没有的拓扑错误（3个可见孔，GT为2）。FFL效应较小且混合：验证AR16 O、C平均IoU略低于BASE，字母B略高；逐例O5改善/2退步/1平，C5/3，字母B5/3。均不能称稳定全面改进。

固定训练12（每字母4例）的改善与验证分开：GDL/FFL的字母B训练AR16全部4例比BASE改善，C训练AR16也有3/4例改善；SNR的O训练AR16四例均比BASE退步。该12例面板不是全48训练片段的拟合验收。

## 棱角、运动、孔洞与限制

已查看O/C/B训练和旧失败的六张AR板。四候选仍存在O朝向/角部偏差、C欠转或外形变化、B位置与朝向误差。不能把原始IoU或轮廓误差的变化直接解释为棱角保真改善百分比，它们混合位置、方向、形变与遮挡。

验证H16各臂形状状态完全相同：O=8unknown；C=2fail+6unknown；字母B=2fail+6unknown。O旋转有7unknown和1not_required，C/字母B各8unknown。因此当前没有合格证据证明O/C刚性保角或物理方向已解决。RH每一时刻使用真实历史，是独立单步；它不能冒充只有起点真图的AR16闭环。

旧失败B在H16，BASE/GDL仍只有1孔（GT2），SNR/FFL有2孔且拓扑状态pass；但后两者形状仍unknown，H16 IoU分别.289979/.302575，仍低于E2的.371849，位置误差分别6.001008/5.773714px。单个末帧孔数正常不能称完整运动或保形修复。旧C开口拓扑pass也不证明开口形状与朝向准确。蓝杆在每臂1248个目标中都未出现empty_pusher，但没有新增接触/穿透物理验证，不能据此称物理关系正确。

旧失败三例及全部逐例得失均保留在索引和JSON，不能挑最好seed或只报正例。单seed、128步、固定系数和多候选重复使用同一seen-val，只是探索筛选，不是收敛、最优loss或独立泛化证明。未新增T/L/X及静止专门试验，不对其下新结论。

## 工程恢复与完整费用

原运行19:17:31（新加坡）在SNR RH第三个批次batch_id8的before_forward写账时遭Windows拒绝os.replace，旧写入器缺少短时重试。具体文件占用者未查明，不归咎用户或其他项目。原failure/孤立part和源码保持。新恢复写入器只重试同一份已写temp的IO；成本账本改为不可变逐事件快照，模型操作不重试。恢复实际IO重试计数0，热ledger替换已避免。

复用了8份已持久SNR raw，未重训BASE/SNR。原SNR缓存批的actual-epsilon观察回执随进程内存丢失；原冻结capture_noise代码在raw持久前已逐次校验actual与replay，本次同seed/B4独立复算哈希。该来源在SNR metrics.noise_provenance明确标注，不假称做了新的原生模型观察。

累计计账图像前向23220，已完成inner forward23216，仍低于原总上限23296；差4是写账失败后未进入inner body、仍保留计费的reservation。原已完成但未落盘的28个denoiser-image计算重算费用也保留。batch计账5901、完成5900；正式backward2048，预检模型梯度6、tensor-only梯度3且未重跑。4992帧是独立最终输出；加丢失前缀重算的8个采样完成帧，实际采样完成总数5000，不能隐去重算开销。

按用户要求恢复所需的第二次模型实例另记，任何时刻仅一个CUDA数值进程；原预检E2 model-only加载1次、恢复SNR最终model-only加载1次、四臂完整E2恢复累计4次。没有四套Adam磁盘拷贝或新环境/数据资产。

原worker计时297.939s，恢复worker317.447s，合计活跃worker墙钟615.386s（约10分15秒）；中断等待时间另算，墙钟不是纯GPU账单。训练loop BASE76.107s、SNR96.232s、GDL61.658s、FFL63.359s；评测BASE67.711s、恢复SNR49.158s（另有原中断评测费用包含原worker）、GDL52.557s、FFL52.597s。恢复完成于2026-11-02 20:14:18新加坡时间。原R1失败33/R1正式33101以及更旧R3B33.047s仅历史，不混入本次费用。

原已保存峰快照allocated1,853,849,088/reserved2,157,969,408B；恢复完整阶段峰allocated1,795,704,832/reserved2,027,945,984B。两次worker自身CPU user/system分别160.672/44.266s、213.484/52.531s，均BelowNormal、各数值库/Torch intra+inter/OpenCV1，0其他项目进程/系统环境修改、0CPU模型回退。

保存结果审计仅执行一次，真实退出码0：312份raw和4份轻量权重SHA通过，12张PNG头/尺寸及安全basename映射通过，原失败未变、更新/AMP/RNG/恢复和累计费用通过。审计0模型/GPU/物理/新渲染调用。root约750.5MB<900MB，共享参考约8.597GB，加256MiB原子与64MiB增长guard仍<10GB。

## 保存指标

下表数值为gray IoU / 可见质心误差imagepx。所有时距和逐例对照在原REPORT.txt、HTML及grouped-all16/per-case JSON；下列突出H1/4/8/16。

### train / real_history

|Letter/H|E2|BASE|SNR|GDL|FFL|
|---|---|---|---|---|---|
|O H1|0.978573/0.064641|0.975802/0.075841|0.973082/0.081607|0.973223/0.117581|0.973987/0.097646|
|O H4|0.915156/0.183242|0.926897/0.186960|0.926840/0.175476|0.931311/0.158496|0.930363/0.173940|
|O H8|0.924522/0.179459|0.939054/0.109628|0.936486/0.118176|0.940083/0.123843|0.936346/0.115260|
|O H16|0.944227/0.181871|0.943901/0.147780|0.944889/0.156120|0.951147/0.117059|0.943816/0.155833|
|C H1|0.949082/0.187111|0.959630/0.113636|0.958592/0.118886|0.957384/0.129603|0.960677/0.118282|
|C H4|0.944073/0.174869|0.945083/0.202948|0.946117/0.210920|0.947498/0.190586|0.946117/0.210920|
|C H8|0.960445/0.185338|0.955508/0.213624|0.955555/0.215653|0.951701/0.259077|0.955508/0.213624|
|C H16|0.885714/0.224963|0.900976/0.247848|0.899930/0.247100|0.901980/0.178539|0.900860/0.230541|
|B H1|0.978981/0.081795|0.978073/0.067811|0.978078/0.063968|0.979631/0.055290|0.978073/0.067811|
|B H4|0.940578/0.135644|0.942367/0.134547|0.942367/0.134547|0.942315/0.130957|0.942326/0.139673|
|B H8|0.911736/0.243170|0.923146/0.172456|0.924686/0.182511|0.930582/0.202661|0.926853/0.175710|
|B H16|0.986691/0.054238|0.984297/0.059855|0.984305/0.064231|0.985081/0.052042|0.984297/0.059855|


### train / autoregressive

|Letter/H|E2|BASE|SNR|GDL|FFL|
|---|---|---|---|---|---|
|O H1|0.978573/0.064641|0.975802/0.075841|0.973082/0.081607|0.973223/0.117581|0.973987/0.097646|
|O H4|0.733397/1.003403|0.756992/0.846202|0.755660/0.877547|0.752509/0.929846|0.757096/0.844788|
|O H8|0.547632/2.075703|0.579605/1.936457|0.563108/1.946592|0.594172/1.891703|0.590062/1.840680|
|O H16|0.379033/3.657807|0.335674/4.120006|0.312884/4.260578|0.337875/4.101860|0.339779/4.179215|
|C H1|0.949082/0.187111|0.959630/0.113636|0.958592/0.118886|0.957384/0.129603|0.960677/0.118282|
|C H4|0.793980/0.533689|0.821679/0.524340|0.820822/0.533874|0.824000/0.545700|0.818232/0.533670|
|C H8|0.673184/1.209356|0.677624/1.244469|0.674348/1.272329|0.693079/1.127853|0.679498/1.251202|
|C H16|0.195370/6.392374|0.264316/4.430930|0.229687/5.193719|0.274154/4.561048|0.284783/4.143431|
|B H1|0.978981/0.081795|0.978073/0.067811|0.978078/0.063968|0.979631/0.055290|0.978073/0.067811|
|B H4|0.801101/0.561115|0.832826/0.494750|0.829104/0.529180|0.827271/0.531987|0.827879/0.538201|
|B H8|0.548302/2.099368|0.594680/1.693062|0.589007/1.723204|0.606822/1.680776|0.593004/1.720620|
|B H16|0.406972/3.117852|0.493603/2.153573|0.493379/2.120349|0.531896/1.964469|0.509333/2.122112|


### seen_val / real_history

|Letter/H|E2|BASE|SNR|GDL|FFL|
|---|---|---|---|---|---|
|O H1|0.995746/0.039797|0.990113/0.072272|0.991039/0.064783|0.990569/0.072513|0.989640/0.071474|
|O H4|0.954097/0.116063|0.957628/0.096586|0.956735/0.100551|0.958111/0.089146|0.956699/0.096707|
|O H8|0.943622/0.155753|0.943543/0.165225|0.943984/0.159751|0.943540/0.149090|0.942640/0.158131|
|O H16|0.942950/0.147911|0.948747/0.078513|0.948280/0.080967|0.947433/0.096418|0.947805/0.092011|
|C H1|0.984084/0.090155|0.984674/0.075202|0.985872/0.065194|0.985286/0.075112|0.984674/0.075202|
|C H4|0.924478/0.199006|0.911609/0.231456|0.910677/0.232124|0.909713/0.238806|0.908991/0.223242|
|C H8|0.926570/0.293565|0.928995/0.248618|0.931889/0.232486|0.925513/0.205871|0.929516/0.239235|
|C H16|0.940310/0.194744|0.941596/0.180930|0.940412/0.196757|0.938714/0.146039|0.940995/0.174175|
|B H1|0.989752/0.050183|0.985776/0.056402|0.987357/0.053571|0.984628/0.073223|0.985383/0.057122|
|B H4|0.944056/0.163407|0.944191/0.165767|0.943471/0.174422|0.945656/0.157790|0.944501/0.177030|
|B H8|0.937412/0.135610|0.938617/0.150495|0.938357/0.163917|0.938733/0.139675|0.938304/0.155835|
|B H16|0.946402/0.131656|0.950757/0.174922|0.950379/0.167202|0.950327/0.179751|0.951119/0.183086|


### seen_val / autoregressive

|Letter/H|E2|BASE|SNR|GDL|FFL|
|---|---|---|---|---|---|
|O H1|0.995746/0.039797|0.990113/0.072272|0.991039/0.064783|0.990569/0.072513|0.989640/0.071474|
|O H4|0.852754/0.500406|0.854977/0.478686|0.852345/0.467971|0.849604/0.507686|0.854441/0.477095|
|O H8|0.675611/1.322679|0.700833/1.202369|0.695754/1.241478|0.705733/1.199943|0.700174/1.225434|
|O H16|0.456442/3.042651|0.466823/2.919192|0.451512/3.021972|0.473135/2.889234|0.463589/2.963109|
|C H1|0.984084/0.090155|0.984674/0.075202|0.985872/0.065194|0.985286/0.075112|0.984674/0.075202|
|C H4|0.630272/0.518774|0.651441/0.454861|0.646465/0.461172|0.642264/0.494876|0.651261/0.434630|
|C H8|0.442016/1.045470|0.445381/1.160152|0.443580/1.229892|0.429409/1.297127|0.442871/1.175190|
|C H16|0.253279/3.880715|0.249992/3.702187|0.246523/3.686839|0.234253/4.230793|0.248479/3.747198|
|B H1|0.989752/0.050183|0.985776/0.056402|0.987357/0.053571|0.984628/0.073223|0.985383/0.057122|
|B H4|0.758849/0.923860|0.778506/0.826559|0.776744/0.831002|0.772484/0.849797|0.774582/0.851683|
|B H8|0.633382/1.508409|0.647624/1.295695|0.643567/1.297326|0.657480/1.299777|0.648509/1.276862|
|B H16|0.514797/2.108555|0.523932/1.667576|0.527388/1.679867|0.552540/1.622913|0.530842/1.620927|


### original_failure / real_history

|Letter/H|E2|BASE|SNR|GDL|FFL|
|---|---|---|---|---|---|
|O H1|0.977099/0.039751|0.980916/0.055183|0.977099/0.039751|0.980916/0.055183|0.980916/0.055183|
|O H4|0.883803/0.107626|0.856643/0.299523|0.860140/0.342684|0.843750/0.351635|0.856643/0.299523|
|O H8|0.897527/0.325401|0.900356/0.318848|0.900356/0.318848|0.893238/0.370167|0.900356/0.318848|
|O H16|0.992674/0.052639|0.992674/0.052639|0.992674/0.052639|0.989011/0.082972|0.992674/0.052639|
|C H1|0.995050/0.051169|0.995050/0.051169|0.995050/0.051169|0.995050/0.051169|0.995050/0.051169|
|C H4|0.933014/0.285312|0.942029/0.268554|0.942029/0.268554|0.951456/0.277929|0.942029/0.268554|
|C H8|0.950980/0.087514|0.935961/0.137226|0.950495/0.108516|0.955446/0.087225|0.940887/0.132855|
|C H16|0.970588/0.100122|0.970443/0.121463|0.970443/0.121463|0.975369/0.119881|0.975369/0.119881|
|B H1|0.956923/0.027051|0.929231/0.237957|0.926154/0.239707|0.923313/0.272568|0.923313/0.274552|
|B H4|0.924699/0.248629|0.935583/0.148595|0.935583/0.148595|0.938650/0.132558|0.935583/0.148595|
|B H8|0.936170/0.157299|0.935780/0.173680|0.935780/0.173680|0.938650/0.171990|0.935780/0.173680|
|B H16|0.938272/0.137386|0.953271/0.124199|0.953271/0.124199|0.947205/0.086811|0.953271/0.124199|


### original_failure / autoregressive

|Letter/H|E2|BASE|SNR|GDL|FFL|
|---|---|---|---|---|---|
|O H1|0.977099/0.039751|0.980916/0.055183|0.977099/0.039751|0.980916/0.055183|0.980916/0.055183|
|O H4|0.639241/1.408446|0.670968/1.054002|0.679739/0.971286|0.665595/0.910745|0.661290/1.025672|
|O H8|0.504178/2.489634|0.559078/1.769312|0.553623/1.840900|0.537572/1.929327|0.559078/1.830348|
|O H16|0.480000/2.547891|0.511111/2.166518|0.515406/2.155112|0.439678/2.454672|0.490358/2.160960|
|C H1|0.995050/0.051169|0.995050/0.051169|0.995050/0.051169|0.995050/0.051169|0.995050/0.051169|
|C H4|0.546154/0.605854|0.610442/0.221385|0.596000/0.560735|0.612000/0.165503|0.616935/0.058658|
|C H8|0.310145/0.703053|0.384375/0.520576|0.396825/0.565778|0.373457/0.500800|0.381250/0.499641|
|C H16|0.129353/5.517083|0.139594/4.544727|0.140306/4.482175|0.133166/4.706590|0.140306/4.549468|
|B H1|0.956923/0.027051|0.929231/0.237957|0.926154/0.239707|0.923313/0.272568|0.923313/0.274552|
|B H4|0.834783/0.238603|0.844575/0.127882|0.852071/0.159146|0.847059/0.216860|0.844118/0.140774|
|B H8|0.692913/0.516525|0.680108/1.148926|0.678378/1.065413|0.688172/0.985146|0.684636/1.048764|
|B H16|0.371849/4.304287|0.296137/5.671804|0.289979/6.001008|0.287281/4.683177|0.302575/5.773714|


## Artifacts and same-case media

- REPORT.txt (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/REPORT.txt`)
- report/index.html (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/index.html`)
- report/summary.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/summary.json`)
- report/comparison-checks.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/comparison-checks.json`)
- report/grouped-all16.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/grouped-all16.json`)
- report/per-case-comparisons.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/per-case-comparisons.json`)
- continuation-saved-audit.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/continuation-saved-audit.json`)
- continuation-audit-result.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/continuation-audit-result.json`)
- BASE-vs-R1-A128.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/BASE-vs-R1-A128.json`)
- continuation-completion.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/continuation-completion.json`)
- completion.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/completion.json`)
- continuation-config.json (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/continuation-config.json`)
- SOURCE_TEST_SPEC.txt (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/SOURCE_TEST_SPEC.txt`)
- train O real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-HMF-AC-O-01-25-a3-4699e73600c89832bad99262.png`)
- train O autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-HMF-AC-O-01-25-a3-b8a6733f613549eb046ae7a2.png`)
- train C real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-HMF-AC-C-03-01-a19-7facef5673a2fda468921f03.png`)
- train C autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-HMF-AC-C-03-01-a19-641b8e0ad014670d3f3c74bb.png`)
- train B real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-HMF-AC-B-03-27-a3-3682133356bc536e08ea8858.png`)
- train B autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-HMF-AC-B-03-27-a3-996ca00b28c7b5329e485902.png`)
- original_failure O real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-AC-O-01-08_AC-O-01-08_a3_H16-35da4809ea674ade07eafe0c.png`)
- original_failure O autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-AC-O-01-08_AC-O-01-08_a3_H16-850b4e866cd5f6bad2a37307.png`)
- original_failure C real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-AC-C-01-24_AC-C-01-24_a3_H16-c909eede4df35a7e974e43e0.png`)
- original_failure C autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-AC-C-01-24_AC-C-01-24_a3_H16-9c023fe61a33f0d98d5ef545.png`)
- original_failure B real_history (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-AC-B-00-08_AC-B-00-08_a19_H1-efbc80fee185c247d2ee46c8.png`)
- original_failure B autoregressive (external historical reference: `__LOCAL_WORKSPACE__/Tdyssey-raw-tl-portable-main/artifacts/loss-screen-20261102/report/p-AC-B-00-08_AC-B-00-08_a19_H1-481b8bfa2cd7b6491a86a6b5.png`)

No further training/deployment is authorized. After actual final callback: waiting_user_review, ended-task static-registration cleanup and monitor deletion.
