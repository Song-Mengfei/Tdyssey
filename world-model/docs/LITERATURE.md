# Literature and project applicability

This page reformats the saved reviews; no new literature search was performed during repository preparation. Inspection depth is explicit. A paper being listed does not mean its full method was implemented or trained. The first review focuses on weighting/appearance, while the later review addresses what should count as error.

## Error definitions: later corrected review

### D01 — Elucidating the Design Space of Diffusion-Based Generative Models

Source: [2022](https://arxiv.org/abs/2206.00364). Family: 扩散目标的兼容性.

Inspection: 本轮摘要；上轮预条件公式及作者EDMLoss源码核对

Original method: 预条件去噪网络与噪声尺度加权重建目标。

Project assessment: 我们的native残差MSE是有明确去噪含义的目标；它没有直接定义物体拓扑/刚体误差。修改目标后仍用原sampler是否可行需要实测，不能仅凭一个可微公式判断。

Actual test status: 原baseline；本轮只读复用

### D02 — Denoising Diffusion Probabilistic Models

Source: [2020](https://arxiv.org/abs/2006.11239). Family: 扩散目标的兼容性.

Inspection: 摘要/目标定位

Original method: 将扩散训练与去噪score matching联系起来。

Project assessment: 纯mask损失有许多RGB等价解；不能直接把任何几何回归器视作原扩散概率模型的等价训练。保留native的组合与完全替换分别测试。

Actual test status: 文献评估；未实施

### D03 — V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation

Source: [2016](https://arxiv.org/pdf/1606.04797). Family: 占据区域.

Inspection: 方法深读：第3节平方分母Dice及其梯度

Original method: 以2 sum(p*g)/(sum(p²)+sum(g²))定义区域重叠，处理小前景与大背景失衡。

Project assessment: 本轮把预测RGB和GT RGB变成相同的可微灰/蓝前景量，等权平均两类Dice误差；采用相同soft GT使完全相同图像有零几何误差。没有新增分割网络。

Actual test status: 定义诊断与G_ONLY/NATIVE_G训练已完成；纯G使验证C长时退化；NATIVE_G仅C短时方向条件通过，长时仍退步。

### D04 — Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations

Source: [2017](https://arxiv.org/abs/1707.03237). Family: 占据区域.

Inspection: 摘要筛选

Original method: 用类别体积相关权重平衡分割类别。

Project assessment: 能降低背景主导，但额外类权重不是本轮核心。灰色字母和蓝杆均需准确，先用可解释的两类等权几何定义；不再做一轮权重搜索。

Actual test status: 文献评估；未实施

### D05 — Tversky loss function for image segmentation using 3D fully convolutional deep networks

Source: [2017](https://arxiv.org/abs/1706.05721). Family: 占据区域.

Inspection: 摘要筛选

Original method: 用不对称FP/FN代价改变精确率与召回率取舍。

Project assessment: 若偏重召回，可能通过增厚/填孔获益；本项目没有授权改成宁可膨胀的目标，不直接选作主配方。

Actual test status: 文献评估；未实施

### D06 — The Lovász-Softmax loss

Source: [2018](https://arxiv.org/pdf/1705.08790). Family: 占据区域.

Inspection: 方法深读：Eq11–13、排序求Jaccard扩展、per-image与dataset口径

Original method: 以排序误差和Lovasz扩展构造IoU的分段线性替代目标。

Project assessment: 可从RGB派生soft前景后应用，但不是对RGB张量直接套公式；hard GT与抗锯齿soft GT的差异要声明。IoU损失仍不能区分完全无重叠的远近错位。

Actual test status: 文献评估；未实施

### D07 — Boundary loss for highly unbalanced segmentation

Source: [2019](https://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf). Family: 轮廓/距离场.

Inspection: 方法深读：Eq2–6，有符号GT距离场与常数项

Original method: 以GT有符号距离场phi和预测前景s的积分衡量轮廓变化，通常结合区域损失。

Project assessment: 可惩罚离真实轮廓很远的前景。原LB可为负；减去GT常数不改梯度。它不是Hausdorff最大距离，也不是把RGB边缘加权。可训练概率映射、离散距离场定义必须清楚。

Actual test status: 定义响应诊断已完成；未列入本轮训练臂，不能报为训练无效。

### D08 — Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks

Source: [2019](https://arxiv.org/pdf/1904.10030). Family: 轮廓/距离场.

Inspection: 方法深读：Eq4–8及侵蚀/卷积替代路线

Original method: 距离变换型损失以(p-q)²乘GT/预测边界距离幂；另给侵蚀与多半径卷积近似。

Project assessment: 强调最坏局部误差，比平均RGB更贴近毛刺/孤块。真正预测DT含离散步骤，不能把detach DT叫作完整可微SDF；只用GT DT则是另一适配公式。

Actual test status: 文献评估；未实施

### D09 — A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation

Source: [2023](https://arxiv.org/abs/2302.03868). Family: 轮廓/距离场.

Inspection: 摘要筛选

Original method: 针对区域重叠高而表面距离差的问题，提出数值与类别平衡改进。

Project assessment: 支持将轮廓误差单列研究；医学分割的效果不能转述为我们的96px RGB预测有效。

Actual test status: 文献评估；未实施

### D10 — Locating Objects Without Bounding Boxes

Source: [2019](https://arxiv.org/pdf/1806.07564). Family: 点集/定位距离.

Inspection: 方法深读：Eq5–7、两项防退化解释

Original method: weighted Hausdorff以soft位置热图和GT点集比较，不要求预测点数固定；两个方向的项避免全空/全满解。

Project assessment: 用于中心定位不能约束完整字母形状；若改成全部轮廓点，监督语义与成本均改变。硬提取预测轮廓点无法直接传RGB梯度。

Actual test status: 文献评估；未实施

### D11 — A Point Set Generation Network for 3D Object Reconstruction From a Single Image

Source: [2017](https://robotics.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf). Family: 点集/定位距离.

Inspection: 作者论文方法摘读：Chamfer/EMD公式

Original method: Chamfer双向最近邻平方距离与一一匹配的EMD用于无序点集。

Project assessment: 轮廓Chamfer可表达几何偏差，但最近邻允许多对一，不保证面积、孔或刚体。直接对预测hard边缘采点会切断mask梯度，不是零成本替换。

Actual test status: 文献评估；未实施

### D12 — Sinkhorn Distances: Lightspeed Computation of Optimal Transport

Source: [2013](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html). Family: 空间输运.

Inspection: 原论文摘要与作者方法说明

Original method: 熵正则输运通过矩阵缩放加速求解。

Project assessment: 必须将像素坐标作为ground cost；只比颜色直方图会对物体错位失明。全96²支持点的密集两两代价有显存/迭代成本，不能把1D投影实验冒称已测完整Sinkhorn。

Actual test status: 文献评估；未实施

### D13 — Interpolating between Optimal Transport and MMD using Sinkhorn Divergences

Source: [2019](https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf). Family: 空间输运.

Inspection: 方法深读：Eq3、20–27，去偏项、梯度、显存

Original method: S_eps(a,b)=OT_eps(a,b)-0.5 OT_eps(a,a)-0.5 OT_eps(b,b)，消除熵正则自比较偏置。

Project assessment: 比裸正则OT更合适作几何差异候选；质量归一化仍会隐藏物体面积，需单独处理。作者大规模GPU能力不等于我们已安装或验证对应库。

Actual test status: 文献评估；未实施

### D14 — Learning Generative Models with Sinkhorn Divergences

Source: [2018](https://proceedings.mlr.press/v84/genevay18a.html). Family: 空间输运.

Inspection: 原论文摘要筛选

Original method: 用可微Sinkhorn比较分布，兼顾不相交支持与计算可行性。

Project assessment: 我们需要逐条件、逐样本下一帧空间分布匹配；不能改成全batch生成图集合匹配，否则可能不同动作间互换结果仍低loss。

Actual test status: 文献评估；未实施

### D15 — Sinkhorn Divergences for Unbalanced Optimal Transport

Source: [2019](https://arxiv.org/abs/1910.12958). Family: 空间输运.

Inspection: 摘要与作者说明；PDF提取超时，未深读公式

Original method: 将输运与质量产生/消失共同处理，允许总质量不同。

Project assessment: 适合同时测错位和物体丢失，但质量销毁代价太小可能允许抹掉难预测物体。当前不用新库/CPU迭代实现；未测，不能宣称不适用。

Actual test status: 文献评估；未实施

### D16 — Generative Modeling using the Sliced Wasserstein Distance

Source: [2018](https://arxiv.org/pdf/1803.11188). Family: 空间输运.

Inspection: 方法深读：一维排序输运与投影平均

Original method: 将高维分布投影到一维，平均各投影Wasserstein，避免昂贵高维匹配。

Project assessment: 项目改成固定16空间方向的weighted W1，加面积差；原文SW2生成器训练不等于本适配。有限方向不是完整二维距离，仍需原始IoU/拓扑验证。

Actual test status: 定义诊断与NATIVE_T训练已完成；距离响应验证成立，三字母训练支持未通过；零质量梯度敏感。

### D17 — Computational Optimal Transport

Source: [2019](https://arxiv.org/html/1803.00567). Family: 空间输运.

Inspection: 专题方法摘读：第2.6节1D CDF/quantile公式；非整本通读

Original method: 一维W1等于两CDF差绝对值的积分；加权离散共同支持可精确累积求和。

Project assessment: 本轮坐标排序固定，梯度只经过像素质量，不硬采样预测前景；避免把空间weighted histogram错误实现为等权颜色排序。作为公式基础文献，非新实验论文。

Actual test status: 文献评估；未实施

### D18 — Topology-Preserving Deep Image Segmentation

Source: [2019](https://www3.cs.stonybrook.edu/~cvl/content/papers/2019/Hu_NeurIPS19.pdf). Family: 拓扑.

Inspection: 方法深读：持久图、Eq2.2、分维匹配与零loss定理条件

Original method: 匹配预测与GT持久同调图，分别处理连通块与孔，未匹配结构推向对角线。

Project assessment: 直接对应B丢孔/O断裂；有资格可见mask才有对应监督。只数Euler数会让块和孔错误抵消；正确拓扑也允许错位/严重变形。现有CPU压力/无新依赖下本轮不跑PH，不把clDice冒称PH复现。

Actual test status: 文献评估；未实施

### D19 — clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation

Source: [2021](https://arxiv.org/pdf/2003.07311). Family: 拓扑.

Inspection: 方法深读：前景/背景骨架条件、soft skeleton；作者PyTorch源码核对

Original method: 用mask与骨架交集的precision/sensitivity及其调和平均表达连通性，soft形态学提供梯度。

Project assessment: GPU可用且无需新网；本轮固定10次侵蚀作响应诊断。二值前景/背景满足条件且clDice=1的定理，不能扩大成有限训练后所有RGB输出保证孔数正确；厚度/角部仍需几何目标。

Actual test status: 定义响应诊断已完成；未列入本轮训练臂，不能报为训练无效。

### D20 — Centerline Boundary Dice Loss for Vascular Segmentation

Source: [2024](https://arxiv.org/abs/2407.01517). Family: 拓扑.

Inspection: 原始摘要筛选

Original method: 在中心线比较外加入边界/半径相关因素，关注几何与拓扑。

Project assessment: 提示纯骨架比较不足；血管形态与实心字母不同，不能直接声称对O/C/B合适，需要另行精确适配。

Actual test status: 文献评估；未实施

### D21 — FourierLoss: Shape-Aware Loss Function with Fourier Descriptors

Source: [2023](https://arxiv.org/abs/2309.12106). Family: 形状/刚体结构.

Inspection: 原始摘要筛选

Original method: 比较分割对象的Fourier轮廓描述子并学习形状细节的重要性。

Project assessment: 这和上轮RGB复数频谱FFL不是同一损失。轮廓参数化、起点、旋转归一化会影响是否忽略真实角度错误；本轮不未经核实移植。

Actual test status: 文献评估；未实施

### D22 — Fused Gromov-Wasserstein Distance for Structured Objects

Source: [2020](https://www.mdpi.com/1999-4893/13/9/212). Family: 形状/刚体结构.

Inspection: 原文方法摘读：定义8、离散Eq24、非凸求解

Original method: 结合点属性/位置间距离与点对内部距离，比较特征和结构。

Project assessment: 内部距离适合刚体形状约束，但纯GW允许等距变换，错位/镜像可不受罚；须另加绝对运动约束。大规模非凸匹配和RGB到点集的可微性未验证。

Actual test status: 文献评估；未实施

### D23 — SE3-Nets: Learning Rigid Body Motion using Deep Neural Networks

Source: [2017](https://arxiv.org/abs/1606.02378). Family: 形状/刚体结构.

Inspection: 原始摘要筛选

Original method: 从深度/动作/点关联学习分部SE3变换，以结构限制刚体运动。

Project assessment: 属于表示/架构改变，依赖深度和关联，不能当RGB GPC的drop-in loss；本轮不重新引入用户已拒绝的pose/template机制。

Actual test status: 文献评估；未实施

### D24 — RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

Source: [2020](https://arxiv.org/abs/2003.12039). Family: 运动/时间.

Inspection: 原始摘要筛选

Original method: 学习逐像素光流，提供运动场估计。

Project assessment: 可启发流误差，但RAFT是网络而非现成物理loss。均匀色块内部对应/遮挡不唯一，外部流估计器也会出错；新资产与监督不在本轮。

Actual test status: 文献评估；未实施

### D25 — Learning to Simulate Complex Physics with Graph Networks

Source: [2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html). Family: 运动/时间.

Inspection: 本轮摘要；上轮训练方法/补充材料已核对

Original method: 以粒子图预测动力学，用训练噪声改善长时误差积累。

Project assessment: 粒子加速度误差是另一种有效定义，但需要粒子状态表示；不能把图像模型变成状态模型后仍称原基线优化。RGB序列速度误差可以另定义，需真实时间对齐与预测历史。

Actual test status: 文献评估；未实施

### D26 — Contrastive Learning of Structured World Models

Source: [2020](https://arxiv.org/abs/1911.12247). Family: 学习表征/任务距离.

Inspection: 原始摘要筛选

Original method: 在对象结构latent中以对比目标学习转移，减少对像素重建的依赖。

Project assessment: 改变表征和负例目标，需要编码器/对象图与匹配基线；不是只改现有输出loss。对比可区分状态但不直接保证毫米级位置和孔形状。

Actual test status: 文献评估；未实施

### D27 — DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Source: [2025](https://arxiv.org/abs/2411.04983). Family: 学习表征/任务距离.

Inspection: 原始摘要筛选（v2 2025）

Original method: 在预训练视觉patch特征上预测世界演化、用于规划。

Project assessment: 改变所预测的量，比改MSE名字更实质；但目前必须保留RGB输出与原输入/采样，需要额外编码/解码或匹配实验，不擅自改为latent任务。

Actual test status: 文献评估；未实施

### D28 — Value-Aware Loss Function for Model-based Reinforcement Learning

Source: [2017](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf). Family: 学习表征/任务距离.

Inspection: 方法摘读：Eq3的transition对value期望误差及未知V问题

Original method: 依据转移误差对价值函数的影响定义模型误差。

Project assessment: 和最终控制相关，但单一覆盖/价值相同不意味着几何和动力学正确；当前缺已验证价值函数，不以它取代用户的形状/运动验收。

Actual test status: 文献评估；未实施

### D29 — Physics-Informed Diffusion Models

Source: [2024](https://arxiv.org/html/2403.14404v3). Family: 物理约束.

Inspection: 方法深读：3.2–3.3、mean/sample估计与Jensen gap、残差/离散化一致性

Original method: 将物理方程残差与数据目标结合，讨论在均值去噪图上施加非线性约束与真实采样分布的区别。

Project assessment: RGB不是压力/速度等物理场，不能直接套PDE，也不能强加受驱动推块的能量守恒。该文提醒本轮D(y+noise)的几何损失不等于采样轨迹几何；必须以raw AR评测。

Actual test status: 文献评估；未实施

## Weighting and appearance: earlier review

This earlier screen does not answer the full error-definition question. Original methods, project adaptation and actual implementation status remain separate.

### P01 — Elucidating the Design Space of Diffusion-Based Generative Models

Source: [NeurIPS 2022](https://arxiv.org/abs/2206.00364). Family: 扩散噪声加权.

Inspection: 论文摘要+作者EDMLoss源码核对

Method: EDM式图像误差已有噪声权重；当前GPC残差MSE等价于带c_out权重的重建误差。

Project assessment: 作为原基线解释；不重复添加整套EDM权重。

Actual status: 基线

### P02 — Efficient Diffusion Training via Min-SNR Weighting Strategy

Source: [ICCV 2023 / arxiv v3](https://arxiv.org/html/2303.09556v3). Family: 扩散噪声加权.

Inspection: 正文+附录D.1.1公式核对

Method: 截断SNR重分配不同噪声级别的训练贡献；附录明确给出EDM乘法系数。

Project assessment: 选入SNR臂；用原GPC sigma_eff定义名义SNR，属于项目适配。

Actual status: 本轮GPU筛选

### P03 — Perception Prioritized Training of Diffusion Models

Source: [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html). Family: 扩散噪声加权.

Inspection: 原论文第3.2节公式核对

Method: P2按(k+SNR)^(-gamma)调整权重，强调部分噪声区间。

Project assessment: 与Min-SNR同属噪声分配，首轮不叠加或同时换采样分布。

Actual status: 未训练

### P04 — Analyzing and Improving the Training Dynamics of Diffusion Models

Source: [CVPR 2024](https://arxiv.org/abs/2312.02696). Family: 扩散噪声加权.

Inspection: 原始摘要初筛

Method: 从幅值稳定、层设计和EMA等改善扩散训练。

Project assessment: 整套EDM2涉及网络/训练改造，不能当纯换loss来替代GPC基线。

Actual status: 范围外整体改造

### P05 — Improved Noise Schedule for Diffusion Training

Source: [2024预印本 / ICCV 2025](https://arxiv.org/abs/2407.03297). Family: 扩散噪声加权.

Inspection: 原始摘要+会议论文方法摘录初筛

Method: 通过log-SNR重要采样增加关键噪声区间训练。

Project assessment: 会改变noise抽样，与本轮只改loss不一致；保留为独立后续因素。

Actual status: 未训练

### P06 — Deep multi-scale video prediction beyond mean square error

Source: [ICLR 2016](https://arxiv.org/html/1511.05440). Family: 图像梯度与稳健误差.

Inspection: 全文方法第2.3/2.4节公式核对

Method: GDL惩罚预测与GT的相邻像素梯度幅值差，配合像素损失。

Project assessment: 选入GDL臂；在可微去噪重建图上计算，不是旧边缘mask加权MSE。

Actual status: 本轮GPU筛选

### P07 — Focal Frequency Loss for Image Reconstruction and Synthesis

Source: [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.html). Family: 频域.

Inspection: 论文方法+作者代码逐项核对

Method: 自适应强调困难频率；作者实现保留复数相位，权重矩阵detach。

Project assessment: 选入FFL臂；FFT用FP32，保持原模型forward精度。

Actual status: 本轮GPU筛选

### P08 — Fourier Space Losses for Efficient Perceptual Image Super-Resolution

Source: [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Fuoli_Fourier_Space_Losses_for_Efficient_Perceptual_Image_Super-Resolution_ICCV_2021_paper.html). Family: 频域.

Inspection: 原始摘要初筛

Method: 用Fourier空间监督改善高频内容，并研究频域判别。

Project assessment: 支持频域方向；首轮用无需新增判别器的FFL，不移植超分架构。

Actual status: 未训练

### P09 — Image Quality Assessment: From Error Visibility to Structural Similarity

Source: [IEEE TIP 2004](https://www.ece.uwaterloo.ca/~z70wang/publications/ssim.html). Family: 局部结构统计.

Inspection: 作者论文/项目页核对

Method: SSIM比较局部亮度、对比度和结构，而非逐像素平方差。

Project assessment: 可微且便宜；大白底和小物体需检验，不能代替孔洞/角度验收。

Actual status: 未训练

### P10 — Multi-scale Structural Similarity for Image Quality Assessment

Source: [Asilomar 2003](https://ece.uwaterloo.ca/~z70wang/publications/msssim.html). Family: 局部结构统计.

Inspection: 作者原始摘要初筛

Method: 多尺度组合结构相似度。

Project assessment: 96px小物体的孔洞在下采样中可能消失，需另行验证尺度。

Actual status: 未训练

### P11 — Loss Functions for Neural Networks for Image Processing

Source: [arxiv 2015](https://arxiv.org/abs/1511.08861). Family: 图像梯度与稳健误差.

Inspection: 原始摘要/方法概述初筛

Method: 比较L2、L1、结构相似等恢复损失，讨论组合目标。

Project assessment: 图像恢复收益不等价于动作/物理准确；可作为后续L1/结构组合依据。

Actual status: 未训练

### P12 — A General and Adaptive Robust Loss Function

Source: [CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.html). Family: 图像梯度与稳健误差.

Inspection: 原论文公式与导数核对

Method: 连续稳健性参数统一L2、Charbonnier等，改变大残差的影响。

Project assessment: 大误差也可能来自重要接触运动；不能把其自动当离群点降权。

Actual status: 未训练

### P13 — Perceptual Losses for Real-Time Style Transfer and Super-Resolution

Source: [ECCV 2016](https://arxiv.org/abs/1603.08155). Family: 学习式感知.

Inspection: 原始摘要初筛

Method: 使用预训练视觉特征衡量图像差异。

Project assessment: 需要额外特征网络/资产；自然图像感知相似不等价于刚体精确。

Actual status: 未训练

### P14 — The Unreasonable Effectiveness of Deep Features as a Perceptual Metric

Source: [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html). Family: 学习式感知.

Inspection: 原论文摘要/方法概述核对

Method: LPIPS以校准的深特征匹配人类感知判断。

Project assessment: 可能帮助外观；需检验96px字母/孔洞域偏差和新增网络费用。

Actual status: 未训练

### P15 — Image Quality Assessment: Unifying Structure and Texture Similarity

Source: [arxiv 2020](https://arxiv.org/abs/2004.07728). Family: 学习式感知.

Inspection: 原始摘要核对

Method: DISTS结合深特征结构/纹理统计，容忍部分几何变化。

Project assessment: 其平移/变形不敏感性与本项目精确位置要求有张力，不优先。

Actual status: 未训练

### P16 — DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data

Source: [NeurIPS 2023](https://arxiv.org/abs/2306.09344). Family: 学习式感知.

Inspection: 原始摘要/出版信息核对

Method: 强调前景、布局和语义等中层感知关系。

Project assessment: 比纯纹理更相关，但仍需额外模型及字母域验证，未证明保孔。

Actual status: 未训练

### P17 — The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks

Source: [CVPR 2018](https://arxiv.org/abs/1705.08790). Family: 轮廓与拓扑.

Inspection: 原始摘要初筛

Method: 通过可优化替代函数直接针对分割IoU。

Project assessment: GPC输出RGB而非分割概率；需明确可微RGB→mask及遮挡资格。

Actual status: 未训练

### P18 — Boundary loss for highly unbalanced segmentation

Source: [MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html). Family: 轮廓与拓扑.

Inspection: 原论文摘要/方法概述核对

Method: 使用距离/边界表示补充区域损失，面向前景不平衡。

Project assessment: 对小灰物体有针对性；需要可靠概率图、GT距离图与单位适配。

Actual status: 未训练

### P19 — Topology-Preserving Deep Image Segmentation

Source: [NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/2d95666e2649fcfc6e3af75e09f5adb9-Abstract.html). Family: 轮廓与拓扑.

Inspection: 原始摘要/方法概述初筛

Method: 用持久同调相关目标鼓励预测与GT拓扑一致。

Project assessment: 直接关联孔洞，但RGB概率图、遮挡与同调计算开销尚未验证。

Actual status: 未训练

### P20 — clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation

Source: [CVPR 2021](https://arxiv.org/abs/2003.07311). Family: 轮廓与拓扑.

Inspection: 原始摘要与适用条件核对

Method: 基于分割和骨架交集，soft-clDice用于连通结构。

Project assessment: 管状结构结论不自动推广到O/C/B的孔洞和刚体朝向。

Actual status: 未训练

### P21 — Centerline Boundary Dice Loss for Vascular Segmentation

Source: [arxiv 2024](https://arxiv.org/abs/2407.01517). Family: 轮廓与拓扑.

Inspection: 原始摘要初筛

Method: cbDice融合中心线、边界和尺度，兼顾拓扑与几何。

Project assessment: 比只保连通更相关，但仍是血管分割，需另做RGB与遮挡适配。

Actual status: 未训练

### P22 — Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion

Source: [NeurIPS 2024](https://arxiv.org/abs/2407.01392). Family: 时序生成范式.

Inspection: 原始摘要/训练范式初筛

Method: 对序列token使用独立噪声，结合因果预测和序列扩散。

Project assessment: 改变conditioning/noise/序列训练范式，不是本轮H1 loss小替换。

Actual status: 范围外整体改造

### P23 — Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

Source: [2025](https://arxiv.org/html/2506.08009). Family: 时序生成范式.

Inspection: 正文/分布匹配与消融方法核对

Method: 自生成历史与整体分布匹配损失，包含DMD/SiD/GAN等路径。

Project assessment: 旧AR4-MSE并非本论文复现；整体移植需教师/判别等新组件。

Actual status: 未训练

### P24 — Foresight Diffusion: Improving Sampling Consistency in Predictive Diffusion Models

Source: [2025 / 2026 v2](https://arxiv.org/abs/2505.16474). Family: 时序生成范式.

Inspection: 最新版原始摘要核对

Method: 把条件理解与目标去噪解耦，改善预测采样一致性。

Project assessment: 需要独立预测流/表征，属于结构与训练改造，非纯loss对照。

Actual status: 范围外整体改造

### P25 — Distilling Physical Priors into Streaming World Models

Source: [arxiv 2026](https://arxiv.org/abs/2608.07981). Family: 物理与状态表征.

Inspection: 原始摘要初筛

Method: 物理视频监督、教师蒸馏与时序信用分配/RL组合。

Project assessment: 涉及新数据/大型教师/训练流程，不能在当前授权内直接移植。

Actual status: 范围外整体改造

### P26 — GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks

Source: [ICML 2018](https://proceedings.mlr.press/v80/chen18a.html). Family: 多目标梯度平衡.

Inspection: 原始摘要/方法概述核对

Method: 动态调整多个目标的梯度幅值与学习速度。

Project assessment: 适合后续检查native与辅助项平衡；需单独控制额外梯度和自适应权重。

Actual status: 未训练

### P27 — Gradient Surgery for Multi-Task Learning

Source: [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html). Family: 多目标梯度平衡.

Inspection: 原始摘要/方法概述核对

Method: 处理多个任务间冲突梯度。

Project assessment: 仅在实测冲突后才有针对性；不是把两个loss数值归一化就等价。

Actual status: 未训练

### P28 — RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

Source: [ECCV 2020](https://arxiv.org/abs/2003.12039). Family: 光流运动.

Inspection: 原始摘要初筛

Method: 估计稠密像素运动，可支持flow监督或运动比较方向。

Project assessment: 是光流架构而非现成GPC loss；额外教师及遮挡/刚体精度需验证。

Actual status: 未训练

### P29 — Learning to Simulate Complex Physics with Graph Networks

Source: [ICML 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html). Family: 物理与状态表征.

Inspection: 原论文摘要+补充材料训练目标核对

Method: 粒子图模拟器；讨论一步噪声训练与长滚动目标的取舍。

Project assessment: 表征/监督不同，不能用其物理成功替代当前image-based GPC任务。

Actual status: 范围外整体改造
