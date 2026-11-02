# Loss literature used in the project

Primary sources from the recorded review. Reading depth and project testing
status are explicit; no new search or training was performed for this PR.

| Method | Source | Inspection | Project status |
| --- | --- | --- | --- |
| 扩散目标的兼容性 | [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) | 本轮摘要；上轮预条件公式及作者EDMLoss源码核对 | 原baseline；本轮只读复用 |
| 扩散目标的兼容性 | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 摘要/目标定位 | 文献评估；未实施 |
| 占据区域 | [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/pdf/1606.04797) | 方法深读：第3节平方分母Dice及其梯度 | 定义诊断与G_ONLY/NATIVE_G训练已完成；纯G使验证C长时退化；NATIVE_G仅C短时方向条件通过，长时仍退步。 |
| 占据区域 | [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237) | 摘要筛选 | 文献评估；未实施 |
| 占据区域 | [Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/abs/1706.05721) | 摘要筛选 | 文献评估；未实施 |
| 占据区域 | [The Lovász-Softmax loss](https://arxiv.org/pdf/1705.08790) | 方法深读：Eq11–13、排序求Jaccard扩展、per-image与dataset口径 | 文献评估；未实施 |
| 轮廓/距离场 | [Boundary loss for highly unbalanced segmentation](https://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf) | 方法深读：Eq2–6，有符号GT距离场与常数项 | 定义响应诊断已完成；未列入本轮训练臂，不能报为训练无效。 |
| 轮廓/距离场 | [Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks](https://arxiv.org/pdf/1904.10030) | 方法深读：Eq4–8及侵蚀/卷积替代路线 | 文献评估；未实施 |
| 轮廓/距离场 | [A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation](https://arxiv.org/abs/2302.03868) | 摘要筛选 | 文献评估；未实施 |
| 点集/定位距离 | [Locating Objects Without Bounding Boxes](https://arxiv.org/pdf/1806.07564) | 方法深读：Eq5–7、两项防退化解释 | 文献评估；未实施 |
| 点集/定位距离 | [A Point Set Generation Network for 3D Object Reconstruction From a Single Image](https://robotics.stanford.edu/~haosu/papers/SI2PC_arxiv_submit.pdf) | 作者论文方法摘读：Chamfer/EMD公式 | 文献评估；未实施 |
| 空间输运 | [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) | 原论文摘要与作者方法说明 | 文献评估；未实施 |
| 空间输运 | [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf) | 方法深读：Eq3、20–27，去偏项、梯度、显存 | 文献评估；未实施 |
| 空间输运 | [Learning Generative Models with Sinkhorn Divergences](https://proceedings.mlr.press/v84/genevay18a.html) | 原论文摘要筛选 | 文献评估；未实施 |
| 空间输运 | [Sinkhorn Divergences for Unbalanced Optimal Transport](https://arxiv.org/abs/1910.12958) | 摘要与作者说明；PDF提取超时，未深读公式 | 文献评估；未实施 |
| 空间输运 | [Generative Modeling using the Sliced Wasserstein Distance](https://arxiv.org/pdf/1803.11188) | 方法深读：一维排序输运与投影平均 | 定义诊断与NATIVE_T训练已完成；距离响应验证成立，三字母训练支持未通过；零质量梯度敏感。 |
| 空间输运 | [Computational Optimal Transport](https://arxiv.org/html/1803.00567) | 专题方法摘读：第2.6节1D CDF/quantile公式；非整本通读 | 文献评估；未实施 |
| 拓扑 | [Topology-Preserving Deep Image Segmentation](https://www3.cs.stonybrook.edu/~cvl/content/papers/2019/Hu_NeurIPS19.pdf) | 方法深读：持久图、Eq2.2、分维匹配与零loss定理条件 | 文献评估；未实施 |
| 拓扑 | [clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://arxiv.org/pdf/2003.07311) | 方法深读：前景/背景骨架条件、soft skeleton；作者PyTorch源码核对 | 定义响应诊断已完成；未列入本轮训练臂，不能报为训练无效。 |
| 拓扑 | [Centerline Boundary Dice Loss for Vascular Segmentation](https://arxiv.org/abs/2407.01517) | 原始摘要筛选 | 文献评估；未实施 |
| 形状/刚体结构 | [FourierLoss: Shape-Aware Loss Function with Fourier Descriptors](https://arxiv.org/abs/2309.12106) | 原始摘要筛选 | 文献评估；未实施 |
| 形状/刚体结构 | [Fused Gromov-Wasserstein Distance for Structured Objects](https://www.mdpi.com/1999-4893/13/9/212) | 原文方法摘读：定义8、离散Eq24、非凸求解 | 文献评估；未实施 |
| 形状/刚体结构 | [SE3-Nets: Learning Rigid Body Motion using Deep Neural Networks](https://arxiv.org/abs/1606.02378) | 原始摘要筛选 | 文献评估；未实施 |
| 运动/时间 | [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039) | 原始摘要筛选 | 文献评估；未实施 |
| 运动/时间 | [Learning to Simulate Complex Physics with Graph Networks](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html) | 本轮摘要；上轮训练方法/补充材料已核对 | 文献评估；未实施 |
| 学习表征/任务距离 | [Contrastive Learning of Structured World Models](https://arxiv.org/abs/1911.12247) | 原始摘要筛选 | 文献评估；未实施 |
| 学习表征/任务距离 | [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://arxiv.org/abs/2411.04983) | 原始摘要筛选（v2 2025） | 文献评估；未实施 |
| 学习表征/任务距离 | [Value-Aware Loss Function for Model-based Reinforcement Learning](https://proceedings.mlr.press/v54/farahmand17a/farahmand17a.pdf) | 方法摘读：Eq3的transition对value期望误差及未知V问题 | 文献评估；未实施 |
| 物理约束 | [Physics-Informed Diffusion Models](https://arxiv.org/html/2403.14404v3) | 方法深读：3.2–3.3、mean/sample估计与Jensen gap、残差/离散化一致性 | 文献评估；未实施 |
