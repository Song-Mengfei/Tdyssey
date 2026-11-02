# R2C1 同窗原RGB评估

灰色物体为主；蓝推杆指标独立。端点失败、途中失败、缺测及形状unknown不剔除。

|Cohort|H|N|copy灰error|原GPC灰error|训练后灰error|对copy胜/平/负|相对改善|训练后端点/路径失败|形状unknown|
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
|train_visual|1|12|未评|未评|未评|0/0/0 (未评12)|undefined|0/0|12|
|train_visual|4|12|未评|未评|未评|0/0/0 (未评12)|undefined|0/0|12|
|val_natural|1|12|未评|未评|未评|0/0/0 (未评12)|undefined|0/0|12|
|val_natural|4|12|未评|未评|未评|0/0/0 (未评12)|undefined|0/0|12|
|val_motion|1|12|未评|未评|未评|0/0/0 (未评12)|undefined|0/0|12|
|val_motion|4|12|未评|未评|未评|0/0/0 (未评12)|undefined|0/0|12|

端点数学沿用冻结协议；H2/H3失败另列，all-step-valid胜数不能被H4恢复掩盖。有限越界原值评估；PNG仅编码截幅时明确标注，未修色/修形。

near-static/small漂移、灰ROI RGB/IoU/原轮廓及蓝推杆分别见JSON与CSV。TRAIN拟合并非泛化，val均6seen字母且轨迹刻意含运动，natural与motion仍可能同轨迹时间重叠。

所有36窗的H1/H4都在media/index.html；主图不混入原始T任务GPC，原始基线独立页面。
