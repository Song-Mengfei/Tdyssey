# R3B试玩第一步诊断：逐帧对照

本报告只读取本次保存的实际轨迹和原始float32预测，不训练、不修改权重，不隐藏碎块、补孔、修边或贴回模板。
当前包含40组保存片段，320个A/B逐帧记录；有0个缺少raw的失败片段记录保留。选择是定向诊断，不是总体失败率。
**物理分组必须分开读：7499模型仍是旧动态推杆/0.1秒半衰期训练；新默认原始运动学推杆结果是跨物理预览，不是旧协议故障的动力学精确重放，也不能单独证明训练与递推分布不一致。**

A每控制重新使用真实四帧历史，有真实历史校正；B从相同真实起点连续预测四步，后续只喂自己raw预测。
每步动作/seed配对，首帧独立调用并检查一致。现有UI的H4每批重新取真实历史，不是从最初场景无限递推。
GT未来只作图像/误差标签。图为同坐标整数nearest放大；原始raw数值不被显示clip影响。

逐帧图与同步慢放视频 (external historical reference: `report/index.html`) · 全部逐帧指标 (external historical reference: `report/frame-metrics.csv`) · 结构化原始引用 (external historical reference: `report/summary.json`)


## 原始物理与持续推进验收

6个已训练字母的实际body/material/geometry与原source参考一致；T、C逐前后state15最大误差均0。O中心和近墙执行hold2/push6/withdraw6/继续hold10，继续真实推进，不清速度、不回写位姿。中心首次整控无AO为9，近墙为10；末速度0不等于严格零位移，首脱离控仍有.130603/.036994世界像素残余位移/接触位置修正。近墙有真实OW接触，最大墙穿透.5031px；中心原运动学推时AO穿透达2.291px，未用旧finiteforce保护隐藏。源物理damping0与旧0.1秒半衰期训练协议不同。

参数/子步一致性/耗费完整记录 (external historical reference: `/diagnostics-root/physics-alignment/validated-v2/summary.json`) · 速度/角速度/接触曲线 (external historical reference: `/diagnostics-root/physics-alignment/validated-v2/separation-curves.png`) · 中心短片 (external historical reference: `/diagnostics-root/physics-alignment/validated-v2/O-center-browser.mp4`) · 近墙短片 (external historical reference: `/diagnostics-root/physics-alignment/validated-v2/O-near-wall-browser.mp4`)。


## 固定片段去噪3/8/16对照

下表只比较同物理、同case/anchor、同动作及逐步seed都有3/8/16结果的固定代表片段，不能拿3步全部片段平均与8/16的子集平均直接比较。

| case | anchor | 分支 | 去噪步 | 四帧平均ROI | H4 ROI | 4次调用外层秒 | 最早可见异常 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-inner_corner | 7 | A | 3 | 0.1505 | 0.0842 | 0.400 | 4 |
| B-inner_corner | 7 | A | 8 | 0.1469 | 0.0648 | 0.816 | none listed |
| B-inner_corner | 7 | A | 16 | 0.1482 | 0.0659 | 1.602 | none listed |
| B-inner_corner | 7 | B | 3 | 0.2795 | 0.3029 | 0.448 | 2 |
| B-inner_corner | 7 | B | 8 | 0.2767 | 0.3002 | 0.830 | 4 |
| B-inner_corner | 7 | B | 16 | 0.2788 | 0.3011 | 1.537 | 4 |
| C-inner_corner | 7 | A | 3 | 0.0541 | 0.0077 | 0.245 | 1 |
| C-inner_corner | 7 | A | 8 | 0.0389 | 0.0074 | 0.845 | none listed |
| C-inner_corner | 7 | A | 16 | 0.0376 | 0.0078 | 1.536 | none listed |
| C-inner_corner | 7 | B | 3 | 0.1873 | 0.2099 | 0.256 | 1 |
| C-inner_corner | 7 | B | 8 | 0.1630 | 0.1821 | 0.877 | none listed |
| C-inner_corner | 7 | B | 16 | 0.1570 | 0.1752 | 1.905 | none listed |
| O-inner_corner | 7 | A | 3 | 0.1360 | 0.0714 | 0.267 | 1 |
| O-inner_corner | 7 | A | 8 | 0.1312 | 0.0665 | 1.361 | none listed |
| O-inner_corner | 7 | A | 16 | 0.1320 | 0.0683 | 2.542 | none listed |
| O-inner_corner | 7 | B | 3 | 0.2431 | 0.2979 | 0.595 | 1 |
| O-inner_corner | 7 | B | 8 | 0.2417 | 0.2940 | 0.833 | none listed |
| O-inner_corner | 7 | B | 16 | 0.2405 | 0.2885 | 1.594 | none listed |
| X-inner_corner | 7 | A | 3 | 0.0735 | 0.0111 | 0.486 | 3 |
| X-inner_corner | 7 | A | 8 | 0.0774 | 0.0109 | 0.940 | 3 |
| X-inner_corner | 7 | A | 16 | 0.0778 | 0.0117 | 1.873 | 3 |
| X-inner_corner | 7 | B | 3 | 0.1743 | 0.2096 | 0.539 | none listed |
| X-inner_corner | 7 | B | 8 | 0.1852 | 0.2201 | 0.873 | none listed |
| X-inner_corner | 7 | B | 16 | 0.1892 | 0.2226 | 1.778 | none listed |

配对原始明细：paired-sampler-comparison.csv (external historical reference: `report/paired-sampler-comparison.csv`)。耗时读取实际runner逐调用`seconds`字段，兼容旧`sampler_s`；缺失或不完整显示N/A，不能填0。全片段实际调用外层耗时合计44.880482秒，320/320条有效调用记录。 外层合计44.880482秒与model-proof内层sampler计时44.836587秒边界不同，差0.043895秒；外层另含输入copy、返回索引和finite扫描，不能强行对齐或当纯GPU费用。 不含轨迹、图表或视频生成时间，不给模型时间硬上限，不删除更多步数反而更差的结果。

## 六张截图对应

| 截图 | 字母 | 状态 | 说明 |
| --- | --- | --- | --- |
| 1 | B | 已复现同类 | 新原始物理下同类组成现象；可能跨片段/步，不称整张同帧复现或旧协议精确重放；证据见JSON |
| 2 | X | 已复现同类 | 新原始物理下同类组成现象；可能跨片段/步，不称整张同帧复现或旧协议精确重放；证据见JSON |
| 3 | C | 已复现同类 | 新原始物理下同类组成现象；可能跨片段/步，不称整张同帧复现或旧协议精确重放；证据见JSON |
| 4 | C | 已复现同类 | 新原始物理下同类组成现象；可能跨片段/步，不称整张同帧复现或旧协议精确重放；证据见JSON |
| 5 | O | 已复现同类 | 新原始物理下同类组成现象；可能跨片段/步，不称整张同帧复现或旧协议精确重放；证据见JSON |
| 6 | O | 已复现同类 | 新原始物理下同类组成现象；可能跨片段/步，不称整张同帧复现或旧协议精确重放；证据见JSON |

截图的多个组成现象若在不同片段/不同步出现，会逐项链接，不能冒称同一帧精确还原。
O/C圆化以人工逐帧观察确认并记录，没有用不可靠raw角点数量直接认定“棱角恢复率”。

[实际逐图人工观察](review/VISUAL-REVIEW.md) · 只标注实际看过的逐帧记录 (external historical reference: `review/manual-observations.json`)。人工观察保留首步已有圆化/形变、后续累积、真实历史校正也出现碎块等例外；没有异常标注不等于刚体通过。

## 可见异常最早发生步

manual_*为实际看过同坐标保存图的人工形状判断，与自动palette可测性分开保留；遮挡unknown不会盖过这些可见形变。

| 片段 | 字母 | 去噪步 | 分支 | 最早异常步 | 可见异常 |
| --- | --- | --- | --- | --- | --- |
| clip000 | B | 3 | A | 2 | pusher_disappeared |
| clip000 | B | 3 | B | 2 | pusher_disappeared |
| clip001 | B | 3 | A | 1 | manual_deformation,manual_hole_merge |
| clip001 | B | 3 | B | 1 | manual_deformation,manual_hole_merge |
| clip002 | B | 3 | A | 1 | pusher_disappeared |
| clip002 | B | 3 | B | 1 | pusher_disappeared |
| clip003 | B | 16 | A | none listed | No rigidity pass inferred |
| clip003 | B | 16 | B | 4 | pusher_disappeared |
| clip004 | B | 3 | A | 4 | pusher_disappeared |
| clip004 | B | 3 | B | 2 | manual_deformation,manual_hole_merge |
| clip005 | B | 8 | A | none listed | No rigidity pass inferred |
| clip005 | B | 8 | B | 4 | pusher_disappeared |
| clip006 | B | 3 | A | none listed | No rigidity pass inferred |
| clip006 | B | 3 | B | 3 | open_field_gray_fragment,manual_gray_fragment |
| clip007 | B | 3 | A | none listed | No rigidity pass inferred |
| clip007 | B | 3 | B | none listed | No rigidity pass inferred |
| clip008 | B | 3 | A | 1 | visible_hole_count_change,manual_deformation |
| clip008 | B | 3 | B | 1 | visible_hole_count_change,manual_deformation |
| clip009 | B | 3 | A | none listed | No rigidity pass inferred |
| clip009 | B | 3 | B | 2 | open_field_gray_fragment,manual_gray_fragment |
| clip010 | C | 3 | A | none listed | No rigidity pass inferred |
| clip010 | C | 3 | B | 4 | pusher_disappeared,visible_hole_count_change,manual_deformation |
| clip011 | C | 3 | A | 1 | manual_rounding |
| clip011 | C | 3 | B | 1 | manual_rounding |
| clip012 | C | 3 | A | 2 | manual_rounding |
| clip012 | C | 3 | B | 2 | manual_rounding |
| clip013 | C | 16 | A | none listed | No rigidity pass inferred |
| clip013 | C | 16 | B | none listed | No rigidity pass inferred |
| clip014 | C | 3 | A | 1 | manual_rounding |
| clip014 | C | 3 | B | 1 | manual_rounding |
| clip015 | C | 8 | A | none listed | No rigidity pass inferred |
| clip015 | C | 8 | B | none listed | No rigidity pass inferred |
| clip016 | C | 3 | A | none listed | No rigidity pass inferred |
| clip016 | C | 3 | B | 3 | open_field_gray_fragment,manual_gray_fragment |
| clip017 | C | 3 | A | none listed | No rigidity pass inferred |
| clip017 | C | 3 | B | none listed | No rigidity pass inferred |
| clip018 | C | 3 | A | 1 | manual_rounding |
| clip018 | C | 3 | B | 1 | manual_rounding |
| clip019 | C | 3 | A | 1 | manual_rounding |
| clip019 | C | 3 | B | 1 | manual_rounding |
| clip020 | O | 3 | A | 1 | manual_rounding |
| clip020 | O | 3 | B | 1 | manual_rounding |
| clip021 | O | 3 | A | 1 | manual_rounding |
| clip021 | O | 3 | B | 1 | manual_rounding |
| clip022 | O | 3 | A | 2 | manual_rounding |
| clip022 | O | 3 | B | 2 | manual_deformation |
| clip023 | O | 16 | A | none listed | No rigidity pass inferred |
| clip023 | O | 16 | B | none listed | No rigidity pass inferred |
| clip024 | O | 3 | A | 1 | manual_rounding |
| clip024 | O | 3 | B | 1 | manual_rounding |
| clip025 | O | 8 | A | none listed | No rigidity pass inferred |
| clip025 | O | 8 | B | none listed | No rigidity pass inferred |
| clip026 | O | 3 | A | 3 | open_field_gray_fragment,manual_gray_fragment |
| clip026 | O | 3 | B | none listed | No rigidity pass inferred |
| clip027 | O | 3 | A | none listed | No rigidity pass inferred |
| clip027 | O | 3 | B | none listed | No rigidity pass inferred |
| clip028 | O | 3 | A | 1 | manual_rounding |
| clip028 | O | 3 | B | 1 | manual_rounding |
| clip029 | O | 3 | A | 1 | manual_rounding |
| clip029 | O | 3 | B | 1 | manual_rounding |
| clip030 | X | 3 | A | 3 | open_field_gray_fragment,manual_gray_fragment |
| clip030 | X | 3 | B | 3 | open_field_gray_fragment,manual_gray_fragment |
| clip031 | X | 3 | A | none listed | No rigidity pass inferred |
| clip031 | X | 3 | B | none listed | No rigidity pass inferred |
| clip032 | X | 3 | A | none listed | No rigidity pass inferred |
| clip032 | X | 3 | B | 3 | large_visible_area_difference |
| clip033 | X | 16 | A | 3 | open_field_gray_fragment |
| clip033 | X | 16 | B | none listed | No rigidity pass inferred |
| clip034 | X | 3 | A | 3 | open_field_gray_fragment |
| clip034 | X | 3 | B | none listed | No rigidity pass inferred |
| clip035 | X | 8 | A | 3 | open_field_gray_fragment |
| clip035 | X | 8 | B | none listed | No rigidity pass inferred |
| clip036 | X | 3 | A | none listed | No rigidity pass inferred |
| clip036 | X | 3 | B | 3 | open_field_gray_fragment,manual_gray_fragment |
| clip037 | X | 3 | A | none listed | No rigidity pass inferred |
| clip037 | X | 3 | B | none listed | No rigidity pass inferred |
| clip038 | X | 3 | A | none listed | No rigidity pass inferred |
| clip038 | X | 3 | B | 3 | pusher_disappeared,manual_deformation |
| clip039 | X | 3 | A | 2 | open_field_gray_fragment,manual_gray_fragment |
| clip039 | X | 3 | B | 2 | open_field_gray_fragment,manual_gray_fragment |

## 解释边界

- 第1步A/B相同且已错，说明该误差不需要多步递推才能出现；仍不能仅凭截图归因网络结构或训练分布。
- 同一轨迹/seed下B后续恶化而A较好，支持预测历史递推对本片段的影响；不能直接证明具体训练机制或所有场景。
- 形状计数区分可见孔洞、独立灰块、近蓝色边缘的小区域/抗锯齿和遮挡。全局occlusion不会抹掉清楚的推杆消失、可见孔变化/碎块；unknown不算pass。
- 原始角点数受96像素栅格、位置朝向及遮挡影响，只作诊断。明显圆化需结合原始同坐标图与人工记录。
- 质心/相对位置是可见图像测量，不冒充真实物理位姿；实际平移、转角与接触由保存标签单列。推杆消失时相关位置误差不可测，失效计数仍保留。
- 3/8/16只在固定代表片段使用同checkpoint/动作/seed比较；所有慢/坏结果保留，不保证去噪更多更好。
- 确认的UI/输入/时序问题及修复证据由本阶段接口审计另附；报告不会把鼠标预览十字当作实际四步执行目标。

采样速度、per-step异常和全部分母见step-summary.csv/summary.json；不得以缺失值过滤失败后宣称模型可靠。
完成本诊断后STOP waiting_user_approval，不开始新训练、采集或模型方案。
