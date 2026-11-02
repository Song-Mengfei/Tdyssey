# R3B 完整首轮与配对评测

已确认启动故障原因：Default Windows spawn selected the venv redirector as sys._base_executable; its extra actual-Python child had invalid duplicated ProcessPool queue synchronization handles (WinError6 after successful initializer).；最小修复：Use the actual base Python executable for multiprocessing while preserving the existing venv launcher identity; no dataset or worker-count changes.；本次1,499批与23,984个不同目标已完整完成，本轮自然 H4 灰物体 ROI 误差低于同窗口 6000 和 copy-last。

**已完成本次批准的首轮 1,499 批、23,984 个不同目标窗口；没有进入第二轮。**
正式更新成功 1499，AMP skip 0；旧模型 lineage6000 与本阶段 probe/正式计数分开。
仅使用原 R3A，CP1/CP2未混入，八个 heldout 字母未访问。LR保持1e-5、clean/weak各50%micro、micro4×4有效16。

本轮自然 H4 灰物体 ROI 误差低于同窗口 6000 和 copy-last。 这是一轮固定窗口结果，不能据此宣称收敛、形状刚性已解决、模型无效或继续训练无用。

旧失败run_id=R3B-firstepoch-0c0e39d5b67940799c2211eca058017f保留完整原记录和33.047秒费用；CPU后台诊断证据 (external historical reference: `diagnostics/diag02-repaired/gate.json`)绑定SHA 77f04815e668a59ff620048874c42e1b502f0bbc1671edad0afc119394ba1465。模型恢复后的同一两worker配置另有CPU取批验证，随后才运行独立记账的GPU工程分支。诊断、工程和正式数据暴露分列，正式目标不重复。

本次模型流水线记录 1456.531秒；engineering {"abandoned": 0, "attempts": 1, "skips": 0, "successes": 1}；formal {"abandoned": 0, "attempts": 1499, "skips": 0, "successes": 1499}。仅统计用时，没有模型时间上限。

自然验证每个144seen-val family预先均匀抽取一个合法H4anchor，H1/H4同anchor；16stress和旧4回归单列。
H4仅反馈原始预测。表中误差越低越好、IoU越高越好；形状为未检出失败/失败/未知，未知保留完整分母。

| 自然H4 | N | Copy ROI | 6000 ROI | Epoch1 ROI | 6000/Epoch1灰IoU | 6000→Epoch1形状P/F/U |
| --- | --- | --- | --- | --- | --- | --- |
| all | 144 | 0.0901 | 0.0771 | 0.0462 | 0.680 / 0.795 | 50/13/81 -> 51/10/83 |
| T | 24 | 0.0617 | 0.0556 | 0.0346 | 0.781 / 0.856 | 15/0/9 -> 14/0/10 |
| L | 24 | 0.0985 | 0.0729 | 0.0391 | 0.701 / 0.836 | 7/1/16 -> 8/0/16 |
| O | 24 | 0.0950 | 0.0949 | 0.0575 | 0.615 / 0.762 | 9/2/13 -> 8/2/14 |
| C | 24 | 0.0905 | 0.0912 | 0.0519 | 0.591 / 0.736 | 5/5/14 -> 7/3/14 |
| X | 24 | 0.0930 | 0.0678 | 0.0441 | 0.694 / 0.780 | 8/2/14 -> 10/1/13 |
| B | 24 | 0.1016 | 0.0803 | 0.0503 | 0.698 / 0.798 | 6/3/15 -> 4/4/16 |

O: ROI 0.0949→0.0575，灰 IoU 0.615→0.762；形状未检出失败/失败/未知 9/2/13→8/2/14（分母24）。 C: ROI 0.0912→0.0519，灰 IoU 0.591→0.736；形状未检出失败/失败/未知 5/5/14→7/3/14（分母24）。

自然H1 epoch1 ROI误差 0.0175，灰IoU 0.931。
H4 epoch1物理位姿可测 122/144；转角误差只在可测且按形状固有对称周期的样本上计算。
遮挡、越界和不可测不会算通过。棱角数/孔洞/碎片/面积/轮廓与蓝推杆逐项见[完整表](report/summary.csv)、形状与可测性 (external historical reference: `report/shape-and-observability.csv`)。

| 自然近静止组 | N | Copy drift px | 6000 drift px | Epoch1 drift px |
| --- | --- | --- | --- | --- |
| H1 | 70 | 0.0000 | 0.4242 | 0.0725 |
| H4 | 52 | 0.0000 | 1.0128 | 0.1541 |

O/C图和旧O/B/C/X回归均为GT/copy-last/6000/epoch1同坐标并排。近景只作同一GT定义整数crop和最近邻放大；
不配准、不平滑、不修边。预览如需限幅会标注，原float32和评分保持原值。

原6000 checkpoint只读保留；本阶段单份滚动完整last.pt保留Adam/scaler、实际sampler cursor及三条conditioning RNG。
仅报告本轮改善/退步及剩余形状问题，不自动续训或更改采样。后续实施必须等待用户明确审批。

O/C与旧四例图 (external historical reference: `report/index.html`) · [配对逐指标胜/平/负与有效分母](report/paired-metrics.csv) ·
[机器结果](result.json) · 原始float32索引t0 (external historical reference: `evaluation/t0/index.json`) · epoch1索引 (external historical reference: `evaluation/epoch1/index.json`)
