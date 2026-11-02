# R2C2：固定300更新权重的原生采样次数诊断

本阶段只比较同一个可信300更新检查点的3/8/16次去噪。3次及copy-last完全引用R1，新增8/16各36次sampler批调用、144原始float32四步帧及72个H1/H4端点评分。合计72批、288帧、144新端点评分；实际inner_model forward共864次，初始和结束全部296参数/298state张量摘要相同。无训练、参数更新或重复权重。

主要结果：3/8/16次去噪的motion H4灰IoU依次0.531987/0.504268/0.489620；nearstatic4窗H4灰error依次0.235675/0.240562/0.252964。H4形状failure/unknown分别12/24、13/23、12/24，更多步未修复O/B磨圆、C/X笔画变化。局部指标改善仍保留，例如8次motion H4 ROI MAE略好，8/16的nearstatic H1质心漂移低于3次，但未转为H4整体保形或静止保持通过。

完整逐组/逐窗/非明显运动8窗/近静止4窗/孔洞碎片/形状failure和unknown结果见[固定口径报告](report/REPORT.md)及全部24页同窗图库 (external historical reference: `report/index.html`)。真实调用、耗时和显存见model-work.json及report/RUNTIME.md，raw和运行内评分在run/evaluation；冷态未补评分。source-identity.json保留133当前源码及快照，input-identities.json绑定2932旧保护文件，独立审计见cold-integrity-review.json。所有旧产物和原105300718字节检查点保留，SHA为3c397035857fe98b896ea5083b52f32c129f1787c1be6884d4c740a9e58e5be0。

**300更新仅说明初期学习信号。本实验只能检验固定300权重上的采样因素，不证明训练充分、收敛、能力上限或保形无望，也不能断言继续训练无用。** 六seen字母/当前36窗，natural和motion同轨迹重叠；无未见泛化或独立测试声明。若同训练点采样改善支持采样是因素；训练长度是否为因素须看固定采样的纵向曲线，二者可以共同作用。

本次实际模型耗时86.907000秒、全局1469.125000秒均仅统计，record_only_v2，没有人类时间预算或旧1200/900/300/5400停止规则。旧550次训练完成和全部先前失败费用不动；本阶段新增训练0。物理41976/76536不动，0新数据/physics/env/阻尼比较/未见八字母/服务器/安装下载。旧发布JSON三项CRLF/LF差异已知且未修改，不声称通用检查全绿。

下一阶段只准备[R2C3完整提案](R2C3_PROPOSAL.md)和项目configs/native_rgb_r2c3_proposed.json：从本谱系300完整状态恢复至1000/3000/6000观察点，横向3/16、纵向训练长度交叉，cos到3000后hold eta_min，扩展抽样顺序及旧attempt守卫须明确版本化，不冒充人类总限额。空间先按最坏字节预算，不删除交付权重、不复制数据。父验收R2C2后下达执行，本阶段不自动续训或跳转。服务器完整训练始终由用户手动执行。

阶段文件<=128MiB/shared<=3GiB含256MiB原子余量；最终storage-final.json。完成后按直接人类长期授权实际回传父对话__LOCAL_CHAT__，保存真实receipt后waiting_parent STOP，不轮询。
