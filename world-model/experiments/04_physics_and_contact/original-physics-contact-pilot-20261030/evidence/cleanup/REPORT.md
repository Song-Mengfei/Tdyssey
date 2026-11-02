# 旧资产清理结果

负责人已实际删除7个已退役checkpoint，精确逻辑回收 **596,831,938字节（569.18MiB）**，逐文件绝对路径、大小、SHA和删除回执见清单 (external historical reference: `checkpoint-deletion-receipt.json`)。原best/last并非相同文件，而是同一已退役旧实验的不同历史选择；相关原报告、raw输出、指标和费用记录保留。旧manifest是历史清单，不能再声称其所有权重仍可加载；旧generic脚本默认best路径已失效，不会静默换成别的模型。

保留核验通过：retained-assets-verification.json (external historical reference: `retained-assets-verification.json`)。6000与当前7499完整checkpoint、原始GPC初始化的SHA均匹配。用户B录像的Downloads副本和项目副本均匹配a9477b...7948；3,369个原始NPZ全部重新核对hash无变化，用户录制视频仍可读。唯一官方策略文件完整读取计算SHA、未加载模型；本次没有早先预期SHA可比较，因此不冒称该项“匹配”。组员原始Tdyssey/tdyssey_pushing_data目录存在，处于全部删除目标之外，未读取其payload。启动器和当前会话/clip/raw保护记录也保留。

**本轮没有删除旧训练数据。** 774个R3A TRAIN NPZ共58,388,513字节仍保留，另有1,314,026字节JSON sidecar；全量旧loader与物理audit依赖其原manifest，负责人选择先保留，避免为了约59MB破坏旧可复现输入。这不是最小子集提取，未复制全量备份。144个seen-validation family及其sidecar共11,014,405字节保留；未解码封存测试payload。

删除动作紧邻的main artifacts大小1,659,438,029→1,062,612,120字节；磁盘free160,844,292,096→161,439,395,840字节。目录差值、free差值可能包含并发诊断写入，实际删除逻辑字节以7个文件的精确和为准。

2026-10-30 22:30:06新加坡时间只读快照：main artifacts1,111,334,967字节，共14,296文件；共享去重1,219,655,778字节，扣256MiB原子写入余量后仍余1,733,134,238字节，磁盘free161,385,033,728字节。快照早于本回执自身及后续试采，不冒充最终静态占用。负责人最终交付前可只读重跑verify_retained_assets.py更新数字。

本盘点/核验模型加载0、GPU/模型调用0、物理控制0、优化器更新0。初次文件扫描72.495秒，保留资产hash核验79.334秒，属于CPU/磁盘I/O而非模型费用。未删除源码、Git树、环境、第三方源码、其他作者数据或用途不明的唯一资产。详见[PLAN.md](PLAN.md)、inventory.json (external historical reference: `inventory.json`)。
