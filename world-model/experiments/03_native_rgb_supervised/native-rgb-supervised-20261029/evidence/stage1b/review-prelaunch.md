# 阶段1B采集前独立审查

结论：**可执行已放行8槽**；不代表真实重接触或moving结果已通过。

- 6配对的15维初始状态/hash、原seed、原scene和原staging均与旧记录一致。初始coverage只使用原sequence严格之前的独立接触控制；不叠加先跑的1B结果。
- 两新X场景为center seed2026102991及right gap18px seed2026103991，已冻结并与旧54场景/seed区分，不按结果换场景。
- 旧证据逐项核验131文件hash；旧source快照全部保留，除获批sampler外15个旧输入/核心当前SHA一致。
- 采样器的scene/initialization/质心/物理阈值和原阶段表不变；通用free-route仅补零长度Point处理，其余改动集中于通用重接触状态机。无X/B/O专用动作。
- lookahead≤24px且整段自由路径核验；最终接触命令分离并始终朝内，检查真实clipped直线的外部前缀。staging到达不能自证contact，contact必须来自真实碰撞回调。
- active recovery不因固定时间表切换被截断；路径12、final接触8、episode20、至多4次、1次换路、最后63步未完成失败均有上限。
- root runner先占槽再初始化；每次控制先计账；state不变保护和float64十子步/固定2px阈值仍在。审核独立重算初始state，而不信logged true；缺槽、改角色/seed/hash、超cap和改旧证据都失败。

新success口径要求连续3接触且至少2个post-contact moving-contact控制；实际接触、持续接触、moving和blocked必须分别报。若物体在路径中主动碰到推杆，可能尚未观测staging到达，必须如实保留flag，不能补画完整阶段链。真实受阻角落可静止，不应靠改阈值或挑替代场景追求全moving。

预算：旧3456+新最多512=3968≤4000；54+8≤62槽；剩32控制不是额外授权。失败不退款/不替换，6配对与2新X分报，不混入原54稀释缺陷。

CPU检查由sampler负责人报告16、root runner3和report9通过；本审查只读和AST/哈希/已有数组核验，0真实控制、0环境初始化、0GPU/模型。初始hash仍需真实fresh env在首次控制前严格核对。

root完成一次1B后实际回传父对话并停下。阶段2–6、训练、模型推理、服务器和打包仍未放行。
