# 阶段1B重接触配对报告

仅汇总保存的状态、动作、接触与恢复日志；0物理控制、0模型调用。原阶段1与1B分开，两个新X场景单列。

| 槽 | 角色 | 原始步 old/new | H1有效窗 old/new | moving old/new | near_static old/new | 初始完整状态相同 |
|---|---|---:|---:|---:|---:|---|
| stage1b-00 | paired | 64/64 | 61/61 | 18/35 | 41/23 | True |
| stage1b-01 | paired | 64/64 | 61/61 | 15/17 | 46/43 | True |
| stage1b-02 | paired | 64/64 | 61/61 | 20/31 | 38/26 | True |
| stage1b-03 | paired | 64/64 | 61/61 | 5/5 | 49/53 | True |
| stage1b-04 | paired | 64/64 | 61/61 | 14/14 | 46/45 | True |
| stage1b-05 | paired | 64/64 | 61/61 | 48/43 | 11/16 | True |
| stage1b-06 | new | missing/64 | missing/61 | missing/34 | missing/20 | None |
| stage1b-07 | new | missing/64 | missing/61 | missing/23 | missing/33 | None |

所有实际64步和有效61窗的接触×运动表、连续分布、各恢复事件的真实首次接触/连续接触/接触后运动/路径距离及失败见同名JSON。缺失或部分槽保留计划分母64/61，不能当成完成或替代采样。

新版route_completed/staging_arrived仅表示日志中的路径/准备点事件，不能等同真实接触。真实接触和运动由NPZ独立重算；旧版episode只按连续恢复/重接触phase近似切分，不冒充新版状态机。接触后额外观察最多8控制并在下一episode前截止，和状态机成功条件分列。

本报告不合并旧54轨迹稀释1B缺陷，不做显著性声明，也不授权正式采集、训练、模型推理或服务器运行。

## 恢复事件逐项证据

old为连续phase近似；new为记录的状态机episode。时间单位秒，first为实际接触的全局控制索引；post moving不含首次接触控制。

| 槽/版本/episode | 控制/秒 | route/staging | first contact | 最大连续接触 | post contact moving | 最终状态/失败 |
|---|---:|---|---:|---:|---:|---|
| stage1b-00/old/approx-1 | 22/2.2 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-00/old/approx-2 | 12/1.2 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-00/new/1 | 8/0.8 | True/True | 16 | 3 | 7 | success/None |
| stage1b-00/new/2 | 12/1.2 | True/True | 61 | 3 | 2 | success/None |
| stage1b-01/old/approx-1 | 26/2.6 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-01/old/approx-2 | 12/1.2 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-01/new/1 | 10/1.0 | True/True | 20 | 3 | 7 | success/None |
| stage1b-01/new/2 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-01/new/3 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-02/old/approx-1 | 24/2.4 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-02/old/approx-2 | 8/0.8 | None/None | 59 | 1 | 4 | unlogged_old_approximation/None |
| stage1b-02/new/1 | 9/0.9 | True/True | 18 | 3 | 7 | success/None |
| stage1b-02/new/2 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-03/old/approx-1 | 6/0.6 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-03/old/approx-2 | 11/1.1 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-03/old/approx-3 | 7/0.7 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-03/new/1 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-03/new/2 | 12/1.2 | True/True | None | 0 | 0 | failed/trajectory_budget_exhausted |
| stage1b-04/old/approx-1 | 8/0.8 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-04/old/approx-2 | 11/1.1 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-04/new/1 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-04/new/2 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-05/old/approx-1 | 7/0.7 | None/None | None | 0 | 0 | unlogged_old_approximation/None |
| stage1b-05/new/1 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-06/new/1 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-07/new/1 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
| stage1b-07/new/2 | 12/1.2 | False/False | None | 0 | 0 | failed/path_timeout |
