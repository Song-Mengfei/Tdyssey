# R3A 后台采集说明

交接验证通过、后台实际启动后，可以停止本对话并关闭 Codex，采集会由独立 Windows 后台进程继续。电脑必须保持开机且不休眠；关机、重启或休眠会中断采集。此任务未配置开机自动启动。

Root 通过已验证的 Windows 桌面 Explorer 代理启动后台，避免进程仍受 Codex 的 Windows Job 管理。零物理步测试已证明启动器退出后，独立进程的 `IsProcessInJob=false`，并持续更新两次心跳；测试期间没有实际关闭 Codex。是否已启动本次真实采集，以实际启动收据和心跳为准。

后台只继续 R3A 尚未完成的场景采集，不重采已成功的轨迹，也不运行训练、GPU 推理或 R3B。原失败记录、部分状态链和已收费控制步都保留。不要另开第二个采集进程。

所有后台文件位于：

`__LOCAL_WORKSPACE__\Tdyssey-raw-tl-portable-main\artifacts\native-rgb-supervised-20261029\r3a\background\run\`

- `launcher-receipt.json`：启动收据、进程 PID 和日志路径。
- `worker-startup-proof.json`：独立进程验证。只有 `IsProcessInJob=false` 的实际验证通过后才允许采集；验证失败会停止，不会假称后台已启动。
- `collector-start.json`：实际采集子进程的 PID、命令及独立性记录。
- `heartbeat.json`：最近更新时间、进程和子进程 PID、采集计数摘要。
- `status.json`：当前后台状态。
- `completion.json`：后台结束标志，记录返回码和最终摘要。结束不一定代表成功；还需确认返回码、完整采集标志及 912 条成功轨迹。
- `collector.stdout.log`、`collector.stderr.log`：实际采集日志。
- `worker.stdout.log`、`worker.stderr.log`：后台管理进程日志。

数据仍写入原 `r3a\trajectories\`，完整清单及核账分别在 `r3a\manifest.json`、`r3a\index.json` 和 `r3a\resource-ledger.json`。原采集器的结束记录为 `r3a\continue-unfinished-completion.json`。不要手动修改这些文件或清除失败、部分轨迹。

重新打开 Codex 后，让本对话继续。Root 会先核对后台完成情况，再完成数据接口检查、两进程吞吐实测、真实视频、最终报告，并按已有授权主动回传给“制定 Tdyssey 世界模型实施计划”对话。关闭 Codex 期间只进行本地采集，不声称报告已回传。

本次真实交接已验证：启动器退出后，后台管理进程 PID 15776 和采集进程 PID 25876 均为 `IsProcessInJob=false`，采集成功数已从交接前的587继续增长至629（此数字是验证时快照，后续看心跳）。电脑可以锁屏；不要注销当前用户。进程 ID 仅用于这次运行，后续以启动收据为准。

若后台异常退出，保留状态、日志和全部数据，由 Root 核账后决定如何继续；不要自行重跑原采集命令。R3B 仍等待真实运动和视频审查后的明确放行。
