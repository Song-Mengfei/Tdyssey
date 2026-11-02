# 上传与手动复现

当前本地试验未通过物体动力学/形状门槛。此包用于审查和复现，**没有启动服务器训练**；`full`会拒绝附带的失败本地报告。不要编辑报告绕过门槛。

## 已验证的交付

- 文件：`native-gpc-rgb-from-scratch.zip`
- 大小：34,779,384 bytes（33.17 MiB）
- SHA256：`4a0bde0efb21120abfa982f46042490bb820e1d3dd3bfabad6131e0d03043a6e`
- 清单：`upload-manifest.json` / `upload-manifest.md`。201个条目均已逐项解压读取校验大小及SHA256，嵌入清单与外部一致。
- 唯一模型权重为原始GPC发布初始化（35,043,916 bytes），SHA256 `4084ed27c2c8b372aa10858474517ac7fe21b531de4a811059c35c05f8191892`。
- 不含本地训练权重、NPZ数据、虚拟环境、Git历史或大型缓存。本地best1500/last1668仍保存在实验目录的training/，没有删除。

## 1. 上传并检查

只需手动上传ZIP；外部清单和本说明可一并上传便于审阅。进入服务器ZIP所在目录：

```bash
sha256sum native-gpc-rgb-from-scratch.zip
mkdir native-gpc-rgb-review-20261029
unzip native-gpc-rgb-from-scratch.zip -d native-gpc-rgb-review-20261029
cd native-gpc-rgb-review-20261029
```

先确认hash等于上面的值。查看`reports/local/REPORT.md`、`reports/local/report-diagnostics/DIAGNOSIS.md`、`docs/NATIVE_RGB_WORLD_MODEL.md`。文件清单记录每个源文件的本机绝对路径、目标相对路径、大小、SHA256、版本和是否必需。

## 2. 可选的独立服务器环境检查

要求Linux x86_64、Python3.12，目标为RTX PRO6000约96GB。以下命令由你手动运行；当前机器没有替你执行。会创建独立`.venv-native-rgb-server`，不改其他项目环境，先验证上传清单再安装：

```bash
bash run_native_rgb_server.sh prepare --setup
bash run_native_rgb_server.sh preflight
```

Torch2.10/torchvision0.25 CUDA13配对依赖由脚本安装，记录pip检查和完整实际freeze。真实CUDA内核/显存/驱动必须在服务器通过；本地CUDA12.4成功不代表该服务器配置已验证。

## 3. 当前不得扩大训练

`bash run_native_rgb_server.sh full`会读附带本地资格报告并拒绝继续。新的采样或训练方案需要先约定预算/验收，再产生新的有效报告；不提供绕过开关。

已准备的未来完整配置为593,920请求控制步、3个种子5478/5479/5480，每种子50,000尝试或24h。真实全流程批量8/16/32/64探测后，只有两进程端到端吞吐提高超过5%且显存安全才并发2种子；否则1个。实际服务器数据/吞吐/训练均未执行，不能把配置当成果。

最终seen_test/unseen_test需先封存checkpoint、数据、源码和校准；DGKNQRWZ在本地从未查看模型预测。真实点击查看器也未运行，默认命令只输出计划。不要在已用尽的本地34,560控制额度内追加点击或采样。
