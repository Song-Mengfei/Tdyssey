# R2C1独立CPU合同审查

当前代码可交根代理最终冻结和唯一实际运行；审查员没有执行模型/权重反序列化/Torch/GPU/优化器/环境/物理。

7项必要纯mock合同通过：归一化动作原样直传、finite OOR保持、attempt先持久预扣后draw、300/1200与更早tail、旧250/868.484链接不清零、阶段/共享保守空间上界、完整数据队列/RNG拒绝篡改相关边界。原H1 residual函数复用，micro4×4按0.25累积；AdamW/AMP128/unscale/clip1/cosineTmax3000数学与配置一致。

静态核对单一时钟在construct前启动，原始/训练后同36窗和四步raw保存、copylast/端点/中间步CPU评分、保存、一次6state重载验证与失败清理均在finish前。正常只同名last保留t0和最终完整边界；异常半micro不能冒称完整payload，原证据/费用保留，finite OOR不触发重新训练。

共享盘空间每调用检查预留整阶段256MiB+atomic256MiB+非阶段1MiB增长后的上界，并即时查阶段实际bytes/临时文件；60秒或保存前强制全扫。不是每调用都遍历旧文件；旧非阶段roots必须冻结。根最终刷新stage源清单时须把新config/tests算入阶段256MiB，完整共享roots已覆盖它们。

2341个受保护输入和原始35MB权重仅SHA校验均一致，旧globalledger SHA未变；实际运行开始后该ledger获授权增加，但priorcopy和历史结果仍不可改。根需最终source/config/snapshot一致、CUDA占用前检后唯一执行，不再要求用户重复启动许可。

这属于静态和mock审查，不证明实际GPU性能/显存/梯度/指标已通过。同步操作无法硬抢占；实际不足1200秒可用空间或工程失败必须保缺测/收费并停止，不能预算外补评估或重跑。
