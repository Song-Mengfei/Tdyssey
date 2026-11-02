# 新数据默认：0.1秒半衰期（已选）

用户已决定此配置，不再开展阻尼对照、选择或验证。

新native-RGB采集和未来TRAIN/验证/测试数据生成的唯一默认入口为
`configs/native_rgb_default.json` → `configs/native_rgb_selected_physics.json` →
`tdyssey.native_rgb_selected_physics.make_selected_env`。实际协议沿用
`58563255fe5b9023387d1051a3331131da1c299c63ad30a57c2a4237c20f81f6`，D精确
`0.0009765625`。显示名独立写“0.1秒半衰期（已选）”，不更改历史candidate技术名或哈希。
R2B入口强制读取该默认配置，逐槽核对实际配置身份及所有冻结物理参数。

旧 `native_rgb_world_model collect`、旧parallel/stage1/stage2a collection和viewer直接
`make_env`路线保留历史协议，仅用于明确的历史重现，不能作为新数据默认采集路线使用。
在将来通用CLI被单独迁移之前，后续新生成器必须显式使用上述selected入口；不得绕开它
继续产生D0新数据，也不得给已有D0数据改标签。当前没有授权训练或重启旧采集命令。

历史 `NativeRGBDataset._env` / recorded-state renderer不切换：它们重放已保存姿态及原协议身份，
不是继续物理轨迹。所有旧数据/协议/权重/失败/父审产物原样保留。
选定factory保留现有seen-letter访问限制；这不解封8保留字母。未来获明确测试授权时
可以单独扩展访问路由，但物理默认仍是同一选定协议，不是再次选择阻尼。
