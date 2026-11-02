# R2B 接触跟随采样报告

0.1秒半衰期（已选）

计划54槽/1296控制，数据错误0；所有未执行、失败及缺测槽保留。

|Population|Planned|Observed|Valid|≥2px|≥5°|
|---|---:|---:|---:|---:|---:|
|all_controls|1296|1272|1272|174|186|
|h1_windows|1134|1113|1113|174|186|
|h4_windows|972|954|954|450|321|
|push_h4_windows|378|371|371|288|212|
|push_full_history_h4_windows|216|212|212|146|101|

push_h4_windows为未来4控全在push的anchor4..10（7/槽，计划378），历史可含approach；push_full_history_h4_windows为3过去+4未来控全在push的anchor7..10（4/槽，计划216）。实际接触/冲量/阶段另列，计划推送不是已接触推送。

|Letter|Valid H1|≥2px|+≥5°|−≤−5°|Actual lever signs|
|---|---:|---:|---:|---:|---|
|T|189|27|30|15|negative,positive|
|L|189|27|11|46|negative,positive|
|O|189|31|14|11|negative,positive|
|C|189|25|13|5|negative,positive|
|X|168|33|10|4|negative,positive|
|B|189|31|17|10|negative,positive|

松开观测以state14实测速度为10%/5%参考，另列真实脱离时刻。所有自由段保留；碰撞、重接触、观察尾和缺测标删失，不套用半衰期公式填时间。

逐family/真实弧段/有符号力臂/实际wallgap/边界过程/旋转方向及联合目标详见JSON与CSV。未运行环境、渲染、模型、GPU或训练。
