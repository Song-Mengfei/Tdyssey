# 阶段1B：重接触小修正的有界对照

**结论：部分修复，仍有回退，不能据此自动进入正式采集。**
三个旧X样例的首次恢复现在都有真实接触和持续推移；后续恢复仍多次超时，
T对照减少5个moving窗口。全部8槽已完成，余32控制不自动使用，阶段2–6仍未放行。

## 根因与最小改动

旧X中央/左下/右上首次恢复分别持续22/26/24个控制，没有真实接触或物体运动。
这些区间的绝对目标路径长度都是0，即固定接近点；到点距离缓慢减小。
原向内延长条件未触发，不能据此声称目标反复切换。

通用策略现在明确区分自由空间路径、到达staging、实体接触、连续接触和后续物体运动。
活跃恢复不被固定阶段表截断；自由空间有界前视目标整段检查，最终朝内接触单独检查。
4px到达容差、最多24px安全前视、无进展/路径/总episode时限、一次合法替代及
最多4次episode均在实际执行前固定。失败弧段不无限重试，预算终点明确记录结束。
“到达staging”不等于重接触；策略success额外要求3次连续真实接触与2次后续contact-moving。

原始GPC、物理控制器参数、对象几何/质量、力/速度、时间步、96px渲染及2px阈值均未改。
没有按X/B/O写专用动作。详见[SOURCE_CHANGE.md](SOURCE_CHANGE.md)与精确差异 (external historical reference: `minimal-change.diff`)。

## 固定8槽与原始分母

六配对从全新环境恢复旧独立初始场景/完整15维状态及原始接触选点，全部初始bytehash匹配。
初始覆盖计数按旧序列之前的轨迹重建，不混入已执行的1B槽。配对是付费物理重放，
不是6个新独立场景。两新X的seed/姿态/选点在任何实际执行前冻结，未按结果替换。

每槽64个真实控制、61个原始有效H1候选；0无效历史排除。
下表moving与near_static的分母均为61，接触控制的分母为64。

| 槽/场景 | moving旧→新 | near_static旧→新 | 真实接触控制旧→新 | 新恢复：route / 真接触 / success |
| --- | --- | --- | --- | --- |
| 00 X center | 18→35 | 41→23 | 23→43 | 2 / 2 / 2 |
| 01 X left-bottom | 15→17 | 46→43 | 19→22 | 1 / 1 / 1 |
| 02 X right-top | 20→31 | 38→26 | 26→39 | 1 / 1 / 1 |
| 03 B left-top | 5→5 | 49→53 | 30→33 | 1 / 0 / 0 |
| 04 O left-top | 14→14 | 46→45 | 39→34 | 0 / 0 / 0 |
| 05 T center | 48→43 | 11→16 | 53→48 | 0 / 0 / 0 |
| 06 新X center，seed2026102991 | 无旧配对 / 34 | 无旧配对 / 20 | 无旧配对 / 47 | 0 / 0 / 0 |
| 07 新X right，seed2026103991，gap18px | 无旧配对 / 23 | 无旧配对 / 33 | 无旧配对 / 32 | 0 / 0 / 0 |

| 汇总口径 | moving | intermediate | near_static | 接触控制 |
| --- | --- | --- | --- | --- |
| 六配对旧：384全部真实转移 | 123 | 24 | 237 | 190 |
| 六配对新：384全部真实转移 | 148 | 24 | 212 | 219 |
| 六配对旧：366候选 | 120 | 15 | 231 | 178 |
| 六配对新：366候选 | 145 | 15 | 206 | 207 |
| 两新X：128全部真实转移 | 58 | 15 | 55 | 79 |
| 两新X：122候选 | 57 | 12 | 53 | 75 |
| 1B全部512真实转移 | 206 | 39 | 267 | 298 |
| 1B全部488候选 | 202 | 27 | 259 | 282 |

没有删静止、无接触或角落受阻片段。原阶段1的3456/3294分母与本次1B分开保存，
不拿新数据稀释旧缺陷，也不把刻意挑选的6配对或8个槽当18字母显著性证据。

## 恢复是否真的完成

新15个恢复episode：路径/staging完成5次，真实接触4次，连续接触/后续contact-moving成功4次；
其余10次path_timeout、1次trajectory_budget_exhausted。B有一次到staging仍未碰到物体，
正说明不能把路径完成当接触成功。

三个X首次恢复实际在step16/20/18碰到物体，距episode开始0.6/0.8/0.7秒，
都出现3连续接触和后续物体运动。X中央第二次也在step61重接触并成功。
相比旧首次长区间的0接触，这部分修正有真实物理证据。

但超时episode并非全部没有进展：日志最大no-progress为0，到超时时还距目标6.246–22.832px。
12步路径上限仍可能截断有进展的接近。T在step59–63，旧版push_again有5次contact-moving，
新版仍处于无接触恢复并超时，解释了moving减少5。两个新X的3次恢复全部超时，
不能宣称重接触已普遍修复。B/O真实角落静止有价值，不为了配比强行制造运动或判成坏数据。

旧episode仅为连续phase近似，新episode为真实状态机日志；接触与运动由NPZ独立重算。
首次接触、连续接触和非连续事件核对0差异。完整路径距离、首次接触、后续最多8控制的
运动与计时见[逐事件报告](recontact-report/recontact-report.md)和JSON (external historical reference: `recontact-report/recontact-report.json`)。

## 真实连续帧

X中央旧/新同帧号连续GIF (external historical reference: `examples/stage1b-00-old-new.gif`)覆盖frame10–32，含新真实接触和推移；
T对照旧/新同帧号连续GIF (external historical reference: `examples/stage1b-05-old-new.gif`)覆盖frame48–64，保留回退。
均为保存状态离线渲染，0额外物理步或模型调用。早期五帧PNG只覆盖接触前，
不能单独作为成功证明，见examples/render-run.json与recontact-report中的图示元数据。

## 审计、成本和停点

- 28项聚焦CPU检查通过，执行前固定8槽清单和新源码身份。
- 512/512控制、5120子步全部保存；0无效转移、0部分步、0失败槽，无替代槽。
- 最大物体-墙/物体-推杆/推杆-墙穿透0.470797/0.983181/0世界px，阈值仍2px。
- 八槽初始完整状态、配对身份、seed、冻结清单均独立核对；旧131项证据未改。
- 本阶段采集wall time约12.390秒。新旧监督阶段合计3968/4000控制、62/62槽；剩32不自动用。
- 连更早native历史共38528控制；训练仍1899尝试/5352.516秒，新增训练/模型/GPU/未见预测/服务器/打包均0。
- 新旧native持久产物仍低于3GiB。资源账本、物理审计、独立复核和结果JSON记录精确值。

实际执行命令：

```powershell
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_supervised_stage1b.py prepare --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_supervised_stage1b.py collect --execute --physics-root ../Tdyssey/third_party/diffusion_policy
../Tdyssey/.venv/Scripts/python.exe -X utf8 -B scripts/native_rgb_supervised_stage1b.py audit --physics-root ../Tdyssey/third_party/diffusion_policy
```

报告命令和日志见report-stdout.txt；冻结清单frozen-slots.json，源码source-identity.json，
旧证据baseline-preservation.json，成本resource-ledger.json，物理结果physics-audit.json，
详细局限DIAGNOSIS.md。本轮完成后必须主动回传父对话，再停下等审查。
**阶段2正式采集仍未放行，不因X部分改善而自动进入下一步。**
