# R3B 第一阶段诊断、原始物理与录像验收

默认试玩已回到原始物理，六类截图中的同类异常在跨物理预览中已复现。部分圆化、孔洞变化第一步就发生；四步递推还会积累形变、碎块和推杆失踪。录像与完整原始记录通过两轮真实浏览器验收。本阶段训练更新 **0**，停止等待用户审批。

## 两类结果分开读

- [原首轮训练报告](../../../03_native_rgb_supervised/native-rgb-supervised-20261029/evidence/r3b/attempts/recovery-20261030/REPORT.md)：旧动态推杆/0.1秒半衰期；1499批/23984不同目标，6000→7499。
- 本次逐帧报告、40组图及同步慢放 (external historical reference: `runs/original-20261030-v1/report/index.html`)：冻结7499，原始运动学推杆/damping0，属于跨物理预览，不能混入旧协议训练改善。
- [人工观察](runs/original-20261030-v1/review/VISUAL-REVIEW.md)、raw组索引 (external historical reference: `runs/original-20261030-v1/group-index.json`)、最终独立审计 (external historical reference: `final-readonly-audit.json`)。

旧截图缺完整历史、动作和随机状态，只能复现同类组成现象，可能跨片段/步，不能称同帧还原或旧协议精确重放。最终六图都有同类证据；pipeline回执保留初版快照，最终以report/reference-correspondence.json为准，不改写历史。

## 已训练首轮：O、C改善

同一自然验证H4窗口，每字母N24；图均GT/copy-last/6000/epoch1同坐标，不配准、不修边。

| 指标 | O：6000→7499 | C：6000→7499 |
| --- | --- | --- |
| 物体区域误差 | .09494→.05746，降低39.5% | .09121→.05189，降低43.1% |
| 轮廓误差 | .9408→.4572图像像素，降低51.4% | .9137→.4675，降低48.8% |
| 灰物体IoU | .6147→.7618 | .5912→.7364 |
| 形状未检出失败/失败/未知 | 9/2/13→8/2/14 | 5/5/14→7/3/14 |
| 共同可测子集质心误差 | .6020→.5318px，N13/24 | .8807→.2197px，N10/24 |
| 同一子集角误差 | 5.3846→2.4615° | 5→.5° |

轮廓误差减半不等于棱角恢复率或刚性通过。O失败仍2/24、C失败3/24，各14/24未知。明显运动组N66区域误差只降低18.8%，形状失败5→8，53/66未知。近静止H4伪漂移1.0128→.1541px，但copy-last为0；自然144窗口仅75个优于copy-last。来源：[配对表](../../../03_native_rgb_supervised/native-rgb-supervised-20261029/evidence/r3b/attempts/recovery-20261030/report/paired-metrics.csv)、[运动分组复核](../../../03_native_rgb_supervised/native-rgb-supervised-20261029/evidence/r3b/attempts/recovery-20261030/review/METRICS-REVIEW.md)。

## 新原始物理：首步与四步

A每控制重新取真实四帧历史；B从相同起点连续四步，仅反馈raw预测。同动作/seed，首帧独立调用且数组完全一致；GT未来只作标签。定向诊断不能当总体失败率。

- C clip011 (external historical reference: `runs/original-20261030-v1/report/assets/clip011-frames.png`)：首步两支都圆化；A第三/四步恢复直边，B仍圆化。
- O clip028 (external historical reference: `runs/original-20261030-v1/report/assets/clip028-frames.png`)：首步方环变圆；B第三/四步蓝推杆失踪。
- B clip001 (external historical reference: `runs/original-20261030-v1/report/assets/clip001-frames.png`)：首步双孔近单孔，B后续开口成C；A校正也有失效。
- X clip036 (external historical reference: `runs/original-20261030-v1/report/assets/clip036-frames.png`)：B第三步出现并保留独立灰碎块；另有A/B都出碎块的反例，未删去。

固定8片段/字母、3去噪步，末步可见质心误差C A=.4777px/B=4.4220px，O A=.5839/B=9.2273px；推杆消失C 0/2、O 0/1。[完整分母](runs/original-20261030-v1/review/OC-fixed-diagnostic-summary.json)。这些是校正与递推差异，不是新训练改善。

固定4代表片段、3/8/16去噪步的H4-B区域误差：C .2099/.1821/.1752，O .2979/.2940/.2885，B .3029/.3002/.3011，X .2096/.2201/.2226。更多去噪步仅部分改善，也可能更差。逐调用耗时字段修复后的完整表见逐帧报告；旧报告0计时版本保留。

## 原始物理与撤杆

实际用固定原source的PushTEnv.step、KINEMATIC推杆/DYNAMIC物体、damping0、原PD/100Hz物理/10Hz控制/solver10、原材料/墙/质量/惯量。没有半衰期、力上限、限速、脱离清速度或回写位姿。

六字母参数/几何匹配独立参考；T/C各8控制/80子步逐前后state15差0。O中心/近墙各24控制，撤杆后继续推进10控制；首次整控无推杆接触为9/10，最终速度/角速度0。分离不等于严格零位移：中心首脱离控有.130603世界像素积分残余；近墙后续最多每控.0369944世界像素接触位置修正。近墙墙接触168子步、最大墙穿透.5031px；中心运动学推时穿透2.291px，未用旧有限力保护隐藏。

曲线 (external historical reference: `physics-alignment/validated-v2/separation-curves.png`) · 中心视频 (external historical reference: `physics-alignment/validated-v2/O-center-browser.mp4`) · 近墙视频 (external historical reference: `physics-alignment/validated-v2/O-near-wall-browser.mp4`) · 完整验证 (external historical reference: `physics-alignment/validated-v2/summary.json`)。首次脚本重心复制顺序错误在0控制时被捕获，原失败保留；只修脚本后一次完整验证通过。

## 实际录像及接口

试玩 http://127.0.0.1:8767/ ，验收PID24644。点“开始录制”，操作H1/H4，再点“停止并保存录像”，可打开/下载WebM及同ID完整raw。仅录真实可见左右画布，无桌面/麦克风、插帧或未显示H4中间帧。松鼠标停止新推动批次，不结束录制；模式/重置/session边界有标记。

两次真实Chrome流程各H1=1控制/H4=4控制；开始/停止/计时无额外推进或模型调用。第二次录制中B→C重置。两份raw均closed/raw_complete，无缺batch；实际下载、打开、播放、本地CPU解码及raw SHA全通过：

| 录像ID | WebM/客户端秒 | 文件 |
| --- | --- | --- |
| recording-046f19d4a24d47a4a77ad4ebef3c4f48 | 3.779/3.900 | WebM (external historical reference: `recordings/recording-046f19d4a24d47a4a77ad4ebef3c4f48.webm`) / raw (external historical reference: `recordings/recording-046f19d4a24d47a4a77ad4ebef3c4f48/raw/index.html`) |
| recording-07c96e4799284ecaa39af409b80a2f73 | 3.510/3.624 | WebM (external historical reference: `recordings/recording-07c96e4799284ecaa39af409b80a2f73.webm`) / raw (external historical reference: `recordings/recording-07c96e4799284ecaa39af409b80a2f73/raw/index.html`) |

工程问题与失败证据保留：终帧未提交便停止造成约.5秒尾部丢失，修复后尾差.121/.114秒；事件记录抢主场景锁造成409，已把纯事件移至轻锁，并验证慢状态响应不持有场景锁。均不改模型/物理/动作。浏览器文件来源保护保持启用，下载后本地CPU解码验画面；未预测时右画布空白如实保留。

实际浏览器回执 (external historical reference: `browser-validation/browser-check.json`) · 解码 (external historical reference: `browser-validation/video-decode-check.json`) · 并发验证 (external historical reference: `event-lock-concurrent-check.json`) · 源码及9项CPU检查 (external historical reference: `ui-capture-implementation-check.json`)。42个报告视频，图片无损坏；实际播放逐帧/撤杆视频成功。旧API仅能导出可见历史，不能称完整续跑；新恢复包含完整公开body/controller/history，但不序列化内部接触缓存，不能承诺接触中未来逐位一致。

## 成本、身份与停止

主诊断152真实控制、40组/320调用/1536去噪前向；内层采样44.836587秒，外层含复制/扫描合计44.880482秒；worker70.065055秒，pipeline含初版报告94.058212秒。CPU物理验证80控制/800子步/5.900081秒；事件并发验收9控制/90子步另列。浏览器验收旧进程20调用/3.160642秒、新进程10调用；恢复的历史计数不重复相加，活服务未结束不编造最终总秒数。旧首轮训练及33.047秒失败费用保留在原报告，不混入本阶段。

checkpoint文件与模型参数SHA前后一致，未构造optimizer，更新0：model-proof (external historical reference: `runs/original-20261030-v1/model-proof.json`)。89冻结输入、40 raw、320记录、6参考图、8UI源码审计通过。共享存储快照1,733,135,152B、扣256MiB保留后余1,219,654,864B，快照早于末尾报告/回执等少量元数据，不冒充包含自身的最终字节数。

无队友数据、heldout、补采、远端工作、安装或新模型副本。服务停在O/0步、无活动录像，可继续试玩；不自动推动或训练。实际回传后保存真实回执并STOP waiting_user_approval，下一步须用户明确批准。

附加仓库源检查scripts/check_repo.py未通过：既有发布证据custom-L-summary.json、pretrained-T-summary.json、palette-T-summary.json的size/hash与发布清单不符。这些文件不属于本次修改，未为使检查变绿而重写旧发布证据。此项与本阶段专门的物理/模型身份/录像验收分列，不能称全仓库检查通过。
