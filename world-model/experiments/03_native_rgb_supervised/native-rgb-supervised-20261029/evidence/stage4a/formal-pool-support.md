# 正式TRAIN分层采样CPU准备

仅使用冻结状态/动作metadata；0Dataset构造、0渲染、0模型/权重/原生采样/优化器/物理。原始数据与静止帧保留。

正式TRAIN396轨迹/25344真实控制，24156个原有效H1候选；独立validity与原dataset catalogue逐anchor一致，joint顺序/hash与2A+2B引用一致。

四层唯一窗口：moving14692、intermediate1017、near-static-contact1176、near-static-no-contact7271。

|字母|M窗口/轨迹|I窗口/轨迹|SC窗口/轨迹|SN窗口/轨迹|
|---|---:|---:|---:|---:|
|A|888/22|49/20|51/20|354/22|
|B|813/22|39/21|93/21|397/22|
|C|759/22|51/22|73/22|459/22|
|E|811/22|74/22|82/22|375/18|
|F|808/22|74/19|88/20|372/19|
|H|816/22|43/20|49/22|434/21|
|I|798/22|58/22|48/21|438/22|
|J|873/22|51/20|47/21|371/22|
|L|798/22|67/18|95/17|382/22|
|M|745/22|57/22|72/22|468/22|
|O|760/22|65/22|66/18|451/22|
|P|795/21|36/18|104/16|407/22|
|S|859/22|59/20|73/21|351/21|
|T|879/22|58/22|48/20|357/22|
|U|906/22|55/21|47/16|334/21|
|V|793/22|48/19|64/21|437/22|
|X|783/22|66/22|30/17|463/22|
|Y|808/22|67/18|46/20|421/22|

每180槽严格每字母7M+1I+1SC+1SN，由独立seed5478 slot RNG洗牌；每层先trajectory均衡队列，再该轨迹层内window无放回轮转，耗尽才洗牌。较小轨迹层可能更常重复，属于训练曝光；不增加物理证据。

真实metadata只模拟360draws：360unique窗口、292独立轨迹。176处完整JSON状态恢复完全相同，下个16batch跨180边界，两个周期各126/18/18/18；无训练账本消费。合成测试额外流只来自假metadata。

原始初始9空间H1分母：{"center": 2684, "left": 2684, "right": 2684, "top": 2684, "bottom": 2684, "left-top": 2684, "right-top": 2684, "left-bottom": 2684, "right-bottom": 2684}。

实际边界行为：{"interior": 7210, "approaching_wall": 3228, "near_static_contact": 636, "boundary_other": 5850, "along_wall_moving": 3493, "release_from_boundary": 3199, "likely_wall_blocked_proxy": 540}。

letter×stratum×初始空间空格共1，详见JSON；全72个letter-stratum均非空，绝不向别层借样本。边界静止/墙阻挡为保存状态和接触proxy，不是wall arbiter或成功证明。

低窗口支持（窗口/独立轨迹）：X near_static_contact 30/17；P intermediate 36/18；B intermediate 39/21；H intermediate 43/20；Y near_static_contact 46/20；U near_static_contact 47/16；J near_static_contact 47/21；V intermediate 48/19；T near_static_contact 48/20；I near_static_contact 48/21；A intermediate 49/20；H near_static_contact 49/22。

未来500次有效更新预估8000draws只是假设；44完整周期+80尾部，尾部和实际尝试/完成/unique/repeats将另记。此准备不授权模型执行或下一阶段。
