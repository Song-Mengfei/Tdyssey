"""Render a cited literature review; stdlib only, no model or data processing."""
import collections
import html
import json
import math
import pathlib

P=pathlib.Path(__file__).resolve().parent
b=json.loads((P/'BIBLIOGRAPHY.json').read_text(encoding='utf-8'))
papers=b['entries'];byid={r['id']:r for r in papers}
result_path=P/'TEST_RESULTS.json'
results=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
progress_path=P/'EXECUTION_PROGRESS.json'
progress=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else None
def link(pid):
    r=byid[pid]
    return f'<a href="{html.escape(r["primary_url"])}">[{pid}] {html.escape(r["title"])}</a>'

mu=-1.2;std=1.2
Phi=lambda z:.5*(1+math.erf(z/math.sqrt(2)))
p_low=Phi((math.log(.2)-mu)/std)
expected_w=1-.8*p_low+20*math.exp(2*mu+2*std*std)*Phi((math.log(.2)-mu-2*std*std)/std)
# The tiny native lower/upper sigma-clamp correction is omitted in this analytic
# illustration; execution records actual weights, not this approximation.
intro=f'''Tdyssey：loss文献综述与有界GPU筛选
综述日期：2026-11-02（新加坡时间）
范围：{len(papers)}篇原始论文/方法，{len(set(r['family'] for r in papers))}类；其中包括原基线、直接loss方法、相关但需改架构/训练范式的方法。不是穷尽所有论文的系统综述，也不是29种都做了训练。
原始摘要初筛、正文/公式核对、作者源码核对分层记录在每个条目，不把只读过摘要冒充通读全文。只以原论文、会议/作者页面和作者代码为依据；搜索出现的二手解读不作技术依据。

用户目标和基线
原GPC图像+动作预测下一图像，白底/灰字母/蓝推杆，保持原物体形状、刚性和正确运动。当前筛选限定在原物理OCB数据和划分，4帧RGB/4动作、96RGB、原3次去噪、原归一化与评价不变。原动作敏感性已被有限配对检验证实，不能把失败全归因于动作未输入。
当前HMF的原loss是H1扩散预条件残差MSE；最近AR4实验则额外反传四步raw RGB MSE，虽改善部分训练案例，未达到预设验证支持条件，C更差。它不等于Self Forcing的分布匹配，也不代表已经试过所有多步目标。
本轮不加新数据/物理、不换表示、额外teacher/资产下载或新网络。执行仍由原6.1Sol负责人负责，本聊天综述与独立复核。

文献给出的方向与项目判断
1. 噪声级别重新加权：Min-SNR/P2可在不增网络forward下重新分配不同去噪任务的贡献。当前残差目标已经隐含EDM式权重，不能再整套重复乘一次。噪声重新分配可能改善训练，也可能损害低噪声细节，不能引用自然图像FID收益作为项目保证。[P01-P05]
2. 图像梯度/稳健误差：GDL直接比较GT与预测的梯度幅值；不同于“在边缘mask内多算MSE”。L1/Charbonnier等改变大残差的影响，重要接触运动也可能是大残差，不能自动当离群点压低。[P06,P11,P12]
3. 频域：正交FFT的完整复数无权L2由Parseval等式等价于像素MSE，单纯搬空间不是新目标。只比幅度谱会丢位置信息。FFL通过detach的动态频率权重改变重点，同时保留相位；它关注困难频率，不是无条件只强调高频。[P07,P08]
4. SSIM/MS-SSIM：比较局部结构统计、实现便宜；小物体/孔洞在多尺度下可能被淡化。需要与位置、拓扑一起验，不以SSIM高替代物理正确。[P09-P11]
5. 学习式感知：VGG/LPIPS/DreamSim等提供感知特征，可考虑，但预训练域、额外权重/显存和小灰字母细节须实测。DISTS的部分几何容忍与精确位置要求尤其有张力。不能笼统宣布它们不可用，也不能假设自然图像主观相似就能保孔。[P13-P16]
6. IoU/轮廓/拓扑：Lovasz、Boundary、TopoLoss、clDice/cbDice更直接关联小前景与孔洞/连通。但GPC输出RGB，不是分割概率；必须先定义可微颜色→概率映射、GT边界/距离与遮挡资格。对硬阈值mask/整数孔数直接算误差通常不能提供有效梯度。分割理论不保证有限训练后的刚体朝向。[P17-P21]
7. 时序范式：Diffusion Forcing、Self Forcing、ForeDiff分别涉及独立token噪声、自生成分布匹配、条件理解与去噪解耦。不是替换一个MSE即可完成复现；可能需要改conditioning、教师或预测流。本轮不把这些改造混入loss对照。[P22-P24]
8. 物理监督与光流：PhyS/GNS/RAFT提供物理学习、状态或运动信息方向，但数据/表征/网络与当前目标不同。不能直接改为粒子状态模拟器，也不能把“连续帧尽量不变”当运动一致性——那会鼓励静止。[P25,P28,P29]
9. 多目标梯度：GradNorm/PCGrad有助处理不同loss贡献和冲突，但应先测梯度而非仅比loss数值。loss数值大不等于参数梯度主导，本轮不同时加入自适应权重或梯度手术。[P26,P27]

为何先选这三个
Min-SNR、GDL、FFL代表噪声分配、局部边界、频域三个不同机制；都不要求新标签、分割head、teacher或额外模型forward，能保留原初始化/数据/训练次序作配对对照。它们不重复旧C/D区域加权或旧AR4分支。选择是受预算限制的首轮筛选，不是认定其余方法无效。

冻结的实际公式与迁移注意
共同：原残差r=(y-c_skip*z)/c_out，base=MSE(F,r)。D=c_skip*z+c_out*F.float()，用同一次网络输出得到未clamp/量化的可微去噪重建。q=D/.5、v=y/.5仅为新增loss内部单位；模型输入仍[0,1]，动作与GT预处理不变。
BASE：原base，沿用原mean约简顺序。
SNR：mean_sample(w_i * mean_CHW((F_i-r_i)^2))，rho=.25/(sigma_draw²+.01)，w=min(rho,5)/rho。不再加EDM权重、不对batch权重归一化、不改sigma采样。这里使用GPC原sigma_eff=sqrt(draw²+.1²)，rho是项目预条件下名义SNR，非经验测得方差。借鉴Min-SNR的EDM附录，是GPC适配而非原论文逐字复现。
GDL：base + .5*(mean(abs(abs(dy(q))-abs(dy(v)))) + mean(abs(abs(dx(q))-abs(dx(v)))))，alpha=1、辅助系数1、无padding。梯度幅值可对某些颜色反转/偏移不敏感，所以保留base，不能单独当形状证书。
FFL：base + mean(M*|FFT(q)-FFT(v)|²)，正交完整复数FFT；M=|Delta|/max_HW|Delta|，每sample/channel归一化，零误差矩阵设0并完整detach。alpha=1，其余作者默认patch1、平均谱/log/batch-matrix均false。96不是2的幂，FFT明确FP32并关闭该段autocast；原模型AMP不变。
新增系数与单位均在看结果前固定，不搜索超参。GDL/FFL施加在可微单次重建，不宣称直接训练了纯噪声起点的AR16。

一个项目特有的参数化差异
由原代码可推得：base等价于重建误差乘1/c_out²=4+1/(draw²+.01)。因此原loss并非所有噪声级别等权。
本轮SNR仅在draw<.2时降权；权重最低约.20008，其他范围为1。按原log-normal抽样（忽略极小clamp修正）约{100*p_low:.2f}%的抽样处于该范围，平均权重约{expected_w:.4f}。实际抽样权重另由执行记录。若错误使用draw而非sigma_eff，低噪声权重会相差很大。这是核对GPC适配的必要细节，不证明此权重必然有益。

测试与验收
BASE/SNR/GDL/FFL各128次尝试，eff16，同完整E2 model/Adam/scaler/RNG独立恢复、原HMF首128批和noise配对。每臂2048原H1曝光，合计8192，0额外GT曝光。预检最多128原模型图像级forward，正式总上限23296；不另warmup/benchmark/选seed。原loss新跑并与保存R1 A128交叉核验。
四臂终点各评train12/seen-val24/old3的RH及raw AR16：共4992预测/14976图像级denoiser。共同E2引用已保存且与原HMF逐项一致的基线，不重推。原绝对门槛、unknown资格不变；方向性筛选沿用R1预设，并单列AR8/16、逐例得失、字母与split。train12不能代表全48拟合。
工程可用（梯度/参数更新/显存）与科学有益（运动、形状/拓扑、验证保留）分开。128步/单seed/多候选仅筛选；反复使用seen-val也不能称一次全新独立泛化测试。无候选通过也完整报告，不自动延长或调权重。
CPU各库/Torch/OpenCV1、单进程BelowNormal，CUDA模型及可等价GPU评分；不操作其他项目。执行新root上限900MB，共享10GB含原子预留/增长guard。无模型的图文报告问题在本轮已有结果内修复，不为报告重推模型。

执行与结果状态
'''
if results:
    intro+=results.get('summary','结果已回传；见TEST_RESULTS.json及独立核对。')+'\n'
elif progress:
    intro+=progress['summary']+'\n'
else:
    intro+='候选/预算已实际交给原负责人；本综述文件生成时尚未收到本轮GPU启动/终态回报，不把交接当训练完成。完成后本聊天独立核对源评分，再更新本节。\n'

lines=[('实测结论\n'+results.get('executive_summary','')+'\n\n') if results else '',intro,'\n逐篇文献与适用性（状态“本轮GPU筛选”只表示纳入，完成与否以上节/执行结果为准）\n']
for r in papers:
    lines.append(f"{r['id']} | {r['family']} | {r['year_venue']}\n{r['title']}\n{r['authors']}\n核对程度：{r['inspection']}\n方法：{r['method']}\n项目判断：{r['project_assessment']}\n状态：{r['status']}\n原始来源：{r['primary_url']}\n")
lines.append('\n作者代码：\n'+'\n'.join(b['selected_code_sources'])+'\n')
(P/'REVIEW.txt').write_text('\n'.join(lines),encoding='utf-8')

page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Tdyssey loss文献综述与测试</title><style>
body{max-width:1160px;margin:28px auto;padding:0 20px;font:16px/1.65 system-ui;color:#20252b;background:#fcfcfd}pre{white-space:pre-wrap;font:inherit}a{color:#176ba5}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d9dee3;padding:9px;vertical-align:top}th{background:#eef2f6}input{padding:10px;width:95%;font:inherit;margin:12px 0}details{border:1px solid #dde2e8;padding:12px;margin:10px 0;border-radius:7px}summary{cursor:pointer;font-weight:600}.tag{color:#566471;font-size:13px}img{max-width:100%}</style>
<h1>Tdyssey：loss文献综述与GPU筛选</h1><p>29篇原始论文/方法，逐项标明阅读深度与项目迁移边界。完整版本：<a href="REVIEW.txt">文本</a> · <a href="BIBLIOGRAPHY.json">文献账本</a> · <a href="TEST_SPEC.txt">预先冻结测试方案</a> · <a href="EXECUTION_RECEIPT.json">真实交接回执</a></p>'''
if results:
    page+='<h2>实测结论</h2><p>'+html.escape(results.get('executive_summary',''))+'</p><p><a href="SCREEN_RESULTS.txt">直接查看完整测试表</a></p>'
page+='<pre>'+html.escape(intro)+'</pre>'
page+='<img src="GPC-MinSNR-weight-profile.png" alt="本项目预条件下的原权重和Min-SNR权重示意；非模型结果">'
page+='<p>关键方法原始来源：'+link('P02')+'；'+link('P06')+'；'+link('P07')+'</p>'
if results:
    for name in results.get('figures',[]):
        page+='<img src="'+html.escape(name)+'">'
    for name,path in results.get('links',{}).items():
        href=pathlib.Path(path).as_uri() if pathlib.Path(path).is_absolute() else path
        page+='<p><a href="'+html.escape(href)+'">'+html.escape(name)+'</a></p>'
page+='<h2>筛选矩阵</h2><table><tr><th>类别</th><th>对当前项目的作用</th><th>本轮处理</th></tr>'
matrix=[('扩散噪声权重','改变不同sigma梯度贡献；注意原预条件已有权重','测试Min-SNR；不混改noise分布'),('梯度/稳健误差','直接惩罚边界变化，或改变大残差影响','测试GDL；L1/Charbonnier未训练'),('频域','自适应频率误差；相位关系到位置','测试FFL；不把普通FFT-MSE当新目标'),('结构统计','局部/多尺度相似；小孔和白底需检验','SSIM/MS-SSIM未训练'),('感知特征','自然图像感知与严格几何可能不一致','未训练；不宣称不适用'),('分割/拓扑','更直接面对轮廓、连接与孔洞','未训练；先验证RGB概率和遮挡'),('时序/物理/光流','可能改变训练范式、表征或引入teacher','本轮不整体移植'),('多目标平衡','处理梯度幅值或冲突','未训练；先实测再单独控制')]
for row in matrix:page+='<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in row)+'</tr>'
page+='</table><h2>逐篇证据</h2><input id="filter" placeholder="按方法、类别或状态筛选，例如：拓扑、GPU、未训练">'
for r in papers:
    text=' '.join(str(v) for v in r.values())
    page+='<details class="paper" data-text="'+html.escape(text.lower(),quote=True)+'"><summary>'+html.escape(r['id']+' · '+r['title'])+'</summary>'
    page+='<p class="tag">'+html.escape(r['year_venue']+' · '+r['family']+' · '+r['status'])+'</p>'
    for k,label in [('authors','作者'),('inspection','核对程度'),('method','方法'),('project_assessment','项目判断')]:page+='<p><b>'+label+'：</b>'+html.escape(r[k])+'</p>'
    page+='<a href="'+html.escape(r['primary_url'])+'">原始论文/作者来源</a></details>'
page+='''<script>document.querySelector('#filter').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.paper').forEach(d=>{d.hidden=!d.dataset.text.includes(q)});});</script></html>'''
(P/'REVIEW.html').write_text(page,encoding='utf-8')
(P/'REVIEW_STATE.json').write_text(json.dumps(dict(literature_screened=len(papers),families=len(set(r['family'] for r in papers)),selected=['SNR','GDL','FFL'],test_results_received=results is not None,models_called_by_review=0),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(papers=len(papers),review=str(P/'REVIEW.html'),test_results_received=results is not None,nominal_sigma_low_probability=p_low,approx_expected_SNR_weight=expected_w),ensure_ascii=False))
