"""Small independent figures/report from saved arrays; no simulation/model calls."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[k]='1'
import hashlib
import html
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

P=pathlib.Path(__file__).resolve().parent
RUN=P.parent.parent/'loss-rollout-ab-20261102-r1'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
audit=read(P/'AUDIT.json')
rows=read(P/'INDEPENDENT_ALL16.json')
own={(r['label'],r['role'],r['letter'],r['mode'],r['horizon']):r for r in rows}
labels=['E2','A128','A256','B128','B256']
data={label:read(RUN/f'evaluation/{label}/metrics.json') for label in ['E2','A256','B256']}
fig,axs=plt.subplots(2,3,figsize=(12.5,7.3),sharex=True)
for j,letter in enumerate('OCB'):
    for label,color,style in [('E2','#777777','--'),('A256','#2469a9','-'),('B256','#df7c12','-')]:
        selected=[own[(label,'seen_val',letter,'autoregressive',h)] for h in range(1,17)]
        axs[0,j].plot(range(1,17),[r['IoU'] for r in selected],label=label,color=color,linestyle=style)
        axs[1,j].plot(range(1,17),[r['position'] for r in selected],color=color,linestyle=style)
    axs[0,j].set_title(letter)
    axs[0,j].set_ylim(0,1.02)
    axs[1,j].set_ylim(0,6)
    axs[1,j].set_xlabel('Raw self-feedback target horizon (10 Hz)')
    for ax in axs[:,j]:ax.grid(alpha=.2);ax.set_xticks([1,4,8,12,16])
axs[0,0].set_ylabel('Gray object IoU (higher is better)')
axs[1,0].set_ylabel('Visible centroid error (image px)')
axs[0,0].legend(frameon=False)
fig.suptitle('Seen validation: 8 fixed families per letter; original loss A vs AR4 auxiliary B',fontsize=12)
fig.tight_layout(rect=[0,0,1,.95])
fig.savefig(P/'validation-all16-A-vs-AR4.png',dpi=140)
plt.close(fig)

# Three missing old-case panels are independent new review artifacts. The failed
# execution report, frozen source, raw data and completion state remain untouched.
spec=read(RUN/'evaluation-spec.json')
old=[r for b in spec['batches'] for r in b['rows'] if r['selection_role']=='original_failure']
fontpath=pathlib.Path('C:/Windows/Fonts/arial.ttf')
font=ImageFont.truetype(str(fontpath),16) if fontpath.exists() else ImageFont.load_default()
manifest=[]
for row in old:
    letter=row['letter'];sid=row['sample_id'];gtfile=pathlib.Path(row['raw_path'])
    gtsha=hashlib.sha256(gtfile.read_bytes()).hexdigest()
    assert gtsha==row['raw_sha256']
    with np.load(gtfile,allow_pickle=False) as z:gt=z['rgb'][row['anchor']+1:row['anchor']+17].astype(np.float32)/255
    assert gt.shape==(16,3,96,96)
    arrays={'GT':gt};references=[]
    for label,d in data.items():
        ref=next(r for r in d['raw_files'] if r['sample_id']==sid and r['mode']=='autoregressive')
        path=pathlib.Path(ref['path']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==ref['sha256']
        with np.load(path,allow_pickle=False) as z:arrays[label]=z['raw']
        assert arrays[label].shape==(16,3,96,96)
        references.append(dict(label=label,path=str(path),sha256=digest,bytes=path.stat().st_size))
    w,h=948,966
    canvas=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(canvas)
    draw.text((10,8),f'Old {letter}: {row["family_id"]}; original saved raw AR; no new inference',fill='black',font=font)
    for col,horizon in enumerate([1,4,8,16]):draw.text((114+col*208,35),f'H{horizon}',fill='black',font=font)
    for i,(label,a) in enumerate(arrays.items()):
        y=64+i*222;draw.text((5,y+65),label,fill='black',font=font)
        for j,horizon in enumerate([1,4,8,16]):
            im=Image.fromarray((np.clip(a[horizon-1],0,1).transpose(1,2,0)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST)
            x=102+j*208;canvas.paste(im,(x,y))
            if label!='GT':
                metric=next(r['metrics'] for r in data[label]['rows'] if r['sample_id']==sid and r['input_mode']=='autoregressive' and r['horizon']==horizon)
                draw.text((x,y+194),f'IoU {metric["raw_mask_iou"]:.3f} pos {metric["object_position_error_image_px"]:.2f}',fill='black',font=font)
    filename=f'old-{letter}-saved-raw-AR.png';canvas.save(P/filename)
    manifest.append(dict(sample_id=sid,family_id=row['family_id'],letter=letter,anchor=row['anchor'],
                         GT_source=str(gtfile),GT_sha256=gtsha,prediction_sources=references,
                         panel=str(P/filename),display='Entire96x96 frame; clip[0,1] only for display, nearest2x; raw unchanged'))
(P/'SAVED_PANELS_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

def trajectory_table(role):
    out=['letter | E2 | A128 | A256 | B128 | B256 (IoU / imagepx)']
    for letter in 'OCB':
        cells=[]
        for label in labels:
            r=own[(label,role,letter,'autoregressive',16)]
            cells.append(f'{r["IoU"]:.5f}/{r["position"]:.5f}')
        out.append(letter+' | '+' | '.join(cells))
    return '\n'.join(out)

text=f'''LOSS-AR4-R1：独立分析聊天的终态复核
日期：2026-11-02（新加坡时间）
run_id：LOSS-AR4-20261102-R1
configSHA256：{audit['config_sha256']}

结论
不建议直接采用这版AR4辅助loss或按原配方继续加步数。该具体方案在固定训练子集上有收益，尤其字母B的长滚动；但预设的三个字母支持条件全部不满足，验证AR4位置都较原loss差，强运动真实历史单步IoU全部较差，C的验证长滚动更坏，三个旧失败最终均低于共同起点。
这不是“多步loss全部无效”的结论，更不能证明loss不是瓶颈。它否定的是本轮具体配置能在此预算内稳健改善这些目标的预期。保持暂停，等待用户审批后续。

实际测试了什么
A：原生H1扩散预条件残差MSE。
B：A + 4×mean(四步原生raw RGB预测与GT的像素平方差)。额外分母为1序列×4时刻×3×96×96。四步生成历史保持跨步梯度；量化前向不变，反向采用STE近似导数。它不同于旧B/D的detach历史，也不是旧C/D的区域/细节加权。
两臂独立从同一E2完整模型、Adam、scaler和RNG恢复。各256次尝试/256次成功，AMP跳步0，原H1曝光各4096（768唯一、3328重复）；B另有1024辅助GT曝光（192个唯一四步块加64次重复块）。
这是等更新、等原H1曝光的对照，B总监督曝光多25%，含重算训练前向10240对A4096，B训练时间481.403s对A125.600s，约3.83倍。不能把任何收益单独归因于跨步梯度或称等计算效率优势；额外监督/权重分配也属于方案效果。

独立核对范围
直接读取五组源metrics，6240条逐帧评分，独立汇总1485组/4455个主要均值，全部与保存的分组结果一致；分组拓扑/形状/旋转状态计数一致。
五个检查点各20组输入/槽位/seed/实际epsilon证据完全配对；256对训练conditioning RNG与branch记录一致；A/B完整恢复保存证明全true。
本轮新评共同E2的3744个主要数值与原HMF起点对应评分逐项完全一致，最大差0。这提供了使用原基线/评分口径的交叉证据，未把改造版本冒充原基线。
另核九份旧失败预测档案及三份对应GT源SHA，并只用这些保存数组生成本目录三张旧失败同案图。没有重新评分全部图像，没有进行390份raw全量hash，没有新模型调用、训练、物理或采集。读取、统计和绘图均为串行BelowNormal，绘图库数值线程1。
该独立工作没有修改执行root、冻结代码、completion或报告器；本目录报告是新的独立分析产物，不能把原报告失败改记成功。

验证结果：原seen-val24，每字母8家族
AR16：
{trajectory_table('seen_val')}

B256相对A256：O的IoU .50860→.54833，位置2.32550→2.08803，平均改善；C的IoU .19405→.18945，位置4.61499→5.66112，更差；字母B的IoU .54492→.54233略降，位置1.88278→1.79987略好，指标分化。
与E2相比，B256逐例AR16 IoU改善/退步 O7/1、C2/6、字母B5/3；与A256相比 O6/2、C4/4、字母B4/4。因此平均优势不是所有案例均受益，C不能被ALL均值掩盖。

辅助目标对应的AR4也没有稳健改善。A256→B256：
O IoU .87723→.87807，位置 .36525→.44209；
C IoU .65628→.63591，位置 .51851→.67677；
字母B IoU .81611→.81734，位置 .59765→.65378。
三字母位置都更差，O/字母B的IoU优势很小。O在AR4还新增1个拓扑失败（原A为0）。
真实历史强运动H1的IoU A→B：O .94971→.93489、C .91317→.90037、字母B .93431→.92736，全部下降。位置O/字母B也更差，C位置略好，不能称全面单步改善。
原numeric_directional_support三字母false；不是事后提高门槛。原绝对all_passed在E2/A128/A256/B128/B256的三个split全部false，unknown仍不算pass。

训练子集：仅固定train12，每字母4家族
AR16：
{trajectory_table('train')}

B256对A256的三字母AR16平均IoU和位置都改善，字母B最明显：.52668→.69546、1.86571→.80473px；其4个家族IoU均改善。O为2改善2退步，C为3/1。该子集改善不能称全部train48拟合或验证通过。
短滚动存在代价：固定train12的AR4位置在三字母上也都比A差。训练图像中O的长时外轮廓仍变圆/角度偏差明显，C示例位置改善但边缘变圆，字母B示例长时位置和双孔较好、外形/朝向仍有偏差。没有“棱角改善百分比”的合格测量，不根据IoU推导刚性恢复。
中期128点有起伏：C训练AR16 B128 .34846，高于B256 .31461；O训练B128 .42335也高于B256 .40807。不能宣称稳定收敛，也不选中期最好点代替预先指定终点。

旧失败：每字母仅1个家族
AR16：
{trajectory_table('original_failure')}

三个B256终点均低于E2。旧B的H16仍拓扑/孔洞fail；旧C位置从E2的5.51708恶化至7.25586px。旧O可见轴与GT偏差大，但原generic rotation资格仍unknown，不能把未经校准的轴直接宣布为物理旋转方向测定。
本目录 old-O/C/B-saved-raw-AR.png 来自本轮已保存预测，展示同一GT与E2/A256/B256的H1/4/8/16，不借旧实验图冒充本轮结果。

可以与不可以据此判断什么
可以：梯度预检最终通过，H4到预测H1的非零范数.0007490100、完整辅助参数梯度范数.3430057已落盘；实际A/B更新完成，故本轮不是因跨步梯度完全断开而没有测试到辅助项。训练长滚动存在局部收益，同时有真实历史精度、短滚动和验证保留代价。
不可以：证明背景主导梯度、证明lambda=4太大、证明STE一定是失败根因、断言所有rollout loss无效、断言只需再多训练即可修好。这些机制没有被单独控制。当前辅助项仍是全图RGB MSE，没有专门刚体/拓扑/旋转目标；不能把“加了多步”视为已经对齐所有物理验收。
没测新T/L/X、未见字母、接触或穿透，不作相关新结论。RH每个目标都重新读真实历史，是16次单步；原AR是自身预测反馈，不混称。

工程状态与费用
R1的预检45、训练14336（含B辅助3072与重算3072）、评测18720图像级前向，总33101，低于本次33120上限。含旧预检失败33次，历史累计33134；旧失败记录没有清零。6240张正式预测、两臂512次成功更新已完成。
worker墙钟1619.979s，约27分钟；自身CPU user1194.906s、system186.813s，属于单线程低优先级的实际CPU消耗，不是CPU零使用或纯GPU账单。本进程峰CUDA allocated1799176192B/reserved2065694720B，不是整卡使用量。无其他项目/系统环境修改、无CPU模型回退。
worker完成时新root约1.002GB<1.2GB，共享参考约7.845GB加原子预留/外部增长guard仍低于10GB；不是本分析时的实时全库审计。

整体completion仍failed=true。训练和五组评测完成后，report.py把含冒号的旧失败sample_id直接作为Windows文件名，首张旧O图写入报Errno22。原REPORT.txt、HTML和summary缺失；六张训练图、分组指标、配对证明、全部五组metrics/raw引用已保存。该报告缺项不抹去保存的模型结果，也不能把整项执行记为成功。无需为了分析这些结果重跑训练/评测。

当前决定
不采用B作为已验收模型，不自动调lambda、增加步数或另起批次。保持等待用户审批。若后续要区分机制，应另行提出明确的单因素、预算和验收；本轮没有下发或启动该工作。

证据
AUDIT.json / analyze_saved.py：独立源评分、噪声/初始化、冻结SHA与费用核对。
INDEPENDENT_ALL16.json / PAIRED_CASES.json / PRESET_CHECKS.json：全部分组、逐例与预设门槛重算。
SAVED_PANELS_MANIFEST.json：九份预测与三份GT源SHA、原sample_id/anchor映射和显示方法。
validation-all16-A-vs-AR4.png：验证全16步曲线。
执行root：{RUN}
'''
(P/'REPORT.txt').write_text(text,encoding='utf-8')
page='<!doctype html><meta charset="utf-8"><title>LOSS-AR4-R1独立复核</title><style>body{max-width:1100px;margin:32px auto;padding:0 18px;font:16px/1.65 system-ui}pre{white-space:pre-wrap;font:inherit}img{max-width:100%}a{margin-right:18px}</style>'
page+='<p><a href="AUDIT.json">独立核对账本</a><a href="SAVED_PANELS_MANIFEST.json">同案图来源与SHA</a></p>'
page+='<img src="validation-all16-A-vs-AR4.png" alt="验证全16步曲线"><pre>'+html.escape(text)+'</pre>'
for letter in 'OCB':page+=f'<h2>旧失败 {letter}：本轮保存raw</h2><img src="old-{letter}-saved-raw-AR.png">'
page+='<h2>执行root已保存训练图</h2><p>'
for letter,sid in [('O','HMF-AC-O-01-25-a3'),('C','HMF-AC-C-03-01-a19'),('B','HMF-AC-B-03-27-a3')]:
    path=RUN/f'report/{sid}-autoregressive.png';page+=f'<a href="{html.escape(path.as_uri())}">{letter}训练同案图</a>'
page+='</p>'
(P/'REPORT.html').write_text(page,encoding='utf-8')
print(json.dumps(dict(report=str(P/'REPORT.html'),new_model_calls=0,raw_archives_hashed=9,GT_sources_hashed=3,new_review_bytes=sum(f.stat().st_size for f in P.iterdir() if f.is_file())),ensure_ascii=False))
