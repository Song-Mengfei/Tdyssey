"""Build independent results from completed saved evidence. No model work."""
import json, pathlib, statistics, math, html

ROOT=pathlib.Path(__file__).resolve().parent
RUN=ROOT.parent.parent/'loss-definition-screen-20261102'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def write(name,v):(ROOT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    audit=read(ROOT/'AUDIT.json');curves=read(ROOT/'INDEPENDENT_ALL16.json');checks=read(ROOT/'INDEPENDENT_CHECKS.json')
    own={(r['label'],r['role'],r['letter'],r['mode'],r['horizon']):r for r in curves}
    assert not audit['missing_evaluations'] and all(r['matches'] for r in audit['source_checks'])
    owner_checks=read(RUN/'report/comparison-checks.json')
    for row in checks:
        other=next(r for r in owner_checks if (r['arm'],r['letter'])==(row['arm'],row['letter']))
        assert other['numeric_directional_support']==row['predeclared_directional_support']
    assert audit['config_sha256']=='f4983a1138d14aeba369fb58286b3b6ad1f10937b360e589b7731a578653636a'
    done=audit['owner_completion'];pf=audit['preflight'];coeff=audit['calibration_coefficients_recomputed_from_saved_parameter_norms']
    scales=[]
    for arm in ['G_ONLY','NATIVE_G','NATIVE_T']:
        log=read(RUN/f'training/{arm}-log.json');micro=[r for s in log for r in s['microbatch_losses']]
        rows=[]
        for a in micro:
            key='T' if arm=='NATIVE_T' else 'G'
            add=coeff['c'+key]*a[key]
            rows.append(dict(native=a['native'],new_error=a[key],scaled_new_error=add,
                scalar_ratio=add/a['native'] if a['native'] else None,W1=a['W1'],relative_area=a['relative_area']))
        scales.append(dict(arm=arm,microbatches=len(rows),native_mean=statistics.mean(r['native'] for r in rows),
            new_error_mean=statistics.mean(r['new_error'] for r in rows),
            scaled_new_error_mean=statistics.mean(r['scaled_new_error'] for r in rows),
            scalar_ratio_median=statistics.median(r['scalar_ratio'] for r in rows),
            scalar_ratio_range=[min(r['scalar_ratio'] for r in rows),max(r['scalar_ratio'] for r in rows)],
            W1_mean=statistics.mean(r['W1'] for r in rows),relative_area_mean=statistics.mean(r['relative_area'] for r in rows)))
    gradient_scales=[]
    for r in pf['rows']:
        n=r['objectives']['native']['gradient_norm']
        gradient_scales.append(dict(microbatch=r['microbatch'],native=n,
            scaled_G=coeff['cG']*r['objectives']['G']['gradient_norm'],
            scaled_T=coeff['cT']*r['objectives']['T']['gradient_norm'],cosines=r['parameter_gradient_cosines']))
    supplement=dict(training_loss_scalar_scales=scales,initial_scaled_parameter_norms=gradient_scales,
        caveat='Small scalar value/coefficient does not imply small parameter gradient. Initial units were comparable; subsequent component gradient norms were not measured. No causal attribution from these records alone.',
        inspected_panels=['original_failure O autoregressive','original_failure C autoregressive','original_failure B autoregressive'],
        visual_observation='O does not follow GT rotation, old C loses its intended corner arrangement, old B remains visibly distorted at H16. These are visual observations, not calibrated angle/rigidity pass rates.',
        all_owner_directional_conditions_independently_matched=True,all_frozen_source_hashes_match=True)
    write('ADDITIONAL_ANALYSIS.json',supplement)
    # Standard scientific figure, single CPU thread configured by invoking shell.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(3,2,figsize=(11.2,10.2),sharex=True)
    colors={'E2':'#999999','BASE':'#222222','G_ONLY':'#c83e48','NATIVE_G':'#1976bc','NATIVE_T':'#24936a'}
    names={'E2':'E2 cached','BASE':'Native BASE128','G_ONLY':'Geometry only','NATIVE_G':'Native + geometry','NATIVE_T':'Native + transport'}
    for row,letter in enumerate('OCB'):
        for label in ['E2','BASE','G_ONLY','NATIVE_G','NATIVE_T']:
            rr=[own[(label,'seen_val',letter,'autoregressive',h)] for h in range(1,17)]
            for col,metric in enumerate(['IoU','position']):
                axes[row,col].plot(range(1,17),[r[metric] for r in rr],color=colors[label],label=names[label],
                    linestyle='--' if label=='E2' else '-',linewidth=2 if label=='BASE' else 1.55)
        axes[row,0].set_ylabel(f'{letter}: mask IoU (higher better)');axes[row,1].set_ylabel(f'{letter}: centroid error, px (lower better)')
        for ax in axes[row]: ax.grid(alpha=.2);ax.set_xticks([1,4,8,12,16]);ax.set_xlim(1,16)
        axes[row,0].set_ylim(0,1.02)
    axes[-1,0].set_xlabel('Raw autoregressive horizon');axes[-1,1].set_xlabel('Raw autoregressive horizon')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.52,.967),frameon=False)
    fig.suptitle('Loss definitions: validation trajectories, 8 cases per letter',y=.995,fontsize=14)
    fig.text(.5,.005,'Same E2 state, order and noise; 128 updates per arm. E2/BASE reused. Single seed, exploratory screen.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.025,1,.925]);fig.savefig(ROOT/'validation-all16.png',dpi=150);plt.close(fig)
    # Text is independent analysis, not a copied owner final.
    t=[
      'Tdyssey loss定义：独立实验结论',
      '2026-11-02 | 三臂均完成；分析仅使用保存证据，零新模型/训练/物理',
      '',
      '结论：找到能表达当前MSE盲区的定义，但本轮没有找到可直接替换现有loss、并改善三字母长时预测的赢家。',
      '纯平方Dice几何、native+Dice几何、native+空间输运三臂均可计算/反传/训练；科学有效性没有因此成立。仅native+geometry的C满足原短时方向性子条件，16步仍退化。',
      '',
      '文献覆盖与本轮实际测试',
      '综述有29项原文来源，含2篇原扩散目标背景与1部公式参考；逐篇注明深读/摘读/摘要筛选。覆盖占据区域、轮廓/距离场、点集距离、空间输运、拓扑、刚体结构、运动、学习表征/价值、物理残差。',
      '训练实测只有两种新错误定义：G=灰/蓝占据量平方分母Dice；T=16个固定空间方向weighted-CDF W1+相对面积差。PH/GW/RAFT/PDE等没有训练，不能据本轮失败判为无效。边界/clDice/moments仅作定义响应诊断。',
      '',
      '为什么这些定义有意义，以及它们仍缺什么',
      '276张受控错误图：24个有资格的2/4/8px平移三元组，T全部严格递增；首个样本6个位移导数均指回GT。相同零重叠率近/远错位的Dice都是1，T为0.085019/0.170038。输运确实补充了“错多远”的信息。',
      '这仍不是学习动力学：定义响应不等于模型会用动作预测正确运动。孔数相同、质心相同或刚体对齐后相同，也都允许错误下一态。旧O/C/B的保存板中仍能看见旋转、角部和孔边失真。',
      'RGB线性颜色解混不是一一映射：颜色差矩阵的零空间会使不同RGB具有相同mask。clamp在严格越界处零梯度；原native提供颜色/去噪anchor，但其是否足以处理这些盲区尚未建立。',
      '',
      '原验证集raw AR16，IoU（越高越好，每字母8例）',
      '定义                     O         C         B',
    ]
    for label in ['BASE','G_ONLY','NATIVE_G','NATIVE_T']:
        vals=[own[(label,'seen_val',l,'autoregressive',16)]['IoU'] for l in 'OCB']
        t.append(f"{label:20s} "+'  '.join(f'{v:.6f}' for v in vals))
    t += ['',
      '纯几何G_ONLY：C AR16 IoU 0.249992→0.204634，下降4.536个百分点；位置3.702→4.020px。训练C/B的H16各4/4例也退步，因此不能只解释为泛化变差。它没有在本128步发生全白/空物体采样崩溃，但仍不支持用它替换native。',
      'NATIVE_G：C AR4 IoU 0.651441→0.659662、位置0.455→0.387px，短时确有局部支持；H16却为0.249992→0.244821，6/8例IoU退步，且O出现一个新的额外孔。短时增益与长时失败必须并列。',
      'NATIVE_T：B H16 IoU提高0.198个百分点且6/8例改善，但位置1.668→1.695px变差；O/C IoU没有均值收益。不能把小的mask面积/位置取舍报成动力学修复。',
      '所有候选所有split的原绝对all_passed均false；9个候选×字母短时子条件仅1个通过，三候选均无跨三字母完整方向性支持。H16大量shape/rotation unknown仍为unknown，未计算角部/旋转改善率。',
      '',
      '新的数值发现及可解释范围',
      '全白诊断下T梯度有限非零，但RGB梯度范数约943万，说明接近零质量时归一化敏感；这条具体输运适配需要数值稳健性研究，不能仅用finite/nonzero判定理想。没有证据表明该极端状态实际造成了本轮长时退步。',
      'cG=0.0233782366559374，cT=0.0000385030905733752；系数只用首train16的参数梯度单位固定一次。尽管cT很小，初始缩放后T参数梯度与native同一量级，所以不能以“系数小”断言输运不起作用。后续未测各分量参数梯度，不能从loss标量大小推断梯度占比或根因。',
      '所有新loss作用于加噪GT的单次连续去噪D，而非纯噪声采样轨迹；非线性mask约束对这个D有效，不等于对原3步sampler或多步动力学有效。本轮结果不能区分这一训练/采样错配、颜色映射、目标本身和128步预算分别造成多少影响。',
      '',
      '独立核验',
      f"从{audit['source_score_rows']}行保存原始评分重算{audit['groups']}组，核对{audit['source_means_compared']}项owner均值及shape/topology/rotation计数；384个训练step的conditioning noise/分支配对均一致。四批范数比中位数、9个方向性条件已独立重算。所有冻结的本轮Python源码hash匹配。",
      '独立分析没有重推/重训/改物理，也没有再次完整重算raw像素评分；负责人一次真实saved-only审计exit0，验证234个raw SHA、3份weights及24张PNG映射。已目视旧O/C/B三张同案AR板。',
      'GPU执行384成功更新、6144正式H1曝光、3744独立新预测；17408模型图像forward。每臂仍只有768个唯一原H1目标，eff16/micro4，单seed128步。CPU只一个数值进程、所有库线程1、BelowNormal；无CPU模型回退或其它项目操作。',
      '本轮worker约9分53秒，显存allocated峰值约1.94GB（十进制）；这不是纯GPU耗时。共享约9.173GB加原预留仍在10GB内；本分析文件与静态图合计远小于30MB。',
      '',
      '当前建议与任务状态',
      '不建议现在直接换成纯Dice；几何/输运可保留为待研究方向，不能宣布它们已解决问题，也不能据此排除loss定义是瓶颈。完整PH拓扑、边界/刚体/真实轨迹运动目标尚未训练验证。',
      '后续若批准，应围绕“定义是否真的约束预测的物体运动/形状”设计下一项可比较验证，而非自动追加这三臂步数或扫描系数。本轮不派新训练、不改物理、不扩采；等待用户审批。',
      '',
      '详细原文与公式：REVIEW.html；逐步曲线：validation-all16.png；完整独立数值：AUDIT.json/INDEPENDENT_ALL16.json/PAIRED_CASES.json；运行前定义：TEST_SPEC.txt。'
    ]
    (ROOT/'RESULTS.txt').write_text('\n'.join(t),encoding='utf-8')
    esc=html.escape
    css='body{max-width:1080px;margin:32px auto;padding:0 24px;font:16px/1.8 system-ui,sans-serif;background:#f7fafc;color:#172738}h1{font-size:28px}h2{font-size:21px;margin-top:26px}.callout{border-left:5px solid #b74c40;background:#fff0ec;padding:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d5dfe8;padding:9px;text-align:left}th{background:#e8eef6}img{width:100%;height:auto}a{color:#125b9d}section{background:white;padding:12px 22px;margin-bottom:16px;border:1px solid #dbe4ec;border-radius:8px}'
    out=[f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(t[0])}</title><style>{css}</style></head><body><h1>{esc(t[0])}</h1>',
         '<p class="callout">找到能表达MSE盲区的错误定义；本轮尚无候选能直接替换现有loss并改善三字母长时预测。</p>',
         '<p><a href="REVIEW.html">原文综述与方法适配</a> · <a href="TEST_SPEC.txt">运行前方案</a> · <a href="AUDIT.json">独立核验记录</a> · <a href="RESULTS.txt">纯文本结果</a></p>',
         '<h2>验证集raw AR16 IoU，每字母8例</h2><table><tr><th>实际目标</th><th>O</th><th>C</th><th>B</th></tr>']
    for label in ['BASE','G_ONLY','NATIVE_G','NATIVE_T']:
        out += ['<tr><td>'+esc(label)+'</td>'+''.join(f'<td>{own[(label,"seen_val",l,"autoregressive",16)]["IoU"]:.6f}</td>' for l in 'OCB')+'</tr>']
    out += ['</table><p>IoU越高越好。BASE为原native loss，同E2、数据顺序和噪声，128更新；未换评价口径。</p>',
            '<img src="validation-all16.png" alt="O C B validation all16 IoU and centroid error curves">']
    # Keep connected paragraphs in the full text, with simple heading detection.
    headings={'文献覆盖与本轮实际测试','为什么这些定义有意义，以及它们仍缺什么','新的数值发现及可解释范围','独立核验','当前建议与任务状态'}
    opened=False
    for line in t[7:]:
        if line in headings:
            if opened:out.append('</section>')
            out += ['<section><h2>'+esc(line)+'</h2>'];opened=True
        elif line and not line.startswith('原验证集raw AR16') and not line.startswith('定义                     ') and not any(line.startswith(a+' ') for a in ['BASE','G_ONLY','NATIVE_G','NATIVE_T']):
            out.append('<p>'+esc(line)+'</p>')
    if opened:out.append('</section>')
    out += ['<p>原文例：<a href="https://arxiv.org/pdf/1606.04797">V-Net区域Dice</a> · <a href="https://arxiv.org/html/1803.00567">一维输运公式</a> · <a href="https://arxiv.org/html/2403.14404v3">非线性约束与去噪均值/真实采样的差异</a>。完整29项来源与阅读深度见综述。</p></body></html>']
    (ROOT/'RESULTS.html').write_text('\n'.join(out),encoding='utf-8')
    state=read(ROOT/'STATE.json');state.update(status='文献综述与有界GPU适配测试完成，保存结果已独立复算；无候选通过三字母完整支持，等待用户审批。',results_received=True,
        execution_config_sha256=audit['config_sha256'],source_score_rows_reaggregated=6240,groups_reaggregated=1485,
        source_means_compared=audit['source_means_compared'],native_RNG_pairs_checked=384,
        numerical_directional_passes=1,total_letter_conditions=9,root_model_calls=0,automatic_followup_experiments=False)
    write('STATE.json',state)
    print(json.dumps(dict(independent_results='complete',all_frozen_sources_match=True,means_compared=audit['source_means_compared'],
        formal_clips={a:r['clip_count'] for a,r in audit['training'].items()},scalar_scale_observation=scales,
        report_bytes=sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file())),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
