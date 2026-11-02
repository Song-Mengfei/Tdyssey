"""Attach independently audited saved results to the cited literature report."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[k]='1'
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=pathlib.Path(__file__).resolve().parent
RUN=P.parent.parent/'loss-screen-20261102'
read=lambda n:json.loads((P/n).read_text(encoding='utf-8'))
rows=read('INDEPENDENT_SCREEN_CURVES.json');audit=read('SCREEN_AUDIT.json');paired=read('PAIRED_SCREEN_CASES.json')
own={(r['label'],r['role'],r['letter'],r['mode'],r['horizon']):r for r in rows}
arms=[a for a in ['BASE','SNR','GDL','FFL'] if any(r['label']==a for r in rows)]
fig,axes=plt.subplots(2,3,figsize=(12.6,7.4),sharex=True)
styles=[('E2','#777777','--'),('BASE','#236aa7','-'),('SNR','#23894b','-'),('GDL','#dc821f','-'),('FFL','#9a4ba5','-')]
for j,letter in enumerate('OCB'):
    for label,color,line in styles:
        if label not in ['E2']+arms:continue
        v=[own[(label,'seen_val',letter,'autoregressive',h)] for h in range(1,17)]
        axes[0,j].plot(range(1,17),[r['IoU'] for r in v],color=color,linestyle=line,label=label)
        axes[1,j].plot(range(1,17),[r['position'] for r in v],color=color,linestyle=line)
    axes[0,j].set_title(letter);axes[0,j].set_ylim(0,1.02)
    axes[1,j].set_xlabel('Raw self-feedback horizon (10 Hz)')
    for ax in axes[:,j]:ax.grid(alpha=.2);ax.set_xticks([1,4,8,12,16])
maxpos=max(r['position'] for r in rows if r['role']=='seen_val' and r['mode']=='autoregressive' and isinstance(r['horizon'],int) and r['position'] is not None)
for ax in axes[1,:]:ax.set_ylim(0,maxpos*1.05)
axes[0,0].set_ylabel('Gray object IoU (higher is better)')
axes[1,0].set_ylabel('Visible centroid error (image px)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.005))
fig.suptitle('Matched loss screen: 128 updates each, seen validation 8 fixed families/letter')
fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(P/'SCREEN-validation-all16.png',dpi=140);plt.close(fig)

def table(role,mode='autoregressive',h=16):
    labels=['E2']+arms
    out=['letter | '+' | '.join(labels)+' (IoU / imagepx)']
    for letter in 'OCB':
        cells=[]
        for label in labels:
            r=own[(label,role,letter,mode,h)];cells.append(f'{r["IoU"]:.6f}/{r["position"]:.6f}')
        out.append(letter+' | '+' | '.join(cells))
    return '\n'.join(out)

support=audit['predeclared_support']
all_support={a:all(r['predeclared_directional_support'] for r in support if r['arm']==a)
             for a in arms if a!='BASE'}
summary='本轮已完成的候选：'+','.join(arms)+'。缺失终点评测：'+str(audit['missing_evaluations'])+'。\n'
summary+='工程终态与科学结果分别报告；完整原始completion/预检见独立账本及执行报告。\n'
summary+='预先冻结的三字母方向性支持（全三字母同时满足）：'+str(all_support)+'。\n'
summary+='原绝对验收状态：'+str(audit['absolute_acceptance'])+'。unknown仍不算通过。\n\n'
summary+=f"独立重算{audit['source_metric_means_compared']}个主要均值与负责人分组吻合，拓扑/形状/旋转状态计数吻合；检查{audit['native_rng_step_pairs_checked']}对正式更新的RNG/branch配对。每臂实际使用同{audit['unique_training_target_indices_each']}个唯一H1目标，2048为曝光次数，不是独立新数据。\n\n"
summary+='seen-val AR4：\n'+table('seen_val',h=4)+'\n\n'
summary+='seen-val AR16：\n'+table('seen_val')+'\n\n'
summary+='seen-val RH强运动H1（跨全部16目标的真正强H1，非第一目标可能静止的H1）：\n'+table('seen_val','real_history','strong_H1')+'\n\n'
summary+='固定train12 AR16（每字母4家族；不代表全部48训练轨迹）：\n'+table('train')+'\n\n'
summary+='旧失败3 AR16（每字母1家族）：\n'+table('original_failure')+'\n\n'
summary+='逐例AR16 IoU变化，相对本轮BASE：\n'
for r in paired:
    if r['mode']=='autoregressive' and r['horizon']==16:
        summary+=f"{r['arm']} {r['role']} {r['letter']}: IoU改善/退步={r['IoU_better']}/{r['IoU_worse']}，位置改善/退步={r['position_better']}/{r['position_worse']}\n"
summary+='\n预设条件逐项：\n'
for r in support:
    summary+=f"{r['arm']} {r['letter']}: {r['conditions']}；拓扑fail不增={r['topology_no_worse']}；合计支持={r['predeclared_directional_support']}\n"
summary+='\n各组128步的loss数值定义不同，不能用训练loss大小跨方法排名。SNR会乘小于1的权重，GDL/FFL添加非负项；统一用原项目评价判定。一次短预算筛选不能否定整类方法，也不能宣称收敛。\n'
summary+='本轮未对新候选重做动作干预、接触或穿透物理验证；旧E2/HMF的动作敏感性证据不自动保证新权重正确响应动作。\n'
summary+='本轮没有新增GT曝光差异，但新增loss运算/反传成本不同；源日志和资源账本分别记录，不能称等FLOP。没有新物理、数据、模型结构、额外teacher或新安装。\n'

costs=[]
for arm in arms:
    log=json.loads((RUN/f'training/{arm}-log.json').read_text(encoding='utf-8-sig'))
    norms=[r.get('gradient_norm_before_clip') for r in log]
    values=[x for x in norms if isinstance(x,(int,float))]
    cost=audit['training_results'].get(arm,{})
    costs.append(dict(arm=arm,attempts=len(log),training_seconds=cost.get('training_seconds'),
                      gradient_clip_events=sum(x>1 for x in values),gradient_norm_max=max(values) if values else None))
summary+='\n训练时间与clip观察（loss梯度竞争机制尚未被单独隔离）：\n'+json.dumps(costs,ensure_ascii=False)+'\n'
summary+='GDL组42/128次更新的总参数梯度范数超过原clip=1阈值（32.8%，最大2.4371）；BASE/SNR/FFL均0次，BASE最大仅.09164。这是保存日志中额外发现的优化行为差异。它说明辅助项的系数/单位及梯度尺度值得检查，但没有分解各项梯度或测夹角，不能断言这就是C退化原因，也不能据此保证调小系数会解决。\n'
summary+='同一个预检B4、同一权重/噪声下，BASE/GDL总参数梯度范数分别.0464208/.4115463，GDL为8.87倍；FFL为1.26倍，SNR为.966倍。这个受控单批观察进一步支持尺度值得诊断；仍不能推广为所有训练批的组件梯度比或C退化因果证明，原始数值见FIXED_BATCH_GRADIENT_OBSERVATION.json。\n'
summary+='结合文献，下一步更有针对性的选择是先诊断native/辅助项的梯度幅值与方向、按预先规则做单变量定标；轮廓/soft-IoU/拓扑项也是未验证候选，应先核实RGB概率映射与遮挡资格。没有启动任何这类后续测试，也不把未测试方法称已有效。\n'
summary+='BASE对旧R1 A128的raw/主指标复现：'+json.dumps(audit['BASE_cross_check'],ensure_ascii=False)+'\n'
summary+='独立核对范围：源评分重聚合、同输入/槽/规范噪声哈希值（SNR缓存来源例外见下）、训练RNG和恢复证明；不把这些算新模型调用，也不宣称已将全部312份raw重新评分/全量hash。\n'
terminal=audit['whole_execution_completion'];ledger=terminal['ledger'];source_audit=audit['owner_saved_only_audit']
summary+='\n工程恢复与完整成本：\n'
summary+='原worker因WinError5原子替换热ledger失败，具体文件占用者未知。人类在负责人聊天明确要求解决并恢复、全部结束再统一汇报，已由本聊天read_thread核实；原中断记录保留。恢复采用新IO/journal实现，仅续未完SNR评测和GDL/FFL，不重训BASE/SNR、不重跑预检。原loss/数据/物理/精度/步数不变；两次模型实例串行，第二次属于用户批准的恢复。\n'
summary+=f"最终complete={terminal['complete']}/failed={terminal['failed']}。累计记账图像forward{audit['charged_image_forwards']}，实际完成{audit['completed_image_forwards']}，差4是原before-body写账失败仍保留的reservation；其中28次已完成但未落盘的denoiser重算成本未丢弃。最终独立评测4992张；含丢失前缀重算实际完成采样帧5000。总记账低于原23296 cap。正式训练512成功/8192曝光/2048反传，AMPskip0；预检20forward/6模型梯度与3tensor-only梯度只做一次。\n"
summary+=f"原活跃worker{terminal['original_worker_wall_seconds']:.3f}s，加恢复{terminal['continuation_wall_seconds']:.3f}s，累计活跃{terminal['combined_active_worker_wall_seconds']:.3f}s（约10分15秒）；中断等待和文献/准备时间另计，非端到端总历时或纯GPU账单。CPU与存储详见源completion，单数值进程BelowNormal、线程1，无CPU模型回退或其他项目操作。\n"
summary+=f"负责人saved-only审计通过：{source_audit['raw_archives_SHA_verified']}份raw、{source_audit['lightweight_checkpoints_SHA_verified']}份轻量weights的SHA，以及{source_audit['panels_verified']}张PNG/映射。此为负责人真实工具退出回执支持的审计；本聊天没有重复全量hash。\n"
summary+='SNR有两批RH训练缓存（8份raw）的原始实际噪声观察回执随内存丢失。冻结源码在保存raw前曾检查实际张量，恢复对同seed/B4独立复算哈希；metrics.noise_provenance明确这不是新的模型观察。规范噪声值与其他臂相同，但不把重建哈希冒充原丢失回执。验证与AR比较对应组为新实际观察，主要验证结论不依赖这两批训练RH缓存。\n'
summary+='论文证据与项目判断见上文逐篇条目；候选不是原论文数据/架构的完整复现，已预先固定GPC参数化与单位适配。等待用户审批后续，不自动调gamma/系数/步数或部署。\n'
summary+='\n执行root：'+str(RUN)+'\n'
(P/'SCREEN_RESULTS.txt').write_text(summary,encoding='utf-8')
executive='已筛选29篇/方法，并完成原loss对照及3个候选各128次更新；三个候选工程上均可计算与更新，但均未通过预设逐字母科学筛选，原绝对验收也未通过。GDL使字母B验证AR16 IoU从.523932升至.552540（8/8提高），C却从.249992降至.234253（6/8退步）。GDL还有42/128次更新触发原梯度裁剪，BASE/SNR/FFL均0次，提示应优先诊断梯度尺度，尚不能认定退化根因。Min-SNR/FFL有小幅局部得失，旧B孔数正确仍伴随较大运动误差。现有结果不足以直接替换原loss或扩训。本轮不自动派下一轮。'
result=dict(status='terminal_saved_results',run_root=str(RUN),available=arms,missing=audit['missing_evaluations'],executive_summary=executive,
    summary=summary,all_three_letters_directional_support=all_support,costs=costs,
    figures=['SCREEN-validation-all16.png'],links={'独立测试详细结果':str(P/'SCREEN_RESULTS.txt'),'独立核对账本':str(P/'SCREEN_AUDIT.json'),'同一预检批梯度观察':str(P/'FIXED_BATCH_GRADIENT_OBSERVATION.json'),'直接人类恢复指令核实':str(P/'CONTINUATION_AUTHORIZATION_VERIFIED.json'),'执行完整图表索引':str(RUN/'report/index.html'),'恢复最终completion':str(RUN/'continuation-completion.json'),'原中断历史completion':str(RUN/'completion.json')})
(P/'TEST_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(support=all_support,costs=costs,report=str(P/'REVIEW.html')),ensure_ascii=False))
