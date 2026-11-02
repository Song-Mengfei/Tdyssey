"""Final Chinese review from saved JSON only; no model/tensor/physics work."""
import json,statistics,collections,ctypes,math
from pathlib import Path
from ctypes import wintypes
from common import ROOT,read,write,write_bytes,sha,output_io,utc
def main():
    k=ctypes.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE;k.SetPriorityClass.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.SetPriorityClass.restype=wintypes.BOOL;assert k.SetPriorityClass(k.GetCurrentProcess(),0x4000)
    c=read(ROOT/'frozen-config.json');done=read(ROOT/'completion.json');audit=read(ROOT/'saved-audit.json');summary=read(ROOT/'report/summary.json');diagnosis=read(ROOT/'diagnosis/summary.json');pre=read(ROOT/'preflight.json');checks=read(ROOT/'report/comparison-checks.json');changes=read(ROOT/'report/per-case-comparisons.json');groups=read(ROOT/'report/grouped-all16.json');measure=read(ROOT/'diagnosis/measurements.json')
    assert done['complete'] and not done['failed'] and audit['passed'] and read(ROOT/'audit-result.json')['actual_exit_code']==0
    labels=['E2','BASE']+c['arms'];lookup={(x['label'],x['role'],x['letter'],x['mode'],x['horizon']):x for x in groups}
    runs={a:read(c['E2_metrics'] if a=='E2' else c['BASE_metrics'] if a=='BASE' else ROOT/f'evaluation/{a}/metrics.json') for a in labels}
    unique=len({i for b in read(ROOT/'schedule.json')['native_h1_batches'] for i in b})
    support=[x for x in checks if x['numeric_directional_support']]
    headline='三臂各128尝试/128成功更新、2048正式原生H1曝光，累计384更新/6144曝光，AMP跳步0；3744帧评测完整，saved-only审计实际退出0。三臂均未通过综合方向性条件和原绝对验收；仅NATIVE_G的C通过短时方向性子条件，C的H16仍退步。'
    lines=['# LOSS-DEFINITION-20261102 最终复核','',headline,'',
        '本轮已实际完成于2026-11-02 22:06:15新加坡。原用户要求改变“什么定义为预测错误”：因此这次测试RGB派生的占据几何与空间输运，和上一轮SNR/GDL/FFL分开。以下不把工程通过当模型通过。','',
        '## 实际方案与比较边界','',
        '- G_ONLY = cG × 平方分母Dice几何损失，明确替换native；没有暗加RGB/残差锚点。NATIVE_G = native+cG×G；NATIVE_T = native+cT×(16方向空间投影W1+相对面积差)。灰物体与蓝杆等权，白背景不作为第三类。',
        '- 每臂独立从完整E2恢复model/Adam/scaler/global与conditioning RNG。训练使用原HMF前128批、effective16/micro4、LR1e-5、clip1、原FP16 forward/FP32loss、50/50真实历史clean/weak。物理、数据、split、动作、曝光顺序、原3去噪raw推理和原评测资格/门槛未变。',
        '- D=c_skip*z+c_out*F来自同一次native forward，其中z含加噪GT。新目标优化的是该单次连续重建；不能说直接优化了纯噪声AR16采样轨迹。无STE/输出warp/模板头/privileged未来状态或隐藏轮廓。',
        '- E2和原BASE128只读复用；BASE的旧R1A128原raw SHA和主指标exact配对证据被审计保留，未为本轮换baseline或重训BASE。',
        f'- 每臂正式曝光2048、原目标唯一数{unique}；不是2048条独立轨迹。评测train12/seen-val24/旧失败3共39家族，每臂RH与AR各16时刻，共1248帧。RH给每时刻真实历史后独立预测一步；AR16只回灌自身原始输出，无未来真值。train12不是全部48训练片段验收。',
        '- T/L/X与其它字母、其它seed、其它loss系数/方向数、长训练收敛、独立泛化均未新测。文献中的PH/GW/RAFT/PDE等没有在本轮运行，不能报成实验负结果。','',
        '## 结果与已见失败','',
        'NATIVE_G的C：seen-val AR4 IoU由BASE 0.651441到0.659662（+0.822个百分点），位置0.454861→0.386860像素（减少14.95%）；强H1两项不差且拓扑fail未增加，所以该字母短时条件通过。但C AR16 IoU 0.249992→0.244821、位置3.702187→3.831008，6/8例IoU退步。不能称C整体修复。',
        'NATIVE_G的O：AR4仅小增，RH强运动位置不达不差条件，H16新增验证例HMF-AC-O-05-08-a3的2孔/GT1孔，拓扑fail由0/8变1/8。O/C棱角、方向或刚性没有因新几何loss得到可靠修复。',
        'G_ONLY：训练面板O AR16 IoU由0.335674到0.412462，但训练面板C/B均下降；验证C AR16由0.249992到0.204634（-4.536个百分点），6/8例下降，位置3.702187→4.019568。原旧O例0.511111→0.398396，形状pass→fail，预测仍不随GT正确转动。纯几何没有在这128步内导致全白/空蓝杆崩溃，但明显没有解决运动，不能据此推荐替换score目标。',
        'NATIVE_T：大部分变化小。验证B AR16 IoU 0.523932→0.525916（+0.198个百分点，6/8例提高），但位置1.667576→1.695223变差；O/C AR16均值仍略降。它未通过任一字母完整方向性条件。新loss数值或梯度非零不等于实际输运/动力学学会。',
        '旧B H16：GT2孔；BASE/NATIVE_G/NATIVE_T各1孔，G_ONLY2孔。但G_ONLY IoU 0.259494、位置6.774768px，较BASE 0.296137/5.671804更差，且明显变形，shape unknown。NATIVE_G/T IoU均0.306843仍低于E2 0.371849；不能把末帧孔数或IoU局部收益当运动/刚体正确。',
        '训练与旧失败同案板已实际目视6张AR板：O仍有不转/反向趋势与孔框变圆；旧C的侧边及角点逐步变成折线/弯角，运动方向不跟GT；旧B后期孔洞与边界明显失真。目视不替代有资格的角度/形状统计。',
        'H16验证：G_ONLY/NATIVE_T的O均8/8 shape unknown；NATIVE_G为1fail+7unknown。所有三臂C/B均2fail+6unknown；O旋转7unknown+1not_required，C/B旋转均8unknown。不能给“棱角或角度改善百分比”，也不能把unknown当通过。所有候选各split原绝对all_passed均false。','',
        '### 验证集AR16：IoU / 物体位置误差px（每字母8例）','',
        '|字母|E2|BASE128|G_ONLY|NATIVE_G|NATIVE_T|','|---|---|---|---|---|---|']
    f=lambda x:'unknown' if x is None else f'{x:.6f}'
    for letter in ['O','C','B']:
        lines.append('|'+letter+'|'+'|'.join(f(lookup[a,'seen_val',letter,'autoregressive',16]['IoU'])+' / '+f(lookup[a,'seen_val',letter,'autoregressive',16]['position']) for a in labels)+'|')
    lines+=['','### 逐字母方向性条件','',f'实际通过{len(support)}/9个候选×字母子条件；各候选须三字母共同满足才算整体验证支持。','', '|候选|字母|AR4 IoU提高|AR4位置不差|RH强H1 IoU不差|RH强H1位置不差|拓扑fail不增|子条件通过|','|---|---|---|---|---|---|---|---|']
    for row in checks:
        q=row['conditions'];lines.append('|'+row['arm']+'|'+row['letter']+'|'+'|'.join(str(q[k]) for k in ['AR4_IoU','AR4_position','RH_strong_IoU','RH_strong_position','topology_fail_not_increased'])+'|'+str(row['numeric_directional_support'])+'|')
    lines+=['','### H16逐例IoU得失，相对BASE','', '|候选|分组|字母|改善|退步|相同|未知|','|---|---|---|---|---|---|---|']
    for arm in c['arms']:
        for role in ['train','seen_val','original_failure']:
            for letter in ['O','C','B']:
                counts=collections.Counter(x['IoU_status'] for x in changes if x['arm']==arm and x['reference']=='BASE' and x['role']==role and x['letter']==letter and x['mode']=='autoregressive' and x['horizon']==16)
                lines.append('|'+arm+'|'+role+'|'+letter+'|'+'|'.join(str(counts[k]) for k in ['improved','regressed','tied','unknown'])+'|')
    lines+=['','## 定义响应与适配限制','',
        f'GT-only诊断实际12个原训练样本、{diagnosis["constructed_images"]}张构造、11种定义/分量，tensor-only梯度{diagnosis["tensor_only_gradient_calls"]}次；无新模型调用，不是物理counterfactual、训练augmentation或新模型预测。来源RGB固定三色已从renderer核实，GT重建残差最大0.006946、原hard灰mask与新阈值mask每图差0–4像素；新mask没有取代旧评价mask。',
        '全部24个有资格x/y平移2/4/8px三元组中T严格递增；首个O样本x/y各三点的连续位移梯度为正，梯度下降方向指回GT（6次导数，不能冒充所有样本/任意误差的方向保证）。首例T约0.004723/0.009447/0.018893。',
        '相同无重叠率的近/远错位：G均1.0，T为0.085019/0.170038，说明输运提供Dice所缺的距离响应。一个同质心异形例的质心误差恰0，而G=0.058824、T=0.112939；质心不等于形状或刚体位姿。',
        '全白p=0时G=1、T=1.151614，梯度均有限非零；但原W1/T RGB梯度范数约9.432百万，显示极小质量归一化的强敏感性，不能只凭有限/非零说数值理想。clDice_background该例loss为0却梯度不为0，暴露空结构/soft骨架代理的边界行为，不是拓扑保证。',
        '孔填充只有8例有可见孔、4例无孔unknown；未用蓝杆遮挡区域冒充可见孔。本组没有命中蓝杆触孔的合格填孔例。切断连接12例实际分量数增加；复制块11例仍相连、1例裁切，全部保留unknown，没有谎称12个独立增生正例。',
        '颜色盲区的解析反例（未新增tensor实验）：三色差向量的法向量n=(-11730,15300,-2210)，与灰/蓝方向的整数点积均恰0。故P*n=0；在灰色像素沿n作足够小RGB扰动可不越界，m(I)不变而RGB变，G/T无法识别该色差。这不是实测额外构造，也没有把276报成288。',
        'Kervadec式signed boundary使用RGB二值GT距离场，保存原signed值与相对同GT参考的常数偏移；与软抗锯齿目标不完全同义。soft-clDice骨架10轮、有限16方向W1、moments均各有盲区，没有PH/Betti、刚体、可见完整轮廓或2D最优输运保证。','',
        '## 校准、clamp和采样风险','',
        f'只在首16训练样本、四个B4上一次定单位：cG={summary["coefficients"]["cG"]:.15g}，cT={summary["coefficients"]["cT"]:.15g}。四个norm(native)/norm(G或T)的普通中位数固定到结束，没有按val/结果调系数。原native loss和逐参数梯度bitwise一致；三个最终组合参数梯度都有限非零。',
        '初始四批native-G cosine约0.147–0.193；native-T约-0.0444–0.0671；G-T约0.580–0.928。它们在初始方向上并不等同，单位匹配不保证训练中持续平衡，更不构成根因证明。',
        'clamp严格越界梯度为0，精确0/1边界PyTorch梯度为1，实测probe [0,1,1,1,0]。预检连续D的u严格越界约45–55%，精确边界0；其中可有白背景附近微小RGB偏差，不能直接说物体有一半错误。推理精确白背景会产生大量u=0边界，不能把u<=0或>=1比例解读成零梯度比例。',
        '每臂全部1248输出均无灰物体/蓝杆面积<1像素的empty，也无RGB越界；这是保存输出检查，不是推杆接触/穿透验证。纯G_ONLY本次未发生全空采样崩溃，仍不证明可在长训练中维持原扩散score或色彩/背景正确。GT白背景上的误差也可能来自运动错位物体，不宜直接解释成背景色污染。','',
        '|候选|输出数|灰empty|蓝empty|平均GT背景RGB MSE|平均u严格越界|平均u精确边界|','|---|---|---|---|---|---|---|']
    risk_summary=[]
    for arm in c['arms']:
        rr=runs[arm]['auxiliary_RGB_mask_diagnostics'];a=dict(arm=arm,n=len(rr),gray_empty=sum(x['gray_empty'] for x in rr),blue_empty=sum(x['blue_empty'] for x in rr),mean_background_MSE=statistics.mean(x['background_MSE'] for x in rr),mean_u_strict_outside=statistics.mean(x['u_strict_outside_fraction'] for x in rr),mean_u_exact_boundary=statistics.mean(x['u_exact_boundary_fraction'] for x in rr))
        risk_summary.append(a);lines.append('|'+arm+'|'+str(a['n'])+'|'+str(a['gray_empty'])+'|'+str(a['blue_empty'])+'|'+f(a['mean_background_MSE'])+'|'+f(a['mean_u_strict_outside'])+'|'+f(a['mean_u_exact_boundary'])+'|')
    lines+=['','## 全部分组H1/4/8/16，同案原指标','', '每格为IoU / 物体位置误差px；其它轮廓、蓝杆、拓扑、shape/rotation资格、全部16时刻及逐例数值在原JSON完整保留。']
    for role in ['train','seen_val','original_failure']:
        for mode in ['real_history','autoregressive']:
            lines+=['',f'### {role} / {mode}','','|字母/时距|'+'|'.join(labels)+'|','|---|'+'|'.join(['---']*len(labels))+'|']
            for letter in ['O','C','B']:
                for h in [1,4,8,16]:lines.append('|'+letter+f' H{h}|'+'|'.join(f(lookup[a,role,letter,mode,h]['IoU'])+' / '+f(lookup[a,role,letter,mode,h]['position']) for a in labels)+'|')
    lines+=['','## 实际费用、恢复和保存核验','',
        f'- 实际{audit["image_forwards"]}图像forward / 上限17504（训练6144、评测11232、预检32），{audit["batch_forwards"]}batch forward /4544。正式反传1536个microbatch，预检参数梯度16次，独立tensor梯度51次；不把反向、tensor构造或审计误当模型预测。',
        '- 仅1个模型构建、1次预检E2 model-only load、3次正式完整状态恢复，384尝试/384成功，AMP0；0新物理/新数据，0额外seed/步数/模型输出修补。预检32图不是额外正式训练曝光。',
        f'- worker活跃混合墙钟{done["wall_seconds"]:.3f}s（约9分53秒），不是纯GPU账单；CPU user{done["own_CPU_seconds"]["user"]:.3f}s/system{done["own_CPU_seconds"]["system"]:.3f}s。训练G_ONLY/NATIVE_G/NATIVE_T约93.355/121.554/80.850s，评测80.868/89.462/57.705s；定义诊断7.177s、预检3.771s另列，剩余为源核验/IO/报告等。',
        f'- GPU峰值allocated{done["results"]["peak_cuda"]["allocated"]}B、reserved{done["results"]["peak_cuda"]["reserved"]}B。单数值进程BelowNormal；Torch intra/inter、OpenCV和数值库线程1；无CPU模型回退、其它进程或系统环境修改。实际IO重试{done["io_replace_retries"]}。',
        f'- saved-only审计一次实际exit0：{audit["raw_archives_SHA_verified"]}个FP32 raw SHA（{audit["raw_bytes"]}B）、3轻量weights SHA、24PNG头/尺寸/安全路径、12845不可变计账事件，以及更新/AMP/完整恢复/conditioning RNG与E2/BASE评测噪声配对。审计0模型/物理/新渲染；没有重跑审计。',
        '- 上轮loss-screen23220计账/23216完成、旧R1失败33/正式33101、更旧R3B失败33.047s仅历史费用，未混入本轮。',
        '- 新root审计时574.55MB<650MB；共享9.173GB，加256MiB原子与64MiB增长预留后仍<10GB。24MiB终态报告预留保留；最终发送回执后再记录实际体积。',
        '- 本轮无Windows系统任务注册，原active exec中worker与一次审计均实际退出0。不操作任何任务或进程。新30分钟监控先被拒，用户后续直接授权后才实际创建；这一授权与真实工具回执保留。最终真实回传后删除监控，waiting_user_review，不自动续训。','',
        '## 可点击完整证据与同案图','']
    links=[('原完整报告',ROOT/'REPORT.txt'),('完整HTML索引',ROOT/'report/index.html'),('全部16步分组',ROOT/'report/grouped-all16.json'),('逐例改善/退步',ROOT/'report/per-case-comparisons.json'),('原方向性条件',ROOT/'report/comparison-checks.json'),('RGB/空物体/蓝杆风险',ROOT/'report/RGB-sampling-risks.json'),('定义诊断与公式',ROOT/'diagnosis/summary.json'),('每构造完整结果',ROOT/'diagnosis/measurements.json'),('实际完成记录',ROOT/'completion.json'),('真实saved-only审计',ROOT/'saved-audit.json'),('审计工具回执',ROOT/'audit-result.json'),('系数与参数梯度',ROOT/'preflight.json'),('配对与恢复',ROOT/'pairing-proof.json')]
    for title,path in links:lines.append(f'- [{title}]({path.as_posix()})')
    for panel in summary['panels']:lines.append('- ['+panel['role']+' '+panel['letter']+' '+panel['mode']+']('+Path(panel['path']).as_posix()+')')
    # Boundary constant offsets are computed only from already-saved scalars.
    baseline={x['sample_id']:x['losses']['signed_boundary'] for x in measure if x['variant']=='correct_RGB'}
    boundary_offsets=[dict(sample_id=x['sample_id'],variant=x['variant'],qualification=x['qualification'],raw_signed_boundary=x['losses']['signed_boundary'],GT_constant=baseline[x['sample_id']],relative_to_same_GT=x['losses']['signed_boundary']-baseline[x['sample_id']]) for x in measure]
    write(ROOT/'report/boundary-constant-offsets.json',boundary_offsets)
    assertions=dict(dot_gray=(-136)*(-11730)+(-119)*15300+(-102)*(-2210),dot_blue=(-190)*(-11730)+(-150)*15300+(-30)*(-2210))
    assert all(x==0 for x in assertions.values())
    result=dict(run_id=c['run_id'],headline=headline,per_arm_support=summary['per_arm_support'],directional_letter_support=[{'arm':x['arm'],'letter':x['letter']} for x in support],risk_summary=risk_summary,color_nullspace_counterexample=dict(vector=[-11730,15300,-2210],exact_integer_dot_products=assertions,scope='Analytic counterexample only; not a newly run tensor/model experiment'),review_model_calls=0,review_new_renders=0,review_new_physics=0,at_utc=utc())
    write(ROOT/'owner-review-summary.json',result);text='\n'.join(lines)+'\n';receipt=write_bytes(ROOT/'FINAL_REVIEW.md',text.encode('utf-8'))
    assert (ROOT/'FINAL_REVIEW.md').read_text(encoding='utf-8')==text
    write(ROOT/'owner-review-verification.json',dict(UTF8_roundtrip=True,receipt=receipt,new_model_calls=0,new_inference=0,new_physics=0,original_saved_audit_not_rerun=True,at_utc=utc()))
    print(json.dumps(dict(review=receipt,headline=headline,storage=output_io().guard(refresh=True)),ensure_ascii=True))
if __name__=='__main__':main()
