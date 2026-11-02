"""Small static report from the independent saved-result audit."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']:
    os.environ[name] = '1'
import html
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = pathlib.Path(__file__).resolve().parent
RUN = P.parent.parent / 'hmf-inference-diagnostics-20261102'
def read(name):
    return json.loads((P / name).read_text(encoding='utf-8'))
e = read('ENDPOINT_TRAJECTORIES.json')
curves = read('INDEPENDENT_CURVES.json')
audit = read('AUDIT.json')
pairs = read('PAIRED_CHANGES.json')
B = read('INDEPENDENT_ACTION_NOISE.json')
labels = ['0', '250', '500', '1000', '2000', 'final']
xs = [0, 250, 500, 1000, 2000, 3000]
fig, axs = plt.subplots(2, 3, figsize=(12.6, 7.5), sharex=True)
styles = [('fixed_train12', 'fixed train (4 families)', '#2474b5', '--'),
          ('seen_val', 'seen validation (8 families)', '#e17c13', '-'),
          ('original_failure', 'old failure (1 family)', '#b32938', ':')]
for col, letter in enumerate('OCB'):
    for role, legend, color, line in styles:
        row = next(x for x in e if (x['role'], x['letter'], x['mode']) == (role, letter, 'autoregressive'))
        axs[0,col].plot(xs, row['iou'], label=legend, color=color, linestyle=line, marker='o', markersize=4)
        axs[1,col].plot(xs, row['position'], color=color, linestyle=line, marker='o', markersize=4)
    axs[0,col].set_title(letter)
    axs[0,col].set_ylim(0, 1)
    axs[1,col].set_ylim(0, 8.5)
    axs[1,col].set_xlabel('HMF training updates (saved checkpoints)')
    for ax in axs[:,col]:
        ax.grid(alpha=.2)
        ax.set_xticks([0, 500, 1000, 2000, 3000])
axs[0,0].set_ylabel('Gray object IoU at raw AR16')
axs[1,0].set_ylabel('Visible centroid error at AR16 (image px)')
handles, legends = axs[0,0].get_legend_handles_labels()
fig.legend(handles, legends, loc='lower center', ncol=3, bbox_to_anchor=(.5, .015), frameon=False)
fig.suptitle('Same saved cases and paired epsilon; raw self-feedback, 16 targets at 10 Hz', fontsize=13)
fig.tight_layout(rect=[0, .065, 1, .95])
fig.savefig(P/'independent-checkpoint-comparison.png', dpi=140)
plt.close(fig)

table_lines = []
for letter in 'OCB':
    r = next(x for x in e if (x['role'],x['letter'],x['mode']) == ('seen_val',letter,'autoregressive'))
    table_lines.append(f"{letter}：" + ' / '.join(f'{v:.5f}' for v in r['iou']))
old_lines = []
for letter in 'OCB':
    r = next(x for x in e if (x['role'],x['letter'],x['mode']) == ('original_failure',letter,'autoregressive'))
    old_lines.append(f"{letter}：" + ' / '.join(f'{v:.5f}' for v in r['iou']))

text = f'''A+B 推理诊断：本分析聊天的独立复核
对象：{audit['run_id']}
日期：2026-11-02，新加坡时间。状态：等待用户审批，无后续训练授权。

结论
补充诊断解决了两个先前未知点：C 的验证退步在最早保存的 250 更新点已经出现；这些被测历史中，当前动作确实影响可见灰物体输出。它们没有解决原训练拟合与闭环验收问题。建议继续保持暂停；现有证据不足以支持直接按原配方追加 1000 更新并期待同时修复验证和旧失败。

本聊天如何独立复核
没有运行模型、训练、物理或采集。使用既有六组 HMF 源评分和本轮 A 源评分，共 7488 条同案逐帧评分，独立重算 1728 个分组；5184 个主要指标均值与负责人曲线吻合。核对 588 个输入字段、4525 个实际噪声字段/配对检查，B 的同历史、槽位与噪声配对一致。72 组动作对照从保存的逐对测量重新汇总；另核验六份本次新 raw 档案的 SHA，未重复全库哈希。独立脚本只有串行 stdlib 统计、少量文件哈希和一张静态图，继承自身 BelowNormal 优先级，绘图数值库线程设 1。
这些核对确认保存证据的内部一致性；主要图像评分沿用原定义，本聊天没有把全部图像重新评分或把负责人日志当作整机独立监控。

A：中期验证和旧失败曲线
以下均为原 raw 自反馈第 16 步的灰物体 IoU，越高越好。顺序：0 / 250 / 500 / 1000 / 2000 / 3000 更新。
seen-val 原 24 家族，每字母 8 家族：
{chr(10).join(table_lines)}

O：最终较起点 +0.12760，8 例中 6 改善、2 退步；质心误差 3.04265→1.87959 imagepx。
C：250 时已从 .25328 降至 .20060，所有新增中期点及 final 均低于起点；1000 最低 .17253，此后恢复但未回到起点。最终 5/8 例退步；质心误差 3.88072→4.85175。没有 0–249 更新的检查点，不能说从第一个更新就已退化。
B：最终较起点 +0.04644，6/8 例改善、2/8 退步；IoU 在 1000 时 .56556，final .56124，略低 .00432，但质心误差 final 最佳 1.58462。因此“1000 一定整体优于 final”也不成立。

固定训练 12 例（每字母仅 4 家族）AR16 IoU 的 0→final：O .37903→.49132，C .19537→.47809，B .40697→.71048；三字母都在 final 达到该固定子集的最高均值。不能用这 12 例冒充全部训练 48 例。尤其 C 的训练子集逐步改善、验证从早期退步，是不均匀适应/保留代价的证据，不能简化成训练还没学会或只需更久。

旧失败 3 例，每字母仅 1 轨迹，AR16 IoU：
{chr(10).join(old_lines)}
旧 O 的全部后续点均未胜起点；旧 C 在 500 时 IoU/位置略好，但 final 更坏；旧 B 在 2000 时 IoU .39905、质心误差 3.31021，好于起点，但此时拓扑仍失败，final 再次退步。没有哪个保存的中期点把三例同时修复。旧 B 的 H16 拓扑 0/500 为 pass，250/1000/2000/final 为 fail；不能把 .399 的略好 IoU 当双孔恢复。

真实历史 RH 每个目标都重新喂真实 4 帧，16 个目标对应 16 次独立单步，不能称 16 步闭环。C 验证强运动 RH 的灰 IoU .91541→.89771，同时质心误差 .27784→.20835 反而改善；图形重叠与位置精度在分化。闭环最终误差却增至 4.85px。这支持局部图像误差与自反馈共同限制长滚动；仅靠“模型完全忽略动作”或“位置单步一直变差”均不能解释。此前发现的 C 姿态曝光偏斜仍是相关证据，本次没有隔离其因果贡献。

B：动作敏感性
12 个冻结强运动历史，每字母 3 个训练家族 + 1 个旧失败家族；E2 与 HMF final、3 当前动作、3 噪声，共 216 张。真实 4 帧和过去 3 动作固定，仅替换当前第 4 动作；hold 为当时真实推杆目标，alternative 来自保存的合法目标。没有裁剪动作，也没有生成反事实物理 GT。

recorded 对 hold：两模型各 12 个案例的灰物体质心、轮廓与排除蓝杆的灰区 RGB 动作差，均大于同动作换噪声的均值。质心这一指标更满足：全部 24 个模型-案例组中，三次配对换动作的最小变化，都超过同动作两端全部六个换噪声比较的最大变化。这是本次有限 3 噪声的描述性结果，不是新的统计门槛或总体显著性检验。
E2 平均质心动作差 .17955–.63047 imagepx；final .13548–1.02253。final 11 例 action/noise 比有限，范围 4.16–14.52；1 例噪声质心差为零，比值未定义，不记无限或自动通过。E2 本来已有敏感性，因此不能把“开始使用动作”记为 HMF 新收益。

可据此排除：这 12 个被测历史中，模型完全忽略当前动作，或只有蓝杆输出改变。
仍未证明：灰物体真实运动方向、角度、幅度、接触约束和穿透正确。可见灰图像变化还可能含遮挡/形变，不能直接解释为刚体位移。recorded 有同状态 GT，hold/alternative 没有；三噪声不是三独立家族。没有测试 T/L/X 或未见字母，不能推广到全部状态和动作。
C 旧失败历史的 recorded 单步 IoU .84627→.85430，但质心误差 .20218→.33283；这也不是全面单步改善。

验收和计算费用
原全训练拟合、长滚动形状/旋转与旧失败验收未通过；unknown 沿用，不能当 pass。此前全 train48 仅 7/48 个 AR16 达 IoU≥.85 的结论不因本轮补中期而改变。
本次计算完成 3672/3672 张，11016/11016 图像级去噪前向，950 次 sampler、2850 次 batch forward；0 训练/反向/优化器/物理/新采集。六次 state-only 权重加载。GPU FP32、原 3 次去噪，无 CPU 模型回退；自身 CPU 单数值线程/BelowNormal，没有操作另一个项目。worker 墙钟 246.059s（约 4 分 6 秒），own CPU user+system 172.844s，不是 CPU 零使用；采样+噪声记录 114.913s 也不是纯 GPU 费用。

整体运行必须保持失败：最终存储检查 501,952,724B 超过新目录 500,000,000B，随后失败 completion 写入又增至 516,100,792B。原因是派生报告/完成元数据没有完整纳入写入保护。停止后负责人无损 gzip 一个逐例 JSON，证明解压字节一致，raw 留存，记录当时恢复至 497,773,330B；没有重跑、改限额或追溯改通过。本聊天没有进行实时全库大小扫描。这个资源边界失守要单列；它发生在预测与报告生成之后，并不自动抹去已保存的诊断证据。

当前处理决定
负责人的 PHASE_STATE 已为 waiting_user_review，训练/新推理授权均 false。用户已批准的两项诊断结果现已整理；后续实验、续训、物理修改与扩采都等待用户另行审批。本聊天没有下发新的执行指令。

证据
AUDIT.json：核对方法、源计数、配对检查、六份 raw SHA、费用和失败状态。
INDEPENDENT_CURVES.json / ENDPOINT_TRAJECTORIES.json：独立逐字母逐分组曲线。
STRONG_RH_CURVES.json：真实历史强运动单步汇总，保留原分母。
PAIRED_CHANGES.json：同案改善/退步，H1/4/8/16 分开。
INDEPENDENT_ACTION_NOISE.json / B_RECORDED_GT_ACCURACY.json：动作噪声独立汇总与 recorded GT。
independent-checkpoint-comparison.png：固定 train12、seen-val24、旧失败3 的 checkpoint 曲线。
负责人原证据目录：{RUN}
'''
(P/'REPORT.txt').write_text(text, encoding='utf-8')
links = [('原完整图索引', RUN/'report/index.html'), ('负责人失败说明', RUN/'FINAL_REVIEW.md'),
         ('独立核对账本', P/'AUDIT.json'), ('完整旧例B同案图', RUN/'report/media/AC-B-00-08-autoregressive-H1-H4-H8-H16.png')]
page = '<!doctype html><meta charset="utf-8"><title>A+B独立复核</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui}pre{white-space:pre-wrap;font:inherit}img{max-width:100%}a{margin-right:20px}</style>'
page += '<p>' + ''.join(f'<a href="{html.escape(path.as_uri())}">{html.escape(name)}</a>' for name,path in links) + '</p>'
page += '<img src="independent-checkpoint-comparison.png" alt="独立重算的中期checkpoint曲线">'
page += '<pre>' + html.escape(text) + '</pre>'
(P/'REPORT.html').write_text(page, encoding='utf-8')
print(json.dumps(dict(report=str(P/'REPORT.html'), plot=str(P/'independent-checkpoint-comparison.png'),
                      report_bytes=(P/'REPORT.txt').stat().st_size), ensure_ascii=False))
