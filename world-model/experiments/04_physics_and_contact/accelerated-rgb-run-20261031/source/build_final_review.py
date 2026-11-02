"""Saved-only final report assembly. Standard library; no model/physics imports."""
from pathlib import Path
import html
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from common import atomic_json,read,sha,check_storage,utc

def link(path,label):return f'[{label}]({Path(path).as_posix()})'

def main():
    c=read(ROOT/'runtime-config.json');end=read(ROOT/'background/completion.json')
    if not end['completed'] or end['failed'] or end['config_sha256']!=sha(ROOT/'runtime-config.json'):
        raise ValueError('Completion/config identity mismatch')
    data=read(ROOT/'selection/summary.json');pred=read(ROOT/'postrun-prediction-review.json')
    training=read(ROOT/'training/result.json');collection=read(ROOT/'collection/completion.json')
    def group(letter,h=16,subset='all'):
        return next(x for x in pred['groups'] if x['letter']==letter and x['horizon']==h and x['subset']==subset)
    def val(g,method,key):return g['methods'][method]['means'][key]['value']
    lines=['# 新数据一轮训练：最终结果与限制','',
      '1536次新尝试均已保存；1339个训练家族、30935个唯一样本完整遍历一轮，1934次成功更新、AMP跳步0。整体H16位置和轮廓改善，但O/C强旋转与棱角仍未解决；不自动继续训练。','',
      f'run_id={c["run_id"]}；配置SHA256={sha(ROOT/"runtime-config.json")}。','',
      '## 同组预测结果','',
      '48个独立seen-val家族，每字母8个。训练前、训练后共用锚点、动作和噪声；未来预测原值逐步回灌，无未来真值或形状修复。误差单位为96px图像像素；位置是可见灰/蓝质心误差，不是真实物理位姿误差。','',
      '| H | 灰物体IoU：前→后（复制） | 物体位置误差：前→后（复制） | 蓝推杆误差：前→后（复制） |',
      '|---|---|---|---|']
    for h in (1,4,8,16):
        g=group('ALL',h)
        cell=lambda key:'→'.join(f'{val(g,m,key):.3f}' for m in ('before','after'))+f'（{val(g,"copy_last",key):.3f}）'
        lines.append(f'| {h} | {cell("raw_mask_iou")} | {cell("object_position_error_image_px")} | {cell("pusher_position_error_image_px")} |')
    lines += ['', 'H16总体轮廓误差1.533→0.998（下降34.9%），物体位置误差3.369→2.132（下降36.7%），蓝推杆1.647→0.396（下降76.0%）。H1整体灰物体指标仍不及复制上一帧，全部48例H1真值运动都低于明显运动展示门槛。','',
      '## O、C棱角和长程运动，B孔洞','',
      '| 对象/子集 | N | H16 IoU：前→后 | 位置误差：前→后 |',
      '|---|---:|---|---|']
    for letter,subset in [('O','all'),('O','obvious_motion'),('C','all'),('C','obvious_motion'),('B','all')]:
        g=group(letter,subset=subset)
        lines.append(f'| {letter}/{"全部" if subset=="all" else "明显运动"} | {g["denominator"]} | {val(g,"before","raw_mask_iou"):.4f}→{val(g,"after","raw_mask_iou"):.4f} | {val(g,"before","object_position_error_image_px"):.3f}→{val(g,"after","object_position_error_image_px"):.3f} |')
    lines += ['',
      '- O整体8例改善，但明显运动4例的位置误差倒退14.1%；强旋转例到H16仍接近旧朝向。',
      '- C在H4/H8有位置改善，H16形状/方向仍可能严重错误；明显运动4例IoU退步。默认C图是无接触、物体静止的撤离场景，不能用于证明接触预测改善。',
      '- B在H16的IoU几乎不变；孔洞数量错误1/8→0/8，仍有旋转不足及位置漂移。孔洞数量正确不等于全部几何正确。',
      '- O/C原始角点计数没有一致改善；该计数受栅格、遮挡影响，不能给出可信的独立“棱角改善百分比”。轮廓误差也混合了位置、朝向和形状误差。',
      '- 每个方法192个评测端点均未出现蓝推杆消失；改善的是位置，不能说消失问题本轮从有到无。没有直接的物理穿透合格证；姿态偏差仍可导致错误接触关系。',
      '- 可见轮廓角度受对称性/接触遮挡影响，不能把该诊断均值直接当真实旋转误差。单次训练、六个已暴露字母，不证明未见字母泛化或策略提升。','',
      '### 真实接触且明显运动的补充对照','',
      '仅从既有48个已评测样本中，按“正冲量接触且H16明显运动”筛出，再取每字母真值绝对转角最大者；不按模型表现选例。每列依次为真值、训练前、训练后一轮、复制上一帧。没有重跑推理。','']
    for m in pred['selected_media']:
        lines += [link(m['path'],m['letter']+' 真实接触强旋转对照'),'',f'真值H16：平移{m["truth_h16_translation_image_px"]:.2f}px，转角{m["truth_h16_signed_rotation_deg"]:+.2f}°；样本{m["sample_id"]}。','']
    lines += ['### 六字母完整逐帧板与10fps视频','', '| 字母 | 默认逐帧图 | 配对视频 |','|---|---|---|']
    report=read(ROOT/'report/summary.json')
    for letter in 'TLOCXB':
        panel=next(p for p in report['panels'] if p['letter']==letter)
        video=next(p for p in report['videos'] if p['letter']==letter)
        lines.append(f'| {letter} | {link(panel["path"],"H1/H4/H8/H16")} | {link(video["path"],"真值/前/后/复制，10fps")} |')
    lines += ['',link(ROOT/'report/index.html','原完整结果页'),'','## 新数据、隔离与覆盖','',
      '实际45969个控制请求、459213个原生.01s子步、初始化积分0；41882个完整干净前缀控制。1536份raw/meta/cost/claim均齐全，pending_raw=0，无替换/重试/额外家族。请求=传入=预先承诺动作，195个冻结绑定哈希无变化。','',
      '结局：1335条跑到任务上限、103硬停（66物体越墙外包络、26推杆撞墙、11圆心进入实体/孔）、63块末未决、15分离未决、20规划不可用。跑到上限并不等于整条训练合格。270个恢复候选事件/211家族均隔离相关后缀；另有74块末未决事件、15分离未决事件、125被其他终止删失的WATCH事件，后者不是仍在运行，也不算恢复。','',
      '自然干净H1/H4/H8/H16窗口为37274/32682/26663/16187，独立家族1531/1529/1406/1252。训练H1去除1468完全重复上下文后保留30935；接触且运动17658（57.08%）、空移蓝推杆11322（36.60%）、接触受阻/微动1009、安静/微动946。未来逐帧完全静止854（2.76%），必要负例保留。未发生>511窗口排除，物理0..512和模型归一化均未裁剪/修改。','',
      '候选验证285个已提前承诺H16窗口/163家族；最终固定48家族用于配对预测。全部家族split隔离。旧120和四例诊断没有混训或改标签。实际正AO冲量占全部已执行原生时间48.31%，不是纯干净训练池接触比例。','',
      '完整干净H16窗口中，原定边界任务有实际OW支持89/192家族，再接触（先正AO、再保存时刻真实几何间隙、再正AO）69/192；B再接触仅7/32。六字母均有双向强旋转、边界和再接触，但并非所有计划都达成。另按任意长度干净前缀，边界正OW为165/192；四角同一原生子步同时两邻墙正冲量的干净证据为9/9/8/10家族（左上/右上/左下/右下），这与完整H16分母不同。无正冲量不等于几何脱离。','',
      '## 开销、检查点与状态','',
      f'模拟时间{collection["simulated_seconds"]:.2f}s；采集墙钟{collection["wall_seconds"]:.2f}s；全流程墙钟{end["wall_seconds"]:.2f}s（{end["wall_seconds"]/60:.2f}min）。训练侧计时{training["wall_seconds"]:.2f}s包含训练前评测108.99s，不是纯GPU核时；训练后评测302.19s另列，不能重复加成CPU/GPU费用。旧R3B失败33.047s仅为历史开销，不混入本轮。','',
      f'CUDA峰值allocated={training["peak_cuda_allocated_bytes"]}bytes，reserved={training["peak_cuda_reserved_bytes"]}bytes。最终阶段共享占用2,027,426,751bytes，新root348,141,432bytes（派生审查文件另见最终资源收据），远低于总10,000,000,000bytes，并保留256MiB原子余量。','',
      f'最终完整边界checkpoint：{link(training["checkpoint"]["path"],"last.pt")}；SHA256={training["checkpoint"]["sha256"]}。初始化来自旧7499完整检查点，累积更新9433；本次1934，不是同分布精确续训。','',
      'training/progress.json留有最后循环的epoch_complete=false旧标志，但曝光/更新已全完成；最终result.json与最终检查点收据明确完成。原sidecar未改写，不因此重跑。后台退出码0。完成回传后状态waiting_user_review；不自动补采、续训、改参数或操作服务器。','',
      '核对依据：'+ ' · '.join(link(ROOT/p,p) for p in ['postrun-collection-review.json','postrun-selection-review.json','postrun-task-coverage.json','postrun-prediction-review.json','postrun-verification.json','training/result.json','background/completion.json'])]
    (ROOT/'FINAL_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    parts=['<!doctype html><meta charset="utf-8"><title>True contact motion comparisons</title><style>body{font:16px sans-serif;max-width:1100px;margin:24px auto}img{max-width:100%}</style><h1>Saved true-contact motion comparisons</h1><p>Existing evaluated48 only; choose maximum absolute TRUE H16 rotation among positive-contact, obvious-motion cases. No new inference. Columns: truth, before, after one epoch, copy last.</p>']
    for m in pred['selected_media']:
        parts += [f'<h2>{html.escape(m["letter"])}</h2><p>{html.escape(m["sample_id"])}</p><img src="{Path(m["path"]).name}">']
    parts += ['<p><a href="../FINAL_REVIEW.md">Final review and limits</a> · <a href="../report/index.html">Original six-letter report</a></p>']
    (ROOT/'postrun-media/index.html').write_text('\n'.join(parts),encoding='utf-8')
    receipt=check_storage(ROOT,c)
    atomic_json(ROOT/'postrun-resource-receipt.json',receipt)
    paths=[ROOT/'runtime-config.json',ROOT/'freeze-receipt.json',ROOT/'FINAL_REVIEW.md',ROOT/'REPORT.md',
      ROOT/'collection/completion.json',ROOT/'collection/ledger.json',ROOT/'selection/train-index.json',
      ROOT/'selection/seen-val-index.json',ROOT/'selection/summary.json',ROOT/'training/result.json',
      ROOT/'training/source-recipe.json',ROOT/'training/last.pt',ROOT/'training/losses.jsonl',
      ROOT/'postrun-resource-receipt.json',ROOT/'postrun-verification.json']
    paths += list(ROOT.glob('postrun-*-review.json'))+list((ROOT/'postrun-media').glob('*'))
    paths += [p for folder in ('report','evaluation') for p in (ROOT/folder).rglob('*') if p.is_file()]
    paths=sorted(set(paths))
    atomic_json(ROOT/'postrun-final-manifest.json',dict(run_id=c['run_id'],runtime_config_sha256=sha(ROOT/'runtime-config.json'),
        at_utc=utc(),saved_only=True,model_calls=0,physics_steps=0,
        raw_collection_audit='postrun-collection-review.json; all1536 raw/meta hashes verified independently',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths if p.is_file()]))
    print({'review':str(ROOT/'FINAL_REVIEW.md'),'manifest_files':len(paths),'resource':receipt['shared_bytes_estimate']})

if __name__=='__main__':main()
