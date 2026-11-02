"""Render local retrospective, plot saved metrics, validate reused media only."""
from pathlib import Path
import json,re,html,os,datetime,subprocess
O=Path(__file__).resolve().parent;R=O.parents[1]
os.environ['MPLCONFIGDIR']=str(O/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

metrics=json.loads((O/'INDEPENDENT_METRIC_AUDIT.json').read_text())
raw=json.loads((O/'INDEPENDENT_RAW_AUDIT.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
for label in 'ABCD':
    hs=[1,4,8,16]
    ax[0].plot(hs,[metrics[label][str(h)]['IoU']for h in hs],marker='o',label=label)
ax[0].set(title='Same 48 seen-val families: saved AR predictions',xlabel='Horizon (steps at 10 Hz)',ylabel='Mean gray IoU',xticks=[1,4,8,16],ylim=(.55,1.01))
ax[0].legend(ncols=4);ax[0].grid(alpha=.25)
for row in raw:
    if row['letter']=='B':ax[1].step(range(1,17),row['holes_by_H1_to16'],where='mid',label=row['label'],alpha=.8)
ax[1].plot(range(1,17),[2]*16,'k--',label='GT')
ax[1].set(title='Fixed B case: AC-B-00-08:a19 (seen-val)',xlabel='Horizon (steps at 10 Hz)',ylabel='Visible holes (frozen palette)',xticks=[1,4,8,12,13,16],yticks=[0,1,2],ylim=(-.1,2.25))
ax[1].legend();ax[1].grid(alpha=.25)
fig.savefig(O/'saved-metric-comparison.png',dpi=150);plt.close(fig)

sources=json.loads((O/'SOURCES.json').read_text())
text=(O/'REPORT.txt').read_text(encoding='utf-8')
text=re.sub(r'\.\./(?:region-weighted-cd-20261101)/[^\s]+',lambda m:str((O/m.group()).resolve()).replace('\\','/'),text)
appendix='\n'.join(f'[{k}] {v["path"].replace(chr(92),"/")}' for k,v in sources.items())
if '<!-- SOURCES_APPENDIX -->'in text:text=text.replace('<!-- SOURCES_APPENDIX -->',appendix)
(O/'REPORT.txt').write_text(text,encoding='utf-8')

def inline(s):
    s=html.escape(s)
    s=re.sub(r'\[(E\d{2})\]',lambda m:f'<a class="ev" href="{Path(sources[m[1]]["path"]).as_uri()}" title="{html.escape(sources[m[1]]["path"],quote=True)}">[{m[1]}]</a>'if m[1]in sources else m[0],s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',s)
    return s

lines=text.splitlines();body=[];toc=[];i=0;code=False;paragraph=[];heading_count=0
def flush():
    if paragraph:body.append('<p>'+inline('\n'.join(paragraph))+'</p>');paragraph.clear()
media=[]
while i<len(lines):
    line=lines[i]
    if line.startswith('```'):
        flush();buf=[];i+=1
        while i<len(lines)and not lines[i].startswith('```'):buf.append(lines[i]);i+=1
        body.append('<pre>'+html.escape('\n'.join(buf))+'</pre>');i+=1;continue
    if line.startswith('#'):
        flush();level=len(line)-len(line.lstrip('#'));title=line[level:].strip();anchor='section-'+str(heading_count);heading_count+=1
        if level<=2:toc.append((anchor,title))
        if title.startswith('06｜'):body.append('<figure><img src="saved-metric-comparison.png" alt="独立重聚合指标与B孔数"/><figcaption>本次CPU重聚合：左为同48个seen-val家族的保存AR指标；右为固定B验证例，GT与C/D原始数组的可见孔数。没有新模型预测。</figcaption></figure>')
        body.append(f'<h{min(level,4)} id="{anchor}">{inline(title)}</h{min(level,4)}>');i+=1;continue
    if line.startswith('|'):
        flush();table=[]
        while i<len(lines)and lines[i].startswith('|'):
            cells=[c.strip()for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c or ' ')for c in cells):table.append(cells)
            i+=1
        body.append('<div class="table-scroll"><table>')
        for j,row in enumerate(table):
            cell='th'if j==0 else'td';body.append('<tr>'+''.join(f'<{cell}>{inline(c)}</{cell}>'for c in row)+'</tr>')
        body.append('</table></div>');continue
    if not line.strip():flush();i+=1;continue
    m=re.search(r'([A-Z]:/[^\n]+\.(?:png|mp4))$',line)
    if m:
        flush();p=Path(m[1]);caption=line[:m.start()].rstrip('：:');media.append(p)
        try:uri=os.path.relpath(p,O).replace('\\','/')
        except ValueError:uri=p.as_uri()
        if p.suffix=='.png':body.append(f'<figure><img src="{html.escape(uri,quote=True)}" alt="{html.escape(caption,quote=True)}" loading="lazy"/><figcaption>{inline(caption)}。原保存图；正式板为seen-val，GT/E2/A/B/C/D，H1/4/8/16。RH板另依原标题，RH每个时刻重新给真历史。</figcaption></figure>')
        else:body.append(f'<figure><video controls preload="none" src="{html.escape(uri,quote=True)}"></video><figcaption>{inline(caption)}；原保存10fps视频，无插帧或速度修改。</figcaption></figure>')
        i+=1;continue
    # Turn the source appendix into directly clickable absolute source entries.
    m=re.fullmatch(r'\[(E\d{2})\] ([A-Z]:/.*)',line)
    if m:
        flush();body.append(f'<p class="source">[{m[1]}] <a href="{Path(m[2]).as_uri()}">{html.escape(m[2])}</a></p>');i+=1;continue
    if line.startswith('- '):
        flush();items=[]
        while i<len(lines)and lines[i].startswith('- '):items.append(lines[i][2:]);i+=1
        body.append('<ul>'+''.join('<li>'+inline(x)+'</li>'for x in items)+'</ul>');continue
    paragraph.append(line);i+=1
flush()
nav=''.join(f'<a href="#{a}">{html.escape(t)}</a>'for a,t in toc)
page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Tdyssey 世界模型全面复盘</title><style>
*{box-sizing:border-box}body{margin:0;color:#172635;background:#f2f5f7;font:16px/1.85 "Microsoft YaHei","Segoe UI",sans-serif}nav{position:fixed;inset:0 auto 0 0;width:255px;overflow:auto;padding:28px 20px;background:#172635;color:white}nav b{font-size:21px}nav a{display:block;color:#d7e7ee;text-decoration:none;font-size:13px;padding:6px 0}main{margin-left:255px;padding:28px 5vw 80px;max-width:1510px;background:#fff;min-height:100vh}h1{font-size:32px;line-height:1.4}h2{margin-top:54px;padding-top:18px;border-top:2px solid #d4e5ed;font-size:24px;scroll-margin-top:20px}h3{font-size:19px;margin-top:30px}p{margin:14px 0;white-space:pre-line;overflow-wrap:anywhere}a{color:#075f87}.ev{font-size:12px;text-decoration:none}pre{background:#eef3f6;padding:18px;overflow:auto;line-height:1.65;font-size:13px}.table-scroll{overflow:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.7;min-width:660px}th,td{border:1px solid #d8e0e6;padding:10px 12px;vertical-align:top}th{background:#eaf2f7}tr:nth-child(2n){background:#f8fafb}figure{margin:28px 0;padding:14px;border:1px solid #dbe4ea;background:#fbfcfd}img,video{max-width:100%;height:auto;display:block;margin:auto}figcaption{font-size:13px;color:#4b6372;margin-top:10px}.source{font-size:12px;line-height:1.6}.banner{background:#eaf4f0;border-left:4px solid #39856b;padding:14px 18px;font-size:14px}.toplinks{font-size:14px}@media(max-width:1000px){nav{position:static;width:auto;max-height:270px}main{margin:0;padding:20px}h1{font-size:27px}}@media print{nav{display:none}main{margin:0;padding:0}h2{break-before:page}figure,table{break-inside:avoid}video{display:none}}
</style></head><body><nav><b>Tdyssey 复盘</b>'''+nav+'''</nav><main><div class="banner">独立只读复盘。运行状态固定截至2026-11-01 23:00:04新加坡时间；没有启动模型、训练、仿真或采集。后台任务继续由原负责人回传，本页不自动刷新。</div><p class="toplinks"><a href="REPORT.txt">纯文本完整报告</a> · <a href="DATA_LEDGER.json">数据总账</a> · <a href="EXPERIMENT_LEDGER.json">实验总账</a> · <a href="CURRENT_SNAPSHOT.json">状态快照</a></p>'''+''.join(body)+'</main></body></html>'
(O/'REPORT.html').write_text(page,encoding='utf-8')

# CPU decode two reused 16-frame clips. No new prediction/rendering, no media copies.
video_checks=[]
try:
    import imageio_ffmpeg
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    for p in [R/'artifacts/region-weighted-cd-20261101/report/formal-O-10fps.mp4',R/'artifacts/region-weighted-cd-20261101/diagnosis/media/AC-B-00-08-a19-10fps.mp4']:
        proc=subprocess.run([exe,'-hide_banner','-i',str(p),'-f','null','-'],capture_output=True,text=True,timeout=30)
        video_checks.append(dict(path=str(p),returncode=proc.returncode,summary=[s.strip()for s in proc.stderr.splitlines()if 'Duration:'in s or 'Video:'in s or 'frame='in s]))
except Exception as e:video_checks.append(dict(error=str(e)))

validation=dict(generated_at_singapore=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    report_characters=len(text),report_lines=len(text.splitlines()),source_count=len(sources),sources_missing=[v['path']for v in sources.values()if not v['exists']],
    reused_media_count=len(media),reused_media_missing=[str(p)for p in media if not p.exists()],tables=page.count('<table>'),
    artifact_files=[dict(path=str(p),bytes=p.stat().st_size)for p in O.iterdir()if p.is_file()],video_decode_checks=video_checks,
    actions=dict(new_model_calls=0,new_training_updates=0,new_physics_steps=0,new_collection=0,checkpoint_deserializations=0,package_installs=0,internet_requests=0),
    limitation='Saved metrics/index reaggregation and selected raw-only CPU topology. No claim of rerunning full model or physical experiments.')
(O/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in validation.items()if k!='artifact_files'},ensure_ascii=False,indent=2))
