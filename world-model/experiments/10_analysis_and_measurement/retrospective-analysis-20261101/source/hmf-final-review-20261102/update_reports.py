"""Update only independent report artifacts. Preserve original snapshot report."""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import html,json,re,datetime
O=Path(__file__).resolve().parent;P=O.parent
old=P/'REPORT_snapshot_20261101.txt'
if not old.exists():old.write_bytes((P/'REPORT.txt').read_bytes())
oldhtml=P/'REPORT_snapshot_20261101.html'
if not oldhtml.exists():oldhtml.write_bytes((P/'REPORT.html').read_bytes())
addon=(O/'REPORT.txt').read_text(encoding='utf-8');original=old.read_text(encoding='utf-8')
prefix='''# 最新状态与独立结论（2026-11-02）
高运动拟合已经于2026-11-01 23:53:45新加坡时间结束：3000成功更新、48000曝光、AMP0；无追加训练，等待用户审阅。旧23:00快照保留为历史证据，不再作为当前运行状态。

独立重聚合全部主评测、固定12六检查点、75案例末帧，并重算原O轴响应：局部单步与部分训练闭环确有学习收益；完整48轨迹仍未拟合，原验证C与旧三个失败例退步。固定训练子集后期仍有趋势，尤其C，但缺少中期验证曲线；不能据此自动续训。优先建议另批有限的中期验证/旧例评测和动作敏感性检验，不运行任何新实验。

下文先列完整新证据分析，再保留原全面复盘。独立补充单页在hmf-final-review-20261102/REPORT.html；旧版文本/HTML为REPORT_snapshot_20261101.*。DATA_LEDGER.json与EXPERIMENT_LEDGER.json已更新终态。

'''
sources=json.loads((P/'SOURCES.json').read_text())
extra='\n\n## 新增证据绝对路径\n'+'\n'.join(f'[{k}] {v["path"].replace(chr(92),"/")}'for k,v in sources.items()if int(k[1:])>=72)
combined=prefix+addon+'\n\n# 以下为原全面复盘历史版本（截至2026-11-01 23:00）\n\n'+original+extra
(P/'REPORT.txt').write_text(combined,encoding='utf-8')
style='''*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#182c39;font:16px/1.85 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1190px;margin:auto;padding:30px 44px 70px;background:white}h1{font-size:29px}h2{margin-top:44px;border-top:2px solid #dbe5eb;padding-top:18px;scroll-margin-top:20px}h3{font-size:19px;margin-top:28px}p{white-space:pre-line;overflow-wrap:anywhere}a{color:#05678d}table{border-collapse:collapse;width:100%;font-size:13px;min-width:650px}td,th{border:1px solid #d6e1e8;padding:9px 11px;vertical-align:top}th{background:#e9f1f6}.table{overflow:auto;margin:22px 0}pre{overflow:auto;background:#f0f4f6;padding:16px;font-size:13px}.banner{background:#edf6f0;border-left:4px solid #317b5a;padding:15px;margin-bottom:22px;font-size:14px}.toc{background:#f3f6f8;padding:20px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;font-size:13px}figure{margin:25px 0;border:1px solid #dce4e9;padding:10px;background:#fcfdfe}img{max-width:100%;height:auto;display:block;margin:auto}figcaption{font-size:13px;line-height:1.7;color:#48616f}details{margin:16px 0;padding:12px;border:1px solid #dae3e9}summary{cursor:pointer;font-weight:bold}.source{font-size:12px}@media(max-width:750px){main{padding:18px}.toc{grid-template-columns:1fr}}'''
def inline(s):
 s=html.escape(s)
 s=re.sub(r'(C:/[^\s<>]+?\.(?:json|md|txt|html|png|mp4))',lambda m:f'<a href="{Path(m[1]).as_uri()}">{m[1]}</a>',s)
 s=re.sub(r'\[(E\d{2})\]',lambda m:f'<a href="{Path(sources[m[1]]["path"]).as_uri()}">[{m[1]}]</a>'if m[1]in sources else m[0],s)
 return s
def render(text,title,gallery=False,main=False):
 lines=text.splitlines();out=[];toc=[];i=0;n=0;para=[]
 def flush():
  if para:out.append('<p>'+inline('\n'.join(para))+'</p>');para.clear()
 while i<len(lines):
  l=lines[i]
  if l.startswith('```'):
   flush();buf=[];i+=1
   while i<len(lines)and not lines[i].startswith('```'):buf.append(lines[i]);i+=1
   out.append('<pre>'+html.escape('\n'.join(buf))+'</pre>');i+=1;continue
  if l.startswith('#'):
   flush();level=min(4,len(l)-len(l.lstrip('#')));t=l.lstrip('#').strip();anchor='s'+str(n);n+=1
   out.append(f'<h{level} id="{anchor}">{inline(t)}</h{level}>')
   if level<=2:toc.append((anchor,t))
   i+=1;continue
  if l.startswith('|'):
   flush();rows=[]
   while i<len(lines)and lines[i].startswith('|'):
    c=[x.strip()for x in lines[i].strip('|').split('|')]
    if not all(re.fullmatch('[-: ]+',x or ' ')for x in c):rows.append(c)
    i+=1
   out.append('<div class="table"><table>')
   for j,row in enumerate(rows):tag='th'if j==0 else'td';out.append('<tr>'+''.join(f'<{tag}>{inline(x)}</{tag}>'for x in row)+'</tr>')
   out.append('</table></div>');continue
  if not l.strip():flush();i+=1;continue
  if l.startswith('- '):
   flush();items=[]
   while i<len(lines)and lines[i].startswith('- '):items.append(lines[i][2:]);i+=1
   out.append('<ul>'+''.join('<li>'+inline(x)+'</li>'for x in items)+'</ul>');continue
  para.append(l);i+=1
 flush()
 navigation='<div class="toc">'+''.join(f'<a href="#{a}">{html.escape(t)}</a>'for a,t in toc)+'</div>'
 galleryhtml=''
 if gallery:
  galleryhtml='<h2>独立重聚合曲线与全75例末帧核查</h2><figure><img src="independent-checkpoint-curves.png"><figcaption>同12训练案例、六个检查点；虚线RH为独立真实历史单步，实线AR为自身历史递推。不是完整48或验证曲线。</figcaption></figure>'
  for role,zh in [('train','48条训练轨迹'),('seen_val','24条原验证轨迹'),('original_failure','3条旧失败例')]:
   galleryhtml+='<details><summary>'+zh+'：展开全部H16末帧</summary>'
   for letter in 'OCB':galleryhtml+=f'<figure><img src="all-{role}-{letter}-H16.png" loading="lazy"><figcaption>{zh}／{letter}。每案例四列：GT、E2 AR16、final3000 AR16、final在t+16的真实历史单步。H16=1.6秒；最后一列不是16步闭环。显示转换不回灌，无修复。</figcaption></figure>'
   galleryhtml+='</details>'
 link='hmf-final-review-20261102/REPORT.html'if main else'../REPORT.html'
 top='<div class="banner">高运动拟合已结束，等待用户审阅。本页为只读独立分析，没有执行建议。<br><a href="'+link+'">'+('独立终态补充单页'if main else'返回更新后的完整复盘')+'</a> · <a href="REPORT.txt">本文TXT</a></div>'
 return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+style+'</style></head><body><main>'+top+navigation+galleryhtml+''.join(out)+'</main></body></html>'
(O/'REPORT.html').write_text(render(addon,'高运动拟合结束后的独立判断',gallery=True),encoding='utf-8')
(P/'REPORT.html').write_text(render(combined,'Tdyssey 全面复盘：含高运动拟合终态分析',main=True),encoding='utf-8')
class QA(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.anchors=[];self.images=[]
 def handle_starttag(self,t,a):
  a=dict(a)
  if'id'in a:self.ids.append(a['id'])
  if t=='a'and a.get('href','').startswith('#'):self.anchors.append(a['href'][1:])
  if t=='img':self.images.append(a['src'])
checks=[]
for p in [O/'REPORT.html',P/'REPORT.html']:
 q=QA();q.feed(p.read_text(encoding='utf-8'));assert not[k for k,v in Counter(q.ids).items()if v>1];assert set(q.anchors)<=set(q.ids);assert all((p.parent/n).exists()for n in q.images)
 checks.append(dict(path=str(p),headings=len(q.ids),images=len(q.images),anchors_valid=True))
for p in O.glob('*.json'):json.loads(p.read_text(encoding='utf-8'))
result=dict(updated_singapore=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),HTML_checks=checks,all75_cases_in_sheets=sum(x['cases']for x in json.loads((O/'SAVED_ONLY_AUDIT.json').read_text())['sheets']),
 archived_old_reports=[str(old),str(oldhtml)],model_calls=0,training=0,physics=0,report_directory_bytes=sum(p.stat().st_size for p in P.rglob('*')if p.is_file()),configuration_changed=False,owner_commands_sent=0)
assert result['all75_cases_in_sheets']==75
(O/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
