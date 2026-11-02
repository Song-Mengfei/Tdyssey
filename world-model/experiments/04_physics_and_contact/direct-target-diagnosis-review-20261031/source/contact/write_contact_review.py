"""Format saved-only contact findings and a non-executable verification proposal."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
SOURCE=REPO/'artifacts/direct-target-acceptance-20261031'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    d=read(HERE/'CONTACT_DIAGNOSIS.json');plan=read(SOURCE/'plan.json')
    d['normal_velocity_timing']='Before=detection/pre_solve relative velocity reconstructed from PD-updated integrator-pre velocities and collision post-integration pose; not immediate solver-pre-impulse. Dynamic velocity integration occurs after pre_solve and damping0 zeros its damping contribution. After=actual post_solve relative velocity at same collision pose; no later position integration is present in that step.'
    write(HERE/'CONTACT_DIAGNOSIS.json',d)
    input_manifest=read(HERE/'INPUT_MANIFEST.json');input_manifest['script_sha256']=sha(HERE/'analyze_saved_contact.py');write(HERE/'INPUT_MANIFEST.json',input_manifest)
    first=[]
    with (HERE/'first-violations.csv').open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            for k in ('violation_pairs','depths','target','native_depths','native_pre_depths','pre_event_counts','post_event_counts'):
                r[k]=json.loads(r[k])
            for k in ('control','substep'):r[k]=int(r[k])
            for k in ('max_depth','time_seconds'):r[k]=float(r[k])
            first.append(r)
    byid={x['attempt_id']:x for x in first}
    # Preserve the root-reviewed v2 table; formatting this report must never
    # regenerate a discarded candidate set or silently alter budgets/hashes.
    proposal=read(HERE/'MINIMAL_VERIFICATION.review.json')
    assert proposal['kind']=='non-executable-minimal-contact-continuation-review.v2'
    assert proposal['review_only'] is True and proposal['execution_authorized'] is False
    assert proposal['candidate_attempt_count']==4
    assert proposal['candidate_total_control_cap']==35
    assert proposal['candidate_total_regular_substep_cap']==350
    assert [x['attempt_id'] for x in proposal['attempts']]==[
        'DT-O-01-0-b0','DT-T-00-0-b0','DT-L-05-0-b0','DT-C-06-1-b0']
    assert all(x['subsequent_original_controls']==4 for x in proposal['attempts'])
    rows=[]
    for dim in ('letter','scenario','mode'):
        values=sorted({x[dim] for x in d['attempts']})
        for value in values:
            group=[x for x in d['attempts'] if x[dim]==value]
            fail=[x for x in first if x[dim]==value]
            rows.append(dict(dimension=dim,value=value,attempt_denominator=len(group),
                independent_families=len({x['family_id'] for x in group}),complete=sum(x['outcome']=='complete' for x in group),
                terminated=len(fail),AO=sum('AO'in x['violation_pairs'] for x in fail),
                AW=sum('AW'in x['violation_pairs'] for x in fail),OW=sum('OW'in x['violation_pairs'] for x in fail),
                first_main_control3=sum(x['control']==3 for x in fail),denominator_type='all actual attempts in thisgroup; modes/scenes not matched causal controls'))
    for phase,n in sorted(Counter(x['phase'] for x in first).items()):
        ff=[x for x in first if x['phase']==phase]
        rows.append(dict(dimension='phase_of_first_violation',value=phase,attempt_denominator=120,
            independent_families=len({x['family_id'] for x in ff}),complete=None,terminated=n,
            AO=sum('AO'in x['violation_pairs'] for x in ff),AW=sum('AW'in x['violation_pairs'] for x in ff),
            OW=sum('OW'in x['violation_pairs'] for x in ff),first_main_control3=sum(x['control']==3 for x in ff),
            denominator_type='120actualattempts globally; phase counts are event counts, not survival-conditioned failure rates'))
    with (HERE/'termination-counts.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    findings='''保存接触数据诊断（只读，未运行物理或模型）

输入是唯一当前DTA120条。46完整、74提前终止，33992个已执行0.01s记录；原始数据、旧非法标记均未改。

一、首个违规的描述性事实
AO64、AW9、OW1；全部pusher中心region仍为outside；49条在control3（首个主段控制）即触发。
最大对深度min1.005882、P101.069848、P501.488232、P902.635481、max4.692251世界单位。
customAO/OW在首个违规处与nativepost_solve负contactdistance的深度一致到浮点误差；AW与nativepre_solve一致。
不是所有native接触距离缺失：9条AW都有pre_solve，全部没有post_solve/正冲量。根源须结合原生infinite-mass pair处理时序解释。
不能用“post_solve”名称推断两物体的位置已完全分离。

二、完整/截断轨迹分布必须保留不同分母
完整46条包含29440microsteps：AO自定义深度P90=.074393、max=.943770；OW与AW所有记录都是0。
截断74条只保存4552microsteps：AO P90=.263927、max4.692251；OW max1.037857，AW max2.192959。
这不是同初态比较；前者长静止尾部较多，后者在>1被删失，不能据此推导模式因果效应或自然恢复概率。
全部分层分布见CONTACT_DIAGNOSIS.json/groups：outcome/mode/scenario/letter，all/nativepost/nativepre/positive_depth分开。
固定模式72条中49终止23完整，移动模式48条中25终止23完整；固定组包含额外分支，不能视为随机配对。
AO首违规目标误差P10/P50/P90=28.494/111.264/238.617worldpx；目标变化12.689/116.154/241.027。
同一步推杆位移1.527/2.715/4.314worldpx，约物体几何直径1.00%/1.83%/2.80%；不是动作标签。

三、已观察到的恢复证据与硬缺口
采用仅用于离线描述的规则：一次深度>.1的excursion触及[.8,1]，之后在保存的合法记录返回<=.1。
共10段/9尝试/9家族：AO9段、OW1段，其中完整5条有6段、后来终止4条各有1段。
例：完整DT-T-02-1-b0在t=1.01s深度.9437699，t=1.17s返回<=.1，确实在已存原物理记录中恢复。
DT-O-05-0-b0的.9753002在.11s内返回<=.1，但该轨迹后来又触发终止，不能将其全程标成正常。
敏感性：近阈值下界.7/.8/.9、恢复上界.1/.2的计数另列。这些数字不是建议终止阈值，也未进入任何配置。
74个>1首违规全部在原记录终点；不存在其继续未来，所以无法证明其中任何一个“放行后能恢复”。
AO64个触发时全部有法向闭合速度；以碰撞检测时的位置、PD后integrator-pre速度重建pre_solve相对速度，57例post法向相对速度约0、3例仍闭合、4例分离。
这里before不是紧邻求解器冲量前的动态速度：原生时序先积分位置/检测/pre_solve，后更新动态速度，再迭代冲量/post_solve。
原space.damping=0在动态速度更新中产生0衰减因子，会把动态旧速度贡献清零；运动学推杆保持其PD速度。此事实不被本诊断更改。
它只说明碰撞时序中的速度响应，不能单独证明下一步位置恢复；post_solve后本步没有再积分位置。相对速度量使用真实CoG和角速度，未把T的原点误作CoG。

四、AW9的具体根因
6条是negative预先故意选择0/512角落；另3条是boundary。9条首违规时的绝对目标都在无重叠推杆中心区域[22,489]外。
边界三例首违规目标分别T的x=512、L的y=501.294820、B的y=504.105396（后两者是朝终点512移动的中间请求）。
输入合法[0,512]不等于半径15圆形推杆中心可达到而不穿墙；不能把这9例全部归为“正常动态接触被误拦”。
此处只解释真实目标与墙几何；没有把动作缩到[22,489]，也不建议偷偷恢复动作限幅。

五、案例分类应按证据，不能把互有重叠的记录强制分成四个互斥分母
测量/坐标明确错误：本分项尚未找到；几何重建代理进一步核验，不能仅因1px欠校准就称几何计算错。
已观察正常接触恢复：上述10个亚阈值局部段，有明确前后观测；不代表74个>1终止点的反事实。
明确不可达目标/墙侵入驱动：AW9，包含6个全域negative压力请求；动作域仍合法，目标物理可达性另回事。
后续截断未知：AO64+OW1的>1点；哪怕已有正冲量或post法向速度归零，也仍未知下一步恢复/恶化。
保存的所有74中心outside，不存在记录直接证明推杆中心穿入实体/封闭孔；也不能据此证明圆盘无挤压或不会穿越。

六、最小验证仅为待批提案
MINIMAL_VERIFICATION.review.json是当前唯一待批四例方案：最接近1的AO、首个AO、最大AO、唯一OW，4个独立家族。
已存DT-T-02-1-b0亚阈值恢复参照只作离线证据，新增尝试和物理调用均0，不纳入新执行表。
若人类以后批准，须原环境完整重建+全公共microstep前缀匹配，不借publicsnapshot延续；目标按原冻结序列原样执行。
每例重建并核对完整原始前缀，继续首触发所在控制的剩余子步，再观察4个后续原冻结输入控制。
按四例原触发位置分别最多8/8/8/11个总控制，合计35controls/350regularsubsteps，已包含重建前缀与失败控制剩余子步；初始化实际另记。
首触发点后的约0.4–0.5s短观察只为了检测早期恢复，是有界待批候选，不是最优保证；仍未恢复就保留删失，不自动加时、重试或改种子。
仅把原>1即时终止改为would_stop观测标记这一点须明确批准；nonfinite/中心进入实体或孔仍立即停，无物理修正。
所有输出diagnostic-only/training_eligible=false，不改老标签、不运行训练。不重复AW明显不可达的继续试验来浪费预算。
该四例清单已与root最终提案一致，当前执行授权仍为false，新增物理/模型/优化器调用均0；任何执行均须人类明确批准。
'''
    (HERE/'FINDINGS.txt').write_text(findings,encoding='utf-8')
    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='OUTPUT_MANIFEST.json']
    assert sum(x['bytes'] for x in files)<8*1024**2
    write(HERE/'OUTPUT_MANIFEST.json',dict(files=files,new_physics_calls=0,model_calls=0,optimizer_updates=0))
    print(json.dumps(dict(proposed_attempts=4,proposed_controls=proposal['candidate_total_control_cap'],proposed_substeps=proposal['candidate_total_regular_substep_cap'],execution_authorized=False,output_bytes=sum(x['bytes'] for x in files),new_physics_calls=0)))

if __name__=='__main__':main()
