"""Build a saved-evidence-only overview. No simulator, model or torch imports."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/original-physics-contact-pilot-20261030'
LETTERS = tuple('TLOCXB')
MODEL_SHA = '23cc0f847fe585f894a7528bd5db1ea3c8541228c6de4396c82f11ce373ce5e7'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def esc(value):
    return html.escape(str(value))


def number(value, digits=3):
    return '—' if value is None else f'{value:,.{digits}f}'


class Evidence:
    def __init__(self, output):
        self.output = output
        self.inputs = {}
        self.errors = []

    def read(self, relative):
        path = self.output / relative
        if not path.is_file():
            return None
        data = path.read_bytes()
        self.inputs[relative] = dict(bytes=len(data), sha256=sha(data))
        try:
            return json.loads(data.decode('utf-8-sig'))
        except (UnicodeError, json.JSONDecodeError) as error:
            self.errors.append(dict(path=relative, error=str(error)))
            return None

    def exists(self, relative):
        return (self.output / relative).is_file()

    def link(self, relative, label):
        return (f'<a href="{esc(relative)}">{esc(label)}</a>' if self.exists(relative) or relative in ('summary.json','index.html','REPORT.md')
                else f'<span class="pending">{esc(label)}（待生成）</span>')

    def mdlink(self, relative, label):
        return (f'[{label}]({relative})' if self.exists(relative) or relative in ('summary.json','index.html','REPORT.md')
                else label+'（待生成）')


def table(headers, rows, *, class_name=''):
    return (f'<div class="table-scroll"><table class="{class_name}"><thead><tr>' +
            ''.join(f'<th>{esc(header)}</th>' for header in headers) +
            '</tr></thead><tbody>' + ''.join(
                '<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>'
                for row in rows) + '</tbody></table></div>')


def version_key(path):
    match = re.fullmatch(r'v(\d+)', path.name)
    return int(match[1]) if match else -1


def protocol_fields(source_path):
    """Read declared literal updates without importing/executing physics code."""
    if not source_path.is_file():
        return {}
    tree = ast.parse(source_path.read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'PROTOCOL' and node.func.attr == 'update'):
            result = {}
            for item in node.keywords:
                try:
                    result[item.arg] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    pass
            return result
    return {}


def aggregate_cases(rows):
    output = []
    for letter in LETTERS:
        selected = [row for row in rows if row['letter'] == letter]
        if not selected:
            continue
        actual_seconds = sum(row.get('observed_physics_seconds', 0) for row in selected)
        intervention = sum(row.get('intervention_seconds', 0) for row in selected)
        penetrations = [max(row['max_penetrations_wall_agentobject_agentwall'][i]
                            for row in selected) for i in range(3)]
        output.append(dict(letter=letter, cases=len(selected),
            safe_cases=sum(bool(row['geometry_safe']) for row in selected),
            attempted_controls=sum(row.get('attempted_controls', row['completed_controls']) for row in selected),
            complete_controls=sum(row['completed_controls'] for row in selected),
            max_penetration_OW_AO_AW=penetrations,
            translation_cases_ge2imagepx=sum(row['cumulative_translation_image'] >= 2 for row in selected),
            rotation_cases_ge5deg=sum(row['cumulative_rotation_deg'] >= 5 for row in selected),
            max_cumulative_translation_image_px=max(row['cumulative_translation_image'] for row in selected),
            max_cumulative_rotation_deg=max(row['cumulative_rotation_deg'] for row in selected),
            intervention_seconds=intervention, observed_seconds=actual_seconds,
            intervention_fraction=intervention/actual_seconds if actual_seconds else None,
            max_final_object_speed=max(row['final_object_speed'] for row in selected),
            max_final_omega_abs=max(abs(row['final_omega']) for row in selected)))
    return output


def get_regressions(evidence):
    root = evidence.output/'physics-regression'
    attempts = []
    if not root.exists():
        return attempts
    for directory in sorted(root.iterdir(), key=version_key):
        if not directory.is_dir() or version_key(directory) < 0:
            continue
        prefix = f'physics-regression/{directory.name}'
        started = evidence.read(prefix+'/started.json')
        if started is None:
            continue
        completed = evidence.read(prefix+'/summary.json')
        progress = evidence.read(prefix+'/progress.json') if completed is None else None
        failed = evidence.read(prefix+'/FAILED.json')
        protocol = protocol_fields(directory/'candidate-source.py.txt')
        if completed is not None:
            status = 'passed' if completed.get('physics_gate_passed') else 'failed'
            rows = completed.get('candidates', [])
            cases = len(rows)
            failed_cases = sum(not row['geometry_safe'] for row in rows)
            cost = completed.get('cost', {})
        else:
            status = 'failed' if failed else 'pending'
            rows = []
            cases = (progress or {}).get('completed_cases', 0)
            failed_cases = None
            cost = (failed or progress or {}).get('cost', {})
        attempts.append(dict(attempt=directory.name, protocol_name=protocol.get('name'),
            protocol_sha256=started.get('candidate_protocol_sha256'),
            source_sha256=started.get('candidate_source_sha256'), status=status,
            completed_cases=cases, failed_cases=failed_cases, cost=cost,
            wall_seconds=(completed or {}).get('wall_seconds'),
            summary_path=prefix+'/summary.json' if completed else prefix+'/progress.json',
            completed=completed, protocol=protocol))
    return attempts


def collection_counts(summary):
    attempts = (summary or {}).get('attempts', [])
    counts = defaultdict(lambda: dict(attempts=0, complete=0, valid_controls=0, actual_controls=0))
    phases = defaultdict(lambda: dict(attempts=0, complete=0, valid_controls=0, actual_controls=0))
    for row in attempts:
        keys = (row['phase'], row['letter'], row['split'])
        for counter in (counts[keys], phases[row['phase']]):
            counter['attempts'] += 1
            counter['complete'] += bool(row['complete'])
            counter['valid_controls'] += row.get('valid_controls', 0)
            counter['actual_controls'] += row.get('actual_controls', 0)
    return [dict(phase=k[0], letter=k[1], split=k[2], **value) for k, value in counts.items()], dict(phases)


def build(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ev = Evidence(output)
    generated = datetime.now(timezone.utc)
    sg_time = generated.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S SGT')
    regressions = get_regressions(ev)
    latest = regressions[-1] if regressions else None
    cases = (latest or {}).get('completed') or {}
    by_letter = aggregate_cases(cases.get('candidates', []))
    original_b = ev.read('physics-design/B-original-replay-final.json')
    b_versions = []
    for path in (output/'physics-design').glob('B-v*-replay-final.json'):
        match=re.fullmatch(r'B-v(\d+)-replay-final.json',path.name)
        if match:
            b_versions.append(int(match[1]))
    b_version=max(b_versions) if b_versions else None
    b_prefix=f'physics-design/B-v{b_version}' if b_version is not None else None
    repaired_b = ev.read(b_prefix+'-replay-final.json') if b_prefix else None
    video_proof = ev.read(b_prefix+'-video-proof.json') if b_prefix else None
    contact_proof = ev.read('physics-design/contact-proof/exact-frame-proof.json')
    corner_comparisons = ev.read('physics-design/B-corner-paired-final/summary.json') or []
    actual_browser = ev.read('ui-validation/actual-browser-check.json')
    after_browser_health = ev.read('ui-validation/after-browser-health.json')
    deployment = ev.read('ui-validation/deployment.json')
    audit_path=f'physics-regression/audit/AUDIT-{latest["attempt"]}.json' if latest else None
    latest_audit = ev.read(audit_path) if audit_path else None
    audit_summary = ({key:latest_audit.get(key) for key in ('passed','physics_gate_passed','protocol_sha256',
        'physics_source_sha256','cost_reconstructed','actual_physical_seconds','post_withdraw_summary','ordinary_pair_summary')}
        if latest_audit else None)
    design_cost = ev.read('physics-design/COST-SUMMARY.json')
    deletion = ev.read('cleanup/checkpoint-deletion-receipt.json')
    inventory = ev.read('cleanup/inventory.json')
    retention = ev.read('cleanup/retained-assets-verification.json')
    storage = ev.read('cleanup/storage-latest.json')
    collection = ev.read('collection/report/summary.json')
    plan = ev.read('collection/plan.json')
    gate = ev.read('physics-gate.json')
    completion = ev.read('completion.json')
    callback = ev.read('final-callback-result.json')
    counts, phases = collection_counts(collection)
    formal_attempts = phases.get('formal', {}).get('attempts', 0)
    formal_complete = phases.get('formal', {}).get('complete', 0)
    expected_formal = (plan or {}).get('primary_trajectories', 360)
    ui = []
    ui_root = output/'ui-validation'
    if ui_root.exists():
        for path in sorted(ui_root.glob('*.json')):
            value = ev.read('ui-validation/'+path.name)
            if isinstance(value, dict) and 'passed' in value:
                ui.append(dict(path='ui-validation/'+path.name, passed=value['passed'],
                    kind=value.get('kind'), tests=value.get('tests_run'),
                    actual_browser=value.get('actual_browser', value.get('browser_used', False)),
                    protocol_sha256=value.get('protocol_SHA256', value.get('protocol_sha256', value.get('actual_environment_protocol_SHA256'))),
                    cost=value.get('cost'), wall_seconds=value.get('wall_seconds'),
                    recorded_inference_calls=value.get('recorded_inference_calls'),
                    service_calls_after=value.get('model_calls_after'),
                    service_calls_not_attributed=value.get('model_calls_not_attributed_to_recorded_test')))
    latest_state = (latest or {}).get('status', 'pending')
    latest_name = (latest or {}).get('protocol_name') or ('物理版本见对应 started/source' if latest else '待开始')
    if latest_state == 'pending':
        headline = '物理回归进行中；尚不表示已通过采集前门槛'
    elif latest_state == 'failed':
        headline = '最新物理回归未通过；不应据此开始正式采集'
    elif formal_attempts < expected_formal:
        headline = f'物理回归已通过；正式采集已尝试 {formal_attempts}/{expected_formal} 条，尚未完成交付'
    elif callback and callback.get('sent'):
        headline = f'正式 {formal_attempts} 条尝试已报告；完整候选 {formal_complete} 条，等待用户审批'
    else:
        headline = f'正式 {formal_attempts} 条尝试已保存；完整候选 {formal_complete} 条，仍须负责人验收与实际回传'
    summary = dict(kind='original-contact-pilot-saved-overview.v1', generated_at_utc=generated.isoformat(),
        generator_source_sha256=sha(Path(__file__).read_bytes()),
        generated_at_singapore=sg_time, headline=headline, latest_regression_attempt=(latest or {}).get('attempt'),
        latest_physics_state=latest_state, latest_physics_protocol_name=latest_name,
        latest_physics_protocol_sha256=(latest or {}).get('protocol_sha256'),
        regressions=[{k: v for k, v in row.items() if k not in ('completed', 'protocol')} for row in regressions],
        latest_per_letter=by_letter, collection_phase_counts=phases, collection_split_counts=counts,
        formal_target=expected_formal, formal_attempts=formal_attempts, formal_complete_trajectories=formal_complete,
        B_original=original_b, B_repaired_version=b_version,B_repaired=repaired_b, B_video_proof=video_proof,
        main_contact_evidence=contact_proof,B_fixed_corner_comparisons=corner_comparisons,
        latest_independent_audit_path=audit_path,latest_independent_audit_summary=audit_summary,
        actual_browser_recording_context=actual_browser,UI_deployment=deployment,
        saved_after_browser_health=after_browser_health,
        cleanup_actual_deleted_files=len((deletion or {}).get('deleted', [])),
        cleanup_actual_deleted_logical_bytes=(deletion or {}).get('actual_deleted_logical_bytes'),
        UI_checks=ui, physics_gate_record=gate, owner_completion_record=completion,
        actual_callback_sent=bool((callback or {}).get('sent')), checkpoint_sha256=MODEL_SHA,
        training_authorized=False, optimizer_updates=0, generator_model_calls=0, generator_physics_steps=0,
        source_read_errors=ev.errors, evidence_inputs=ev.inputs,
        scope='Only reads saved evidence. The overview is not a physics gate, browser receipt, collection execution or training approval.')

    sections = []
    sections.append('<section><h2>当前阶段</h2><p class="notice">'+esc(headline)+'</p>'
        '<p>模型仍是旧动态推杆／0.1秒速度半衰期训练的 frozen7499；环境修复不代表模型已经适配。'
        '<strong>本阶段训练更新为0；图像＋请求动作→图像路线不变。</strong>完成人工验收与实际回传后等待用户审批，不能自动续训。</p>'
        '<p>本页只读取已保存证据，不模拟、不推理、不重新生成物理标签。更新时间 '+esc(sg_time)+'</p>'
        '<p>'+ev.link('PARENT_HANDOFF.txt','完整授权与停止条件')+' · '+ev.link('summary.json','本页机器摘要')+' · '
        +ev.link('collection/report/index.html','采集幅度、接触分层、覆盖图与实时视频')+' · '+ev.link('cleanup/REPORT.md','清理结果')+'</p></section>')
    status_label = {'pending':'进行中','passed':'通过固定回归','failed':'未通过'}
    regression_rows = []
    for row in regressions:
        cost = row['cost']
        regression_rows.append([ev.link(row['summary_path'], row['attempt']), esc(row['protocol_name'] or 'v3（执行身份见归档）'),
            esc(status_label[row['status']]), f'{row["completed_cases"]}/96', esc(row['failed_cases'] if row['failed_cases'] is not None else '待完成'),
            f'{cost.get("attempted_controls",0)}/{cost.get("completed_controls",0)}', esc(cost.get('microsteps',0)), number(row['wall_seconds'])])
    active_protocol=(latest or {}).get('protocol',{})
    integration_label=(f'冻结声明：solver={active_protocol.get("solver_iterations","见源")}, '
        f'physics_hz={active_protocol.get("physics_hz","见源")}, '
        f'max_microstep_dt={active_protocol.get("max_microstep_dt","见源")}s。')
    sections.append('<section id="physics"><h2>物理原因、修复与固定回归</h2>'
        '<p>无限质量的运动学推杆可以请求与墙角约束不相容的运动。原实现已检测到接触，仍被强制推进挤入实体与封闭孔；'
        '不能解释成仅漏了一次碰撞，也不能只丢坏帧就称修复成功。</p>'
        '<p>保留 KINEMATIC 推杆、DYNAMIC 物体、Space.damping=0、原几何／质量／惯量／摩擦／墙／渲染和10Hz绝对目标动作。'
        '新协议明确改变数值积分和接触执行：100Hz原PD请求，solver由10到50；真实多接触面与墙支撑点的单侧速度约束，'
        '限制不可行法向推进并保留可行切向；不增加原请求速度范数，不回写物体位姿或人工清速度。'
        '具体积分与执行规则以最新attempt冻结的源码和协议为准，不能把旧版本设置冒充最终设置。'+esc(integration_label)+'</p>'
        '<p>事前合法性阈值：最大合法数值重叠1.0世界像素（0.1875图像像素）；'
        '&gt;2世界像素属于实质重叠；杆中心入实体或由外部进入封闭孔无论深度均非法。阈值未为结果放宽。'
        '最新名义接触压缩为 '+esc(active_protocol.get('contact_compression_world_px','见冻结源码'))+' 世界像素；'
        '原型使用过0.25，若后续改小会独立版本化，不把它与合法重叠阈值混为一谈。'
        '积分设计的点位移目标不冒充每条任意命令的无条件保证。</p>'
        +table(['执行attempt','候选物理','状态','案例','失败例','控制尝试/完成','实际子步','运行墙钟秒'],regression_rows)+
        '<p>每轮同16场景×6字母：自由接触、偏心、凹角、四墙、四角和快速长目标；每例24控制，包括撤杆后继续推进。'
        '表中控制成本还含24个原始普通动作匹配分支。失败版本保留，不与最终候选合并成一次成功。'
        'v1表中的初版完成数／干预统计缺陷有独立只读审计，不能直接把非空字典当作100%干预。'
        +ev.link('physics-regression/audit/AUDIT-v1.md','首次失败回归的真实分母审计')+' · '
        +ev.link('physics-design/DESIGN-RESULT.md','设计与所有原型失败')+' · '+ev.link('physics-design/focused-CPU-checks.json','保存的纯CPU聚合／身份检查')+'</p>'
        '<p>'+ (ev.link(audit_path,'最新完整回归独立只读审计') if audit_path else '独立审计待生成') +
        ('：passed='+esc(latest_audit.get('passed'))+'，只读真实保存子步和几何，未新增物理／模型。' if latest_audit else '（尚未生成）') +'</p>'
        '<details><summary>最新attempt实际冻结的协议差异（直接读取源码字面量）</summary><pre>'+esc(json.dumps(active_protocol,ensure_ascii=False,indent=2))+'</pre></details></section>')
    if by_letter:
        motion_rows = [[esc(row['letter']), f'{row["safe_cases"]}/{row["cases"]}',
            '/'.join(number(x) for x in row['max_penetration_OW_AO_AW']),
            f'{row["translation_cases_ge2imagepx"]}/{row["cases"]}', f'{row["rotation_cases_ge5deg"]}/{row["cases"]}',
            number(row['max_cumulative_translation_image_px']),number(row['max_cumulative_rotation_deg']),
            number(100*row['intervention_fraction'],2)+'%' if row['intervention_fraction'] is not None else '—'] for row in by_letter]
        sections.append('<section><h2>最新完整回归：真实运动仍保留</h2><p>仅 '+esc(latest_name)+'，各字母固定16例。'
            '下面平移/旋转按案例的累计控制边界路程／绝对角变化判别，不能当作H1/H4运动分布；'
            '正式数据的微步累计与净运动分布另见采集报告。穿透顺序为物体墙OW／杆物体AO／杆墙AW，单位世界像素。</p>'
            +table(['字母','合法案例','最大OW/AO/AW','累计平移≥2图像px案例','累计旋转≥5°案例','最大累计平移图像px','最大累计旋转°','真实时间干预率'],motion_rows)+
            '<p>“withdraw/hold”是请求阶段名，不是实际脱离标签；仍有AO或OW接触时不能把运动全算惯性。'
            '原始阻尼0不等于每个脱离瞬间严格零位移：积分残余与墙接触位置修正须从实际接触和保存子步判断。</p></section>')
        pairs = cases.get('paired_ordinary', [])
        sections.append('<section><h2>普通可行动作与原实现的配对偏差</h2><p>同初始构造、同请求目标，只有普通四方向自由场景；'
            '数值求解变更后不再声称逐步等同原始。末位姿使用body origin距离（不是图像质心），角差未作形状对称约简；'
            '最大state差混合位置、速度与角量纲，只作诊断。</p>'+table(
            ['字母/场景','共同边界状态数','最大state绝对差','末body-origin距离worldpx','末角差°'],
            [[esc(row['letter']+'/'+row['scenario']),esc(row['common_states']),number(row['max_abs_state_difference']),
              number(row['final_body_origin_distance_world']),number(row['final_angle_difference_deg'])] for row in pairs])+ '</section>')

    # Contact at a plotted endpoint requires that exact microstep's arbiter and
    # positive impulse. Whole-control contact time is not substituted here.
    if contact_proof:
        exact_rows=contact_proof.get('selected_exact_contact_frames',[])
        exact_table=[[esc(row['branch']),number(row['physical_time_seconds'],3),
            f'{row["control"]}/{row["substep_index"]+1}',number(row['substep_dt'],6),
            f'{row["AO_post_events"]} / {number(row["AO_impulse_norm_sum"],3)}',
            f'{row["OW_post_events"]} / {number(row["OW_impulse_norm_sum"],3)}',
            number(row['signed_agent_object_gap_world'],6),number(row['object_wall_overlap_world'],6),
            esc(row['pusher_center_region'])] for row in exact_rows]
        free_rows=contact_proof.get('free_contact_frames',[])
        free_table=[[number(row['time_seconds'],3),number(row['AO_impulse'],3),esc(row['OW_events']),
            number(row['signed_gap_world'],6),number(row['H1_centroid_world'],3),
            number(row['H1_centroid_image'],3),number(row['H1_rotation_deg'],3)]for row in free_rows]
        exact_media=(' <img class="evidence-image" src="physics-design/contact-proof/B-TL-exact-contact-proof.png" '
            'alt="B左上角原始与v6的同一物理时间0.320及0.330秒精确接触子步证据">'
            if ev.exists('physics-design/contact-proof/B-TL-exact-contact-proof.png') else '')
        free_media=('<img class="evidence-image" src="physics-design/contact-proof/B-free-sustained-contact-motion.png" '
            'alt="B自由推动在持续正AO冲量下的实际位移">'
            if ev.exists('physics-design/contact-proof/B-free-sustained-contact-motion.png') else '')
        span=contact_proof.get('free_span_seconds')
        positive_span=contact_proof.get('free_positive_AO_span_seconds')
        sections.append('<section id="b-contact"><h2>主接触证据：精确子步接触，同时发生实际运动</h2>'
            '<p class="notice">旧B请求前缀控制31／32右图确实没有AO接触。它只说明新旧轨迹已经分离，'
            '<strong>不能作为接触修复通过图</strong>。下列主证据改用保存的真实接触子步；'
            '整控制累计接触秒数不能代替所示帧是否接触。</p>'
            '<p>固定B左上墙角，t=0.320s与0.330s：原始与v6各自由相同初始／请求前缀发展而来，实际轨迹不同。'
            'v6这两个精确子步均有AO post-solve事件及正冲量J，同时有OW事件及正冲量；'
            '杆中心仍在实体外。AO signed gap仅约−0.009世界像素、OW重叠约0.032世界像素，'
            '原始分支同时间已明显穿实体。<strong>这些结论指该子步，不是用控制4的累计81.5ms倒推其末帧。</strong>'
            '同一控制内t=.340s没有AO，也已在完整0.01s网格CSV明确保留。</p>'+exact_media+
            table(['分支','精确物理秒','控制/1-based子步','子步dt秒','本子步AO事件数/J范数和','本子步OW事件数/J范数和',
                   'AO signed gap worldpx','OW重叠worldpx','杆中心区域'],exact_table)+
            '<p>AO=推杆物体，OW=物体墙。J按真实arbiter记录，未按接触点重复累计。'
            '前后子步dt不同，J只用于确证当帧接触及真实冲量，不能直接当作力度改善百分比。'
            '图像由保存状态作纯渲染，不重新推进，也未对齐／插值／修补像素。</p><p>'
            +ev.link('physics-design/contact-proof/exact-frame-proof.json','精确帧完整证明与输入hash')+' · '
            +ev.link('physics-design/contact-proof/selected-frame-metrics.csv','所选精确帧CSV')+' · '
            +ev.link('physics-design/contact-proof/TL-control4-all-0_01-grid.csv','控制4全部0.01秒采样点（含无AO帧）')+'</p>'
            '<h3>连续接触仍能推动物体</h3><p>B自由推动实际(.30,.60]秒内，正AO冲量时间 '+number(positive_span,3)+'/'+number(span,3)+
            '秒，对应600个真实0.0005秒子步；不是只挑四张接触照片推断连续性。'
            '所示H1真实质心位移约2.1–3.4图像像素／0.1秒，无OW，不能把“没有穿透”解释为推杆根本没推动。'
            '这是一组接触与运动例证，不是正式采集有效接触率或总体通过率。</p>'+free_media+
            table(['所示端点物理秒','本子步AO J','本子步OW事件','AO signed gap worldpx','H1质心位移worldpx','H1质心位移imagepx','H1角变°'],free_table)+
            '<p>H1是实际连续0.1秒净位移；同坐标图未对齐。完整600子步检查和原始输入hash见同一精确帧证明。</p></section>')
    else:
        sections.append('<section id="b-contact"><h2>主接触证据待生成</h2><p>不能将旧请求前缀31／32的轨迹分离图当作当帧接触通过证据，也不能用整控接触累计代替末帧接触。</p></section>')

    if corner_comparisons:
        corner_rows=[]
        corner_media=[]
        for row in corner_comparisons:
            name=row['name'];prefix='physics-design/B-corner-paired-final/'+name
            corner_rows.append([ev.link(prefix+'.json',name),esc(row['initial_state15_match']),esc(row['requested_actions_match']),
                '/'.join(number(v,4)for v in row['original_max_penetration']),
                '/'.join(number(v,4)for v in row['v6_max_penetration'])])
            if ev.exists(prefix+'-before-after-10Hz.mp4'):
                corner_media.append('<h3>'+esc(name)+'</h3><video controls preload="metadata" src="'+prefix+'-before-after-10Hz.mp4"></video>')
        sections.append('<section><h2>B固定墙角完整前后轨迹</h2><p>TL／TR配对使用相同完整初始state15及相同请求动作，'
            'v6原始数据直接复用最终96例回归，不重复模拟。两支轨迹随后不同，不能声称同一中间接触状态下只替换一步；'
            'TR或其他所选末帧不一定仍有AO，接触判断必须回到精确子步表。</p>'+table(
            ['案例','初始state完全相同','请求完全相同','原始最大OW/AO/AW worldpx','v6最大OW/AO/AW worldpx'],corner_rows)+''.join(corner_media)+
            '<p>真实10Hz保存RGB，无插帧／加速；25帧含t0对应2.4秒物理轨迹加末端一帧显示。'
            +ev.link('physics-design/B-corner-paired-final/summary.json','墙角配对与视频hash')+' · '
            +ev.link('physics-design/B-corner-paired-final/geometry-audit.json','配对几何只读审计')+'</p></section>')

    b_rows = []
    for label, row in [('原始完整前缀重放',original_b),(f'v{b_version}同初始／同请求',repaired_b)]:
        if row:
            b_rows.append([esc(label),f'{row["completed"]}/{row["attempted"]}',
                '/'.join(number(v,6) for v in row['max_penetration']),
                number(row.get('intervention_seconds',0)),esc(row.get('illegal_center_microsteps','见原始穿孔证据')),
                number(row.get('control_state15_max_diff_from_recorded'),9)])
    media = ''
    if repaired_b and ev.exists(b_prefix+'-before-after-10Hz.mp4'):
        media += '<video controls preload="metadata" src="'+b_prefix+'-before-after-10Hz.mp4"></video>'
    if repaired_b and ev.exists(b_prefix+'-before-after-control32.png'):
        media += '<img class="evidence-image" src="'+b_prefix+'-before-after-control32.png" alt="原始与所标版本在同请求控制32时各自的真实状态">'
    b_matches=bool(repaired_b and repaired_b.get('protocol_sha256')==(latest or {}).get('protocol_sha256'))
    b_scope=f'所示为保存的v{b_version}子实验；'+('与最新回归协议相同。' if b_matches else '与最新回归协议不相同，仅为历史版本证据。') if repaired_b else '尚无保存的修复分支B子实验。'
    sections.append('<section id="b-prefix"><h2>旧用户请求前缀：精确复现及轨迹分离</h2>'
        '<p>原始构造＋3个真实hold控制＋37个用户请求动作，原始重放的37个控制state15及370个post子步与用户raw最大差均0。'
        '同一完整前缀精确再现OW18.898176／AO20.641872世界像素重叠及中心入实体、入封闭孔。'
        '修复分支从相同构造和请求开始；轨迹因新执行规则从第一控起改变，不能冒称在原始control32相同中间状态原位修好。</p>'
        '<p class="notice">B-v6控制31／32的右图无AO，属于历史轨迹分离图，不能证明该时刻接触修复通过。'
        '3.7秒内的累计干预或接触秒数也不能改变该帧无接触的事实。主接触证明见上方精确子步图。</p><p>'+esc(b_scope)+' 子实验合法不等于所选帧在接触。</p>'
        +table(['分支','完成/请求控制','最大OW/AO/AW worldpx','干预秒（分母3.7s；非当帧接触）','非法中心子步','对用户raw控制state最大差'],b_rows)+
        '<details><summary>保留旧请求前缀视频／控制31–32图：非当帧接触通过证据</summary>'+media+'</details>'+
        '<p>视频为真实保存RGB的10fps／10Hz，无插帧或加速；含t0共38帧，对应3.7秒物理轨迹及末端一帧显示时间。'
        '旧电影是独立版本证据，不冒称最终物理验收。'
        +ev.link('physics-design/B-original-replay-final.json','原始前缀身份与精确重放')+' · '
        +(ev.link(b_prefix+'-replay-final.json',f'v{b_version}实际结果')+' · '+ev.link(b_prefix+'-video-proof.json','该版本视频回执') if b_prefix else '修复分支回执待生成')+'</p></section>')

    data_rows = [[esc(row['phase']),esc(row['letter']),esc(row['split']),esc(row['attempts']),esc(row['complete']),
                  esc(row['actual_controls']),esc(row['valid_controls'])] for row in counts]
    data_text = ('尚无保存的采集标签报告；计划不等于已执行。' if collection is None else
        f'已保存全部阶段 {collection["attempt_count"]} 条尝试，完整 {collection["complete_trajectories"]} 条；'
        f'实际控制 {collection["actual_controls"]}，有效 {collection["valid_controls"]}，'
        f'实际物理 {number(collection["actual_physics_seconds"])} 秒。校准、正式、补采分别计数。')
    sections.append('<section id="data"><h2>候选数据验收：计划与实际分开</h2><p>'+esc(data_text)+'</p>'
        '<p>正式目标每字母48 train-candidate＋12 seen-validation，共360独立家族、每条约64个10Hz控制。'
        '校准18＋最多一次18修订，有限补采最多每字母12；未执行的保留槽不能算数据。初始家族与split事先冻结，'
        '仅T/L/O/C/X/B；不混旧动态数据、组员原始数据或封存字母。采集依赖状态／几何的先验会披露，模型输入仍是图像和请求动作。</p>'
        +table(['阶段','字母','split','轨迹尝试','完整轨迹','实际控制','有效控制'],data_rows)+
        '<p>设计目标：接触且实际运动时间≥60%，有用无接触／静止约10–20%，边界相关约20–30%；'
        '不是过滤后虚报的通过线。H1严格0.1秒、H4严格0.4秒，实际面积质心用96/512换算，'
        '连续展开角变化、微步累计路程／绝对角位移和净量均保留；不跳帧拉长H1。'
        '全尝试、有效物理时间和最终候选窗口三种分母各自列明。</p><p>'
        +ev.link('collection/report/index.html','逐字母幅度直方图／分位数、接触阶段、边界、无效、实际弧段与实时视频')+' · '
        +ev.link('collection/report/summary.json','保存标签机器结果')+' · '+ev.link('collection/plan.json','冻结独立场景与配额')+' · '
        +ev.link('collection/DESIGN.md','指标／分母／有限补采规则')+'</p>'
        '<p>正式候选实际弧段覆盖与全部阶段覆盖分图；不能借校准或补采弧段冒充正式覆盖。'
        '随机完整视频使用固定选择seed；清楚平移／旋转／边界及弱运动／失败例的来源阶段与轨迹ID应以详细报告为准。'
        '缺少代表性运动时须报告不足，不能只挑最好片段或凭一个达标率建议训练。</p></section>')

    ui_rows = [[ev.link(row['path'],Path(row['path']).name),esc(row['passed']),esc(row['tests']),
                esc(row['actual_browser']),esc(row['protocol_sha256'] or '见原始回执'),esc(row['cost'] or '见原始回执')] for row in ui]
    browser_text='实际浏览器回执待完成。'
    if actual_browser:
        browser_text=('真实IAB录制／保存／下载／打开播放已'+('通过'if actual_browser.get('passed')else'未通过')+
            '，raw完整性及hash均以实际回执为准。本次不变录像包含 '+str(actual_browser.get('recorded_inference_calls'))+
            ' 次神经推理（B H1=1、C H4=4，5控制／0.5秒物理时间）；同期服务计数为 '+str(actual_browser.get('model_calls_after'))+
            '，其中另 '+str(actual_browser.get('model_calls_not_attributed_to_recorded_test'))+
            ' 次不归本录像测试，检测到用户同时操作／重置，不能把85次全部计为测试费用。'
            '录像约94.6秒是含暂停的真实UI墙钟，不是采集10Hz轨迹时长；完整模型秒数未暴露，不编造。')
    health_text=(f'存档验收时服务PID{after_browser_health.get("pid")}，模型与v6协议身份记录在after-browser-health；'
        '这是保存时快照，不实时推断用户当前场景。'if after_browser_health else '')
    sections.append('<section id="ui"><h2>实际试玩、录制和会话保存</h2><p>修改前已保存用户C场景、公开body/controller状态、模型历史和最近13批原始动作；'
        '旧ring更早已淘汰的472批没有伪造恢复。修改生效前是否有更新会话／活动录制，以负责人最新备份与实际浏览器回执为准。</p>'
        '<p>'+esc(browser_text)+' '+esc(health_text)+'</p><p>'+ev.link('ui-validation/actual-browser-check.json','真实IAB录像／播放／raw核验回执')+' · '+ev.link('ui-validation/deployment.json','部署与旧会话保存')+'</p>'
        +table(['回执','passed','检查数','真实浏览器','物理协议','实际成本（Spy不是神经模型）'],ui_rows)+
        '<p>CPU Spy接口通过不能代替真实浏览器、录像下载／打开／播放和同ID raw完整性验收。'
        '请求绝对目标保留为模型动作，实际推杆端点、无约束／执行速度、干预与接触时间单独记录；'
        '接触显示使用真实毫秒／100ms，不能硬套10子步分母。'
        'v5控制器的上次microstep ticks也是完整公开状态的一部分，导出恢复时要保留；旧协议／缺控制状态恢复必须先拒绝并保留当前会话。'
        '求解器内部接触缓存未序列化，从头前缀重放证明不等于任意接触中restore未来逐位一致。</p><p>'
        +ev.link('live-before/health.json','修改前模型／进程身份')+' · '+ev.link('live-before/session-export.json','修改前完整公开会话导出')+' · '
        +ev.link('live-before/saved-recent-clip.json','实际最近raw保存回执')+'</p></section>')

    removed = (deletion or {}).get('actual_deleted_logical_bytes')
    removed_count = len((deletion or {}).get('deleted', []))
    storage_text = ('最新大小快照：共享 '+number(storage['shared_bytes'],0)+'B，扣256MiB原子保留后余 '
                    +number(storage['headroom_after_atomic_reserve_bytes'],0)+'B；磁盘free '+number(storage['filesystem']['free'],0)+'B。'
                    if storage else '尚未刷新末尾大小快照，清理报告中有时间戳明确的历史占用。')
    sections.append('<section id="cleanup"><h2>已执行清理与保留核验</h2><p>实际删除 '+str(removed_count)+' 个退役权重，精确逻辑回收 '
        +number(removed,0)+'B'+(f'（{removed/1024**2:.2f}MiB）' if removed is not None else '')+'。'
        '逐文件Resolve-Path、无reparse祖先、大小与SHA先核对后LiteralPath删除；没有全量备份。'
        '旧best和last并非重复文件，而是已退役实验的不同历史选择；旧manifest不再表示所有大权重仍可加载，'
        '旧generic默认best路径已失效，报告结论／raw／费用仍保留。</p>'
        '<p>保留当前7499和6000完整检查点、唯一GPC／官方初始化、用户视频与B穿透raw、关键O/C/B/X对照、'
        '源码／配置／split／暴露／报告／指标／cost、封存测试、组员数据和当前启动／录制文件。'
        '旧TRAIN774个NPZ约58.4MB仍由完整loader/audit依赖，本轮未删除，未假称最小子集提取。'
        '保留核验曾逐字节hash3369个原NPZ；官方唯一策略完整读取校验，没有旧预期SHA时不冒称匹配。</p>'
        '<p>'+esc(storage_text)+'快照不包含其写出后新增的报告／回执。</p><p>'
        +ev.link('cleanup/REPORT.md','清理简报')+' · '+ev.link('cleanup/checkpoint-deletion-receipt.json','实际逐文件删除回执')+' · '
        +ev.link('cleanup/retained-assets-verification.json','保留资产真实核验')+' · '+ev.link('cleanup/storage-latest.json','最新大小快照')+'</p></section>')

    costs = []
    if design_cost:
        costs.append(['设计原型与B复现（含失败／warmup）',esc(design_cost['attempted_control_slots_including_warmup']),
            esc(design_cost['completed_controls_including_warmup']),esc(design_cost['known_substeps']),
            number(design_cost['measured_environment_run_wall_seconds']),ev.link('physics-design/COST-SUMMARY.json','详细计时边界')])
    for row in regressions:
        cost=row['cost']
        costs.append(['96例回归 '+esc(row['attempt'])+'／'+esc(row['status']),esc(cost.get('attempted_controls')),
            esc(cost.get('completed_controls')),esc(cost.get('microsteps')),number(row['wall_seconds']),ev.link(row['summary_path'],'原记录')])
    sections.append('<section id="cost"><h2>运行费用、失败和边界</h2>'+table(
        ['范围','实际控制尝试','完整控制','实际子步','墙钟秒','证据'],costs)+
        '<p>各scope分开，不将I/O扫描、图表／压缩／视频、CPU Spy或旧训练秒数当GPU模型费用。'
        '设计原始writer失败whole-tool4.0499113s与环境内部计时不同；首次可变dt爆炸、后来墙角失败、统计身份修正均保留。'
        '此页不重新累计可能范围重叠的总“GPU费用”。旧首轮训练及33.047秒失败费用仍在原阶段报告，不挪入本次。</p>'
        '<p>checkpoint保留身份 SHA256 <code>'+MODEL_SHA+'</code>。本页生成器模型调用／物理步／训练更新均0；'
        '实际试玩推理调用须从浏览器／service回执另列，环境修复和候选采集没有优化器更新权限。</p></section>')
    if ev.errors:
        sections.append('<section><h2>读取异常</h2><p>并发写入或损坏文件未作零值／通过处理，请负责人核对后重新生成。</p><pre>'+esc(json.dumps(ev.errors,ensure_ascii=False,indent=2))+'</pre></section>')
    sections.append('<section><h2>下一步与主动回传</h2><p>本页不作训练许可。负责人须读真实采集规模、完整分母、运动幅度、接触比例、'
        '弱例与失败视频，实际send_message_to_thread回传父聊天并保存真实回执，最终waiting_user_approval。'
        '父聊天仅整理展示，必须等待用户明确审批才可训练；不自动扩大采样或启动服务器任务。</p><p>'
        +ev.link('final-callback-result.json','实际回传回执')+' · '+ev.link('completion.json','负责人停止／完成记录')+'</p></section>')
    style = '''body{margin:0;background:#f4f6f9;color:#172a3a;font:16px/1.65 system-ui,Segoe UI,sans-serif}main{max-width:1250px;margin:auto;padding:28px}h1{line-height:1.25}h2{font-size:1.35rem}section{background:white;border:1px solid #dbe3ed;border-radius:12px;margin:20px 0;padding:24px}nav{display:flex;flex-wrap:wrap;gap:18px}a{color:#145a86}code,pre{word-break:break-all;white-space:pre-wrap}.notice{padding:12px 16px;border-left:5px solid #df951f;background:#fff5df}.pending{color:#866f56}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dbe3ed;text-align:left;vertical-align:top}th{background:#eef3f8}.table-scroll{overflow:auto}video,.evidence-image{display:block;width:100%;max-width:1050px;margin:16px auto}small{color:#5e7180}td{max-width:450px;word-break:break-word}@media(max-width:700px){main{padding:10px}section{padding:14px}}'''
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tdyssey 原始物理接触采样验收</title><style>'+style+'</style></head><body><main><h1>Tdyssey：运动学推杆接触修复与候选数据验收</h1><nav>'+''.join(
        f'<a href="#{anchor}">{label}</a>' for anchor,label in [('physics','物理与回归'),('b-contact','B精确接触证明'),('b-prefix','旧前缀／非接触图'),('data','真实数据'),('ui','试玩／录像'),('cleanup','清理'),('cost','费用／限制')])+ '</nav>'+''.join(sections)+'</main></body></html>'
    (output/'index.html').write_text(document,encoding='utf-8')
    md = '# Tdyssey 原始物理接触修复与候选采样验收\n\n'+headline+'。\n\n'
    md += ev.mdlink('index.html','浏览完整HTML报告')+' · '+ev.mdlink('collection/report/index.html','采集运动／接触／覆盖／视频')+' · '+ev.mdlink('cleanup/REPORT.md','清理报告')+' · '+ev.mdlink('summary.json','机器摘要')+'\n\n'
    md += f'生成时间：{sg_time}。这是已保存证据的只读汇总，不执行物理、推理或训练。\n\n'
    md += '当前最新回归：'+str((latest or {}).get('attempt'))+' / '+latest_name+' / '+latest_state+'；各版本失败与费用分别保留。\n\n'
    if contact_proof:
        md += ('**主接触证据已纠正：** 旧B-v6控制31／32右图没有AO，仅是轨迹分离，不能作接触修复通过图。'
            '现改用B左上角t=.320/.330秒精确子步：v6同帧AO正冲量＋OW正冲量、杆中心在实体外，'
            'AO重叠约.009世界像素、OW约.032；整控制累计接触不能替代当帧判断，.340秒无AO也保留。\n\n'
            +ev.mdlink('physics-design/contact-proof/B-TL-exact-contact-proof.png','精确接触前后图')+' · '
            +ev.mdlink('physics-design/contact-proof/exact-frame-proof.json','同子步事件／冲量／signed gap证明')+' · '
            +ev.mdlink('physics-design/contact-proof/selected-frame-metrics.csv','所选帧CSV')+'\n\n'
            '另有B自由推动(.30,.60]的600子步／0.3秒连续正AO冲量，实际H1位移2.1–3.4图像像素，不是以不接触冒充修复。'
            +ev.mdlink('physics-design/contact-proof/B-free-sustained-contact-motion.png','连续接触与真实运动图')+'。这些是定向例证，不是总体接触率。\n\n')
    if repaired_b:
        md += ('B从完整构造＋3hold＋37请求动作重放：原始分支对用户370子步最大状态差0，精确再现OW18.898176/AO20.641872世界像素重叠；'
               f'保存的v{b_version}分支同请求{repaired_b["completed"]}/{repaired_b["attempted"]}完整，最大OW/AO/AW='+ '/'.join(number(v,6) for v in repaired_b['max_penetration'])+'，非法中心子步'+str(repaired_b.get('illegal_center_microsteps','见原记录'))+'。'
               '轨迹从第一控起改变，不是同中间状态原位修复。\n\n'
               +b_scope+' 旧请求前缀所选31／32图无AO，保留为轨迹分离证据，不能当接触修复通过图。\n\n'
               +ev.mdlink(b_prefix+'-before-after-10Hz.mp4','历史请求前缀实际10Hz视频')+' · '+ev.mdlink(b_prefix+'-before-after-control32.png','控制32轨迹分离图（非接触通过）')+'\n\n')
    if actual_browser:
        md += browser_text+' '+health_text+'\n\n'
    md += f'正式目标{expected_formal}条；已尝试{formal_attempts}条，完整{formal_complete}条。校准／补采与正式分列；计划不等于执行，完整不等于可训练。\n\n'
    if removed is not None:
        md += f'实际清理{removed_count}个退役checkpoint，释放{removed:,}B（{removed/1024**2:.2f}MiB）；7499/6000/唯一初始化及关键证据保留。旧TRAIN仍依赖loader/audit，本轮未删。\n\n'
    md += ('物理保留KINEMATIC+damping0与原几何／质量／惯量／材料，改变积分／solver／不可行动作执行并版本化。'
           '7499模型仍是旧物理训练，属于跨物理预览；请求目标和实际端点／速度干预分别保存。公开控制状态含上次microstep ticks，内部solver缓存不序列化。\n\n'
           '**训练更新0。** 实际回传父聊天后等待用户审批，不自动训练、扩大采样或启动远端任务。\n')
    (output/'REPORT.md').write_text(md,encoding='utf-8')
    # Only emit links to files that existed when this snapshot was built. Pending
    # labels above deliberately have no link. Every active local link is checked.
    links = re.findall(r'(?:href|src)="([^"]+)"',document)
    missing=[]
    for target in links:
        if target.startswith('#') or '://' in target:
            continue
        if target == 'summary.json':
            continue  # Written just below, after recording this check.
        resolved=(output/target).resolve()
        if not resolved.is_relative_to(output) or not resolved.is_file():
            missing.append(target)
    summary['overview_local_link_check']=dict(passed=not missing,checked_local_links=sum(not t.startswith('#') and '://' not in t for t in links),missing=missing)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'overview-link-check.json').write_text(json.dumps(summary['overview_local_link_check'],indent=2)+'\n',encoding='utf-8')
    if missing:
        raise ValueError('Invalid active local links: '+str(missing))
    print(json.dumps(dict(headline=headline,latest_physics_state=latest_state,formal_attempts=formal_attempts,
        formal_complete=formal_complete,deleted_bytes=removed,local_links_passed=True,generator_model_calls=0,generator_physics_steps=0,optimizer_updates=0),ensure_ascii=False))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    build(parser.parse_args().output)
