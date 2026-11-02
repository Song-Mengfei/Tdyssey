"""Offline stdlib-only replay. No project imports, native Reader or payload scan."""
import collections
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).absolute().parents[2]
REVIEW = Path(__file__).absolute().parent
RUN = ROOT/'artifacts/gpc-checkpoint500-evaluation-20261026'
G = RUN/'guardian';L = RUN/'ledger'
ISSUES = [];INPUTS = {}


def check(condition, label, detail=None):
    if not condition:ISSUES.append(dict(check=label, detail=detail))


def raw(path):
    value = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
    return value


def js(path):return json.loads(raw(path))
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def iso(value, offset=0):return datetime.fromtimestamp(value,timezone(timedelta(hours=offset))).isoformat()
def envelope(status):
    return {k:status.get(k) for k in ('at_unix','at_monotonic','phase','model_instances','model_loads',
        'optimizer_updates','forecasts','euler_steps','eval_batches','cuda_allocated_bytes','cuda_reserved_bytes',
        'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes','cuda_sample_state','cuda_sample_at_monotonic')}


def replay_ledger(attempt):
    source=raw(L/'highwater.jsonl');lines=source.splitlines(keepends=True)
    rows=[json.loads(line) for line in lines];previous='0'*64;last_wall=last_mono=-math.inf
    counts=collections.Counter();per={r:collections.Counter() for r in ('frozen','adapted')}
    pending=None;active=None;max_active=0;released=[];loads=[];milestones=[];plans={r:[] for r in per}
    for seq,(line,row) in enumerate(zip(lines,rows),1):
        check(line.endswith(b'\n'),'ledger.complete_line',seq)
        check(row['sequence']==seq and row['previous_sha256']==previous,'ledger.sequence_chain',seq)
        previous=hashlib.sha256(line).hexdigest();kind=row['kind'];data=row['data'];counts[kind]+=1
        check(finite(row['at_unix']) and finite(row['at_monotonic']),'ledger.finite_time',seq)
        check(row['at_unix']>=last_wall and row['at_monotonic']>=last_mono,'ledger.monotone_time',seq)
        check(row['at_unix']<attempt['deadline_unix'] and row['at_monotonic']<attempt['deadline_monotonic'],'ledger.original_deadline',seq)
        last_wall,last_mono=row['at_unix'],row['at_monotonic']
        if kind=='created':check(seq==1,'ledger.created_first')
        elif kind=='reserve_model_load':
            check(pending is None and active is None,'ledger.serial_load_admission',seq)
            role=data['role'];check(role==('frozen','adapted')[len(loads)],'ledger.load_order',seq)
            if role=='adapted':check(released==['frozen'],'ledger.frozen_released_before_adapted')
            loads.append(role);active=role;max_active=max(max_active,1);pending=(seq,'model_load',role)
            milestones.append(dict(sequence=seq,kind=kind,role=role,at_monotonic=last_mono,at_unix=last_wall))
        elif kind in ('reserve_eval','reserve_forecast'):
            meta=data['metadata'];role=meta['role']
            check(pending is None and active==role,'ledger.operation_active_role',seq)
            if kind=='reserve_eval':
                check(data['batch_size']==1 and meta['example']==per[role]['reserve_eval'],'ledger.loss_order',seq)
                check(per[role]['reserve_forecast']==0,'ledger.loss_before_forecasts',seq)
                pending=(seq,'eval',role)
            else:
                check(data['actual_count']==1 and data['euler_per_image']==3,'ledger.three_euler',seq)
                check(per[role]['complete_eval']==16,'ledger.sixteen_loss_before_forecasts',seq)
                plans[role].append((*meta['case'],meta['mode'],meta['horizon']));pending=(seq,'forecast',role)
            per[role][kind]+=1
        elif kind.startswith('complete_'):
            wanted=kind.removeprefix('complete_')
            check(pending is not None and pending[0]==data['operation_id'] and pending[1]==wanted,'ledger.exact_completion',seq)
            if pending:per[pending[2]][kind]+=1
            if wanted=='model_load':milestones.append(dict(sequence=seq,kind=kind,role=active,at_monotonic=last_mono,at_unix=last_wall))
            pending=None
        elif kind=='model_released':
            check(pending is None and data['role']==active and data['cleanup'] is False,'ledger.successful_release',seq)
            check(per[active]['complete_eval']==16 and per[active]['complete_forecast']==1998,'ledger.complete_role_release',seq)
            released.append(active);milestones.append(dict(sequence=seq,kind=kind,role=active,at_monotonic=last_mono,at_unix=last_wall));active=None
        elif kind=='evaluation_finished':
            check(pending is None and active is None and released==['frozen','adapted'],'ledger.finished_quiescent',seq)
        else:check(False,'ledger.forbidden_event',dict(sequence=seq,kind=kind))
    check(len(rows)==8064,'ledger.exact_8064',len(rows));check(pending is None,'ledger.no_pending',pending)
    expected=dict(created=1,reserve_model_load=2,complete_model_load=2,reserve_eval=32,complete_eval=32,
        reserve_forecast=3996,complete_forecast=3996,model_released=2,evaluation_finished=1)
    check(dict(counts)==expected,'ledger.exact_event_counts',dict(counts))
    check(plans['frozen']==plans['adapted'],'ledger.identical_role_plan')
    check(hashlib.sha256(encoded(plans['frozen'])).hexdigest()==attempt['forecast_order_sha256'],'ledger.sealed_plan_order')
    check(rows[-1]['kind']=='evaluation_finished','ledger.terminal_finished')
    return dict(rows=len(rows),bytes=len(source),sha256=hashlib.sha256(source).hexdigest(),tip_sha256=previous,
        counts=dict(counts),per_role={k:dict(v) for k,v in per.items()},pending=pending,max_active=max_active,
        active_final=active,load_roles=loads,released_roles=released,milestones=milestones,
        optimizer_updates_reserved=0,optimizer_updates_completed=0,new_training_examples=0,
        forecasts_reserved=counts['reserve_forecast'],forecasts_completed=counts['complete_forecast'],
        euler_reserved=3*counts['reserve_forecast'],euler_completed=3*counts['complete_forecast'],
        unique_physical_branches=len({p[:2] for p in plans['frozen']}),noise_seeds=sorted({p[2] for p in plans['frozen']}))


def replay_status(report):
    source=raw(G/'evaluation-status.journal');header=struct.Struct('>8sIQ');offset=sequence=0;frames=[]
    phases=collections.Counter();states=collections.Counter();last_mono=-math.inf;max_gap=0;cuda_ages=[];payload_max=0
    bytes_keys=('cuda_allocated_bytes','cuda_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')
    maxima={k:0 for k in bytes_keys};last_peaks={};transitions=[];before=None
    while offset<len(source):
        if len(source)-offset<header.size:break
        magic,length,seq=header.unpack_from(source,offset);total=header.size+length+32+4
        check(magic==b'GPCEVS01' and 0<length<=1536 and total<=1592,'status.valid_header',seq)
        if not 0<length<=1536 or magic!=b'GPCEVS01':break
        if offset+total>len(source):break
        h=source[offset:offset+header.size];body=source[offset+header.size:offset+header.size+length]
        check(seq==sequence+1,'status.contiguous_sequence',seq)
        check(hashlib.sha256(h+body).digest()==source[offset+header.size+length:offset+total-4],'status.frame_checksum',seq)
        check(source[offset+total-4:offset+total]==b'DONE','status.commit_marker',seq)
        v=json.loads(body);payload_max=max(payload_max,length);sequence=seq
        check(v['actual_worker_pid']==report['actual_worker_pid'] and v['nonce']==report['nonce'],'status.pid_nonce',seq)
        check(finite(v['at_unix']) and finite(v['at_monotonic']) and v['at_monotonic']>=last_mono,'status.monotone_time',seq)
        if frames:max_gap=max(max_gap,v['at_monotonic']-last_mono)
        last_mono=v['at_monotonic'];phases[v['phase']]+=1;states[v['cuda_sample_state']]+=1
        check(v['evaluation_only'] is True and v['training_applicable'] is False and v['training_complete'] is False
            and v['source_checkpoint_update']==500 and v['optimizer_updates']==0,'status.zero_training_contract',seq)
        for key,cap in dict(model_instances=1,model_loads=2,forecasts=3996,euler_steps=11988,eval_batches=32,eval_examples=32).items():
            check(type(v[key]) is int and 0<=v[key]<=cap,'status.counter_cap',dict(sequence=seq,key=key))
        for key in ('environment_creations','resets','restores','controls','physics_substeps','real_renders','policy_sampling'):
            check(type(v[key]) is int and v[key]==0,'status.forbidden_zero',dict(sequence=seq,key=key))
        if v['cuda_sample_state']=='unavailable':
            check(v['phase'] in ('setup','model_loading') and all(v.get(k) is None for k in (*bytes_keys,'cuda_sample_at_unix','cuda_sample_at_monotonic')),'status.legal_unknown_cuda',seq)
        else:
            check(v['cuda_sample_state']=='fresh','status.cuda_fresh_label',seq)
            age=v['at_monotonic']-v['cuda_sample_at_monotonic'];cuda_ages.append(age)
            check(0<=age<=5 and -.5<=v['at_unix']-v['cuda_sample_at_unix']<=5,'status.cuda_sample_age_5s',seq)
            for key,cap in zip(bytes_keys,(536870912,1073741824,536870912,1073741824)):
                check(type(v[key]) is int and 0<=v[key]<=cap,'status.cuda_cap',dict(sequence=seq,key=key));maxima[key]=max(maxima[key],v[key])
            for key in bytes_keys[2:]:
                check(v[key]>=last_peaks.get(key,0),'status.cumulative_peak_nondecreasing',dict(sequence=seq,key=key));last_peaks[key]=v[key]
        marker=(v['phase'],v['model_loads'],v['model_instances'])
        if marker!=before:transitions.append(dict(sequence=seq,**envelope(v)));before=marker
        frames.append(v);offset+=total
    check(offset==len(source),'status.no_partial_tail',len(source)-offset)
    check(sequence==len(frames) and sequence<=10000 and len(source)<=16*1024**2,'status.record_byte_caps')
    check(max_gap<=5,'status.max_publication_gap_5s',max_gap)
    terminal=frames[-1]
    check(terminal==report['last_status'],'status.terminal_equals_guard_last')
    check(terminal['phase']=='complete' and terminal['model_instances']==0 and terminal['model_loads']==2
        and terminal['forecasts']==3996 and terminal['euler_steps']==11988 and terminal['eval_batches']==32,'status.terminal_counts')
    end_mono=report['binding']['started_monotonic']+report['wall_seconds']
    check(0<=end_mono-terminal['at_monotonic']<=5,'status.terminal_age_5s',end_mono-terminal['at_monotonic'])
    return dict(frames=len(frames),last_sequence=sequence,bytes=len(source),sha256=hashlib.sha256(source).hexdigest(),
        parsed_bytes=offset,partial_bytes=len(source)-offset,max_payload_bytes=payload_max,max_interpublication_seconds=max_gap,
        phases=dict(phases),cuda_states=dict(states),max_cuda_age_at_publication=max(cuda_ages),
        terminal_age_at_guard_end=end_mono-terminal['at_monotonic'],all_status_maxima=maxima,
        terminal=envelope(terminal),model_transitions=transitions),frames


def monitor_audit(report):
    source=raw(G/'monitor.jsonl');lines=source.splitlines(keepends=True);rows=[json.loads(x) for x in lines]
    kinds=collections.Counter(r['kind'] for r in rows);samples=[r for r in rows if r['kind']=='sample']
    status_ages=[];cuda_ages=[];current_cuda=collections.defaultdict(int)
    host_max=collections.defaultdict(int);minima={};gpu_ages=[]
    observations=[report['startup'],report['last_resources']]
    for row in samples:
        d=row['detail'];v=d.get('status');res=d['resources'];observations.append(res)
        if v:
            when=datetime.fromisoformat(row['at_utc']).timestamp();age=when-v['at_unix'];status_ages.append(age)
            check(-.5<=age<=5,'monitor.latest_status_age_5s',age)
            if v['cuda_sample_state']=='fresh':
                age=when-v['cuda_sample_at_unix'];cuda_ages.append(age)
                check(-.5<=age<=5,'monitor.cuda_sample_age_5s',age)
            for key in ('cuda_allocated_bytes','cuda_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'):
                if type(v.get(key)) is int:current_cuda[key]=max(current_cuda[key],v[key])
    for res in observations:
        for key in ('private_bytes','working_set_bytes','process_peak_working_set_bytes'):
            if type(res.get(key)) is int:host_max[key]=max(host_max[key],res[key])
        for key in ('available_ram_bytes','commit_headroom_bytes','gpu_free_bytes','disk_free_bytes'):
            if type(res.get(key)) is int:minima[key]=min(minima.get(key,res[key]),res[key])
        if finite(res.get('gpu_age_seconds')):gpu_ages.append(res['gpu_age_seconds'])
    check(kinds.get('heartbeat_first_error',0)==0,'monitor.no_heartbeat_first_error')
    check(rows[-1]['kind']=='final' and rows[-1]['detail']['status']=='complete','monitor.final_complete')
    check(all(line.endswith(b'\n') for line in lines),'monitor.no_partial_json_tail')
    check(host_max['private_bytes']<=3*1024**3 and host_max['process_peak_working_set_bytes']<=2*1024**3,'monitor.host_caps',dict(host_max))
    check(max(gpu_ages)<=15,'monitor.native_gpu_age_15s',max(gpu_ages))
    for key,value in host_max.items():check(report['observed_peaks'][key]==value,'monitor.report_host_peak_matches',key)
    for key,value in minima.items():check(report['min_free'][key]==value,'monitor.report_minima_match',key)
    actual=len(source);declared=report['telemetry']['monitor_bytes'];delta=actual-declared
    check(delta==len(lines[-1]),'monitor.report_snapshot_excludes_only_final_event',dict(delta=delta,final_bytes=len(lines[-1])))
    return dict(events=len(rows),samples=len(samples),kinds=dict(kinds),bytes=actual,
        reported_snapshot_bytes=declared,final_append_bytes=delta,max_status_age_at_monitor=max(status_ages),
        max_cuda_age_at_monitor=max(cuda_ages),sampled_current_and_peak_fields=dict(current_cuda),
        sampled_host_maxima=dict(host_max),native_min_free=minima,max_whole_device_gpu_read_age=max(gpu_ages),
        note='Private bytes are sampled current maxima; process_peak_working_set is the native OS peak counter.')


def main():
    report=js(REVIEW/'execute.stdout.json');stop=js(G/'stop-record.json');receipt=js(REVIEW/'execute-receipt.json')
    claim=js(RUN/'attempt.json');attempt=js(L/'attempt.json');handoff=js(G/'handoff.json')
    worker=js(G/'worker.json');ack=js(G/'worker-ack.json');first=js(G/'heartbeat-error.json')
    completion=js(RUN/'completion.json');binding=report['binding']
    expected_limits=dict(max_updates=0,max_train_examples=0,max_eval_batches=32,max_eval_examples=32,
        max_batch_size=1,max_forecasts=3996,max_euler_steps=11988,max_model_loads=2)
    for key,value in expected_limits.items():check(attempt['limits'][key]==value,'attempt.fixed_limit',key)
    check(attempt['binding']==claim['binding']==binding,'attempt.same_bound_identity')
    check(attempt['binding_sha256']==hashlib.sha256(encoded(binding)).hexdigest(),'attempt.binding_hash')
    check(binding['source_commit']=='042f7186e6eb5b3f5b2e170f50b80e4ce8aa41b9','attempt.reviewed_source')
    check(binding['candidate_id']=='gpc-checkpoint500-evaluation-01' and binding['mode']=='eval_only'
        and binding['source_checkpoint_update']==500 and binding['max_updates']==0,'attempt.eval_only_identity')
    check(binding['deadline_monotonic']-binding['started_monotonic']==3600
        and binding['deadline_unix']-binding['started_unix']==3600,'attempt.original_3600_deadline')
    check(binding['training_deadline_applicable'] is False and binding['training_deadline_monotonic'] is None
        and binding['training_deadline_unix'] is None,'attempt.no_training_phase')
    check(attempt['deadline_monotonic']<=binding['deadline_monotonic'] and attempt['deadline_unix']<=binding['deadline_unix'],'attempt.no_deadline_extension')
    check(first=={},'guardian.empty_first_error',first)
    check(report['status']==stop['status']=='complete' and report['guard_reason'] is None and stop['guard_reason'] is None,'guardian.complete_no_reason')
    check(not any(report['injected'].values()) and report['native_integration_verified'] is True,'guardian.native_not_injected')
    check(report['job_active_processes_after_cleanup']==0 and report['single_model_lock_retained'] is False
        and report['cleanup_errors']==[] and report['worker_exit_code']==0 and report['launcher_exit_code']==0,'guardian.exact_cleanup')
    check(worker==dict(pid=report['actual_worker_pid'],parent_pid=report['launcher_pid'],nonce=report['nonce'])
        and ack==dict(pid=report['actual_worker_pid'],nonce=report['nonce']),'guardian.actual_worker_ack')
    check(handoff['actual_worker_pid']==report['actual_worker_pid'] and handoff['guard_pid']==report['guard_pid']
        and handoff['nonce']==claim['nonce']==report['nonce'] and claim['guard_pid']==report['guard_pid']
        and handoff['ownership']=='native_job_before_ack' and handoff['kill_on_job_close'] is True
        and handoff['job_memory_bytes']==report['job_memory_bytes']==3*1024**3,'guardian.native_handoff')
    ledger=replay_ledger(attempt);status,frames=replay_status(report);monitor=monitor_audit(report)
    state=completion['budget']
    check(state['journal_sequence']==8064 and state['journal_tip_sha256']==ledger['tip_sha256'],'completion.journal_tip')
    check(state['pending'] is None and state['active_models']==0 and state['evaluation_complete'] is True
        and state['released_model_roles']==['frozen','adapted'],'completion.quiescent_budget')
    for key,value in dict(spent_reserved_updates=0,completed_updates=0,spent_train_examples=0,
        spent_reserved_model_loads=2,completed_model_loads=2,spent_reserved_eval_batches=32,completed_eval_batches=32,
        spent_eval_examples=32,spent_reserved_forecasts=3996,completed_forecasts=3996,
        spent_reserved_euler_steps=11988,completed_euler_steps=11988).items():check(state[key]==value,'completion.exact_budget',key)
    check(completion['old_forecasts_reused']==0 and completion['new_optimizer_updates']==0
        and completion['new_training_examples']==0 and completion['optimizer_state_restored'] is False
        and completion['training_rng_restored'] is False and completion['forecast_records']==3996,'completion.no_training_or_old_reuse')
    releases=[]
    for role_index,milestone in enumerate([m for m in ledger['milestones'] if m['kind']=='model_released'],1):
        candidates=[(abs(v['at_monotonic']-milestone['at_monotonic']),i,v) for i,v in enumerate(frames,1)
            if v['model_instances']==0 and v['model_loads']==role_index and v['phase']=='evaluation']
        distance,seq,value=min(candidates,key=lambda item:item[0]);releases.append(dict(role=milestone['role'],
            ledger_sequence=milestone['sequence'],ledger_at_monotonic=milestone['at_monotonic'],status_sequence=seq,
            time_distance_seconds=distance,status=envelope(value)))
        check(value['model_instances']==0 and value['cuda_sample_state']=='fresh' and distance<=1,
            'serial_release.zero_instance_marker_near_ledger',milestone['role'])
        check(0<=value['cuda_allocated_bytes']<=536870912 and 0<=value['cuda_reserved_bytes']<=1073741824,
            'serial_release.allocator_within_fixed_caps',milestone['role'])
    guardian_paths=[p for p in G.iterdir() if p.is_file()]
    ledger_paths=[p for p in L.iterdir() if p.is_file()]
    sizes={p.name:p.stat().st_size for p in guardian_paths};ledger_sizes={p.name:p.stat().st_size for p in ledger_paths}
    actual=dict(status_journal=sizes['evaluation-status.journal'],ledger=sum(ledger_sizes.values()),
        monitor=sizes['monitor.jsonl'],stdout=sizes['worker-output.log'])
    actual['fixed']=sum(v for k,v in sizes.items() if k not in ('evaluation-status.journal','monitor.jsonl','worker-output.log'))+(RUN/'attempt.json').stat().st_size
    caps=dict(status_journal=16*1024**2,ledger=16*1024**2,fixed=1024**2,stdout=2*1024**2,monitor=29*1024**2)
    for key,value in actual.items():check(value<=caps[key],'telemetry.bucket_cap',dict(bucket=key,actual=value,cap=caps[key]))
    check(sum(caps.values())==64*1024**2 and sum(actual.values())<=64*1024**2,'telemetry.total_64MiB')
    check(actual['stdout']==report['telemetry']['stdout_bytes'],'telemetry.stdout_exact')
    log=raw(G/'worker-output.log').decode('utf-8',errors='replace')
    release_peaks={key:report['last_status'][key] for key in ('cuda_allocated_bytes','cuda_reserved_bytes','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')}
    result=dict(kind='gpc_checkpoint500_guardian_offline_audit.v1',audit_revision=2,passed=not ISSUES,issues=ISSUES,
        scope='Only targeted metadata/ledger/status/monitor/log. No project imports/native Reader/process/GPU query, no NPY/checkpoint content read.',
        identity=dict(candidate=binding['candidate_id'],authorization=binding['authorization_id'],source_commit=binding['source_commit'],
            config_sha256=binding['config_sha256'],pid=report['actual_worker_pid'],launcher_pid=report['launcher_pid'],guard_pid=report['guard_pid'],nonce=report['nonce']),
        timing=dict(guard_wall_seconds=report['wall_seconds'],durable_stop_wall_seconds=stop['wall_seconds'],receipt=receipt,
            started_utc=report['started_utc'],ended_utc=report['ended_utc'],started_sgt=iso(binding['started_unix'],8),
            deadline_utc=iso(binding['deadline_unix']),deadline_sgt=iso(binding['deadline_unix'],8),
            guard_end_seconds_before_original_deadline=3600-report['wall_seconds'],
            ledger_started_after_guard_seconds=attempt['started_monotonic']-binding['started_monotonic'],
            ledger_header_seconds=attempt['limits']['deadline_seconds']),
        ledger=ledger,status=status,monitor=monitor,release_evidence=releases,
        peak_distinctions=dict(guardian_sampled_peaks=report['observed_peaks'],terminal_fresh_allocator=release_peaks,
            all_status_peaks=status['all_status_maxima'],host_note='Private max is sampled; native working-set peak counter is separate. CUDA tracked peaks are lifetime allocator counters, not sampled current allocation.'),
        release_interpretation='Both role releases have a zero model-instance marker and zero ledger-active count, but allocator residuals are nonzero. This supports the recorded serial lifecycle; it does not prove zero retained CUDA tensors or identify the residual owner. Zero allocator bytes is not a frozen acceptance condition. No extra introspection was performed.',
        audit_revision_note='Initial audit incorrectly required zero allocated/reserved bytes at release. Its two false-positive findings are preserved as guardian-audit-initial-zero-assumption.json; revision2 applies the frozen count/cap rules and reports residuals explicitly.',
        cleanup={k:report[k] for k in ('job_active_processes_after_cleanup','single_model_lock_retained','worker_exit_code','launcher_exit_code','cleanup_errors','native_integration_verified','injected')},
        telemetry=dict(precharged_caps=caps,actual_end_file_bytes=actual,total_actual_end_bytes=sum(actual.values()),
            guardian_file_sizes=sizes,ledger_file_sizes=ledger_sizes,report_snapshot=report['telemetry'],
            note='Actual final monitor includes its last charged final event, which follows the report telemetry snapshot.'),
        cumulative_costs=dict(old_three_pilots=dict(model_loads=3,updates_reserved=502,updates_completed=501,actual_forecasts=333,
            forecasts_reserved=333,forecasts_completed=332,euler_executed=999,euler_completed=996,loss_completed=16,guard_seconds=394.485),
            this_eval=dict(model_loads=2,updates_reserved=0,updates_completed=0,actual_forecasts=3996,euler_executed=11988,loss_completed=32),
            cumulative_model_jobs=dict(model_loads=5,updates_reserved=502,updates_completed=501,actual_forecasts=4329,
                forecasts_reserved=4329,forecasts_completed=4328,euler_executed=12987,euler_completed=12984,
                loss_completed=48,guard_seconds=394.485+report['wall_seconds']),sdk_separate_guard_seconds=3.860,
            note='Distinct authorized attempts; old failure charges retained, no refund. Three noise seeds are repeated measurements, not independent physical branches.'),
        stdout_lines=log.splitlines(),input_metadata=INPUTS,
        durability_note='Reader verifies visible complete frames. Writer fsync-success is inferred from frozen control flow, empty first-error slot and clean worker/guardian completion; Reader alone does not prove fsync return.')
    destination=REVIEW/'guardian-audit.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],issues=ISSUES,ledger_rows=ledger['rows'],status_frames=status['frames'],
        guard_wall_seconds=report['wall_seconds'],host_peaks=monitor['sampled_host_maxima'],cuda_terminal=release_peaks,
        actual_telemetry_bytes=sum(actual.values()),output=str(destination)),indent=2))
    return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
