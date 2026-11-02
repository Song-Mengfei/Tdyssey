"""Freeze existing high-motion RGB segments before any model work. No physics."""
from collections import Counter,defaultdict
import hashlib,json,math,random,zlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
AR1=ROOT.parent/'accelerated-rgb-run-20261031';SEED=2026110107;LETTERS=('O','C','B')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    from common import atomic_json
    atomic_json(path,value)
def token(*values):return hashlib.sha256(json.dumps([SEED,*values],separators=(',',':')).encode()).hexdigest()
def centers(letter,states):
    from tdyssey.envs.official import official_shape
    local=np.array(official_shape(letter).centroid);c,s=np.cos(states[:,4]),np.sin(states[:,4])
    return states[:,2:4]+np.column_stack((c*local[0]-s*local[1],s*local[0]+c*local[1]))

def motion(states,letter,a,h):
    points=centers(letter,states[a:a+h+1]);angles=states[a:a+h+1,4]
    return dict(net_translation_imagepx=float(np.linalg.norm(points[-1]-points[0])*96/512),
        cumulative_translation_imagepx=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()*96/512),
        signed_rotation_deg=float(np.rad2deg(angles[-1]-angles[0])),
        cumulative_rotation_deg=float(np.abs(np.rad2deg(np.diff(angles))).sum()))

def target_motion(states,letter,a):
    out=[]
    for k in range(16):
        m=motion(states,letter,a+k,1);t=m['net_translation_imagepx'];rot=m['signed_rotation_deg'];strong=t>=1 or abs(rot)>=3
        dominant='translation' if strong and t>=abs(rot)/3 else 'rotation' if strong else 'transition'
        out.append(dict(step=k+1,translation_imagepx=t,signed_rotation_deg=rot,strong=strong,obvious=strong,
            translation_dominant=dominant=='translation',rotation_dominant=dominant=='rotation',dominance=dominant))
    return out

def contacts(telemetry,a,letter):
    ao=ow=seconds=0.;inner=False;bins=set()
    for control in telemetry[a:a+16]:
        for step in control.get('substeps',[]):
            if not step.get('executed'):continue
            dt=float(step['dt']);seconds+=dt;positive=set()
            for e in step.get('contacts',[]):
                if e.get('phase')!='post_solve' or np.linalg.norm(e.get('total_impulse',[0,0]))<=0:continue
                positive.add(e.get('kind'))
                if e.get('kind')=='agent_object':
                    order=e.get('shapes_order',[])
                    if 'object' not in order:continue
                    j=order.index('object');normal=np.asarray(e.get('normal',[0,0]),float)*(1 if j==0 else -1)
                    state=np.asarray(step['post_state']);center=centers(letter,state[None])[0]
                    for point in e.get('points',[]):
                        q=point.get('point_a' if j==0 else 'point_b')
                        if q is None:continue
                        radial=np.asarray(q)-center;inner |= bool(np.dot(normal,radial)<-1e-8)
                        bins.add(int((math.atan2(radial[1],radial[0])%(2*math.pi))/(2*math.pi)*16)%16)
            if 'agent_object' in positive:ao+=dt
            if 'object_wall' in positive:ow+=dt
    return dict(actual_seconds=seconds,positive_AO_seconds=ao,positive_OW_seconds=ow,
        positive_AO_fraction=ao/seconds if seconds else None,actual_wall_contact=ow>0,
        concave_facing_contact_candidate=inner,actual_contact_polar_bins16=sorted(bins))

def _scan(records,split,excluded=()):
    byfile=defaultdict(list)
    for r in records:
        if r['letter'] in LETTERS and r['family_id'] not in excluded:byfile[r['raw_path']].append(r)
    candidates=[];bindings={}
    for text,rows in sorted(byfile.items()):
        path=Path(text);meta_path=path.with_suffix('.json');meta=read(meta_path);first=rows[0]
        if meta['split']!=split:raise ValueError('Original split differs')
        expected={r['raw_sha256'] for r in rows};actual=sha(path)
        if expected!={actual}:raise ValueError('Frozen raw changed')
        bindings[str(path)]=actual;bindings[str(meta_path)]=sha(meta_path)
        with np.load(path,allow_pickle=False) as z:
            states=z['states'];actions=z['passed_actions'];requested=z['requested_actions'];valid=z['control_valid'];times=z['control_time_seconds'];positive=z['contact_positive'];packed=z['telemetry_zlib'].tobytes()
        if not np.array_equal(actions,requested):raise ValueError('Absolute action was transformed')
        anchors={int(r['anchor']):r for r in rows};pending=[]
        for a in sorted(anchors):
            if split=='train' and not set(range(a,a+16)).issubset(anchors):continue
            if a<3 or a+16>=len(states) or a+16>int(meta['clean_control_count']) or not valid[a-3:a+16].all():continue
            if not np.allclose(np.diff(times[a-3:a+17]),.1,rtol=0,atol=1e-9):raise ValueError('Non-10Hz clean window')
            action=actions[a-3:a+16]
            if not np.isfinite(action).all() or (action<0).any() or (action>511).any():continue
            block=next((b for b in meta['segments'] if b['start_control']<=a and a+16<=b['end_control']),None)
            if block is None:continue
            offset=a-block['start_control']
            if not np.array_equal(np.asarray(block['committed_actions'])[offset:offset+16],actions[a:a+16]):raise ValueError('Recorded future commitment differs')
            ms={str(h):motion(states,first['letter'],a,h) for h in (1,4,8,16)};targets=target_motion(states,first['letter'],a);strong=sum(t['strong'] for t in targets)
            net=ms['16']
            if net['net_translation_imagepx']<8 and abs(net['signed_rotation_deg'])<20:continue
            if split=='train' and strong<4:continue
            sid=f"HMF-{first['family_id']}-a{a}"
            pending.append(dict(sample_id=sid,window_id=sid,segment_id=sid,family_id=first['family_id'],letter=first['letter'],split=split,
                raw_path=str(path.resolve()),raw_sha256=actual,metadata_path=str(meta_path.resolve()),metadata_sha256=bindings[str(meta_path)],anchor=a,horizon=16,
                selection_role='train_fit' if split=='train' else 'original_seen_val',task=meta['task'],pose_bin=meta['attempt'].get('pose_bin'),
                source_H1_ids=[anchors[a+k]['sample_id'] for k in range(16)] if split=='train' else [],
                strong_H1_count=strong,strong_flags=[t['strong'] for t in targets],target_motion=targets,motion=ms,committed_future=True,feedback_recorded=False,
                positive_AO_control_count=int(positive[a:a+16].sum()),requested_equals_passed=True))
        if pending:
            telemetry=json.loads(zlib.decompress(packed))['telemetry']
            for row in pending:row['contact']=contacts(telemetry,row['anchor'],row['letter'])
            candidates.extend(pending)
    return candidates,bindings

def choose_train(candidates,letter):
    rows=[r for r in candidates if r['letter']==letter];options={}
    for role in ('cw','ccw','translation'):
        byfamily=defaultdict(list)
        for r in rows:
            net=r['motion']['16'];angle=net['signed_rotation_deg'];distance=net['net_translation_imagepx']
            if role=='cw' and angle>=20 or role=='ccw' and angle<=-20 or role=='translation' and distance>=8:byfamily[r['family_id']].append(r)
        options[role]={family:sorted(rs,key=lambda r:(-int(r['contact']['actual_wall_contact']),-int(r['contact']['concave_facing_contact_candidate']),token(letter,role,family,r['anchor'])))[0] for family,rs in byfamily.items()}
    slots=[('cw',i) for i in range(6)]+[('ccw',i) for i in range(6)]+[('translation',i) for i in range(4)]
    family_owner={};assignment={}
    def visit(slot,seen):
        role=slot[0]
        families=sorted(options[role],key=lambda f:(-int(options[role][f]['contact']['actual_wall_contact']),-int(options[role][f]['contact']['concave_facing_contact_candidate']),token(letter,role,slot[1],options[role][f]['pose_bin'],f)))
        for family in families:
            if family in seen:continue
            seen.add(family)
            if family not in family_owner or visit(family_owner[family],seen):family_owner[family]=slot;assignment[slot]=family;return True
        return False
    for slot in sorted(slots,key=lambda x:(len(options[x[0]]),x)):
        if not visit(slot,set()):raise ValueError(f'Insufficient distinct {letter} families for6CW/6CCW/4translation; role support='+str({k:len(v) for k,v in options.items()}))
    selected=[]
    for slot in slots:
        row=dict(options[slot[0]][assignment[slot]],quota=slot[0]);row['mixed_translation_and_rotation']=row['motion']['16']['net_translation_imagepx']>=8 and abs(row['motion']['16']['signed_rotation_deg'])>=20;selected.append(row)
    return selected,{k:len(v) for k,v in options.items()}

def make_order(targets,steps=3000):
    rng=random.Random(SEED);seen=set();families={};orders={};positions={};family_cursor={};draws=[];first_full=None;strong_first=None
    for pool in ('strong','full'):
        for letter in LETTERS:
            byfamily=defaultdict(list)
            for i,t in enumerate(targets):
                if t['letter']==letter and (pool=='full' or t['strong']):byfamily[t['family_id']].append(i)
            fs=sorted(byfamily);rng.shuffle(fs);families[(pool,letter)]=fs;family_cursor[(pool,letter)]=0
            for f,values in byfamily.items():rng.shuffle(values);orders[(pool,letter,f)]=values;positions[(pool,letter,f)]=0
    def next_target(pool,letter):
        key=(pool,letter);fs=families[key];unseen=any(i not in seen for f in fs for i in orders[(pool,letter,f)])
        for _ in range(len(fs)):
            f=fs[family_cursor[key]%len(fs)];family_cursor[key]+=1;k=(pool,letter,f);values=orders[k]
            for __ in range(len(values)):
                i=values[positions[k]%len(values)];positions[k]+=1
                if not unseen or i not in seen:seen.add(i);return i
        raise RuntimeError('Unseen pool target could not be scheduled')
    strong_ids={i for i,t in enumerate(targets) if t['strong']}
    for batch in range(steps):
        letters=list(LETTERS)*4;rng.shuffle(letters);full=[LETTERS[batch%3],*LETTERS];rng.shuffle(full)
        value=[next_target('strong',l) for l in letters]+[next_target('full',l) for l in full];draws.append(value)
        if first_full is None and len(seen)==len(targets):first_full=batch+1
        if strong_first is None and strong_ids<=seen:strong_first=batch+1
    if first_full is None:raise ValueError('Frozen order failed to cover all unique targets')
    for start in range(0,steps,3):
        totals=Counter(targets[i]['letter'] for b in draws[start:start+3] for i in b)
        if len(draws[start:start+3])==3 and totals!=Counter(dict.fromkeys(LETTERS,16)):raise ValueError('Three-batch letter balance changed')
    return dict(seed=SEED,batches=draws,strong_slots=list(range(12)),full_slots=list(range(12,16)),
        first_all_unique_coverage_batch=first_full,first_strong_pool_coverage_batch=strong_first,planned_draws=steps*16,
        rule='Per batch12strong(4perletter)+4full(2/1/1rotating). Per pool family round robin; globally unseen supported targets first; repeat only after that pool/letter supports no unseen targets. Three batches exactly16targets perletter.',
        planned_target_exposures=dict(Counter(str(i) for b in draws for i in b)),planned_family_exposures=dict(Counter(targets[i]['family_id'] for b in draws for i in b)))

def freeze(root,config):
    root=Path(root);folder=root/'selection';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        summary=read(folder/'summary.json')
        if sha(folder/'index.json')!=summary['index_sha256'] or sha(folder/'order.json')!=summary['order_sha256']:raise ValueError('Frozen selection/order changed')
        return summary
    train_path=AR1/'selection/train-index.json';val_path=AR1/'selection/seen-val-index.json'
    original=read(AR1/'evaluation-spec.json')['rows'];failures=[]
    for letter,prefix in (('O','AC-O-01-08:'),('C','AC-C-01-24:'),('B','AC-B-00-08:')):
        matches=[r for r in original if r['sample_id'].startswith(prefix)]
        if len(matches)!=1:raise ValueError('Original failure case identity missing')
        row=dict(matches[0],selection_role='original_failure',segment_id='HMF-FAIL-'+letter,window_id='HMF-FAIL-'+letter);failures.append(row)
    excluded={r['family_id'] for r in failures};train,bindings=_scan(read(train_path),'train');val,vbind=_scan(read(val_path),'seen_val',excluded);bindings.update(vbind)
    selected=[];support={}
    for letter in LETTERS:
        rows,support[letter]=choose_train(train,letter)
        # The first complete B4 perletter is fixed before model output and shows
        # twoCW/oneCCW/one translationquota, rather than four same-direction cases.
        first=rows[:2]+rows[6:7]+rows[12:13];ids={r['segment_id'] for r in first}
        selected.extend(first+[r for r in rows if r['segment_id'] not in ids])
    validation=[]
    for letter in LETTERS:
        families=defaultdict(list)
        for row in val:
            if row['letter']==letter:families[row['family_id']].append(row)
        best=[sorted(v,key=lambda r:(-int(r['strong_H1_count']>=4),token('val-anchor',r['family_id'],r['anchor'])))[0] for v in families.values()]
        best.sort(key=lambda r:(-int(r['strong_H1_count']>=4),token('val-family',r['family_id'])))
        if len(best)<8:raise ValueError(f'Insufficient original seen-val {letter}: {len(best)}/8')
        validation.extend(best[:8])
    if len({r['family_id'] for r in selected+validation+failures})!=75:raise ValueError('75-family isolation or duplicate case failure')
    targets=[]
    for segment_index,row in enumerate(selected):
        for k,m in enumerate(row['target_motion']):
            targets.append(dict(target_index=len(targets),target_id=row['segment_id']+f':h{k+1}',window_id=row['segment_id']+f':h{k+1}',segment_index=segment_index,segment_id=row['segment_id'],family_id=row['family_id'],letter=row['letter'],anchor=row['anchor']+k,phase=k,quota=row['quota'],selection_role='train_fit',**m))
    panel=[r for letter in LETTERS for r in [x for x in selected if x['letter']==letter][:4]]
    index=dict(kind='high-motion-fit-selection.v1',seed=SEED,train48=selected,seen_val24=validation,failures3=failures,fixed_train_panel12=panel,targets=targets,
        semantics='Positive saved physical angle is screen-clockwise in original y-down renderer. Translation quota has no rotation upper bound; mixed motion disclosed. Target dominance compares t/1 toabs(rotation)/3, ties translation; threshold-normalized analysis only, not physical equivalence.',
        val_rule='Original admitted committed H16 with net>=8imagepx ORabsrotation>=20deg;4strongH1 is training-only and preferred but not required for seen-val.')
    write(folder/'index.json',index);order=make_order(targets);order['selection_index_sha256']=sha(folder/'index.json');write(folder/'order.json',order)
    used={r['raw_path']:r for r in selected+validation+failures}
    sources=[train_path,val_path,AR1/'evaluation-spec.json']+[Path(p) for r in used.values() for p in (r['raw_path'],r.get('metadata_path',str(Path(r['raw_path']).with_suffix('.json'))))]
    summary=dict(kind=index['kind'],seed=SEED,train_segments=48,train_families=48,seen_val_segments=24,failure_segments=3,unique_train_targets=len(targets),strong_train_targets=sum(t['strong'] for t in targets),quota_support=support,
        selected_train_quotas=dict(Counter(r['letter']+':'+r['quota'] for r in selected)),val_strong4_count=dict(Counter(r['letter'] for r in validation if r['strong_H1_count']>=4)),
        first_all_unique_coverage_batch=order['first_all_unique_coverage_batch'],index_sha256=sha(folder/'index.json'),order_sha256=sha(folder/'order.json'),
        bindings=[dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(sources))],model_calls=0,physics_calls=0)
    write(folder/'summary.json',summary)
    return summary


def generate_gt(root,config):
    """Render saved RGB only:48future GT clips, normal10Hz; no simulator."""
    import cv2,html,os,subprocess,imageio_ffmpeg
    from common import check_storage
    root=Path(root);folder=root/'selection';out=folder/'gt';out.mkdir(exist_ok=True)
    index=read(folder/'index.json');summary=read(folder/'summary.json');manifest=folder/'gt-manifest.json'
    if manifest.exists():
        value=read(manifest)
        if any(sha(x['path'])!=x['sha256'] for x in value['files']):raise ValueError('Existing GT media changed')
        summary['support_by_letter']=summarize_support(index)
        write(folder/'summary.json',summary)
        return value
    files=[];videos=[];saved={}
    for row in index['train48']:
        path=Path(row['raw_path'])
        if sha(path)!=row['raw_sha256']:raise ValueError('GT source changed')
        a=row['anchor']
        with np.load(path,allow_pickle=False) as z:images=z['rgb'][a+1:a+17].copy()
        saved[row['sample_id']]=images;frames=[]
        for k,image in enumerate(images):
            frame=np.full((240,224,3),255,np.uint8)
            frame[32:224,16:208]=cv2.resize(image.transpose(1,2,0),(192,192),interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame,f'{row["letter"]} {row["quota"]} t+{k+1}={(k+1)/10:.1f}s',(4,12),cv2.FONT_HERSHEY_SIMPLEX,.34,(20,20,20),1)
            m=row['target_motion'][k]
            cv2.putText(frame,f'dx={m["translation_imagepx"]:.2f}px  da={m["signed_rotation_deg"]:+.2f}deg',(4,26),cv2.FONT_HERSHEY_SIMPLEX,.31,(20,20,20),1)
            cv2.putText(frame,'SAVED REAL RGB / TRAIN FIT CASE',(4,237),cv2.FONT_HERSHEY_SIMPLEX,.30,(20,20,20),1)
            frames.append(frame)
        target=out/(row['segment_id']+'.mp4');check_storage(root,config,reserve_bytes=2*1024**2)
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','224x240','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        kw={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        result=subprocess.run(cmd,input=b''.join(x.tobytes() for x in frames),stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw)
        if result.returncode:raise RuntimeError('GT encoding failed '+result.stderr.decode(errors='replace')[-300:])
        cap=cv2.VideoCapture(str(target));fps=cap.get(cv2.CAP_PROP_FPS);n=0
        while cap.read()[0]:n+=1
        cap.release()
        if n!=16 or abs(fps-10)>1e-3:raise ValueError('GT clip timeline differs')
        entry=dict(path=str(target.resolve()),sha256=sha(target),bytes=target.stat().st_size)
        files.append(entry);videos.append(dict(sample_id=row['sample_id'],letter=row['letter'],quota=row['quota'],frames=n,fps=fps,**entry))
    boards=[]
    for letter in LETTERS:
        cases=[r for r in index['fixed_train_panel12'] if r['letter']==letter];rows=[]
        for row in cases:
            line=np.full((222,768,3),255,np.uint8)
            for col,h in enumerate((1,4,8,16)):
                line[28:220,col*192:(col+1)*192]=cv2.resize(saved[row['sample_id']][h-1].transpose(1,2,0),(192,192),interpolation=cv2.INTER_NEAREST)
                cv2.putText(line,f'{row["quota"]} t+{h}',(col*192+4,24),cv2.FONT_HERSHEY_SIMPLEX,.38,(20,20,20),1)
            cv2.putText(line,row['sample_id'],(4,11),cv2.FONT_HERSHEY_SIMPLEX,.31,(20,20,20),1);rows.append(line)
        board=np.concatenate(rows);path=folder/(letter+'-fixed4-GT.png');ok,png=cv2.imencode('.png',cv2.cvtColor(board,cv2.COLOR_RGB2BGR))
        if not ok:raise RuntimeError('GT board encoding failed')
        check_storage(root,config,reserve_bytes=len(png));path.write_bytes(png.tobytes());entry=dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size);files.append(entry);boards.append(dict(letter=letter,**entry))
    distributions={}
    for split,key in (('train','train48'),('seen_val','seen_val24')):
        for letter in LETTERS:
            selected=[r for r in index[key] if r['letter']==letter]
            for h in (1,4,8,16):
                values={field:[r['motion'][str(h)][field] for r in selected] for field in ('net_translation_imagepx','cumulative_translation_imagepx','signed_rotation_deg','cumulative_rotation_deg')}
                distributions[f'{split}/{letter}/H{h}']={field:dict(N=len(v),P10=float(np.quantile(v,.1)),P50=float(np.median(v)),P90=float(np.quantile(v,.9)),mean=float(np.mean(v))) for field,v in values.items()}
    chunks=['<!doctype html><meta charset="utf-8"><title>Frozen high-motion GT48</title><style>body{font:15px system-ui;max-width:1100px;margin:24px auto}img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}video{width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style>',
        '<h1>48 frozen training trajectories: saved ground truth</h1><p>No model predictions, simulator replay or new physical steps. Each clip shows16future frames at10fps (0.1–1.6s). The four true history frames remain in the source archive and model loader. Positive saved angle means screen-clockwise; translation-quota segments may also rotate and are identified as mixed.</p>',
        '<p>Selection seed2026110107. Cases are frozen before looking at model outputs. Each letter has6CW/6CCW/4translation,16independent families. Validation24 and original failures3 are excluded from fitting.</p>']
    for board in boards:chunks += [f'<h2>{board["letter"]}: fixed4 display cases</h2>',f'<img src="{Path(board["path"]).name}">']
    for letter in LETTERS:
        chunks += [f'<h2>All16 {letter} training cases</h2><div class="grid">']
        for row in [r for r in index['train48'] if r['letter']==letter]:
            video=next(v for v in videos if v['sample_id']==row['sample_id']);net=row['motion']['16']
            chunks += ['<div><b>'+html.escape(row['sample_id'])+'</b><p>'+html.escape(row['quota'])+f'; strongH1={row["strong_H1_count"]}/16; H16={net["net_translation_imagepx"]:.2f}px / {net["signed_rotation_deg"]:+.2f}°; AO+={row["contact"]["positive_AO_seconds"]:.2f}s; mixed={row["mixed_translation_and_rotation"]}</p>',f'<video controls preload="none" src="gt/{Path(video["path"]).name}"></video></div>']
        chunks+=['</div>']
    chunks+=['<p><a href="index.json">Frozen75case/768target index</a> · <a href="summary.json">Full motion/contact/source summary</a></p>']
    page=folder/'index.html';page.write_text('\n'.join(chunks),encoding='utf-8');files.append(dict(path=str(page.resolve()),sha256=sha(page),bytes=page.stat().st_size))
    value=dict(kind='saved-real-high-motion-GT48.v1',videos=videos,fixed_boards=boards,files=files,physics_calls=0,model_calls=0,frame_note='16real future frames perclip,10fps; fourhistory sourceframes not invented or repeated')
    write(manifest,value)
    summary.update(GT_index=str(page.resolve()),GT_manifest=str(manifest.resolve()),GT_videos=48,GT_fixed_boards=boards,motion_distributions=distributions,
        training_contact_positive_seconds=sum(r['contact']['positive_AO_seconds'] for r in index['train48']),training_actual_seconds=48*1.6,
        selected_training_strong_targets_by_letter=dict(Counter(t['letter'] for t in index['targets'] if t['strong'])),
        mixed_translation_quota_counts=dict(Counter(r['letter'] for r in index['train48'] if r['quota']=='translation' and r['mixed_translation_and_rotation'])),
        rotation_sign_proof='Original screen coordinates have ydown. R(+pi/2)*(1,0)=(0,1), hence positive saved unwrapped angle is screenclockwise. Same original render transform is retained.',
        quota_does_not_imply_pure_motion=True)
    summary['support_by_letter']=summarize_support(index)
    write(folder/'summary.json',summary)
    return value


def summarize_support(index):
    """Actual selected support, separate from nonoverlapping quota labels."""
    result={}
    for split,key in (('train','train48'),('seen_val','seen_val24')):
        for letter in LETTERS:
            rows=[r for r in index[key] if r['letter']==letter]
            duration=sum(r['contact']['actual_seconds'] for r in rows)
            bins=sorted({b for r in rows for b in r['contact']['actual_contact_polar_bins16']})
            angles=[r['motion']['16']['signed_rotation_deg'] for r in rows]
            target_rows=[t for r in rows for t in r['target_motion']]
            ao=sum(r['contact']['positive_AO_seconds'] for r in rows)
            ow=sum(r['contact']['positive_OW_seconds'] for r in rows)
            result[f'{split}/{letter}']=dict(segments=len(rows),targets=len(target_rows),strong_H1=sum(t['strong'] for t in target_rows),
                transition_H1=sum(not t['strong'] for t in target_rows),actual_seconds=duration,positive_AO_seconds=ao,positive_OW_seconds=ow,
                positive_AO_substep_time_fraction=ao/duration,positive_OW_substep_time_fraction=ow/duration,
                actual_wall_contact_segments=sum(r['contact']['actual_wall_contact'] for r in rows),
                concave_facing_contact_candidate_segments=sum(r['contact']['concave_facing_contact_candidate'] for r in rows),
                contact_polar_bins16=bins,missing_contact_polar_bins16=sorted(set(range(16))-set(bins)),
                net_H16_angle_sign=dict(screen_CW=sum(x>0 for x in angles),screen_CCW=sum(x<0 for x in angles),zero=sum(x==0 for x in angles)),
                mixed_translation_quota=sum(r.get('quota')=='translation' and r.get('mixed_translation_and_rotation',False) for r in rows),
                notes='Actual signed net angle counts are distinct from6CW/6CCW/4translation quotas; translation can be mixed. Contact bins are world polar directions of positive-impulse contact points about the true object centroid, not proof of surface coverage or sealed-hole access.')
    return result
