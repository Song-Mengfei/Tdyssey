"""Recompute saved geometry/contact observations. Never creates a simulator."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zlib

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
SOURCE=REPO/'artifacts/direct-target-acceptance-20261031'
sys.path.insert(0,str(REPO))
from tdyssey.native_rgb_env import shape_geometry


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rotation(t):return np.asarray(((math.cos(t),-math.sin(t)),(math.sin(t),math.cos(t))))
def xform(g,state):return affinity.translate(affinity.rotate(g,float(state[4]),origin=(0,0),use_radians=True),*state[2:4])
def q(values):
    a=np.asarray(values,float)
    return dict(N=len(a),minimum=float(a.min()),P50=float(np.median(a)),P90=float(np.quantile(a,.9)),maximum=float(a.max())) if len(a) else dict(N=0)
def native_kind_info(events,kind):
    values=[p['distance'] for e in events if e['kind']==kind for p in e.get('points',[])]
    return max(0.,-min(values)) if values else None
def to_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    loaded={};records=[];errors=[];shape_rows=[];point_dot_errors=[];native_differences=defaultdict(list)
    geometry_errors=[];surface_errors=[];phase_matches=[];seam_rows=[];event_counter=Counter();all_inputs=[]
    capsule_aw_errors=[];capsule_ow_errors=[];region_errors=[]
    for path in sorted((SOURCE/'collection/attempts').glob('*.json')):
        m=read(path);a=m['attempt'];identifier=a['attempt_id'];identity=m['initial_identity'];phys=identity['physics']
        raw=SOURCE/'collection/raw'/(identifier+'.npz')
        all_inputs.extend([dict(path=str(path),sha256=sha(path)),dict(path=str(raw),sha256=sha(raw))])
        shapes=phys['shapes'];pieces=[]
        for s in shapes:
            if s['role']=='object':
                if s['geometry']['kind']!='poly' or s['geometry']['radius']!=0:errors.append(identifier+':unexpected object geometry')
                pieces.append(Polygon(s['geometry']['vertices']))
        local=unary_union(pieces); wrapper=shape_geometry(a['letter'])
        difference=float(local.symmetric_difference(wrapper).area)
        radius=next(s['geometry']['radius'] for s in shapes if s['role']=='agent')
        walls=[s['geometry'] for s in shapes if s['role']=='wall']
        if radius!=15:errors.append(identifier+':unexpected pusher radius')
        if difference>1e-8:errors.append(identifier+':actual collision union differs from wrapper')
        shape_rows.append(dict(attempt_id=identifier,letter=a['letter'],union_difference_area=difference,
            object_parts=len(pieces),holes=len(local.interiors),agent_radius=radius,object_radius=0.,
            body_cog=phys['object']['center_of_gravity'],area_centroid=[local.centroid.x,local.centroid.y],
            walls=walls))
        with np.load(raw,allow_pickle=False) as z:
            actions=z['requested_actions'].copy();states=z['states'].copy()
            payload=zlib.decompress(z['telemetry_zlib'].tobytes());telemetry=json.loads(payload)['telemetry']
        sequence=[];time=0.
        for control,info in enumerate(telemetry):
            for micro,row in enumerate(info.get('substeps',[])):
                if not row.get('executed'):continue
                dt=float(row['dt']);time+=dt;post=np.asarray(row['post_state'],float);pre=np.asarray(row['pre_state'],float)
                g=xform(local,post);p=Point(*post[:2]);inside=g.contains(p)
                signed=-g.boundary.distance(p) if inside else g.distance(p)
                bounds=g.bounds
                ao=max(0.,radius-signed)
                ow=max(0.,7-bounds[0],7-bounds[1],bounds[2]-504,bounds[3]-504)
                aw=max(0.,22-post[0],22-post[1],post[0]-489,post[1]-489)
                computed=np.asarray((ow,ao,aw));saved=np.asarray(row['penetrations']);geo_error=float(np.max(np.abs(computed-saved)))
                geometry_errors.append(geo_error)
                # Independent native shape capsule distance, not the wrapper's halfplanes.
                aw_shape=max([0.,*[radius+w['radius']-LineString((w['a'],w['b'])).distance(p) for w in walls]])
                ow_shape=max([0.,*[w['radius']-LineString((w['a'],w['b'])).distance(g) for w in walls]])
                capsule_aw_errors.append(aw-aw_shape);capsule_ow_errors.append(ow-ow_shape)
                region='solid' if inside else next((f'hole-{i}' for i,ring in enumerate(g.interiors) if Polygon(ring).contains(p)),'outside')
                if region!=row['pusher_region']:region_errors.append(dict(attempt_id=identifier,control=control,micro=micro,saved=row['pusher_region'],recomputed=region))
                # OW polygon-wall intersection cannot be inferred merely from unsigned
                # distance once a polygon crosses a segment. The halfplane extrema
                # above are exact on the bounded interior side recorded in this run.
                if geo_error>1e-7:errors.append(identifier+f': geometry mismatch {control}/{micro}')
                events=row.get('contacts',[]);pre_events=[e for e in events if e['phase']=='pre_solve'];post_events=[e for e in events if e['phase']=='post_solve']
                ev_norm=None;vpre=None;vpost=None;impulse=0.;normal_impulse=0.;object_point=None
                for e in events:
                    event_counter[(e['kind'],e['phase'])]+=1
                    n=np.asarray(e['normal'],float)
                    for point in e.get('points',[]):
                        pa=np.asarray(point['point_a']);pb=np.asarray(point['point_b']);dist=float(point['distance'])
                        point_dot_errors.append(abs(float((pb-pa)@n)-dist))
                        if e['kind']=='agent_object':
                            oi=e['shapes_order'].index('object');op=pa if oi==0 else pb
                            union_distance=float(g.boundary.distance(Point(*op)))
                            surface_errors.append(union_distance)
                            if union_distance>1e-6:
                                seam_rows.append(dict(attempt_id=identifier,control=control,micro=micro,time=time,
                                    phase=e['phase'],distance_to_union_boundary=union_distance,native_distance=dist,
                                    custom_AO=ao,object_point_x=float(op[0]),object_point_y=float(op[1])))
                    if e['kind']=='agent_object' and e.get('points') and (ev_norm is None or e['phase']=='post_solve'):
                        objfirst=e['shapes_order'][0]=='object';n_out=n if objfirst else -n
                        ep=e['points'][0];op=np.asarray(ep['point_a' if objfirst else 'point_b']);ap=np.asarray(ep['point_b' if objfirst else 'point_a'])
                        cog=post[2:4]+rotation(post[4])@np.asarray(phys['object']['center_of_gravity'])
                        lever=op-cog
                        def relative(s):
                            vo=s[7:9]+s[9]*np.asarray((-lever[1],lever[0]))
                            va=s[5:7] # agent has zero angular velocity and center offset
                            return float((va-vo)@n_out)
                        ev_norm=n_out;vpre=relative(pre);vpost=relative(post);object_point=op
                    if e['kind']=='agent_object' and e['phase']=='post_solve':
                        j=np.asarray(e['total_impulse']);impulse+=float(np.linalg.norm(j));normal_impulse+=abs(float(j@n))
                native={kind:native_kind_info(events,kind) for kind in ('agent_object','agent_wall','object_wall')}
                for kind,depth in native.items():
                    if depth is not None:
                        idx={'object_wall':0,'agent_object':1,'agent_wall':2}[kind]
                        native_differences[kind].append(float(computed[idx]-depth))
                # Same points/distances before and after velocity solve is expected
                # when callbacks use the same detected manifold. Compare exact event
                # shape role order and manifold values, not the saved velocities.
                for pe in pre_events:
                    for po in post_events:
                        if pe['kind']==po['kind'] and pe['shapes_order']==po['shapes_order'] and pe['points']==po['points'] and pe['normal']==po['normal']:
                            phase_matches.append((identifier,control,micro,pe['kind']))
                prev=sequence[-1]['post_state'] if sequence else states[0]
                cog_local=np.asarray((local.centroid.x,local.centroid.y))
                center=post[2:4]+rotation(post[4])@cog_local
                previous=np.asarray(prev);prevcenter=previous[2:4]+rotation(previous[4])@cog_local
                record=dict(attempt_id=identifier,letter=a['letter'],scenario=a['scenario'],mode=a['target_mode'],status=m['cost']['status'],
                    control=control,micro=micro,time=time,valid=row.get('valid'),custom_AO=ao,custom_AW=aw,custom_OW=ow,
                    native_AO=native['agent_object'],native_AW=native['agent_wall'],native_OW=native['object_wall'],
                    native_AO_pre=native_kind_info(pre_events,'agent_object'),native_AO_post=native_kind_info(post_events,'agent_object'),
                    AO_impulse_norm=impulse,AO_normal_impulse_abs=normal_impulse,
                    AO_normal_relative_v_pre=vpre,AO_normal_relative_v_post=vpost,
                    normal_x=float(ev_norm[0]) if ev_norm is not None else None,normal_y=float(ev_norm[1]) if ev_norm is not None else None,
                    object_point_x=float(object_point[0]) if object_point is not None else None,object_point_y=float(object_point[1]) if object_point is not None else None,
                    target_error_pre=float(np.linalg.norm(actions[control]-pre[:2])),target_error_post=float(np.linalg.norm(actions[control]-post[:2])),
                    pusher_step_displacement=float(np.linalg.norm(post[:2]-pre[:2])),object_centroid_step_displacement=float(np.linalg.norm(center-prevcenter)),
                    object_angle_step_degrees=float((post[4]-previous[4])*180/math.pi),custom_recompute_error=geo_error,
                    capsule_AW_depth=aw_shape,poly_segment_unsigned_OW_depth=ow_shape,
                    pre_state=pre.tolist(),post_state=post.tolist())
                sequence.append(record);records.append(record)
        loaded[identifier]=dict(meta=m,local=local,records=sequence)
    # Select the completed trajectory with deepest observed AO contact and at
    # least one observed later <=.1 sample, not a hypothetical continuation.
    recovery_candidates=[]
    for identifier,d in loaded.items():
        if d['meta']['cost']['status']!='completed':continue
        seq=d['records'];best=max(range(len(seq)),key=lambda i:seq[i]['custom_AO'])
        later=next((i for i in range(best+1,len(seq)) if seq[i]['custom_AO']<=.1),None)
        if later is not None:recovery_candidates.append((seq[best]['custom_AO'],identifier,best,later))
    recovery=max(recovery_candidates) if recovery_candidates else None
    cases=[('first_invalid','DT-T-00-0-b0',None)]
    if recovery:cases.append(('observed_legal_recovery',recovery[1],recovery[2]))
    aw_candidates=[d for d in loaded.values() if d['records'] and d['records'][-1]['custom_AW']>1 and d['meta']['attempt']['scenario']=='negative']
    if aw_candidates:cases.append(('wall_target',aw_candidates[0]['meta']['attempt']['attempt_id'],None))
    selected=[]
    os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for title,identifier,centerindex in cases:
        seq=loaded[identifier]['records']
        if centerindex is None:
            centerindex=next((i for i,x in enumerate(seq) if (x['custom_AW']>0 if title=='wall_target' else x['AO_impulse_norm']>0 or not x['valid'])),len(seq)-1)
        lo=max(0,centerindex-12);hi=min(len(seq),centerindex+80)
        if title=='observed_legal_recovery':hi=min(len(seq),max(hi,recovery[3]+10))
        shown=seq[lo:hi];table=[{k:v for k,v in x.items() if k not in ('pre_state','post_state')} for x in shown]
        to_csv(OUT/(title+'-process.csv'),table)
        fig,axes=plt.subplots(5,1,figsize=(10,12),constrained_layout=True)
        for ax in axes[1:4]:ax.sharex(axes[0])
        times=[x['time'] for x in shown]
        for key,label in (('custom_AO','custom AO'),('native_AO','native AO manifold'),('custom_AW','custom AW'),('native_AW','native AW (pre)'),('custom_OW','custom OW')):
            vals=[np.nan if x[key] is None else x[key] for x in shown];axes[0].plot(times,vals,label=label,lw=1.5,marker='o' if key.startswith('native') else None,markersize=3)
        axes[0].axhline(1,color='r',ls=':',label='existing terminal rule=1');axes[0].set_ylabel('depth world units');axes[0].legend(ncol=3,fontsize=8)
        axes[1].plot(times,[x['AO_impulse_norm'] for x in shown],label='AO total impulse norm');axes[1].plot(times,[x['target_error_post'] for x in shown],label='target error post-step');axes[1].set_ylabel('impulse / world units');axes[1].legend(fontsize=8)
        for key,label in (('AO_normal_relative_v_pre','pre_solve-stage relative normal v (reconstructed)'),('AO_normal_relative_v_post','post_solve-stage relative normal v')):
            axes[2].plot(times,[np.nan if x[key] is None else x[key] for x in shown],label=label,marker='o',markersize=3)
        axes[2].axhline(0,color='.6',lw=.7);axes[2].set_ylabel('world units/s');axes[2].legend(fontsize=8)
        axes[3].plot(times,[x['pusher_step_displacement'] for x in shown],label='pusher displacement/.01s');axes[3].plot(times,[x['object_centroid_step_displacement'] for x in shown],label='area-centroid displacement/.01s');axes[3].set_ylabel('world units');axes[3].set_xlabel('actual cumulative simulation seconds');axes[3].legend(fontsize=8)
        for ax in axes[:4]:
            ax.grid(alpha=.25)
            if not seq[-1]['valid']:ax.axvline(seq[-1]['time'],color='r',ls='--',lw=1)
        final=shown[-1];pose=np.asarray(final['post_state']);g=xform(loaded[identifier]['local'],pose)
        axes[4].fill(*g.exterior.xy,color='.75',label='object collision union')
        for ring in g.interiors:axes[4].fill(*ring.xy,color='white')
        from matplotlib.patches import Circle
        axes[4].add_patch(Circle(pose[:2],15,fill=False,color='royalblue',lw=1.5,label='radius15 pusher, final saved pose'))
        axes[4].plot([x['post_state'][0] for x in shown],[x['post_state'][1] for x in shown],color='royalblue',lw=.8,label='pusher path')
        contact=[x for x in shown if x['object_point_x'] is not None]
        if contact:
            axes[4].scatter([x['object_point_x'] for x in contact],[x['object_point_y'] for x in contact],s=7,c='red',label='native object contact points')
            e=contact[-1];axes[4].arrow(e['object_point_x'],e['object_point_y'],e['normal_x']*15,e['normal_y']*15,width=.3,color='red',length_includes_head=True)
        for wall in loaded[identifier]['meta']['initial_identity']['physics']['shapes']:
            if wall['role']=='wall':
                w=wall['geometry'];axes[4].plot([w['a'][0],w['b'][0]],[w['a'][1],w['b'][1]],color='k',lw=2)
        if title=='wall_target':axes[4].set_xlim(pose[0]-35,pose[0]+35);axes[4].set_ylim(pose[1]+35,pose[1]-35)
        else:
            x0,y0,x1,y1=g.bounds;axes[4].set_xlim(min(x0,pose[0]-15)-8,max(x1,pose[0]+15)+8);axes[4].set_ylim(max(y1,pose[1]+15)+8,min(y0,pose[1]-15)-8)
        axes[4].set_aspect('equal',adjustable='box');axes[4].set_xlabel('world x');axes[4].set_ylabel('world y (screen down)');axes[4].legend(fontsize=7,loc='upper left')
        fig.suptitle(title+' | '+identifier+' | saved .01s observations, no replay')
        fig.savefig(OUT/(title+'-process.png'),dpi=140);plt.close(fig)
        selected.append(dict(label=title,attempt_id=identifier,control=seq[centerindex]['control'],micro=seq[centerindex]['micro'],time=seq[centerindex]['time'],depth_AO=seq[centerindex]['custom_AO'],depth_AW=seq[centerindex]['custom_AW'],source_sha256=sha(SOURCE/'collection/raw'/(identifier+'.npz')),csv=title+'-process.csv',plot=title+'-process.png'))
    byletter={}
    for letter in 'TLOCXB':
        sr=[x for x in shape_rows if x['letter']==letter]
        byletter[letter]=dict(attempts=len(sr),max_union_difference_area=max(x['union_difference_area'] for x in sr),object_parts=sr[0]['object_parts'],holes=sr[0]['holes'],body_cog=sr[0]['body_cog'],area_centroid=sr[0]['area_centroid'])
    summary=dict(kind='saved-only-shape-and-contact-geometry-validation.v1',run_id='DTA-20261031-7b4e4819a73b41a78dba24947fd6579b',
        physical_constructions=0,physics_steps=0,model_calls=0,optimizer_updates=0,attempts=len(loaded),native_rows=len(records),
        imported_pymunk='pymunk' in sys.modules,imported_torch='torch' in sys.modules,errors=errors,passed=not errors,
        shape_by_letter=byletter,custom_depth_absolute_error=q(geometry_errors),native_point_dot_normal_absolute_error=q(point_dot_errors),
        custom_minus_independent_circle_wall_capsule_depth=q(capsule_aw_errors),custom_minus_polygon_wall_unsigned_capsule_depth=q(capsule_ow_errors),
        region_recompute_errors=region_errors,
        custom_minus_native_depth={k:q(v) for k,v in native_differences.items()},
        object_contact_point_distance_to_union_boundary=q(surface_errors),possible_internal_component_seam_points=len(seam_rows),
        pre_post_identical_manifold_matches=len(phase_matches),event_counts={str(k):v for k,v in event_counter.items()},
        selected_cases=selected,recovery_selection=dict(depth_AO=recovery[0],attempt_id=recovery[1],peak_time=loaded[recovery[1]]['records'][recovery[2]]['time'],observed_recovery_time=loaded[recovery[1]]['records'][recovery[3]]['time'],observed_recovery_depth=loaded[recovery[1]]['records'][recovery[3]]['custom_AO']) if recovery else None,
        actual_library_version_context='Root independently verified installed Pymunk6.9.0, Chipmunk7.0.3-dfc2fb8ca023ce6376fa2cf4a7f91c92ee08a970. This geometry script imports neither library nor a simulator.',
        caveats=['Native manifold pre/post callbacks and custom after-return geometry are separate observations; this script measures their numerical agreement, not unobserved future recovery.',
                 'For relative contact velocity, pre_solve/detection-stage velocities use saved integrator_pre_state and geometry lever at recorded post-integration positions. Native velocity integration occurs later, before impulses; with damping0 dynamic-object velocity is then zeroed. Thus this pre value is not the immediate pre-impulse-solver velocity. Body velocity is CoG velocity; T local CoG=(0,45) differs from area centroid=(0,40.7142857).',
                 'AW kinematic/static can have pre_solve contacts without post_solve impulse; absent post event is not absence of overlap.',
                 'Unsigned poly-to-segment distance is not a penetration metric after crossing; recorded bounded-interior wall depths are compared to original axis halfplanes.',
                 'No >1 continuation was simulated or inferred.'])
    to_csv(OUT/'native-union-contact-exceptions.csv',seam_rows[:100])
    (OUT/'geometry_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'input-manifest.json').write_text(json.dumps(dict(inputs=all_inputs,sources=[dict(path=str(REPO/'tdyssey/native_rgb_env.py'),sha256=sha(REPO/'tdyssey/native_rgb_env.py')),dict(path=str(REPO/'tdyssey/native_rgb_original_physics.py'),sha256=sha(REPO/'tdyssey/native_rgb_original_physics.py'))]),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('passed','native_rows','custom_depth_absolute_error','custom_minus_native_depth','native_point_dot_normal_absolute_error','possible_internal_component_seam_points','recovery_selection','selected_cases','event_counts')},ensure_ascii=False))

if __name__=='__main__':run()
