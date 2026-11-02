"""Read-only independent joint-limit and execution-endpoint accounting."""
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parent


def errors(state):
    p,g=state['object_body']['pose'],state['goal']
    return math.dist(p['position'],g['position']),abs(math.atan2(
        math.sin(p['angle']-g['angle']),math.cos(p['angle']-g['angle'])))


def main():
    issues=[]
    results=[]
    for scene in (100000,100200):
        data=json.loads((BASE/f'scene-{scene}-resume160-search42/report.json').read_text(encoding='utf-8'))
        states={160:data['search_start_state'],**{s['state']['control_step']:s['state'] for s in data['steps']}}
        actual={s['state']['control_step']:s for s in data['steps']}
        prior=None
        globals_=[d for d in data['decisions'] if not d['continuation']]
        completed=0
        interrupted=[]
        triples=sum(len(d['selected']['stages'])==3 for d in globals_)
        for index,d in enumerate(globals_):
            if not d['precision_mode']:
                continue
            start=d['live_control_step']
            pos,angle=errors(states[start])
            expected=dict(center_limit=max(3.,pos+.25),angle_limit=max(math.radians(2),angle+math.radians(.25)))
            if prior:
                for k in expected:expected[k]=min(expected[k],prior[k])
            expected.update(path_center_limit=max(pos,expected['center_limit'])+3.,
                path_angle_limit=max(angle,expected['angle_limit'])+math.radians(8))
            for k,v in expected.items():
                if abs(d['joint_limits'][k]-v)>1e-7:issues.append(f'{scene}/{start}: limit {k}')
            prior=expected
            sel=d['selected'];n=sel['execution_controls']
            pred=d['selected_predictions'][:n]
            if not 0<n<=len(d['selected_predictions'])<=80:issues.append(f'{scene}/{start}: execution length')
            values=[errors(s) for s in pred]
            tail=values[-3:]
            tp,ta=max(p for p,a in tail),max(a for p,a in tail)
            if abs(sel['execution_position_error']-values[-1][0])>1e-7 or abs(sel['execution_angle_error']-values[-1][1])>1e-7:
                issues.append(f'{scene}/{start}: execution endpoint metric')
            if not sel['native_success']:
                if tp>expected['center_limit']+1e-7 or ta>expected['angle_limit']+1e-7:
                    issues.append(f'{scene}/{start}: terminal constraint')
                if max(p for p,a in values)>expected['path_center_limit']+1e-7 or max(a for p,a in values)>expected['path_angle_limit']+1e-7:
                    issues.append(f'{scene}/{start}: trajectory corridor')
                balanced=max(tp/3,ta/math.radians(2))+.02*sel['completion_penalty']
                if abs(sel['balanced_score']-balanced)>1e-7:issues.append(f'{scene}/{start}: balanced metric')
            stop=start+n
            next_start=globals_[index+1]['live_control_step'] if index+1<len(globals_) else max(states)
            if next_start<stop:
                interrupted.append(dict(start=start,expected_terminal=stop,interrupted_at=next_start))
            elif stop in actual and not sel['native_success']:
                check=actual[stop].get('joint_execution_check')
                if not check or not check['macro_terminal'] or not check['terminal_ok']:
                    issues.append(f'{scene}/{start}: actual terminal check missing')
                completed+=1
        checked=sum(s.get('joint_execution_check') is not None for s in data['steps'])
        for row in data['steps']:
            check=row.get('joint_execution_check')
            if check:
                p,a=errors(row['state'])
                if abs(check['position_error']-p)>1e-7 or abs(check['angle_error']-a)>1e-7 or not check['corridor_ok']:
                    issues.append(f'{scene}: actual step guard')
        results.append(dict(scene=scene,completed_precision_macros=completed,
            interrupted_precision_macros=interrupted,actual_guarded_controls=checked,
            selected_three_contact_macros=triples,final_position_angle=errors(states[max(states)])))
    result=dict(status='passed' if not issues else 'failed',issues=issues,results=results,
        training=False,new_physics_or_search=False)
    (BASE/'GUARD_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if issues:raise SystemExit(1)


if __name__=='__main__':
    main()
