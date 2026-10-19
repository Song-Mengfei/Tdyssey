"""Saved-trajectory score diagnostics; no inference, simulator or dependencies."""


def candidate_score_diagnostics(candidates, future):
    """Keep max/final/sum views separate; final success is absorbing for sums."""
    result={}
    for score in ('max','final','sum'):
        rows=[]
        for candidate in candidates:
            predicted=candidate['predicted_rewards']
            actual=[step['reward'] for step in candidate['actual']]
            if score=='max':a,p=max(actual),max(predicted)
            elif score=='final':a,p=actual[-1],predicted[-1]
            else:
                if candidate['actual'][-1]['success']:
                    actual=actual+[actual[-1]]*(future-len(actual))
                a,p=sum(actual),sum(predicted)
            rows.append({'name':candidate['name'],'predicted':p,'physical':a})
        chosen=max(rows,key=lambda row:row['predicted'])
        best=max(rows,key=lambda row:row['physical'])
        result[score]={'rows':rows,'selected':chosen['name'],'best_physical':best['name'],
                       'regret':best['physical']-chosen['physical'],
                       'predicted_range':max(r['predicted'] for r in rows)-min(r['predicted'] for r in rows),
                       'physical_range':max(r['physical'] for r in rows)-min(r['physical'] for r in rows)}
    return result
