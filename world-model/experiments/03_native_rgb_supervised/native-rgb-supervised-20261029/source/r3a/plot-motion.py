"""Standalone scientific figures from completed recorded R3A measurements."""
import json
from pathlib import Path


def main():
    stage=Path(__file__).resolve().parent
    raw=json.loads((stage/'report/physics-audit.json').read_text(encoding='utf-8'))
    if raw['trajectories']!=912:raise ValueError('Plot the complete real library, not a small favorable subset')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    letters=list('TLOCXB');x=np.arange(len(letters));colors=plt.cm.tab10(np.arange(6))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for row,horizon in enumerate((1,4)):
        for col,quantity in enumerate(('translation_image96px','signed_rotation_deg')):
            ax=axes[row,col]
            for i,letter in enumerate(letters):
                q=raw['motion']['letter:'+letter]['H'+str(horizon)][quantity]
                ax.errorbar(i,q['median'],yerr=[[q['median']-q['p10']],[q['p90']-q['median']]],fmt='o',capsize=5,color=colors[i])
            ax.set_xticks(x,letters);ax.set_title(f'H{horizon}: median and P10–P90, all new trajectories')
            if col==0:ax.set_ylabel('Area-centroid translation (96 px image pixels)');ax.axhline(2,color='.45',ls='--',lw=1)
            else:
                ax.set_ylabel('Signed rotation (degrees)');ax.axhline(5,color='.45',ls='--',lw=1);ax.axhline(-5,color='.45',ls='--',lw=1)
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('R3A recorded H1/H4 movement — no model prediction')
    fig.savefig(stage/'report/motion-distributions.png');plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    for i,letter in enumerate(letters):
        records=raw['control_time_curves']['letter:'+letter];times=[r['simulation_start_seconds']for r in records]
        axes[0].plot(times,[r['translation_image96px']['median']for r in records],label=letter,color=colors[i])
        axes[1].plot(times,[r['contact_fraction']for r in records],color=colors[i])
        axes[2].plot(times,[r['penetration_worldpx']['p95']for r in records],color=colors[i])
    axes[0].set_ylabel('Median per-control translation\n(96 px image pixels)');axes[0].legend(ncol=6)
    axes[1].set_ylabel('Fraction of trajectories\nwith actual contact');axes[1].set_ylim(-.02,1.02)
    axes[2].set_ylabel('P95 control-maximum\npenetration (world pixels)');axes[2].set_xlabel('Recorded control start time (seconds)');axes[2].axhline(2,color='.4',ls='--',label='Abort threshold')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Real 48-control time curves, including stalls and releases')
    fig.savefig(stage/'report/control-time-curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,split in zip(axes,('train','seen_val')):
        for offset,field,label in [(-.25,'translation_H1','Translation ≥ 2 px'),(0,'positive_H1','Rotation ≥ +5°'),(.25,'negative_H1','Rotation ≤ −5°')]:
            values=[raw['independent_support_by_split_and_letter'][split][l][field]for l in letters]
            ax.bar(x+offset,values,.23,label=label)
        ax.set_xticks(x,letters);ax.set_title('TRAIN'if split=='train'else'New seen validation');ax.set_ylabel('Independent trajectories supporting H1 motion');ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Independent support is separate from repeated window exposure')
    fig.savefig(stage/'report/independent-motion-support.png');plt.close(fig)
    print('Three complete-library motion figures saved; 0 model or physics work')


if __name__=='__main__':main()
