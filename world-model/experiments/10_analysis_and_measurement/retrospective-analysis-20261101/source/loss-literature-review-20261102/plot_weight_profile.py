"""Plot frozen loss algebra only; no model/data/GPU or numerical pools."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']:
    os.environ[k]='1'
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=pathlib.Path(__file__).resolve().parent
s=np.geomspace(.002,20,600)
rho=.25/(s*s+.01)
w=np.minimum(rho,5)/rho
native=4+1/(s*s+.01)
fig,axes=plt.subplots(1,2,figsize=(10.5,3.9))
axes[0].semilogx(s,w,color='#d17a18',label='Min-SNR multiplier')
axes[0].set_ylabel('Multiplier on native residual MSE')
axes[0].set_ylim(0,1.08)
axes[1].semilogx(s,native,label='Native equivalent image-MSE weight',color='#2469a9')
axes[1].semilogx(s,native*w,label='After frozen Min-SNR adaptation',color='#d17a18')
axes[1].set_ylabel('Equivalent image-MSE weight')
for ax in axes:
    ax.axvspan(.002,.2,color='#7ba9ca',alpha=.10)
    ax.axvline(.2,color='#777777',linestyle=':',label='draw sigma = 0.2')
    ax.set_xlabel('Original drawn sigma (unchanged distribution)')
    ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
fig.suptitle('GPC nominal SNR: sigma_data=0.5, sigma_eff^2 = sigma_draw^2 + 0.1^2')
fig.tight_layout()
fig.savefig(p/'GPC-MinSNR-weight-profile.png',dpi=140)
