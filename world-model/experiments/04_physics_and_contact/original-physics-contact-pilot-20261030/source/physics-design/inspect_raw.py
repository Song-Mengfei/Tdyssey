from pathlib import Path
import json
import numpy as np
p=Path(__file__).resolve().parents[2]/'r3b-playground-diagnostics-20261030/recordings/recording-7917cd3c57594ca1b4c64401251b5e97/raw'
for name in ['initial-session.npz','batch-00000291.npz']:
 with np.load(p/name,allow_pickle=False) as z:
  print(name,[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
  for k in z.files:
   if z[k].dtype.kind in 'US': print(k,z[k].item()[:15000])
