"""Bounded immutable uint8 segment cache and frozen3000batch sampler. No Torch."""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stat(path):
    s=Path(path).stat();return s.st_size,s.st_mtime_ns,s.st_ctime_ns


class FitDataset:
    def __init__(self,root,config=None):
        self.root=Path(root).resolve();self.index=read(self.root/'selection/index.json');summary=read(self.root/'selection/summary.json')
        if sha(self.root/'selection/index.json')!=summary['index_sha256']:raise ValueError('Frozen selection index changed')
        self.rows=self.index['targets'];self.segments=self.index['train48'];self.cache={};self.source_stats={};self.loads=0;self.cache_bytes=0
        self.maximum_cache_bytes=int((config or {}).get('maximum_fit_cache_bytes',48*1024**2))
        for row in self.segments:self._load(row)
        if len(self.rows)!=768 or len({r['target_id'] for r in self.rows})!=768:raise ValueError('Expected768unique training targets')
    def _load(self,row):
        key=row['segment_id']
        if key in self.cache:return self.cache[key]
        path=Path(row['raw_path']);before=stat(path)
        if sha(path)!=row['raw_sha256']:raise ValueError('Frozen source RGB/action archive changed')
        a=int(row['anchor'])
        with np.load(path,allow_pickle=False) as z:
            rgb=z['rgb'][a-3:a+17].copy();absolute=z['passed_actions'][a-3:a+16].copy();requested=z['requested_actions'][a-3:a+16];valid=z['control_valid'][a-3:a+16]
        if rgb.dtype!=np.uint8 or rgb.shape!=(20,3,96,96) or absolute.shape!=(19,2) or not valid.all() or not np.array_equal(absolute,requested):raise ValueError('20RGB/19action segment alignment differs')
        if not np.isfinite(absolute).all() or (absolute<0).any() or (absolute>511).any():raise ValueError('Original model action contract changed')
        normalized=absolute.astype(np.float32)/np.float32(255.5)-np.float32(1)
        if stat(path)!=before:raise ValueError('Source mutated while caching')
        rgb.setflags(write=False);normalized.setflags(write=False)
        self.cache_bytes+=rgb.nbytes+normalized.nbytes
        if self.cache_bytes>self.maximum_cache_bytes:raise RuntimeError('Bounded fit cache exhausted')
        self.source_stats[str(path)]=before;self.cache[key]=dict(rgb=rgb,action=normalized,raw_path=str(path));self.loads+=1
        return self.cache[key]
    def _indices(self,indices):
        values=[int(x['target_index']) if isinstance(x,dict) else int(x) for x in indices]
        if any(x<0 or x>=len(self.rows) for x in values):raise IndexError('Unknown frozen target')
        return values
    def model_batch(self,indices):
        values=self._indices(indices);images=[];actions=[];touched=set()
        for i in values:
            row=self.rows[i];segment=self.segments[row['segment_index']];data=self.cache[segment['segment_id']];h=int(row['phase'])
            touched.add(data['raw_path']);images.append(data['rgb'][h:h+5]);actions.append(data['action'][h:h+4])
        if any(stat(path)!=self.source_stats[path] for path in touched):raise ValueError('Frozen source changed during fitting')
        return dict(image=np.stack(images).astype(np.float32)/np.float32(255),action=np.stack(actions).astype(np.float32))
    batch=model_batch
    def metadata(self,indices):return [dict(self.rows[i]) for i in self._indices(indices)]
    def evaluation_rows(self,name):
        return self.index[{'train':'train48','seen_val':'seen_val24','failures':'failures3','panel':'fixed_train_panel12'}.get(name,name)]
    def close(self):self.cache.clear();self.cache_bytes=0


class FrozenSampler:
    def __init__(self,root,config=None):
        self.root=Path(root).resolve();path=self.root/'selection/order.json';summary=read(self.root/'selection/summary.json')
        self.order_sha256=sha(path);self.order=read(path)
        if self.order_sha256!=summary['order_sha256'] or self.order['selection_index_sha256']!=sha(self.root/'selection/index.json'):raise ValueError('Frozen order/selection changed')
        self.cursor=0;self.batches=self.order['batches']
        if len(self.batches)!=3000 or any(len(b)!=16 for b in self.batches):raise ValueError('Only frozen3000x16 fitting schedule allowed')
    def next_batch(self):
        if self.cursor>=len(self.batches):raise StopIteration('Authorized3000batches consumed')
        batch=list(self.batches[self.cursor]);self.cursor+=1;return batch
    next16=next_batch
    def state_dict(self):return dict(cursor=self.cursor,order_sha256=self.order_sha256,selection_index_sha256=self.order['selection_index_sha256'],seed=self.order['seed'],rng='All random choices frozen into order; runtime state is cursor only')
    def load_state_dict(self,state):
        if state['order_sha256']!=self.order_sha256 or state['selection_index_sha256']!=self.order['selection_index_sha256'] or state['seed']!=self.order['seed']:raise ValueError('Resume sampler identity differs')
        cursor=state['cursor']
        if type(cursor)is not int or not 0<=cursor<=3000:raise ValueError('Invalid reserved sampler cursor')
        self.cursor=cursor
    def planned_exposure_prefix(self):return dict(Counter(str(i) for b in self.batches[:self.cursor] for i in b))
