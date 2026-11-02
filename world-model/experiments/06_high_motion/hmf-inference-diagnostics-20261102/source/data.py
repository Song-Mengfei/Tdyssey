"""Bounded saved-RGB input/truth cache. No model, physics, workers or media.

Only the root's unique single-thread process may construct/use this object.
Model arrays are original float32 RGB/255 and raw absolute XY actions; poses,
letter IDs, truth features and future truth are kept out of model inputs.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib,json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _stat(path):
    s=Path(path).stat()
    return s.st_size,s.st_mtime_ns,s.st_ctime_ns


def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


class DiagnosticData:
    def __init__(self,root,config=None):
        self.root=Path(root).resolve()
        self.config=config or {}
        self.A_spec=read(self.root/'A-spec.json')
        self.B_spec=read(self.root/'B-spec.json')
        self.checkpoints=read(self.root/'checkpoint-map.json')
        self.cache=OrderedDict()
        self.cache_bytes=0
        self.loads=0
        self.max_bytes=int(self.config.get('diagnostic_cache_bytes',32*1024**2))
        self.source_stats={}
        self.trusted_source_stats=self.config.get('_validated_source_stats',{})
        # Saved calibration only; no new GT feature extraction or geometry.
        cal=read(self.A_spec['GT_calibration_path'])
        self.calibration={x['sample_id']:x for x in cal['rows']}

    def _strip(self,row):
        base=int(row.get('parent_anchor',row['anchor']))
        path=Path(row['raw_path']).resolve()
        key=(str(path),base)
        trusted=self.trusted_source_stats.get(str(path))
        current=_stat(path)
        if not trusted or trusted.get('sha256')!=row['raw_sha256'] or current!=(trusted['bytes'],trusted['mtime_ns'],trusted['ctime_ns']):
            raise ValueError('Raw source differs from the worker validated SHA/stat receipt')
        if key in self.cache:
            if current!=self.source_stats[str(path)]:
                raise ValueError('An actually referenced frozen raw source changed')
            self.cache.move_to_end(key)
            return self.cache[key]
        before=current
        with np.load(path,allow_pickle=False) as z:
            rgb=z['rgb'][base-3:base+17].copy()
            actions=z['passed_actions'][base-3:base+16].copy()
            requested=z['requested_actions'][base-3:base+16]
            valid=z['control_valid'][base-3:base+16].copy()
            pusher_xy=z['states'][base:base+17,:2].copy()
        if rgb.shape!=(20,3,96,96) or rgb.dtype!=np.uint8 or actions.shape!=(19,2):
            raise ValueError('Original20RGB/19action strip alignment differs')
        if not valid.all() or not np.array_equal(actions,requested):
            raise ValueError('Strip crosses invalid or changed action timeline')
        if not np.isfinite(actions).all() or (actions<0).any() or (actions>511).any():
            raise ValueError('Unchanged model action contract exceeded')
        if _stat(path)!=before:
            raise ValueError('Source changed while reading saved RGB strip')
        for value in (rgb,actions,pusher_xy):
            value.setflags(write=False)
        size=rgb.nbytes+actions.nbytes+pusher_xy.nbytes
        if size>self.max_bytes:
            raise RuntimeError('A single saved strip exceeds RAM budget')
        while self.cache and self.cache_bytes+size>self.max_bytes:
            _,old=self.cache.popitem(last=False)
            self.cache_bytes-=old['bytes']
        data=dict(rgb=rgb,actions=actions,pusher_xy=pusher_xy,bytes=size)
        self.cache[key]=data
        self.cache_bytes+=size
        self.source_stats[str(path)]=before
        self.loads+=1
        return data

    @staticmethod
    def _rows(batch):
        return batch['rows'] if isinstance(batch,dict) else batch

    def A_inputs(self,batch,mode):
        if mode not in ('real_history','autoregressive'):
            raise ValueError('Unknown A conditioning mode')
        initial=[];actions=[];histories=[]
        for row in self._rows(batch):
            d=self._strip(row)
            initial.append(d['rgb'][:4].astype(np.float32)/np.float32(255))
            actions.append(d['actions'])
            if mode=='real_history':
                histories.append(np.stack([d['rgb'][k:k+4].astype(np.float32)/np.float32(255) for k in range(16)]))
        result=dict(initial_histories=np.stack(initial),absolute_actions=np.stack(actions))
        if histories:
            result['real_histories']=np.stack(histories)
        return result

    def B_inputs(self,batch,action='recorded'):
        if action not in ('recorded','hold','alternative'):
            raise ValueError('Unfrozen B action')
        initial=[];actions=[]
        for row in self._rows(batch):
            d=self._strip(row);phase=int(row['parent_phase'])
            history=d['rgb'][phase:phase+4]
            absolute=d['actions'][phase:phase+4].copy()
            if array_sha(history)!=row['history4_sha256'] or array_sha(absolute[:3])!=row['past3_actions_sha256']:
                raise ValueError('B frozen history/past3 alignment differs')
            current=np.asarray(row['actions'][action])
            if current.shape!=(2,) or not np.isfinite(current).all() or (current<0).any() or (current>511).any():
                raise ValueError('B current action cannot be clipped or transformed')
            # Preserve subpixel hold coordinates even if source actions are ints.
            absolute=absolute.astype(np.result_type(absolute.dtype,current.dtype),copy=False)
            absolute[3]=current
            initial.append(history.astype(np.float32)/np.float32(255));actions.append(absolute)
        return dict(initial_histories=np.stack(initial),absolute_actions=np.stack(actions))

    def truth_arrays(self,batch,section='A',mode=None):
        """Evaluation-only saved truth. Never merge this into model inputs."""
        if section not in ('A','B'):
            raise ValueError('Unknown diagnostic section')
        results=[]
        for row in self._rows(batch):
            d=self._strip(row)
            original=row.get('original_sample_id',row['sample_id'])
            cal=self.calibration[original]
            phase=int(row.get('parent_phase',0)) if section=='B' else 0
            count=1 if section=='B' else 16
            motion=row.get('target_motion',[])
            # B inherited parent target_motion: take only its selected H1.
            selected_motion=motion[phase:phase+1] if section=='B' else motion
            item=dict(sample_id=row['sample_id'],original_sample_id=original,parent_phase=phase,
                initial_last=d['rgb'][phase+3].astype(np.float32)/np.float32(255),
                targets=d['rgb'][phase+4:phase+4+count].astype(np.float32)/np.float32(255),
                real_previous=d['rgb'][phase+3:phase+3+count].astype(np.float32)/np.float32(255),
                initial_last_uint8=d['rgb'][phase+3],
                targets_uint8=d['rgb'][phase+4:phase+4+count],
                real_previous_uint8=d['rgb'][phase+3:phase+3+count],
                truth_features=deepcopy(cal['truth_features'][phase:phase+count+1]),
                target_motion=deepcopy(selected_motion),
                recorded_rotation_deg=deepcopy(row.get('recorded_rotation_deg',[])),
                pusher_xy=d['pusher_xy'][phase].copy(),
                counterfactual_GT_available=False if section=='B' else None)
            if section=='B':
                item['history4']=d['rgb'][phase:phase+4].astype(np.float32)/np.float32(255)
                item['recorded_rotation_deg']=[0.,float(row['true_H1_motion']['signed_rotation_deg'])]
                item['GT_valid_for_action']='recorded'
            results.append(item)
        return results

    scorer_truth=truth_arrays

    def close(self):
        self.cache.clear()
        self.cache_bytes=0
