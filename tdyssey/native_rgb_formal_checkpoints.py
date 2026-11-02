"""Checkpoint file ownership for formal validation selection; no model imports."""
from pathlib import Path
import os
import hashlib


class FormalCheckpointFiles:
    """Keep last plus at most one different best, with no duplicate last/best."""
    def __init__(self,output,*,write_json,read_json):
        self.output=Path(output).resolve();self.write_json=write_json
        self.rank=None;self.best_update=None;self.best_info=None;self.last_info=None
        metadata=self.output/'best.json'
        if metadata.exists():
            value=read_json(metadata)
            if value.get('eligible'):
                self.rank=tuple(value['rank']);self.best_update=int(value['updates'])
                self.best_info=value['checkpoint'];self._owned(self.best_info['path'])
                path=Path(self.best_info['path'])
                if not path.is_file():raise ValueError('Saved best reference is missing')
                with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                if actual!=self.best_info['sha256'] or path.stat().st_size!=self.best_info['bytes']:
                    raise ValueError('Saved best reference bytes differ')
                self.best_info.setdefault('selection_rank',list(self.rank))

    def _owned(self,path):
        path=Path(path).resolve()
        if path.parent!=self.output or path.name not in ('last.pt','best.pt'):
            raise ValueError('Checkpoint operation must stay in its exact formal output directory')
        return path

    def consider(self,rank,update):
        if rank is None:return False
        rank=tuple(rank)
        if len(rank)!=3 or rank[-1]!=update:raise ValueError('Rank must include its formal update')
        if self.rank is None or rank<self.rank:
            self.rank=rank;self.best_update=int(update)
            return True
        return False

    def progress_state(self):
        return dict(rank=None if self.rank is None else list(self.rank),best_update=self.best_update)

    def verify_restored_selection(self,state):
        if state!=self.progress_state():
            raise ValueError('Restored checkpoint selection and best.json disagree or are missing')

    def _record(self):
        last=self.output/'last.pt'
        self.write_json(self.output/'best.json',dict(kind='formal-last-best-reference.v1',
            eligible=self.best_info is not None,rank=None if self.best_info is None else self.best_info['selection_rank'],
            updates=None if self.best_info is None else self.best_info['updates'],checkpoint=self.best_info,
            best_is_last=self.best_info is not None and Path(self.best_info['path']).resolve()==last,
            no_duplicate_checkpoint_payload=True))

    def save_last(self,update,writer):
        last=self._owned(self.output/'last.pt');best=self._owned(self.output/'best.pt')
        # Move a selected prior last before replacement. No second copy of its
        # large payload is created; an atomic writer supplies the new last.
        if (self.best_info is not None and Path(self.best_info['path']).resolve()==last
                and int(self.best_info['updates'])!=int(update)):
            if not last.is_file():raise ValueError('Selected previous last disappeared')
            os.replace(last,best)
            self.best_info=dict(self.best_info,path=str(best))
            self._record()
        info=writer(last)
        if Path(info['path']).resolve()!=last or info['updates']!=update:
            raise ValueError('Writer returned the wrong complete last boundary')
        self.last_info=info
        if self.best_update==update:
            self.best_info=dict(info,selection_rank=list(self.rank))
            best.unlink(missing_ok=True)
        self._record()
        return info
