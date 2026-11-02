"""Immutable cost snapshots avoid contention on the old hot ledger filename."""
from copy import deepcopy
from collections import defaultdict
from ledger import Ledger

class ResumeLedger(Ledger):
    def __init__(self,io,previous):
        arms=('BASE','SNR','GDL','FFL')
        phases={**{'preflight/'+x:128 for x in ('REFERENCE','BASE','SNR','GDL','FFL')},**{'train/'+a:2048 for a in arms},**{'eval/'+a:(3776 if a=='SNR' else 3744) for a in arms}}
        super().__init__(io,phases,23296,category_caps=dict(preflight=128,train=8192,eval=15008),batch_cap=6016)
        self.counts=defaultdict(lambda:dict(images_attempted=0,images_completed=0,batches_attempted=0,batches_completed=0),deepcopy(previous['counts']))
        for field in ('attempts','updates','exposures','backward'):setattr(self,field,defaultdict(int,deepcopy(previous[field])))
        self.extra=deepcopy(previous['extra']);self.extra['original_aborted_pre_body_reservations']=deepcopy(previous['inflight'])
        self.extra['original_reservation_cost_retained_images']=4
        self.stack=[];self.inflight=[];self.sequence=0
    def persist(self):
        self.sequence+=1
        self.io.write_json(f'continuation-events/{self.sequence:06d}.json',self.snapshot(),exclusive=True)

