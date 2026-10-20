"""Dependency-free limits and metric aggregation for the first visual model pilot."""

from dataclasses import dataclass
import math

from .collection import integer


@dataclass(frozen=True)
class WorldModelConfig:
    epochs: int = 10
    batch_size: int = 8
    future: int = 4
    max_steps: int = 750
    seconds: int = 300
    seed: int = 42
    learning_rate: float = .0003
    max_bytes: int = 32*1024*1024

    def __post_init__(self):
        for value,name,low,high in ((self.epochs,'epochs',1,50),(self.batch_size,'batch size',1,32),
                (self.future,'future',1,4),(self.max_steps,'optimizer steps',1,3000),
                (self.seconds,'seconds',1,900),(self.seed,'seed',0,2**32-1),
                (self.max_bytes,'output bytes',1024*1024,64*1024*1024)):
            integer(value,name,low,high)
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01:
            raise ValueError('Learning rate must be finite and in (0,.01]')


class MetricTotals:
    """Forecast-step weighted means; empty contact groups stay absent, not zero."""

    def __init__(self, future):
        self.rows = {str(h):{name:{'count':0,'sums':{}} for name in ('all','contact','noncontact')}
                     for h in range(1,future+1)}

    def add(self, horizon, group, count, sums):
        row=self.rows[str(horizon)][group]
        row['count']+=count
        for key,value in sums.items():
            if not math.isfinite(value):
                raise ValueError('Non-finite evaluation metric')
            row['sums'][key]=row['sums'].get(key,0)+value

    def result(self):
        return {h:{name:{'forecast_steps':r['count'],**{key:total/r['count'] if r['count'] else None
                    for key,total in r['sums'].items()}} for name,r in groups.items()} for h,groups in self.rows.items()}
