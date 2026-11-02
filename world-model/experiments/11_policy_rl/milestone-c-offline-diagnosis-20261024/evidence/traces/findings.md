# Offline saved-trace diagnosis

Use the figures in `figures-reviewed/`; initial exported figures remain unchanged.

- All20 L recorded trajectories (10cases x2methods) stay below0.491coverage; none reaches0.90. L100002/5479 stays at exactly0coverage for all300controls in both methods.
- T100001/5478 starts at identical0.16896148coverage; base reaches strictsuccess at240, delta reaches its maximum0.9493688 at287 and ends0.9019827 at300. Delta controls286..300 form one15-control near-success segment, without a success.
- T100000/5478 and T100002/5478 show higher delta peak but lower finalcoverage. L100000/5479 shows lower delta peak but higher finalcoverage. These directly show why peak and final summaries can move oppositely.
- At shared ordinal200/201/202 in lostTcase baselinecoverage rises while deltacoverage falls; ordinal alignment does not mean identical physicalstates.
- Peak-to-final losses, zero-coverage spans and command divergence are observed quantities; they do not identify contacts, object orientation or causes.

The observed L plateau is far below the near-success band; it is not a collection of threshold near misses. The traces alone cannot distinguish geometry/contact effects, representation limits, reward effects or insufficient learning.

One-factor proposal only: compare beta=0 against the current beta=1 reward shaping under a separately authorized protocol, keeping interaction caps, training seed/scene schedule, actor subset, optimizer and sampler fixed. Judge predeclared T/L success on held-out scene blocks; do not tune on this development set. No such run is authorized or executed here.
