# Stage2C joint reference and four CPU samples

Passed: source/split/hash-bound396 TRAIN +162 seen_val;0 unseen/probe entries. Original NPZ files were referenced, never copied.

H1/H4 catalogs: {'train': {'h1': 24156, 'h4': 22968, 'trajectories': 396, 'cache_cap_bytes': 1048576, 'initial_cache_bytes': 0}, 'seen_val': {'h1': 9882, 'h4': 9396, 'trajectories': 162, 'cache_cap_bytes': 1048576, 'initial_cache_bytes': 0}}.

CPU batches: {'train': {'image': [2, 5, 3, 96, 96], 'action': [2, 4, 2]}, 'seen_val': {'image': [2, 5, 3, 96, 96], 'action': [2, 4, 2]}}. Images/actions keep originalfloat32[0,1]/[-1,1], five CHW RGB frames and four real actions.

Only four predeclared direct Dataset items were read/rendered, uint8LRU≤1MiB/dataset. Pure training data_identity accepts the canonical joint manifest/audit/split hash.

No physics control, model/denoiser/weight, optimizer, training iterator or index draw. This is data compatibility evidence, not model performance or training authorization.
