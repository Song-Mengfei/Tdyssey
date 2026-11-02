"""Read-only adapter to the exact prior A/B pool/cache, no duplicated library."""
from pathlib import Path
import importlib.util

OLD=Path(__file__).resolve().parents[1]/'generated-history-ab-20261101'
spec=importlib.util.spec_from_file_location('_cd_original_saved_pool',OLD/'pool.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

class SegmentLoader(original.SegmentLoader):
    def __init__(self,root,config):
        super().__init__(OLD,config)
        self.current_root=Path(root)
        if original.sha(self.current_root/'pool-index.json')!=original.sha(OLD/'pool-index.json') or original.sha(self.current_root/'order.json')!=original.sha(OLD/'order.json'):raise ValueError('Current pool/order are not exact unchanged A/B identities')
