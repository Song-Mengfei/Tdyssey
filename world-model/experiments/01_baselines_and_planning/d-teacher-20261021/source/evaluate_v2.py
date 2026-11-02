"""Isolated v2 teacher experiment using the unchanged common evaluator.

The replacement is process-local dependency injection. No v1 source or other
process is changed. Source provenance is appended even if the run raises.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    draft = BASE / 'v2-draft/simulator_search_v2.py'
    spec = importlib.util.spec_from_file_location('tdyssey_teacher_search_v2_experiment', draft)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import tdyssey.simulator_search as original
    failed_attempts = []

    class TrackedV2(module.SimulatorActionSearchV2):
        def choose(self, current_state, *args, **kwargs):
            began, before = time.monotonic(), dict(self.counts)
            try:
                return super().choose(current_state, *args, **kwargs)
            except TimeoutError:
                # The common evaluator already records this outcome.
                raise
            except Exception as exc:
                failed_attempts.append({
                    'control': current_state.control_step, 'completed': False,
                    'timeout': False, 'error': f'{type(exc).__name__}: {exc}',
                    'seconds': time.monotonic()-began,
                    'counts': {key: value-before[key] for key, value in self.counts.items()},
                })
                raise

    original.SimulatorActionSearch = TrackedV2
    evaluator = runpy.run_path(str(ROOT / 'scripts/evaluate_simulator_search.py'))
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    try:
        evaluator['main']()
    finally:
        report_path = output / 'report.json'
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding='utf-8'))
            report.setdefault('planning_attempts', []).extend(failed_attempts)
            report['teacher_implementation'] = {
                'version': 2,
                'kind': 'isolated_generic_contact_search_experiment',
                'base_evaluator_unchanged': True,
                'process_local_dependency_injection': True,
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), draft, ROOT / 'tdyssey/official.py')},
                'training': False,
            }
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
