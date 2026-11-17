"""Actual flushed events, service failure isolation and portable run plans."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.launch_training import build_jobs
from scripts.view_training import read_events
from tdyssey.tracking import RunLogger, TrackingConfig, evaluation_scalars, evaluation_score


class TrackingChecks(unittest.TestCase):
    def test_local_events_are_immediately_readable_and_partial_tail_is_tolerated(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            logger=RunLogger(path)
            logger.log({'train/loss':.5},3)
            result=read_events(path)
            self.assertEqual(result['events'][0]['metrics']['progress/optimizer_step'],3)
            self.assertEqual(result['status'],'running')
            logger.finish('complete')
            with (path/'metrics.jsonl').open('a') as stream:
                stream.write('{"unfinished":')
            self.assertEqual(len(read_events(path)['events']),1)
            self.assertEqual(read_events(path)['status'],'complete')

    def test_remote_failure_does_not_lose_local_metrics_and_event_cap_is_bounded(self):
        service=SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(ConnectionError()),
                                Settings=lambda **kwargs: kwargs)
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules,{'wandb':service}):
            with self.assertWarns(UserWarning):
                logger=RunLogger(root,TrackingConfig(mode='online',max_event_bytes=1024))
            with self.assertWarns(UserWarning):
                for i in range(20):
                    logger.log({'train/loss':.5},i)
            logger.finish('complete')
            self.assertLessEqual((Path(root)/'metrics.jsonl').stat().st_size,1024)
            status=json.loads((Path(root)/'tracking.json').read_text())
            self.assertEqual(status['external_status'],'failed')
            self.assertTrue(status['local_events_capped'])

    def test_shape_metrics_and_matrix_preserve_evaluation_protocol(self):
        episodes=[dict(shape='T',success=True,final_coverage=.96,peak_coverage=.97,controls=20),
                  dict(shape='L',success=False,final_coverage=.1,peak_coverage=.4,controls=300)]
        self.assertEqual(evaluation_scalars(episodes)['eval/L/success'],0)
        self.assertEqual(evaluation_score(episodes),(.5,.53))
        jobs=build_jobs({'eval_seeds':[100000,100001]},[42,43],['nft','advantage_flow'],['lora'])
        self.assertEqual(len(jobs),4)
        self.assertEqual({j['config']['eval_seeds'] for j in jobs},{(100000,100001)})
        self.assertEqual({j['config']['seed'] for j in jobs},{42,43})
        with self.assertRaises(ValueError):
            build_jobs({},[42,42],['nft'],['lora'])


if __name__=='__main__':
    unittest.main()
