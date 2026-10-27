import argparse,contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_pipeline import create_fixture
from trex_force.data import prepare
from trex_force.engine import train

class StopTargetTests(unittest.TestCase):
    def test_target_checkpoint_exits_without_next_step_or_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source';create_fixture(root)
            prep=argparse.Namespace(root=str(root),out=str(Path(temp)/'cache'),finger='left_index',target='force',split='object',seed=42,stride=2,limit_per_split=0)
            with contextlib.redirect_stdout(io.StringIO()):prepare(prep)
            run=Path(temp)/'run'
            args=argparse.Namespace(manifest=str(Path(prep.out)/'manifest.json'),run_dir=str(run),device='cpu',cpu_threads=2,seed=42,lr=3e-4,weight_decay=1e-4,amp=False,amp_dtype='float16',resume=None,epochs=2,data_parallel=False,batch_size=4,workers=0,shuffle_buffer=8,streams_per_finger=2,near_zero_keep=1.,near_zero_threshold=.1,log_every=1,steps_per_epoch=0,eval_batches=0,checkpoint_every=5000,stop_at_step=2,micro_batch_size=2)
            tiny=lambda width,**kwargs:torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(1,width))
            with patch('trex_force.engine.ForceRegressor',tiny),patch('trex_force.engine.evaluate_model',side_effect=AssertionError('Must not validate after stop target')),contextlib.redirect_stdout(io.StringIO()):train(args)
            rows=[json.loads(l) for l in (run/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['global_step'] for r in rows],[1,2])
            cp=torch.load(run/'step_00000002.pt',weights_only=False)
            self.assertEqual(cp['global_step'],2);self.assertIn('data_loader',cp)
            self.assertTrue((run/'TARGET_STEP_COMPLETE.json').is_file());self.assertFalse((run/'COMPLETE.json').exists())
