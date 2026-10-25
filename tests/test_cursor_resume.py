import argparse
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from test_pipeline import create_fixture
from trex_force.data import FrameDataset, prepare
from trex_force.cursor_data import CursorFrameDataset
from trex_force.engine import train


def fixture(temp):
    root = Path(temp)/'source'
    create_fixture(root)
    args = argparse.Namespace(root=str(root),out=str(Path(temp)/'cache'),finger='left_index',
        target='force',split='episode',seed=42,stride=1,limit_per_split=0)
    with contextlib.redirect_stdout(io.StringIO()):prepare(args)
    return Path(args.out)/'manifest.json'


def assert_batches(test, expected, actual):
    test.assertEqual(len(expected),len(actual))
    for a,b in zip(expected,actual):
        test.assertEqual(set(a),set(b))
        for key in a:
            if torch.is_tensor(a[key]):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            else:test.assertEqual(a[key],b[key])


class CursorTests(unittest.TestCase):
    def test_matches_original_sampling_and_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest=fixture(temp)
            for workers in (0,2,4):
                for streams,keep in ((0,1.),(2,1.),(2,.35)):
                    kw=dict(seed=42,shuffle_buffer=7,streams_per_finger=streams,
                            near_zero_keep=keep,near_zero_threshold=100.)
                    old=FrameDataset(manifest,'train',**kw);new=CursorFrameDataset(manifest,'train',**kw)
                    old.epoch=new.epoch=2
                    expected=list(torch.utils.data.DataLoader(old,batch_size=5,num_workers=workers))
                    actual=list(StatefulDataLoader(new,batch_size=5,num_workers=workers))
                    assert_batches(self,expected,actual)

    def test_restore_worker_cursors_with_prefetch_and_partial_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest=fixture(temp)
            for workers in (0,2,4):
                for cut in (1,3,7):
                    def loader():
                        return StatefulDataLoader(CursorFrameDataset(manifest,'train',shuffle_buffer=7,
                            streams_per_finger=2),batch_size=5,num_workers=workers,
                            **({'prefetch_factor':2} if workers else {}))
                    original=loader();it=iter(original)
                    prefix=[next(it) for _ in range(cut)]
                    snapshot=copy.deepcopy(original.state_dict())
                    expected=list(it)
                    restored=loader();restored.load_state_dict(snapshot)
                    actual=list(restored)
                    assert_batches(self,expected,actual)
                    keys=lambda batches:[(int(e),int(f)) for b in batches for e,f in zip(b['episode'],b['frame'])]
                    self.assertFalse(set(keys(prefix)) & set(keys(actual)))
                    self.assertEqual(len(keys(prefix+actual)),48)

    def test_restored_iterator_seeks_only_to_unconsumed_frames(self):
        from trex_force import cursor_data
        with tempfile.TemporaryDirectory() as temp:
            manifest=fixture(temp)
            ds=CursorFrameDataset(manifest,'train',shuffle_buffer=7,streams_per_finger=2)
            it=iter(ds);prefix=[next(it) for _ in range(25)];saved=it.state_dict()
            expected=list(it);restored=iter(ds);restored.load_state_dict(saved)
            # Decoded requests after restore must exclude already emitted samples.
            used={(x['episode'],x['frame']) for x in prefix}
            buffered=restored.buffer[restored.drain_cursor:] if restored.draining else restored.buffer
            for rid,index in buffered:
                row=ds.rows[rid];_,frames,_,_=restored.row_data(rid)
                self.assertNotIn((row['episode_index'],int(frames[index])),used)
            consumed_pts={round(x['video_timestamp'],5) for x in prefix}
            with patch('trex_force.cursor_data.decode_selected',wraps=cursor_data.decode_selected) as decode:
                actual=list(restored)
            requested={round(float(t),5) for call in decode.call_args_list for t in call.args[1]}
            self.assertFalse(consumed_pts & requested)
            assert_batches(self,expected,actual)

    def test_resume_matches_uninterrupted_optimizer_and_losses(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest=fixture(temp)
            args=argparse.Namespace(manifest=str(manifest),run_dir=str(Path(temp)/'full'),
                device='cpu',cpu_threads=2,seed=42,lr=3e-4,weight_decay=1e-4,amp=False,amp_dtype='float16',
                resume=None,epochs=2,data_parallel=False,batch_size=5,workers=4,shuffle_buffer=7,
                streams_per_finger=2,near_zero_keep=1.,near_zero_threshold=.1,log_every=1,
                steps_per_epoch=0,eval_batches=0,checkpoint_every=3)
            def tiny(width,**kwargs):
                return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),
                    torch.nn.Linear(1,8),torch.nn.Dropout(.2),torch.nn.Linear(8,width))
            with patch('trex_force.engine.ForceRegressor',tiny),contextlib.redirect_stdout(io.StringIO()):train(args)
            full=Path(args.run_dir)
            for cut in (3,9,12):
                resumed=Path(temp)/f'resumed{cut}'
                args.run_dir=str(resumed);args.resume=str(full/f'step_{cut:08d}.pt')
                with patch('trex_force.engine.ForceRegressor',tiny),contextlib.redirect_stdout(io.StringIO()):train(args)
                a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
                self.assertEqual(a['global_step'],b['global_step'])
                for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
                for key in a['optimizer']['state']:
                    for field in a['optimizer']['state'][key]:
                        torch.testing.assert_close(a['optimizer']['state'][key][field],b['optimizer']['state'][key][field],rtol=0,atol=0)
                old=[json.loads(l) for l in (full/'steps.jsonl').read_text().splitlines() if json.loads(l)['global_step']>cut]
                new=[json.loads(l) for l in (resumed/'steps.jsonl').read_text().splitlines()]
                self.assertEqual(len(old),len(new))
                for x,y in zip(old,new):
                    for key in ('global_step','epoch','batch','samples','loss','loss_epoch_running_mean','loss_mean_last_100_steps'):
                        self.assertEqual(x[key],y[key],key)
