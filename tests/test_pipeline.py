import argparse
import copy
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from trex_force.data import FINGERS, FrameDataset, decode_selected, finger_offset, prepare, split_episodes
from trex_force.engine import Metrics, train
from trex_force.model import ForceRegressor


def create_fixture(root):
    (root/'meta/episodes/chunk-000').mkdir(parents=True)
    (root/'data/chunk-000').mkdir(parents=True)
    key = 'observation.images.tactile_left_raw_index'
    video = root/f'videos/{key}/chunk-000/file-000.mp4'
    video.parent.mkdir(parents=True)
    with av.open(str(video), 'w') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height = 320, 240
        stream.pix_fmt = 'yuvj420p'; stream.codec_context.color_range = 2
        stream.options = {'qp': '0', 'preset': 'ultrafast'}
        for i in range(72):
            frame = av.VideoFrame.from_ndarray(np.full((240,320), i+30, np.uint8), format='gray')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    info = {'fps': 30, 'total_episodes': 6, 'total_frames': 72}
    (root/'meta/info.json').write_text(json.dumps(info))
    episodes=[]; data=[]
    for eid in range(6):
        episodes.append({'episode_index': eid, 'length': 12, 'object': f'object{eid//2}',
                         'motor_primitive': 'press', f'videos/{key}/chunk_index': 0,
                         f'videos/{key}/file_index': 0, f'videos/{key}/from_timestamp': eid*12/30,
                         f'videos/{key}/to_timestamp': (eid+1)*12/30})
        for i in range(12):
            force = np.zeros(60, np.float32)
            force[6:9] = [i*.1+eid, i*.2-eid, i*.3+1]
            data.append({'episode_index':eid, 'frame_index':i, 'timestamp':np.float32(i/30).item(),
                         'observation.tactile_force':force.tolist()})
    pq.write_table(pa.Table.from_pylist(episodes), root/'meta/episodes/chunk-000/file-000.parquet')
    # Episode 2 deliberately crosses two Parquet files.
    for j, rows in enumerate((data[:31],data[31:])):
        pq.write_table(pa.Table.from_pylist(rows), root/f'data/chunk-000/file-{j:03d}.parquet')
    return video


class Tests(unittest.TestCase):
    def test_all_fingers_episode_split_and_label_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source'; video=create_fixture(root)
            fingers=[f'{side}_{f}' for side in ('left','right') for f in FINGERS]
            meta_file=root/'meta/episodes/chunk-000/file-000.parquet'
            rows=pq.read_table(meta_file).to_pylist()
            for finger in fingers:
                key=f'observation.images.tactile_{finger.split("_")[0]}_raw_{finger.split("_")[1]}'
                target=root/f'videos/{key}/chunk-000/file-000.mp4'
                if target!=video:
                    target.parent.mkdir(parents=True);shutil.copyfile(video,target)
                for r in rows:
                    for field in ('chunk_index','file_index','from_timestamp','to_timestamp'):
                        r[f'videos/{key}/{field}']=r[f'videos/observation.images.tactile_left_raw_index/{field}']
            pq.write_table(pa.Table.from_pylist(rows),meta_file)
            for file in (root/'data').rglob('*.parquet'):
                records=pq.read_table(file).to_pylist()
                for r in records:
                    r['observation.tactile_force']=[float(j*100+k*10+r['episode_index']+r['frame_index']*.1)
                                                   for j in range(10) for k in range(6)]
                pq.write_table(pa.Table.from_pylist(records),file)
            args=argparse.Namespace(root=str(root),out=str(Path(temp)/'all'),finger='all',target='force',
                                    split='episode',seed=42,stride=2,limit_per_split=0)
            with contextlib.redirect_stdout(io.StringIO()):prepare(args)
            manifest=Path(args.out)/'manifest.json';meta=json.loads(manifest.read_text())
            self.assertEqual(len(meta['episodes']),60)
            self.assertEqual(meta['statistics']['count'],240)
            split_ids={}
            for split in ('train','val','test'):
                items=list(FrameDataset(manifest,split,shuffle_buffer=4,streams_per_finger=2))
                self.assertEqual(len(items),sum(r['samples'] for r in meta['episodes'] if r['split']==split))
                self.assertEqual(len(items),len({(r['episode'],r['finger'],r['frame']) for r in items}))
                split_ids[split]={r['episode'] for r in items}
                self.assertEqual({r['finger'] for r in items},set(fingers))
                for r in items:
                    j=fingers.index(r['finger'])
                    np.testing.assert_allclose(r['target'],[j*100+k*10+r['episode']+r['frame']*.1 for k in range(3)],rtol=1e-6)
            self.assertFalse(split_ids['train'] & split_ids['test'])
            self.assertFalse(split_ids['val'] & split_ids['test'])
            ds=FrameDataset(manifest,'train',shuffle_buffer=16,streams_per_finger=2)
            keys=lambda items: [(r['episode'],r['finger'],r['frame']) for r in items]
            expected=keys(list(ds))
            self.assertEqual(expected,keys(list(ds)))
            multi=list(torch.utils.data.DataLoader(ds,batch_size=None,num_workers=2))
            self.assertEqual(sorted(expected),sorted(keys(multi)))
            self.assertEqual(len(expected),len(set(keys(multi))))
            self.assertGreaterEqual(len({r['finger'] for r in multi[:40]}),6)
            from trex_force.cursor_data import CursorFrameDataset
            from torchdata.stateful_dataloader import StatefulDataLoader
            cursor_ds=CursorFrameDataset(manifest,'train',shuffle_buffer=16,streams_per_finger=2)
            cursor_loader=StatefulDataLoader(cursor_ds,batch_size=None,num_workers=2)
            self.assertEqual(keys(multi),keys(list(cursor_loader)))
            cursor_loader=StatefulDataLoader(cursor_ds,batch_size=7,num_workers=4)
            it=iter(cursor_loader)
            for _ in range(5):next(it)
            saved=copy.deepcopy(cursor_loader.state_dict())
            expected_tail=list(it)
            restored=StatefulDataLoader(CursorFrameDataset(manifest,'train',shuffle_buffer=16,streams_per_finger=2),batch_size=7,num_workers=4)
            restored.load_state_dict(saved)
            actual_tail=list(restored)
            self.assertEqual(len(expected_tail),len(actual_tail))
            for a,b in zip(expected_tail,actual_tail):
                for key in ('episode','frame','image','target'):
                    torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                self.assertEqual(a['finger'],b['finger'])

            ds.epoch=1
            self.assertNotEqual(expected,keys(list(ds)))

    def test_step_logs_and_epoch_checkpoint_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source';create_fixture(root)
            prep=argparse.Namespace(root=str(root),out=str(Path(temp)/'cache'),finger='left_index',
                target='force',split='object',seed=42,stride=2,limit_per_split=0)
            with contextlib.redirect_stdout(io.StringIO()): prepare(prep)
            run=Path(temp)/'run'
            args=argparse.Namespace(manifest=str(Path(prep.out)/'manifest.json'),run_dir=str(run),
                device='cpu',cpu_threads=2,seed=42,lr=3e-4,weight_decay=1e-4,amp=False,amp_dtype='bfloat16',
                resume=None,epochs=2,data_parallel=False,batch_size=4,workers=0,shuffle_buffer=8,
                streams_per_finger=2,near_zero_keep=1.,near_zero_threshold=.1,log_every=1,
                steps_per_epoch=0,eval_batches=0,checkpoint_every=2)
            tiny=lambda width, **kwargs: torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),torch.nn.Linear(1,width))
            with patch('trex_force.engine.ForceRegressor',tiny),contextlib.redirect_stdout(io.StringIO()):train(args)
            logs=[json.loads(x) for x in (run/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['global_step'] for r in logs],list(range(1,7)))
            for epoch in (1,2):
                part=[r for r in logs if r['epoch']==epoch]
                self.assertAlmostEqual(part[-1]['loss_epoch_running_mean'],np.mean([r['loss'] for r in part]))
                cp=torch.load(run/f'epoch_{epoch:04d}.pt',weights_only=False)
                self.assertEqual(cp['next_epoch'],epoch)
            self.assertEqual((run/'last.pt').stat().st_ino,(run/'epoch_0002.pt').stat().st_ino)
            self.assertTrue((run/'best.pt').is_file())
            self.assertEqual(sorted(p.name for p in run.glob('step_*.pt')),
                             ['step_00000002.pt','step_00000004.pt','step_00000006.pt'])
            for step, epoch in [(2,0),(4,1),(6,1)]:
                cp=torch.load(run/f'step_{step:08d}.pt',weights_only=False)
                self.assertEqual(cp['global_step'],step)
                self.assertEqual(cp['next_epoch'],epoch)
                self.assertTrue(cp['epoch_incomplete'])
                self.assertEqual(cp['resume_policy'],'restore_data_cursor_v1')
                self.assertIn('data_loader',cp)
                self.assertTrue(cp['optimizer']['state'])
                restored=tiny(3);restored.load_state_dict(cp['model'])
                self.assertTrue(torch.isfinite(restored(torch.ones(1,1,240,320))).all())
            self.assertEqual((run/'last_step.pt').stat().st_ino,(run/'step_00000006.pt').stat().st_ino)
            self.assertFalse(list(run.glob('*.tmp')))


    def test_force_offsets(self):
        self.assertEqual(finger_offset('left_index'), 6)
        self.assertEqual(finger_offset('right_pinky'), 54)
        with self.assertRaises(ValueError): finger_offset('wrong_index')

    def test_group_disjoint_and_deterministic(self):
        rows=[{'episode_index':i,'object':str(i//2)} for i in range(40)]
        a=split_episodes(rows,'object',42); b=split_episodes(rows,'object',42)
        self.assertEqual(a,b)
        sets=[{r['object'] for r in a[s]} for s in ['train','val','test']]
        for i in range(3):
            for j in range(i): self.assertFalse(sets[i] & sets[j])

    def test_decode_time_and_full_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source'; video=create_fixture(root)
            frames=list(decode_selected(video, np.array([.4,.5,1.]),30))
            self.assertEqual(len(frames),3)
            self.assertAlmostEqual(frames[0][1],.4,places=6)
            self.assertEqual(int(frames[0][0][100,100]),42)
            self.assertEqual(int(frames[2][0][100,100]),60)
            with self.assertRaises(ValueError): list(decode_selected(video,np.array([10.]),30))
            args=argparse.Namespace(root=str(root),out=str(Path(temp)/'cache'),finger='left_index',
                target='force',split='object',seed=42,stride=2,limit_per_split=0)
            with contextlib.redirect_stdout(io.StringIO()): prepare(args)
            manifest=Path(args.out)/'manifest.json'; meta=json.loads(manifest.read_text())
            self.assertEqual(len(meta['episodes']),6)
            self.assertEqual(meta['statistics']['count'],12)
            train=np.concatenate([np.load(manifest.parent/r['labels'])['target'] for r in meta['episodes'] if r['split']=='train'])
            np.testing.assert_allclose(train.mean(0),meta['statistics']['mean'],atol=1e-6)
            for split in ['train','val','test']:
                ds=FrameDataset(manifest,split,shuffle_buffer=4)
                items=list(ds); self.assertEqual(len(items),12)
                self.assertEqual(len({(x['episode'],x['frame']) for x in items}),12)
                for item in items:
                    eid,f=item['episode'],item['frame']
                    np.testing.assert_allclose(item['target'],[f*.1+eid,f*.2-eid,f*.3+1],atol=1e-6)
                    self.assertAlmostEqual(float(item['image'][0,100,100]),(eid*12+f+30)/255,places=6)
            with self.assertRaises(FileExistsError): prepare(args)
            # Changing cached labels must not silently change a checkpoint's supervision.
            entry=next(r for r in meta['episodes'] if r['split']=='val')
            with open(manifest.parent/entry['labels'],'ab') as handle:
                handle.write(b'changed')
            with self.assertRaises(RuntimeError): list(FrameDataset(manifest,'val'))

    def test_metrics_weight_by_frames(self):
        m=Metrics(3); m.add(np.array([[1,2,3],[0,0,0]]),np.zeros((2,3)))
        m.add(np.array([[2,0,0]]),np.zeros((1,3)))
        np.testing.assert_allclose(m.result()['mae'],[1,2/3,1])
        np.testing.assert_allclose(m.result()['rmse'],np.sqrt([5/3,4/3,3]))
        self.assertEqual(m.result()['count'],3)

    def test_image_only_model_gradients(self):
        torch.set_num_threads(2); model=ForceRegressor()
        output=model(torch.rand(2,1,240,320))
        self.assertEqual(tuple(output.shape),(2,3))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.encoder[0].weight.grad).all())
        with self.assertRaises(ValueError): model(torch.zeros(2,3,240,320))

    def test_resnet50_checkpoint_gradients_and_bn_buffers(self):
        torch.set_num_threads(2);torch.manual_seed(4)
        normal=ForceRegressor(backbone='resnet50').train()
        recompute=copy.deepcopy(normal);recompute.activation_checkpointing=True
        x=torch.rand(2,1,240,320)
        a=normal(x); b=recompute(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.square().mean().backward();b.square().mean().backward()
        for (name,p),(other,q) in zip(normal.named_parameters(),recompute.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7)
        for (name,p),(other,q) in zip(normal.named_buffers(),recompute.named_buffers()):
            self.assertEqual(name,other)
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        normal.eval();recompute.eval()
        with torch.no_grad():torch.testing.assert_close(normal(x),recompute(x),rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
