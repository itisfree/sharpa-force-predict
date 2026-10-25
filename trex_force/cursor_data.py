"""Checkpointable equivalent of FrameDataset's deterministic interleaved stream.

Snapshots contain descriptors, not decoded pixels. Restore seeks to active streams
and rehydrates only buffered, not-yet-consumed frames; consumed samples are never
replayed through the loader or optimizer.
"""
import pickle
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import get_worker_info

from .data import FrameDataset, decode_selected, digest


class CursorFrameDataset(FrameDataset):
    def __iter__(self):
        return FrameCursor(self)


class FrameCursor:
    VERSION = 1

    def __init__(self, dataset):
        self.ds = dataset
        wi = get_worker_info()
        self.worker, self.workers = (wi.id, wi.num_workers) if wi else (0, 1)
        order = list(range(len(dataset.rows)))
        if dataset.split == 'train':
            random.Random(dataset.seed + dataset.epoch).shuffle(order)
        assigned = order[self.worker::self.workers]
        self.rng = random.Random(dataset.seed + dataset.epoch * 100003 + self.worker * 997)
        self.queues = defaultdict(list)
        for rid in assigned:
            key = dataset.rows[rid].get('finger', 'single') if dataset.streams_per_finger else 'single'
            self.queues[key].append(rid)
        self.active = []
        for key, queue in sorted(self.queues.items()):
            queue.reverse()
            for _ in range(min(dataset.streams_per_finger or 1, len(queue))):
                self.active.append(self.new_stream(key, queue.pop()))
        self.remaining = []
        self.cursor = 0
        self.round_started = False
        self.buffer = []
        self.draining = False
        self.drain_cursor = 0
        self.yielded = 0
        self.images = {}
        self.decoders = {}
        self.labels = OrderedDict()
        self.needs_hydration = False

    @staticmethod
    def new_stream(key, rid):
        return {'key': key, 'rid': rid, 'indices': None, 'pos': 0}

    def __iter__(self):
        return self

    def row_data(self, rid):
        if rid in self.labels:
            self.labels.move_to_end(rid)
            return self.labels[rid]
        row = self.ds.rows[rid]
        path = self.ds.path.parent / row['labels']
        if 'labels_sha256' in row and digest(path) != row['labels_sha256']:
            raise RuntimeError(f'Prepared labels were modified: {path}')
        with np.load(path) as data:
            y, frames, ts = data['target'], data['frame_index'], data['timestamp']
            if y.ndim == 3:
                y = y[:, row['finger_index'], :]
        video = Path(self.ds.meta['root']) / row['video']
        expected = self.ds.meta['source_videos'][row['video']]
        stat = video.stat()
        if (stat.st_size, stat.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
            raise RuntimeError(f'Video modified after manifest creation: {video}')
        value = y, frames, ts, video
        self.labels[rid] = value
        if len(self.labels) > 64:
            self.labels.popitem(last=False)
        return value

    def source_next(self):
        while self.active:
            if not self.round_started:
                if self.ds.streams_per_finger:
                    self.rng.shuffle(self.active)
                self.round_started = True
                self.cursor = 0
                self.remaining = []
            if self.cursor == len(self.active):
                self.active = self.remaining
                self.remaining = []
                self.round_started = False
                continue
            stream = self.active[self.cursor]
            while True:
                rid = stream['rid']
                y, frames, ts, video = self.row_data(rid)
                if stream['indices'] is None:
                    stream['indices'] = np.array([j for j, v in enumerate(y)
                        if np.linalg.norm(v[:3]) >= self.ds.threshold or self.rng.random() < self.ds.near_zero_keep], dtype=np.int64)
                indices = stream['indices']
                if stream['pos'] < len(indices):
                    break
                decoder = self.decoders.pop(rid, None)
                if decoder is not None:
                    decoder.close()
                queue = self.queues[stream['key']]
                if not queue:
                    stream = None
                    break
                stream = self.new_stream(stream['key'], queue.pop())
                self.active[self.cursor] = stream
            self.cursor += 1
            if stream is None:
                continue
            if rid not in self.decoders:
                stamps = ts[indices[stream['pos']:]].astype(np.float64) + self.ds.rows[rid]['from_timestamp']
                self.decoders[rid] = iter(decode_selected(video, stamps, self.ds.meta['fps']))
            gray, pts = next(self.decoders[rid])
            index = int(indices[stream['pos']])
            stream['pos'] += 1
            self.remaining.append(stream)
            descriptor = (rid, index)
            self.images[descriptor] = (gray, pts)
            return descriptor
        raise StopIteration

    def hydrate_buffer(self):
        grouped = defaultdict(list)
        pending = self.buffer[self.drain_cursor:] if self.draining else self.buffer
        for rid, index in pending:
            grouped[rid].append(index)
        for rid, indices in grouped.items():
            indices.sort()
            _, _, ts, video = self.row_data(rid)
            stamps = ts[indices].astype(np.float64) + self.ds.rows[rid]['from_timestamp']
            for index, image in zip(indices, decode_selected(video, stamps, self.ds.meta['fps']), strict=True):
                self.images[(rid, index)] = image
        self.needs_hydration = False

    def emit(self, descriptor):
        rid, index = descriptor
        gray, pts = self.images.pop(descriptor)
        y, frames, _, _ = self.row_data(rid)
        row = self.ds.rows[rid]
        self.yielded += 1
        return {'image': torch.from_numpy(gray).unsqueeze(0).float()/255.,
                'target': torch.from_numpy(y[index].copy()), 'episode': row['episode_index'],
                'frame': int(frames[index]), 'video_timestamp': pts,
                'finger': row.get('finger', self.ds.meta['finger']),
                'object': row['object'], 'primitive': row['primitive']}

    def __next__(self):
        if self.needs_hydration:
            self.hydrate_buffer()
        while not self.draining:
            try:
                item = self.source_next()
            except StopIteration:
                self.rng.shuffle(self.buffer)
                self.draining = True
                break
            if not self.ds.shuffle_buffer:
                return self.emit(item)
            if len(self.buffer) < self.ds.shuffle_buffer:
                self.buffer.append(item)
                continue
            k = self.rng.randrange(len(self.buffer))
            previous, self.buffer[k] = self.buffer[k], item
            return self.emit(previous)
        if self.drain_cursor < len(self.buffer):
            item = self.buffer[self.drain_cursor]
            self.drain_cursor += 1
            return self.emit(item)
        self.close()
        raise StopIteration

    def state_dict(self):
        state = {k: getattr(self, k) for k in ('queues','active','remaining','cursor','round_started',
                 'buffer','draining','drain_cursor','yielded')}
        state.update(version=self.VERSION, epoch=self.ds.epoch, worker=self.worker,
                     workers=self.workers, rng=self.rng.getstate())
        # An immutable packed snapshot avoids mutating in-flight worker snapshots.
        return {'cursor': pickle.dumps(state, protocol=5)}

    def load_state_dict(self, state):
        saved = pickle.loads(state['cursor'])
        if (saved.pop('version'), saved.pop('epoch'), saved.pop('worker'), saved.pop('workers')) != (
                self.VERSION, self.ds.epoch, self.worker, self.workers):
            raise ValueError('Cursor version/epoch/worker topology changed')
        self.close()
        self.rng.setstate(saved.pop('rng'))
        for key, value in saved.items():
            setattr(self, key, value)
        self.images = {}
        self.labels.clear()
        self.needs_hydration = True

    def close(self):
        for stream in self.decoders.values():
            stream.close()
        self.decoders.clear()

    def __del__(self):
        self.close()
