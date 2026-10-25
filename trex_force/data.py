import hashlib
import json
import random
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def finger_offset(finger):
    side, name = finger.split('_', 1)
    if side not in ('left', 'right') or name not in FINGERS:
        raise ValueError(f"Invalid finger: {finger}")
    return (0 if side == 'left' else 30) + 6 * FINGERS.index(name)


def split_episodes(rows, mode, seed):
    """Partition before frame sampling; every finger/frame of an episode stays together."""
    key = 'object' if mode == 'object' else 'episode_index'
    if any(r.get(key) is None for r in rows):
        raise ValueError(f"Missing split key: {key}")
    groups = sorted({r[key] for r in rows})
    if len(groups) < 3:
        raise ValueError('At least three distinct split groups are required')
    random.Random(seed).shuffle(groups)
    n_val = max(1, int(len(groups) * .1))
    n_test = max(1, int(len(groups) * .1))
    sets = {'test': set(groups[:n_test]), 'val': set(groups[n_test:n_test+n_val]),
            'train': set(groups[n_test+n_val:])}
    return {name: [r for r in rows if r[key] in values] for name, values in sets.items()}


def prepare(args):
    if args.finger == 'all':
        from .prepare_all import prepare_all
        return prepare_all(args)
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite prepared data: {out}')
    if args.stride < 1 or args.limit_per_split < 0:
        raise ValueError('stride must be positive; limit_per_split must be nonnegative')
    offset = finger_offset(args.finger)
    width = 3 if args.target == 'force' else 6
    info = json.loads((root / 'meta/info.json').read_text())
    fps = float(info['fps'])
    key = f'observation.images.tactile_{args.finger.split("_", 1)[0]}_raw_{args.finger.split("_", 1)[1]}'
    prefix = f'videos/{key}/'
    columns = ['episode_index', 'length', 'object', 'motor_primitive',
               prefix+'chunk_index', prefix+'file_index', prefix+'from_timestamp', prefix+'to_timestamp']
    meta_files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    if not meta_files:
        raise FileNotFoundError('No episode metadata')
    rows = [r for file in meta_files for r in pq.read_table(file, columns=columns).to_pylist()]
    if len({r['episode_index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode IDs')
    source_meta_hashes = {str(f.relative_to(root)): digest(f) for f in [root/'meta/info.json', *meta_files]}
    partitions = split_episodes(rows, args.split, args.seed)
    # A pilot is a subset of the same deterministic full-data split, never a new split.
    for name, values in partitions.items():
        random.Random(args.seed + {'train': 1, 'val': 2, 'test': 3}[name]).shuffle(values)
        if args.limit_per_split:
            partitions[name] = values[:args.limit_per_split]
    selected = {r['episode_index']: (name, r) for name, values in partitions.items() for r in values}
    stamps = {}
    for _, r in selected.values():
        rel = f"videos/{key}/chunk-{int(r[prefix+'chunk_index']):03d}/file-{int(r[prefix+'file_index']):03d}.mp4"
        path = root / rel
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Required raw video missing: {path}; do not silently change the split')
        r['video'] = rel
        if rel not in stamps:
            stat = path.stat()
            with av.open(str(path)) as c:
                stream = c.streams.video[0]
                if (stream.width, stream.height) != (320, 240):
                    raise ValueError(f'Unexpected raw image size: {path}')
                if abs(float(stream.average_rate) - fps) > 1e-3:
                    raise ValueError(f'Unexpected video rate: {path}')
                duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
            stamps[rel] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'duration': duration}
        duration = stamps[rel]['duration']
        if duration is not None and float(r[prefix+'to_timestamp']) > duration + 1/fps:
            raise ValueError(f'Episode extends past video end: {path}')
    gathered = defaultdict(list)
    wanted = np.array(sorted(selected), dtype=np.int64)
    data_sources = {}
    for file in sorted((root/'data').rglob('*.parquet')):
        pf = pq.ParquetFile(file)
        col = pf.schema_arrow.names.index('episode_index')
        row_groups = []
        for i in range(pf.num_row_groups):
            stats = pf.metadata.row_group(i).column(col).statistics
            if stats is None or not stats.has_min_max or np.any((wanted >= stats.min) & (wanted <= stats.max)):
                row_groups.append(i)
        if not row_groups:
            continue
        before = file.stat()
        t = pf.read_row_groups(row_groups, columns=['episode_index', 'frame_index', 'timestamp', 'observation.tactile_force'])
        ids = t['episode_index'].to_numpy()
        use = np.isin(ids, wanted)
        if not use.any():
            continue
        t = t.filter(use)
        ids = t['episode_index'].to_numpy()
        forces = np.asarray(t['observation.tactile_force'].to_pylist(), dtype=np.float32)
        if forces.ndim != 2 or forces.shape[1] != 60 or not np.isfinite(forces).all():
            raise ValueError(f'Invalid wrench labels in {file}')
        frames, times = t['frame_index'].to_numpy(), t['timestamp'].to_numpy()
        for eid in np.unique(ids):
            keep = ids == eid
            gathered[int(eid)].append((frames[keep], times[keep], forces[keep, offset:offset+width]))
        after = file.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Data changed while reading: {file}')
        data_sources[str(file.relative_to(root))] = {'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}
        print(f'Read {file.name}: selected {len(t)} label rows', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + '.building-' + uuid.uuid4().hex[:8])
    temp.mkdir()
    try:
        (temp/'episodes').mkdir()
        records, train_labels = [], []
        for eid, (split, r) in sorted(selected.items()):
            if eid not in gathered:
                raise ValueError(f'No labels for episode {eid}')
            arrays = [np.concatenate([chunk[i] for chunk in gathered[eid]]) for i in range(3)]
            order = np.argsort(arrays[0]); frames, times, labels = [a[order] for a in arrays]
            if not np.array_equal(frames, np.arange(r['length'])):
                raise ValueError(f'Missing/duplicate frames in episode {eid}')
            if not np.allclose(times, frames / fps, atol=1e-3):
                raise ValueError(f'Unexpected episode-local timestamps in episode {eid}')
            frames, times, labels = frames[::args.stride], times[::args.stride], labels[::args.stride]
            rel = f'episodes/{eid:06d}.npz'
            np.savez_compressed(temp/rel, frame_index=frames, timestamp=times, target=labels)
            records.append({'episode_index': eid, 'split': split, 'object': r['object'],
                            'primitive': r['motor_primitive'], 'samples': len(frames), 'labels': rel,
                            'labels_sha256': digest(temp/rel),
                            'video': r['video'], 'from_timestamp': float(r[prefix+'from_timestamp'])})
            if split == 'train':
                train_labels.append(labels)
        y = np.concatenate(train_labels).astype(np.float64)
        std = y.std(0)
        if np.any(std < 1e-8):
            raise ValueError('Training targets have constant axes; select more varied episodes')
        stats = {'mean': y.mean(0).tolist(), 'std': std.tolist(), 'count': len(y),
                 'force_norm_q90': float(np.quantile(np.linalg.norm(y[:,:3], axis=1), .9))}
        for rel, stamp in stamps.items():
            now = (root/rel).stat()
            if (now.st_size, now.st_mtime_ns) != (stamp['bytes'], stamp['mtime_ns']):
                raise RuntimeError(f'Video changed during preparation: {rel}')
        for rel, sha in source_meta_hashes.items():
            if digest(root/rel) != sha:
                raise RuntimeError(f'Metadata changed during preparation: {rel}')
        manifest = {'version': 1, 'root': str(root), 'finger': args.finger, 'target': args.target,
                    'output_dim': width, 'fps': fps, 'stride': args.stride, 'split_mode': args.split,
                    'seed': args.seed, 'pilot_limit_per_split': args.limit_per_split,
                    'statistics': stats, 'episodes': records, 'source_metadata_sha256': source_meta_hashes,
                    'source_label_files': data_sources, 'source_videos': stamps,
                    'units': 'dataset native; SI units and coordinate calibration not independently verified'}
        (temp/'manifest.json').write_text(json.dumps(manifest, indent=2))
        temp.rename(out)
    except BaseException:
        shutil.rmtree(temp)
        raise
    print(json.dumps({'prepared': str(out), 'splits': {s: {'episodes': sum(r['split']==s for r in records),
          'frames': sum(r['samples'] for r in records if r['split']==s)} for s in ('train','val','test')}}, indent=2))


def decode_selected(path, timestamps, fps):
    """Seek once, decode sequentially, assert the closest-frame time tolerance."""
    if len(timestamps) == 0:
        return
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError('Requested timestamps must be strictly increasing')
    tolerance = .5 / fps + 1e-4
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        seek = max(0, int((float(timestamps[0]) - 1/fps) / float(stream.time_base)))
        container.seek(seek, stream=stream, backward=True, any_frame=False)
        index = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f'Missing video PTS: {path}')
            pts = float(frame.pts * stream.time_base)
            target = float(timestamps[index])
            if pts < target - tolerance:
                continue
            if abs(pts-target) > tolerance:
                raise ValueError(f'Video/label mismatch: {path}, target={target}, pts={pts}')
            if frame.format.name != 'yuvj420p':
                raise ValueError(f'Unexpected tactile pixel format: {frame.format.name}')
            plane = frame.planes[0]
            gray = np.frombuffer(plane, dtype=np.uint8).reshape(frame.height, plane.line_size)[:, :frame.width].copy()
            if gray.shape != (240, 320):
                raise ValueError(f'Unexpected frame shape {gray.shape}')
            yield gray, pts
            index += 1
            if index == len(timestamps):
                return
        raise ValueError(f'Video ended after {index}/{len(timestamps)} requested frames: {path}')


def interleave_rows(rows, factory, rng, streams_per_finger):
    """Exhaust every row once, interleaving multiple episodes for each finger."""
    queues = defaultdict(list)
    for row in rows:
        queues[row.get('finger', 'single')].append(row)
    active = []
    for finger, queue in sorted(queues.items()):
        queue.reverse()
        for _ in range(min(streams_per_finger, len(queue))):
            active.append((finger, iter(factory(queue.pop()))))
    try:
        while active:
            rng.shuffle(active)
            remaining = []
            for finger, stream in active:
                while True:
                    try:
                        item = next(stream)
                        break
                    except StopIteration:
                        if not queues[finger]:
                            stream = None
                            break
                        stream = iter(factory(queues[finger].pop()))
                if stream is not None:
                    remaining.append((finger, stream))
                    yield item
            active = remaining
    finally:
        for _, stream in active:
            if hasattr(stream, 'close'):
                stream.close()


class FrameDataset(IterableDataset):
    def __init__(self, manifest_path, split, seed=42, shuffle_buffer=128, near_zero_keep=1., near_zero_threshold=.1, streams_per_finger=0):
        super().__init__()
        self.path = Path(manifest_path)
        self.meta = json.loads(self.path.read_text())
        self.rows = [r for r in self.meta['episodes'] if r['split'] == split]
        self.split, self.seed, self.epoch = split, seed, 0
        self.shuffle_buffer = shuffle_buffer if split == 'train' else 0
        self.near_zero_keep = near_zero_keep if split == 'train' else 1.
        self.threshold = near_zero_threshold
        self.streams_per_finger = streams_per_finger if split == 'train' else 0

    def __iter__(self):
        wi = get_worker_info(); worker, workers = (wi.id, wi.num_workers) if wi else (0, 1)
        rows = list(self.rows)
        if self.split == 'train':
            random.Random(self.seed + self.epoch).shuffle(rows)
        rng = random.Random(self.seed + self.epoch * 100003 + worker * 997)
        def row_items(row):
            if 'labels_sha256' in row and digest(self.path.parent/row['labels']) != row['labels_sha256']:
                raise RuntimeError(f"Prepared labels were modified: {row['labels']}")
            with np.load(self.path.parent/row['labels']) as data:
                y, frames, ts = data['target'], data['frame_index'], data['timestamp']
                if y.ndim == 3:
                    y = y[:, row['finger_index'], :]
            keep = np.array([np.linalg.norm(v[:3]) >= self.threshold or rng.random() < self.near_zero_keep for v in y])
            y, frames, ts = y[keep], frames[keep], ts[keep]
            if not len(y):
                return
            video = Path(self.meta['root'])/row['video']
            expected = self.meta['source_videos'][row['video']]
            stat = video.stat()
            if (stat.st_size, stat.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
                raise RuntimeError(f'Video modified after manifest creation: {video}')
            decoded = decode_selected(video, ts.astype(np.float64) + row['from_timestamp'], self.meta['fps'])
            for j, (gray, pts) in enumerate(decoded):
                # Keep buffered images in uint8; normalize only when emitting.
                item = {'image': torch.from_numpy(gray).unsqueeze(0),
                        'target': torch.from_numpy(y[j].copy()), 'episode': row['episode_index'],
                        'frame': int(frames[j]), 'video_timestamp': pts,
                        'finger': row.get('finger', self.meta['finger']),
                        'object': row['object'], 'primitive': row['primitive']}
                yield item
        assigned = rows[worker::workers]
        if self.streams_per_finger:
            source = interleave_rows(assigned, row_items, rng, self.streams_per_finger)
        else:
            source = (item for row in assigned for item in row_items(row))
        buffer = []
        def normalized(item):
            item['image'] = item['image'].float()/255.
            return item
        for item in source:
            if self.shuffle_buffer:
                if len(buffer) < self.shuffle_buffer:
                    buffer.append(item)
                    continue
                k = rng.randrange(len(buffer)); previous, buffer[k] = buffer[k], item
                yield normalized(previous)
            else:
                yield normalized(item)
        rng.shuffle(buffer)
        for item in buffer:
            yield normalized(item)
