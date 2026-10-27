import csv
from contextlib import nullcontext
from copy import copy
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from .cursor_data import CursorFrameDataset

from .data import FrameDataset, digest
from .model import ForceRegressor, isolated_bn_recompute


class Metrics:
    def __init__(self, width):
        self.n = 0
        self.absolute = np.zeros(width, np.float64)
        self.square = np.zeros(width, np.float64)
        self.force_vector_error = 0.

    def add(self, pred, target):
        if len(target) == 0:
            return
        err = pred.astype(np.float64) - target.astype(np.float64)
        self.n += len(target)
        self.absolute += np.abs(err).sum(0)
        self.square += (err**2).sum(0)
        self.force_vector_error += np.linalg.norm(err[:,:3], axis=1).sum()

    def result(self):
        if not self.n:
            return {'count': 0, 'mae': None, 'rmse': None, 'force_vector_mae': None}
        return {'count': self.n, 'mae': (self.absolute/self.n).tolist(),
                'rmse': np.sqrt(self.square/self.n).tolist(),
                'force_vector_mae': self.force_vector_error/self.n}


def make_loader(dataset, args, device):
    loader = StatefulDataLoader if isinstance(dataset, CursorFrameDataset) else DataLoader
    extra = {'snapshot_every_n_steps': 1} if loader is StatefulDataLoader else {}
    return loader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                      generator=torch.Generator().manual_seed(args.seed if hasattr(args, 'seed') else 42), **extra,
                      pin_memory=device.type == 'cuda', persistent_workers=False,
                      **({'prefetch_factor': 2} if args.workers else {}))


def evaluate_model(model, loader, mean, std, near_zero, high_force, max_batches=0, csv_path=None, amp_dtype=None):
    device = mean.device
    width = len(mean)
    totals = {k: Metrics(width) for k in ('all', 'near_zero', 'above_near_zero', 'high_force', 'zero_baseline', 'mean_baseline')}
    groups = {k: defaultdict(lambda: Metrics(width)) for k in ('object', 'primitive', 'episode', 'finger')}
    model.eval()
    count_batches = 0
    handle = open(csv_path, 'w', newline='') if csv_path else None
    writer = csv.writer(handle) if handle else None
    axes = ['Fx','Fy','Fz','Mx','My','Mz'][:width]
    if writer:
        writer.writerow(['episode','frame','video_timestamp','object','primitive','finger',
                         *['true_'+a for a in axes], *['pred_'+a for a in axes]])
    try:
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast(device_type=device.type, enabled=amp_dtype is not None, dtype=amp_dtype):
                    normalized = model(batch['image'].to(device))
                prediction = (normalized.float() * std + mean).cpu().numpy()
                target = batch['target'].numpy()
                if not np.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite model predictions')
                force_norm = np.linalg.norm(target[:,:3], axis=1)
                masks = {'all': np.ones(len(target), dtype=bool), 'near_zero': force_norm < near_zero,
                         'above_near_zero': force_norm >= near_zero, 'high_force': force_norm >= high_force}
                for name, mask in masks.items():
                    totals[name].add(prediction[mask], target[mask])
                totals['zero_baseline'].add(np.zeros_like(target), target)
                totals['mean_baseline'].add(np.broadcast_to(mean.cpu().numpy(), target.shape), target)
                for key, group in groups.items():
                    values = [str(v.item()) if torch.is_tensor(v) else str(v) for v in batch[key]]
                    for value in set(values):
                        mask = np.array([v == value for v in values])
                        group[value].add(prediction[mask], target[mask])
                if writer:
                    for i in range(len(target)):
                        writer.writerow([int(batch['episode'][i]), int(batch['frame'][i]),
                                         float(batch['video_timestamp'][i]), batch['object'][i],
                                         batch['primitive'][i], batch['finger'][i], *target[i].tolist(), *prediction[i].tolist()])
                count_batches += 1
                if max_batches and count_batches >= max_batches:
                    break
    finally:
        if handle:
            handle.close()
    if totals['all'].n == 0:
        raise ValueError('Evaluation produced no frames')
    result = {k: m.result() for k, m in totals.items()}
    result['groups'] = {k: {g: m.result() for g, m in values.items()} for k, values in groups.items()}
    result['episode_macro_force_vector_mae'] = float(np.mean([m.result()['force_vector_mae'] for m in groups['episode'].values()]))
    result['batches'] = count_batches
    result['requested_batch_limit'] = max_batches
    result['near_zero_threshold'] = near_zero
    result['high_force_threshold_train_q90'] = high_force
    result['units'] = 'dataset native; no independently verified SI calibration'
    result['axes'] = axes
    return result


def accumulated_backward(model, image, target, mean, std, criterion, scaler, device, amp_dtype, micro_batch_size):
    """Match DataParallel chunks; only the first chunk persists BN buffers."""
    size = micro_batch_size or len(target)
    predictions, losses = [], []
    for begin in range(0, len(target), size):
        end = min(begin + size, len(target))
        context = isolated_bn_recompute([model]) if begin else nullcontext()
        with context:
            with torch.autocast(device_type=device.type, enabled=amp_dtype is not None, dtype=amp_dtype):
                prediction = model(image[begin:end])
                loss = criterion(prediction, (target[begin:end]-mean)/std) * ((end-begin)/len(target))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            predictions.append(prediction.detach())
            losses.append(loss.detach())
    return torch.cat(predictions), torch.stack(losses).sum()


def atomic_checkpoint(path, checkpoint):
    temp = path.with_suffix('.tmp')
    torch.save(checkpoint, temp)
    os.replace(temp, path)


def checkpoint_alias(source, destination):
    """Keep one payload per epoch; best/last are atomic hard-link aliases."""
    temp = destination.with_suffix('.link.tmp')
    temp.unlink(missing_ok=True)
    os.link(source, temp)
    os.replace(temp, destination)


def train(args):
    if not (0 < args.near_zero_keep <= 1) or args.shuffle_buffer < 0:
        raise ValueError('near_zero_keep must be in (0,1]; shuffle_buffer must be nonnegative')
    if getattr(args, 'checkpoint_every', 0) < 0:
        raise ValueError('checkpoint_every must be nonnegative')
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    manifest = Path(args.manifest).resolve()
    meta = json.loads(manifest.read_text()); manifest_sha = digest(manifest)
    run = Path(args.run_dir).resolve()
    if not args.resume and run.exists() and any(run.iterdir()):
        raise FileExistsError('Run directory is nonempty; use a new directory or --resume')
    run.mkdir(parents=True, exist_ok=True)
    # Explicit FP32 mode also disables reduced-precision TF32 kernels.
    if not args.amp:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = ForceRegressor(meta['output_dim'], backbone=getattr(args, 'backbone', 'resnet18')).to(device)
    model.activation_checkpointing = getattr(args, 'activation_checkpointing', False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mean = torch.tensor(meta['statistics']['mean'], dtype=torch.float32, device=device)
    std = torch.tensor(meta['statistics']['std'], dtype=torch.float32, device=device)
    amp_dtype = getattr(torch, getattr(args, 'amp_dtype', 'float16')) if args.amp and device.type == 'cuda' else None
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('Requested BF16 is unsupported on this GPU')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    start, global_step, best = 0, 0, float('inf')
    resume_progress = None
    if args.resume:
        cp = torch.load(args.resume, map_location='cpu', weights_only=False)
        if cp['manifest_sha256'] != manifest_sha:
            raise ValueError('Resume manifest differs from checkpoint')
        for field, default in [('streams_per_finger', 0), ('amp_dtype', 'float16'), ('backbone', 'resnet18'), ('activation_checkpointing', False)]:
            if cp['config'].get(field, default) != getattr(args, field, default):
                raise ValueError(f'Resume setting differs: {field}')
        for field in ('batch_size','workers','shuffle_buffer','near_zero_keep','near_zero_threshold','seed','steps_per_epoch','lr','weight_decay','amp','eval_batches'):
            if cp['config'][field] != getattr(args, field):
                if field == 'batch_size' and getattr(args, 'allow_batch_size_change', False):
                    change = {'resume_checkpoint': str(Path(args.resume).resolve()),
                              'epoch': cp['next_epoch'], 'global_step': cp['global_step'],
                              'old_batch_size': cp['config'][field], 'new_batch_size': args.batch_size,
                              'note': 'User-requested batch change; not a numerically identical continuation'}
                    (run/'resume_batch_change.json').write_text(json.dumps(change, indent=2))
                    print(json.dumps(change), flush=True)
                    continue
                raise ValueError(f'Resume setting differs: {field}; retain the original configuration')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        for state in optimizer.state.values():
            for k, value in state.items():
                if torch.is_tensor(value) and k != 'step':
                    state[k] = value.to(device)
                # Non-capturable AdamW keeps its scalar step on CPU, as in a fresh optimizer.
        scaler.load_state_dict(cp['scaler'])
        start, global_step, best = cp['next_epoch'], cp['global_step'], cp['best_val']
        if cp.get('epoch_incomplete', False):
            if cp.get('resume_policy') != 'restore_data_cursor_v1' or 'data_loader' not in cp:
                raise ValueError('Legacy step checkpoint has no data cursor; refusing to replay trained samples')
            if cp['config']['batch_size'] != args.batch_size:
                raise ValueError('Mid-epoch cursor resume requires the original batch size')
            resume_progress = cp
            print(json.dumps({'resume_policy': 'restore_data_cursor_v1', 'epoch': start+1,
                              'saved_global_step': global_step, 'saved_epoch_batch': cp['epoch_batch'],
                              'note': 'Continue with the next unconsumed batch; no trained batches are replayed.'}), flush=True)
        torch.set_rng_state(cp['torch_rng']); np.random.set_state(cp['numpy_rng']); random.setstate(cp['python_rng'])
        if device.type == 'cuda' and cp['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(cp['cuda_rng'][:torch.cuda.device_count()])
    if args.epochs <= start:
        raise ValueError(f'epochs must exceed the completed epoch count {start}')
    if getattr(args, 'stop_at_step', 0) and global_step >= args.stop_at_step:
        raise ValueError('Stop target must be later than the resume checkpoint')
    base_model = model
    if getattr(args, 'data_parallel', False):
        if device.type != 'cuda' or device.index not in (None, 0) or torch.cuda.device_count() < 2:
            raise ValueError('Data parallel requires cuda:0 and at least two visible CUDA devices')
        model = torch.nn.DataParallel(base_model)
    (run/'COMPLETE.json').unlink(missing_ok=True)
    config = vars(args).copy(); config.pop('func', None)
    (run/'config.json').write_text(json.dumps(config, indent=2))
    (run/'manifest.json').write_text(manifest.read_text())
    trainset = CursorFrameDataset(manifest, 'train', args.seed, args.shuffle_buffer, args.near_zero_keep, args.near_zero_threshold,
                            getattr(args, 'streams_per_finger', 0))
    valset = FrameDataset(manifest, 'val')
    eval_args = copy(args)
    eval_args.batch_size = getattr(args, 'micro_batch_size', 0) or args.batch_size
    trainloader, valloader = make_loader(trainset, args, device), make_loader(valset, eval_args, device)
    criterion = torch.nn.SmoothL1Loss()
    print(json.dumps({'device': str(device), 'parameters': sum(p.numel() for p in model.parameters()),
                      'backbone': getattr(args, 'backbone', 'resnet18'),
                      'parameter_dtype': str(next(model.parameters()).dtype),
                      'amp_enabled': amp_dtype is not None, 'amp_dtype': str(amp_dtype),
                      'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                      'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                      'activation_checkpointing': getattr(args, 'activation_checkpointing', False),
                      'manifest': str(manifest), 'resume_epoch': start, 'input': 'single raw grayscale frame only',
                      'data_parallel_devices': model.device_ids if isinstance(model, torch.nn.DataParallel) else [],
                      'global_batch_size': args.batch_size, 'micro_batch_size': getattr(args, 'micro_batch_size', 0)}), flush=True)
    for epoch in range(start, args.epochs):
        trainset.epoch = epoch; model.train()
        loss_sum, n, batches = 0., 0, 0
        previous_elapsed = 0.
        recent = deque(maxlen=100)
        if resume_progress is not None and epoch == start:
            trainloader.load_state_dict(resume_progress['data_loader'])
            loss_sum = resume_progress['epoch_loss_sum']
            n, batches = resume_progress['epoch_samples'], resume_progress['epoch_batch']
            previous_elapsed = resume_progress['epoch_elapsed_seconds']
            recent.extend(resume_progress['recent_losses'])
        begun = time.monotonic() - previous_elapsed
        iterator = iter(trainloader)
        if resume_progress is not None:
            # Worker startup/restoration must not change model-side RNG state.
            torch.set_rng_state(resume_progress['torch_rng'])
            np.random.set_state(resume_progress['numpy_rng']); random.setstate(resume_progress['python_rng'])
            if device.type == 'cuda' and resume_progress['cuda_rng'] is not None:
                torch.cuda.set_rng_state_all(resume_progress['cuda_rng'][:torch.cuda.device_count()])
            resume_progress = None
        while not args.steps_per_epoch or batches < args.steps_per_epoch:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            image, y = batch['image'].to(device), batch['target'].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, loss = accumulated_backward(model, image, y, mean, std, criterion,
                scaler, device, amp_dtype, getattr(args, 'micro_batch_size', 0))
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            value = float(loss.detach())
            n += len(y); loss_sum += value*len(y); batches += 1; global_step += 1
            recent.append((value * len(y), len(y)))
            if batches == 1 or batches % args.log_every == 0:
                with torch.no_grad():
                    norms = y[:, :3].norm(dim=1)
                    err = (prediction.float()*std+mean-y).abs().mean(0).tolist()
                record = {'epoch': epoch+1, 'batch': batches, 'global_step': global_step,
                          'loss': value, 'loss_mean_last_100_steps': sum(v for v,c in recent)/sum(c for v,c in recent),
                          'loss_epoch_running_mean': loss_sum/n, 'grad_norm_before_clip': float(grad),
                          'batch_samples': len(y), 'samples': n,
                          'batch_sample_sha256': hashlib.sha256(json.dumps(list(zip(
                              batch['episode'].tolist(), batch['finger'], batch['frame'].tolist())),
                              separators=(',', ':')).encode()).hexdigest(),
                          'near_zero_fraction': float((norms < args.near_zero_threshold).float().mean()),
                          'force_norm_mean': float(norms.mean()), 'force_mae_axes': err,
                          'finger_counts': dict(Counter(batch['finger'])),
                          'episode_count': len(set(batch['episode'].tolist())),
                          'epoch_elapsed_seconds': time.monotonic()-begun,
                          'samples_per_second_including_io': n/(time.monotonic()-begun)}
                line = json.dumps(record)
                with open(run/'steps.jsonl', 'a') as f:
                    f.write(line+'\n')
                print(line, flush=True)
            interval = getattr(args, 'checkpoint_every', 0)
            reached_target = bool(getattr(args, 'stop_at_step', 0) and global_step >= args.stop_at_step)
            if reached_target or (interval and global_step % interval == 0):
                checkpoint = {'model': base_model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scaler': scaler.state_dict(), 'next_epoch': epoch,
                              'epoch_incomplete': True, 'epoch_batch': batches, 'epoch_samples': n,
                              'resume_policy': 'restore_data_cursor_v1', 'data_loader': trainloader.state_dict(),
                              'epoch_loss_sum': loss_sum, 'recent_losses': list(recent),
                              'epoch_elapsed_seconds': time.monotonic()-begun,
                              'global_step': global_step, 'best_val': best,
                              'manifest': str(manifest), 'manifest_sha256': manifest_sha, 'metadata': meta,
                              'config': config, 'torch_rng': torch.get_rng_state(),
                              'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                              'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
                step_path = run/f'step_{global_step:08d}.pt'
                atomic_checkpoint(step_path, checkpoint)
                checkpoint_alias(step_path, run/'last_step.pt')
                print(json.dumps({'checkpoint_saved': str(step_path), 'global_step': global_step,
                                  'epoch': epoch+1, 'validation_performed': False}), flush=True)
            if reached_target:
                (run/'TARGET_STEP_COMPLETE.json').write_text(json.dumps({
                    'status': 'target step reached and checkpoint saved', 'global_step': global_step,
                    'checkpoint': str(step_path), 'training_plan_complete': False}, indent=2))
                return
            if args.steps_per_epoch and batches >= args.steps_per_epoch:
                break
        if n == 0:
            raise ValueError('Training produced no frames; inspect sampling and split')
        val = evaluate_model(model, valloader, mean, std, args.near_zero_threshold,
                             meta['statistics']['force_norm_q90'], args.eval_batches, amp_dtype=amp_dtype)
        val['expected_frames'] = sum(r['samples'] for r in valset.rows)
        val['complete_split'] = val['all']['count'] == val['expected_frames']
        (run/f'validation_epoch_{epoch+1:04d}.json').write_text(json.dumps(val, indent=2))
        score = val['all']['force_vector_mae']; improved = score < best; best = min(best, score)
        record = {'epoch': epoch+1, 'global_step': global_step, 'train_samples': n, 'train_batches': batches,
                  'train_loss_standardized': loss_sum/n, 'val_force_vector_mae': score,
                  'validation_complete': val['complete_split'], 'elapsed_seconds': time.monotonic()-begun}
        with open(run/'metrics.jsonl', 'a') as f:
            f.write(json.dumps(record)+'\n')
        checkpoint = {'model': base_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                      'next_epoch': epoch+1, 'global_step': global_step, 'best_val': best,
                      'manifest': str(manifest), 'manifest_sha256': manifest_sha, 'metadata': meta,
                      'config': config, 'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                      'python_rng': random.getstate(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        epoch_path = run/f'epoch_{epoch+1:04d}.pt'
        atomic_checkpoint(epoch_path, checkpoint)
        checkpoint_alias(epoch_path, run/'last.pt')
        if improved:
            checkpoint_alias(epoch_path, run/'best.pt')
        print(json.dumps(record), flush=True)
    (run/'COMPLETE.json').write_text(json.dumps({'status': 'requested training epochs completed',
        'epochs': args.epochs, 'global_step': global_step, 'validation_complete': val['complete_split'],
        'test_evaluated': False, 'pilot_data': bool(meta['pilot_limit_per_split'])}, indent=2))


def evaluate(args):
    torch.set_num_threads(args.cpu_threads)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    manifest = Path(args.manifest or cp['manifest'])
    if digest(manifest) != cp['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from checkpoint; prevents split/statistics drift')
    device = torch.device(args.device)
    meta = cp['metadata']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ForceRegressor(meta['output_dim'], backbone=cp['config'].get('backbone', 'resnet18')).to(device); model.load_state_dict(cp['model'])
    mean = torch.tensor(meta['statistics']['mean'], dtype=torch.float32, device=device)
    std = torch.tensor(meta['statistics']['std'], dtype=torch.float32, device=device)
    if getattr(args, 'data_parallel', False):
        if device.type != 'cuda' or device.index not in (None, 0) or torch.cuda.device_count() < 2:
            raise ValueError('DataParallel evaluation requires cuda:0 and at least two visible GPUs')
        model = torch.nn.DataParallel(model)
    dataset = FrameDataset(manifest, args.split)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite report: {output}')
    csv_path = output.with_suffix('.predictions.csv')
    if csv_path.exists():
        raise FileExistsError(csv_path)
    result = evaluate_model(model, make_loader(dataset, args, device), mean, std,
        cp['config']['near_zero_threshold'], meta['statistics']['force_norm_q90'], args.max_batches, csv_path)
    result.update({'split': args.split, 'checkpoint': str(Path(args.checkpoint).resolve()),
                   'manifest_sha256': cp['manifest_sha256'], 'pilot_data': bool(meta['pilot_limit_per_split']),
                   'expected_frames': sum(r['samples'] for r in dataset.rows)})
    result['complete_split'] = result['all']['count'] == result['expected_frames']
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'report': str(output), 'metrics': result['all'], 'complete_split': result['complete_split']}, indent=2))


def predict(args):
    from PIL import Image
    torch.set_num_threads(args.cpu_threads)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    image = np.asarray(Image.open(args.image))
    if image.dtype != np.uint8 or image.shape != (240, 320):
        raise ValueError('Input must be an original uint8 grayscale image with shape 240x320')
    device = torch.device(args.device); meta = cp['metadata']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = ForceRegressor(meta['output_dim'], backbone=cp['config'].get('backbone', 'resnet18')).to(device)
    model.load_state_dict(cp['model']); model.eval()
    with torch.inference_mode():
        x = torch.from_numpy(image.copy()).float()[None,None].to(device)/255.
        y = model(x)[0].cpu().numpy() * np.array(meta['statistics']['std']) + np.array(meta['statistics']['mean'])
    print(json.dumps({'finger': meta['finger'], 'axes': ['Fx','Fy','Fz','Mx','My','Mz'][:len(y)],
                      'prediction': y.tolist(), 'units': meta['units']}, indent=2))
