import argparse

from .data import prepare
from .engine import evaluate, predict, train


def main():
    parser = argparse.ArgumentParser(description='Single raw tactile image -> per-finger force regression')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Build immutable label cache and split manifest, never modify source data')
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--finger', default='left_index', help='A finger such as left_index, or all for a shared ten-finger model')
    p.add_argument('--target', choices=['force','wrench'], default='force')
    p.add_argument('--split', choices=['object','episode'], default='object')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stride', type=int, default=3, help='Subsample frames after partitioning; 3 means 10 Hz')
    p.add_argument('--limit-per-split', type=int, default=0, help='Pilot episode cap; 0 means all')
    p.set_defaults(func=prepare)
    p = sub.add_parser('train')
    p.add_argument('--manifest', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--backbone', choices=['resnet18', 'resnet50'], default='resnet18')
    p.add_argument('--activation-checkpointing', action='store_true', help='Recompute activations without reducing precision or batch size')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle-buffer', type=int, default=128)
    p.add_argument('--streams-per-finger', type=int, default=0, help='Concurrent episode streams per finger per worker; 0 keeps sequential reading')
    p.add_argument('--near-zero-keep', type=float, default=1., help='Training-only retention probability for near-zero force')
    p.add_argument('--near-zero-threshold', type=float, default=.1, help='Dataset-native force norm, not a calibrated contact label')
    p.add_argument('--steps-per-epoch', type=int, default=0, help='0 exhausts selected training data; nonzero is an explicit step budget')
    p.add_argument('--eval-batches', type=int, default=0, help='Smoke-test limit only; 0 validates the complete selected split')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--checkpoint-every', type=int, default=0, help='Save a full checkpoint every N optimizer steps; 0 disables step checkpoints')
    p.add_argument('--resume', default=None, help='Resume a trusted checkpoint; step checkpoints restore the next unconsumed batch')
    p.add_argument('--allow-batch-size-change', action='store_true', help='Explicitly permit a new batch size on resume; preserve model and optimizer state')
    p.add_argument('--data-parallel', action='store_true', help='Split the global batch across all visible CUDA devices')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--amp-dtype', choices=['float16', 'bfloat16'], default='float16')
    p.set_defaults(func=train)
    e = sub.add_parser('evaluate')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--manifest', default=None)
    e.add_argument('--split', choices=['val','test'], default='test')
    e.add_argument('--output', required=True)
    e.add_argument('--max-batches', type=int, default=0, help='0 evaluates every selected frame')
    e.add_argument('--data-parallel', action='store_true', help='Split global evaluation batch across visible CUDA devices')
    e.set_defaults(func=evaluate)
    for cmd in (p,e):
        cmd.add_argument('--device', default='cuda:0')
        cmd.add_argument('--batch-size', type=int, default=64)
        cmd.add_argument('--workers', type=int, default=4)
        cmd.add_argument('--cpu-threads', type=int, default=4)
    q = sub.add_parser('predict')
    q.add_argument('--checkpoint', required=True)
    q.add_argument('--image', required=True, help='Raw uint8 grayscale PNG, 320x240')
    q.add_argument('--device', default='cpu')
    q.add_argument('--cpu-threads', type=int, default=4)
    q.set_defaults(func=predict)
    args = parser.parse_args()
    if hasattr(args,'batch_size') and (args.batch_size < 1 or args.workers < 0 or args.cpu_threads < 1):
        parser.error('batch-size/cpu-threads must be positive; workers must be nonnegative')
    if hasattr(args,'log_every') and args.log_every < 1:
        parser.error('log-every must be positive')
    for key in ('steps_per_epoch','eval_batches','max_batches','streams_per_finger','checkpoint_every'):
        if hasattr(args,key) and getattr(args,key) < 0:
            parser.error(f'{key} must be nonnegative')
    if hasattr(args,'epochs') and args.epochs < 1:
        parser.error('epochs must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
