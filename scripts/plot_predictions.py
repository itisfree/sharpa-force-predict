"""Plot a complete episode/finger trace from an evaluation CSV (no smoothing)."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(axs, frame, title, fps=30):
    frame = frame.sort_values('frame')
    if frame['frame'].duplicated().any():
        raise ValueError('Duplicate frame keys in the selected episode/finger')
    t = (frame['frame'].to_numpy() - frame['frame'].min()) / fps
    for i, axis in enumerate(('Fx', 'Fy', 'Fz')):
        ax = axs[i]
        ax.plot(t, frame['true_' + axis], color='#2374ab', lw=1.7, label='Ground truth')
        ax.plot(t, frame['pred_' + axis], color='#ed8b36', lw=1.4, alpha=.9, label='Prediction')
        ax.set_ylabel(axis + ' (native units)')
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_title(title, fontsize=12, pad=12)
    axs[-1].set_xlabel('Time (s)')
    axs[0].legend(frameon=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True)
    p.add_argument('--episode', required=True, type=int)
    p.add_argument('--finger', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fps', type=float, default=30)
    a = p.parse_args()
    if a.fps <= 0: p.error('--fps must be positive')
    chunks = []
    for c in pd.read_csv(a.csv, chunksize=100000):
        g=c[(c.episode == a.episode) & (c.finger == a.finger)]
        if len(g): chunks.append(g)
    if not chunks: raise ValueError('No matching episode/finger samples')
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    draw(axs, pd.concat(chunks), f'Episode {a.episode} | {a.finger}', a.fps)
    fig.tight_layout()
    out=Path(a.output);out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160);plt.close(fig)


if __name__ == '__main__': main()
