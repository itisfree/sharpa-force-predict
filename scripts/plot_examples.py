"""Rebuild the README figure from the included numeric traces."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_predictions import draw

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--step', choices=['40000','48000'], default='48000')
args=parser.parse_args()
assets=root/'assets' if args.step=='40000' else root/'assets/step48000'
files=sorted(assets.glob('episode_*.csv'))
if args.step=='48000':
    files=[assets/name for name in ['episode_001461_left_index.csv', 'episode_000075_left_thumb.csv', 'episode_005214_left_middle.csv']]
if not files: raise FileNotFoundError('No example traces')
fig, axs=plt.subplots(3,len(files),figsize=(5.4*len(files),8.8),squeeze=False)
for col,file in enumerate(files):
    df=pd.read_csv(file)
    draw(axs[:,col],df,f"Episode {int(df.episode.iloc[0])} · {df.finger.iloc[0].replace('_',' ')}")
label='40,000' if args.step=='40000' else '48,000 (replayed)'
fig.suptitle(f'Tactile image → 3D force\nResNet-50 · checkpoint step {label}',fontsize=20,y=1.01)
fig.tight_layout()
fig.savefig(assets/'force_comparison.png',dpi=160,bbox_inches='tight')
fig.savefig(assets/'force_comparison.svg',bbox_inches='tight')
plt.close(fig)
