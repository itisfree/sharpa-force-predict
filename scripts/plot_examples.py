"""Rebuild the README figure from the included numeric traces."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_predictions import draw

root=Path(__file__).resolve().parents[1]
files=sorted((root/'assets').glob('episode_*.csv'))
if not files: raise FileNotFoundError('No example traces')
fig, axs=plt.subplots(3,len(files),figsize=(5.4*len(files),8.8),squeeze=False)
for col,file in enumerate(files):
    df=pd.read_csv(file)
    draw(axs[:,col],df,f"Episode {int(df.episode.iloc[0])} · {df.finger.iloc[0].replace('_',' ')}")
fig.suptitle('Tactile image → 3D force\nResNet-50 · checkpoint step 40,000',fontsize=20,y=1.01)
fig.tight_layout()
fig.savefig(root/'assets/force_comparison.png',dpi=160,bbox_inches='tight')
fig.savefig(root/'assets/force_comparison.svg',bbox_inches='tight')
plt.close(fig)
