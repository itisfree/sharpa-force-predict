"""Compute all-test, per-finger and per-episode metrics from evaluation predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(y, p):
    y,p=np.asarray(y,dtype=np.float64),np.asarray(p,dtype=np.float64)
    error=p-y
    ss=((y-y.mean(0))**2).sum(0)
    ratio=np.divide((error**2).sum(0),ss,out=np.full(3,np.nan),where=ss>1e-12)
    v=np.linalg.norm(error,axis=1)
    return dict(count=len(y),mae=np.abs(error).mean(0).tolist(),
        rmse=np.sqrt((error**2).mean(0)).tolist(),bias=error.mean(0).tolist(),
        r2=[float(1-x) if np.isfinite(x) else None for x in ratio],
        pearson=[float(np.corrcoef(y[:,i],p[:,i])[0,1]) if y[:,i].std()>1e-12 and p[:,i].std()>1e-12 else None for i in range(3)],
        force_vector_mae=float(v.mean()),vector_error_p50=float(np.quantile(v,.5)),
        vector_error_p90=float(np.quantile(v,.9)),vector_error_p95=float(np.quantile(v,.95)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--report',help='Optional evaluate output JSON to check full-set coverage and MAE')
    a=parser.parse_args()
    df=pd.read_csv(a.csv,dtype={'episode':'int32','frame':'int32','finger':'category','object':'category','primitive':'category'})
    if df.empty or df.duplicated(['episode','finger','frame']).any():raise ValueError('Empty or duplicated samples')
    y=df[['true_'+v for v in ['Fx','Fy','Fz']]].to_numpy();p=df[['pred_'+v for v in ['Fx','Fy','Fz']]].to_numpy()
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite values')
    out={'all':metrics(y,p),'by_finger':{},'by_episode':{}}
    for dest,col in [('by_finger','finger'),('by_episode','episode')]:
        for name,idx in df.groupby(col,observed=True).indices.items():out[dest][str(name)]=metrics(y[idx],p[idx])
    if a.report:
        r=json.loads(Path(a.report).read_text())
        if not r['complete_split'] or len(df)!=r['expected_frames']:raise ValueError('Incomplete test set')
        np.testing.assert_allclose(out['all']['mae'],r['all']['mae'],rtol=1e-6)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
