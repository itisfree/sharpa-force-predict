"""Prepare ten-finger supervision in one label pass, splitting episodes first."""
import json
import random
import shutil
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from .data import FINGERS, digest, split_episodes


def prepare_all(args):
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if args.stride < 1 or args.limit_per_split < 0:
        raise ValueError('Invalid stride or episode limit')
    fingers = [f'{side}_{finger}' for side in ('left', 'right') for finger in FINGERS]
    width = 3 if args.target == 'force' else 6
    info = json.loads((root/'meta/info.json').read_text()); fps = float(info['fps'])
    keys = {f:f'observation.images.tactile_{f.split("_",1)[0]}_raw_{f.split("_",1)[1]}' for f in fingers}
    columns = ['episode_index', 'length', 'object', 'motor_primitive']
    for key in keys.values():
        columns += [f'videos/{key}/{field}' for field in ('chunk_index','file_index','from_timestamp','to_timestamp')]
    meta_files = sorted((root/'meta/episodes').rglob('*.parquet'))
    rows = [r for file in meta_files for r in pq.read_table(file,columns=columns).to_pylist()]
    if len(rows) != info['total_episodes'] or sum(r['length'] for r in rows) != info['total_frames']:
        raise ValueError('Episode metadata does not cover the full dataset')
    if len({r['episode_index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode IDs')
    hashes = {str(p.relative_to(root)):digest(p) for p in [root/'meta/info.json',*meta_files]}
    partitions = split_episodes(rows,args.split,args.seed)
    for name, values in partitions.items():
        random.Random(args.seed+{'train':1,'val':2,'test':3}[name]).shuffle(values)
        if args.limit_per_split:
            partitions[name]=values[:args.limit_per_split]
    selected = {r['episode_index']:(s,r) for s,values in partitions.items() for r in values}
    required = {}
    for _, r in selected.values():
        r['_videos']={}
        for finger,key in keys.items():
            prefix=f'videos/{key}/'
            rel=f"videos/{key}/chunk-{int(r[prefix+'chunk_index']):03d}/file-{int(r[prefix+'file_index']):03d}.mp4"
            r['_videos'][finger]={'video':rel,'from_timestamp':float(r[prefix+'from_timestamp'])}
            required[rel]=max(required.get(rel,0),float(r[prefix+'to_timestamp']))
    def inspect(entry):
        rel,end=entry;path=root/rel;stat=path.stat()
        if stat.st_size == 0:raise ValueError(f'Empty video: {path}')
        with av.open(str(path)) as c:
            st=c.streams.video[0]
            if (st.width,st.height)!=(320,240) or abs(float(st.average_rate)-fps)>1e-3:
                raise ValueError(f'Unexpected raw video format: {path}')
            duration=float(st.duration*st.time_base) if st.duration is not None else None
        if duration is None or end > duration+1/fps:
            raise ValueError(f'Incomplete video duration: {path}')
        return rel,{'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'duration':duration}
    stamps={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(rel,stamp) in enumerate(pool.map(inspect,required.items()),1):
            stamps[rel]=stamp
            if i%250==0 or i==len(required):print(f'Validated {i}/{len(required)} raw tactile videos',flush=True)
    gathered=defaultdict(list);data_sources={};wanted=set(selected)
    for file in sorted((root/'data').rglob('*.parquet')):
        before=file.stat()
        t=pq.read_table(file,columns=['episode_index','frame_index','timestamp','observation.tactile_force'])
        ids=t['episode_index'].to_numpy(); use=np.isin(ids,list(wanted));t=t.filter(use)
        if not len(t):continue
        ids=t['episode_index'].to_numpy();frames=t['frame_index'].to_numpy();times=t['timestamp'].to_numpy()
        forces=np.asarray(t['observation.tactile_force'].to_pylist(),dtype=np.float32)
        if forces.shape!=(len(t),60) or not np.isfinite(forces).all():raise ValueError(f'Invalid force labels: {file}')
        # Store only the requested axes in fixed left/right, thumb/index/middle/ring/pinky order.
        forces=forces.reshape(-1,10,6)[:,:,:width].copy()
        for eid in np.unique(ids):
            mask=ids==eid;gathered[int(eid)].append((frames[mask],times[mask],forces[mask]))
        after=file.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError(f'Changing labels: {file}')
        data_sources[str(file.relative_to(root))]={'bytes':after.st_size,'mtime_ns':after.st_mtime_ns}
        print(f'Read {file.name}: {len(t)} frames x 10 fingers',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_name(out.name+'.building-'+uuid.uuid4().hex[:8]);temp.mkdir();(temp/'episodes').mkdir()
    try:
        records=[];sums=np.zeros(width);squares=np.zeros(width);count=0;norms=[]
        for num,(eid,(split,r)) in enumerate(sorted(selected.items()),1):
            if eid not in gathered:raise ValueError(f'No labels for episode {eid}')
            chunks=gathered.pop(eid)
            arrays=[np.concatenate([v[i] for v in chunks]) for i in range(3)]
            order=np.argsort(arrays[0]);frames,times,targets=[a[order] for a in arrays]
            if not np.array_equal(frames,np.arange(r['length'])):raise ValueError(f'Bad frame coverage: {eid}')
            if not np.allclose(times,frames/fps,atol=1e-3):raise ValueError(f'Bad timestamps: {eid}')
            frames,times,targets=frames[::args.stride],times[::args.stride],targets[::args.stride]
            rel=f'episodes/{eid:06d}.npz'
            np.savez_compressed(temp/rel,frame_index=frames,timestamp=times,target=targets)
            sha=digest(temp/rel)
            for index,finger in enumerate(fingers):
                records.append({'episode_index':eid,'split':split,'object':r['object'],'primitive':r['motor_primitive'],
                                'finger':finger,'finger_index':index,'samples':len(frames),'labels':rel,
                                'labels_sha256':sha,**r['_videos'][finger]})
            if split=='train':
                y=targets.reshape(-1,width).astype(np.float64)
                sums+=y.sum(0);squares+=(y*y).sum(0);count+=len(y)
                norms.append(np.linalg.norm(y[:,:3],axis=1).astype(np.float32))
            if num%250==0 or num==len(selected):print(f'Cached {num}/{len(selected)} episodes (all fingers)',flush=True)
        mean=sums/count;std=np.sqrt(np.maximum(0,squares/count-mean**2))
        if np.any(std<1e-8):raise ValueError('Constant training target axes')
        stats={'mean':mean.tolist(),'std':std.tolist(),'count':count,'force_norm_q90':float(np.quantile(np.concatenate(norms),.9))}
        for rel,stamp in stamps.items():
            st=(root/rel).stat()
            if (st.st_size,st.st_mtime_ns)!=(stamp['bytes'],stamp['mtime_ns']):raise RuntimeError(f'Video changed: {rel}')
        for rel,sha in hashes.items():
            if digest(root/rel)!=sha:raise RuntimeError(f'Metadata changed: {rel}')
        manifest={'version':2,'root':str(root),'finger':'all','fingers':fingers,'target':args.target,'output_dim':width,
                  'fps':fps,'stride':args.stride,'split_mode':args.split,'seed':args.seed,'pilot_limit_per_split':args.limit_per_split,
                  'statistics':stats,'episodes':records,'source_metadata_sha256':hashes,'source_label_files':data_sources,
                  'source_videos':stamps,'units':'dataset native; SI units and coordinate calibration not independently verified'}
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2));temp.rename(out)
    except BaseException:
        shutil.rmtree(temp);raise
    summary={s:{'episodes':len(partitions[s]),'finger_episode_streams':sum(r['split']==s for r in records),
                'image_samples':sum(r['samples'] for r in records if r['split']==s)} for s in partitions}
    print(json.dumps({'prepared':str(out),'fingers':fingers,'splits':summary},indent=2),flush=True)
