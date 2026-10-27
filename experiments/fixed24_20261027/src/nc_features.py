"""Deterministic, target-free NC instruction features aligned by mapped_nc_line."""
import bisect
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEFAULT_MANIFEST=Path(os.environ.get('TEE_NC_MANIFEST', ROOT/'nc_manifest.json'))
WORD=re.compile(r'(?<![A-Z])([XYZACFS])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',re.I)
GCODE=re.compile(r'(?<![A-Z0-9])G0*([0123])(?=[^0-9]|$)',re.I)
COMMENT=re.compile(r'\([^)]*\)|;.*$')
NC_COLS=[
 'nc_valid','nc_line_progress','nc_mode_rapid','nc_mode_arc','nc_feed','nc_spindle',
 'nc_x','nc_y','nc_z','nc_a','nc_c',
 'nc_tangent_x','nc_tangent_y','nc_tangent_z','nc_segment_length',
 'nc_next_dx','nc_next_dy','nc_next_dz','nc_next_segment_length','nc_next5_max_segment',
 'nc_corner_degrees','nc_next_feed_change','nc_next_rotary_change',
 'nc_pose_offset_x','nc_pose_offset_y','nc_pose_offset_z','nc_pose_offset_norm',
 'nc_upcoming_plunge','nc_line_gap'
]


def read_manifest(path=None):
    p=Path(path or DEFAULT_MANIFEST)
    data=json.loads(p.read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict): raise ValueError('NC manifest must be {JSON filename: NC path}')
    return {str(k): ({str(i):str(Path(p).resolve()) for i,p in v.items()}
                     if isinstance(v,dict) else str(Path(v).resolve())) for k,v in data.items()}


def parse_program(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f'NC source missing: {p}')
    raw=p.read_bytes()
    # NC files may use a machine-local legacy code page; XYZACFS/G tokens are ASCII.
    lines=raw.decode('latin1').splitlines()
    state={a:np.nan for a in 'XYZAC'}; state.update({'F':0.,'S':0.})
    mode=1; moves=[]
    last_xyz=None
    for idx,line in enumerate(lines):
        code=COMMENT.sub('',line.upper())
        gm=GCODE.search(code)
        if gm: mode=int(gm.group(1))
        words={a.upper():float(v) for a,v in WORD.findall(code)}
        state.update(words)
        if not any(a in words for a in 'XYZAC'): continue
        if not all(np.isfinite(state[a]) for a in 'XYZ'): continue
        xyz=np.array([state[a] for a in 'XYZ'],float)
        segment=float(np.linalg.norm(xyz-last_xyz)) if last_xyz is not None else 0.
        moves.append((idx,xyz,np.array([state[a] if np.isfinite(state[a]) else 0. for a in 'AC'],float),mode,state['F'],state['S'],segment))
        last_xyz=xyz.copy()
    if not moves: raise ValueError(f'No valid XYZ motion instructions: {p}')
    return moves,len(lines),hashlib.sha256(raw).hexdigest()


def feature_for_point(line, pose, moves, indices, line_count):
    result=dict.fromkeys(NC_COLS,0.)
    if not np.isfinite(line) or line < 0 or line >= line_count:
        return result
    i=bisect.bisect_right(indices,int(line))-1
    if i<0: return result
    cmd=moves[i]; xyz=cmd[1]; prev=moves[max(0,i-1)]; next_cmd=moves[min(len(moves)-1,i+1)]
    next5=moves[i+1:min(len(moves),i+6)]
    tangent=xyz-prev[1]; tangent_len=float(np.linalg.norm(tangent))
    outgoing=next_cmd[1]-xyz; outgoing_len=float(np.linalg.norm(outgoing))
    if tangent_len>1e-9 and outgoing_len>1e-9:
        cosine=np.clip(np.dot(tangent,outgoing)/(tangent_len*outgoing_len),-1.,1.)
        corner=float(np.degrees(np.arccos(cosine)))
    else: corner=0.
    offset=np.asarray(pose,float)-xyz
    vals={
      'nc_valid':1.,'nc_line_progress':cmd[0]/max(line_count-1,1),
      'nc_mode_rapid':float(cmd[3]==0),'nc_mode_arc':float(cmd[3] in (2,3)),
      'nc_feed':cmd[4],'nc_spindle':cmd[5],
      'nc_x':xyz[0],'nc_y':xyz[1],'nc_z':xyz[2],
      'nc_a':cmd[2][0],'nc_c':cmd[2][1],
      'nc_tangent_x':tangent[0]/max(tangent_len,1e-9),
      'nc_tangent_y':tangent[1]/max(tangent_len,1e-9),
      'nc_tangent_z':tangent[2]/max(tangent_len,1e-9),
      'nc_segment_length':cmd[6],
      'nc_next_dx':outgoing[0],'nc_next_dy':outgoing[1],'nc_next_dz':outgoing[2],
      'nc_next_segment_length':outgoing_len,
      'nc_next5_max_segment':max([m[6] for m in next5],default=0.),
      'nc_corner_degrees':corner,'nc_next_feed_change':abs(next_cmd[4]-cmd[4]),
      'nc_next_rotary_change':float(np.linalg.norm(next_cmd[2]-cmd[2])),
      'nc_pose_offset_x':offset[0],'nc_pose_offset_y':offset[1],'nc_pose_offset_z':offset[2],
      'nc_pose_offset_norm':float(np.linalg.norm(offset)),
      'nc_upcoming_plunge':max(0.,-outgoing[2]),
      'nc_line_gap':max(0.,float(line)-cmd[0]),
    }
    result.update(vals)
    return {k:float(v) if np.isfinite(v) else 0. for k,v in result.items()}


def add_nc_features(df, manifest=None, *, min_coverage=.90, return_metadata=False):
    paths=read_manifest(manifest) if manifest is None or not isinstance(manifest,dict) else manifest
    required={'source_file_name','mapped_nc_line','pose_x','pose_y','pose_z'}
    if not required.issubset(df): raise ValueError('NC input missing columns: '+str(required-set(df)))
    frames=[]; metadata=[]
    group_cols=['source_file_name','original_mapped_file_index'] if 'original_mapped_file_index' in df else ['source_file_name']
    for group_key,group in df.groupby(group_cols,sort=False):
        source=group_key[0] if isinstance(group_key,tuple) else group_key
        original_index=int(group_key[1]) if isinstance(group_key,tuple) and len(group_key)>1 else 0
        if source not in paths: raise ValueError(f'No NC program mapping for {source}. Supply --nc-map with its original NC file.')
        mapping=paths[source]
        if isinstance(mapping,dict):
            if str(original_index) not in mapping:
                raise ValueError(f'NC manifest missing original file index {original_index} for {source}')
            path=Path(mapping[str(original_index)])
        else:
            if original_index!=0: raise ValueError(f'{source} has multiple NC programs; map each original file index separately')
            path=Path(mapping)
        moves,nlines,sha=parse_program(path)
        indices=[m[0] for m in moves]
        vals=[]
        for line,xyz in zip(group['mapped_nc_line'].to_numpy(),group[['pose_x','pose_y','pose_z']].to_numpy()):
            vals.append(feature_for_point(line,xyz,moves,indices,nlines))
        feature_df=pd.DataFrame(vals,index=group.index,columns=NC_COLS)
        coverage=float(feature_df['nc_valid'].mean())
        metadata.append({'source':source,'original_file_index':original_index,'path':str(path),'sha256':sha,'lines':nlines,'motion_blocks':len(moves),
                         'mapped_points':len(group),'valid_coverage':coverage,
                         'median_pose_offset_mm':float(feature_df.loc[feature_df.nc_valid>0,'nc_pose_offset_norm'].median())})
        if coverage<min_coverage: raise ValueError(f'NC line alignment too low: {source} = {coverage:.1%}')
        frames.append(feature_df)
    all_features=pd.concat(frames).loc[df.index]
    output=pd.concat([df,all_features.astype(np.float32)],axis=1)
    if return_metadata: return output,metadata
    return output
