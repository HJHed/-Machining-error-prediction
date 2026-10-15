"""Training-only calibration of a machining-dynamics risk score.

The score describes unusual tool motion/path transitions.  It is not a
probability that the downstream error predictor will fail.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import main as training
import motion_risk as original_risk


ROOT=Path(__file__).resolve().parent.parent
PRIOR=ROOT/'results/exp1_nc_risk_20261015'
ORIGINAL_PREPARE=original_risk.prepare
FRAME_KEYS=['mapped_file_index','mapped_nc_line','mapped_frame_num','mapped_frame_id']
POINT_KEYS=FRAME_KEYS+['point_index','source_file_name']
FEATURES=['risk_linear_acceleration','risk_pose_jump','nc_corner_degrees','nc_next_segment_length']
QUANTILES=[.75,.90,.95,.99]


def extract(df):
    motion=original_risk.motion_features(df,FRAME_KEYS)
    out=pd.DataFrame(index=df.index)
    out['risk_linear_acceleration']=motion['risk_linear_acceleration'].to_numpy(float)
    out['risk_pose_jump']=motion['risk_pose_jump'].to_numpy(float)
    for col in FEATURES[2:]:
        out[col]=pd.to_numeric(df[col],errors='coerce').fillna(0).to_numpy(float)
    out['mapped_file_index']=df['mapped_file_index'].to_numpy()
    return out.replace([np.inf,-np.inf],0).fillna(0)


def fit_spec(train_df):
    # Repeated measurement points in one frame contribute once to thresholds.
    frames=train_df.drop_duplicates(FRAME_KEYS)
    values=extract(frames)
    programs={}
    for file_id,part in values.groupby('mapped_file_index'):
        programs[int(file_id)]={col:{
            'quantiles':np.quantile(part[col].to_numpy(float),QUANTILES),
            'sorted':np.sort(part[col].to_numpy(float)),
        } for col in FEATURES}
    return {'programs':programs,'quantiles':QUANTILES,
            'definition':'extreme acceleration/pose jump, or coupled motion and NC path transition',
            'score_kind':'relative physical-dynamics severity; not failure probability',
            'version':1}


def score(spec,df):
    f=extract(df); n=len(f)
    ranks={c:np.zeros(n,np.float32) for c in FEATURES}
    exceed={c:np.zeros((n,4),bool) for c in FEATURES}
    ids=f.mapped_file_index.to_numpy()
    for file_id in np.unique(ids):
        if int(file_id) not in spec['programs']:
            raise ValueError(f'Physical-risk calibration missing NC program {file_id}')
        idx=np.flatnonzero(ids==file_id)
        for col in FEATURES:
            vals=f[col].to_numpy(float)[idx]
            item=spec['programs'][int(file_id)][col]
            sorted_train=item['sorted']; q=item['quantiles']
            ranks[col][idx]=np.searchsorted(sorted_train,vals,side='right')/len(sorted_train)
            exceed[col][idx]=vals[:,None]>q[None,:]
    a=exceed['risk_linear_acceleration'];j=exceed['risk_pose_jump']
    c=exceed['nc_corner_degrees'];s=exceed['nc_next_segment_length']
    high=a[:,3]|j[:,3]|(a[:,2]&(j[:,0]|c[:,1]))|(a[:,1]&j[:,2])|(a[:,1]&s[:,2])
    ar=ranks['risk_linear_acceleration'];jr=ranks['risk_pose_jump']
    cr=ranks['nc_corner_degrees'];sr=ranks['nc_next_segment_length']
    severity=np.maximum.reduce([np.minimum(ar,jr),np.minimum(ar,cr),
        np.minimum(ar,sr),.8*ar**3,.8*jr**3]).astype(np.float32)
    return severity,high


def prepare(g,d):
    out=Path(g['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    saved=d['saved_artifacts']
    old_cache=PRIOR/'motion_risk_cache.pkl'
    if saved is None and old_cache.exists() and not (out/old_cache.name).exists():
        shutil.copy2(old_cache,out/old_cache.name)
    base=list(ORIGINAL_PREPARE(g,d))
    xs=list(base[:3]);weights=base[3];artifact=base[4];packs=base[5]
    old_weights=[p['weight'].copy() for p in packs]
    tables=[d[s+'_diag_df'] for s in ('train','val','test')]
    if saved is None:
        spec=fit_spec(tables[0])
    else:
        spec=artifact.get('physical_risk')
        if spec is None: raise ValueError('Matching physical-risk calibration is required')
    scored=[score(spec,t) for t in tables]
    # The risk definition and alert are fixed before looking at validation
    # errors. Only a mild training weight is applied to physical extremes.
    strength=float(spec.get('weight_strength',.35))
    mild=float(spec.get('mild_score_strength',.15))
    if saved is None:
        normalizer=float(np.mean(1+mild*scored[0][0]+strength*scored[0][1]))
        spec.update(weight_strength=strength,mild_score_strength=mild,
                    weight_normalizer=normalizer)
    else: normalizer=spec['weight_normalizer']
    new_weights=[((1+mild*severity+strength*high)/normalizer).astype(np.float32)
                 for severity,high in scored]
    scaler=StandardScaler().fit(np.column_stack([scored[0][0],new_weights[0]])) if saved is None else spec['input_scaler']
    if saved is None: spec['input_scaler']=scaler
    for i,split in enumerate(('train','val','test')):
        start=-len(artifact['channel_names'])
        xs[i][:,start:start+2]=scaler.transform(
            np.column_stack([scored[i][0],new_weights[i]])).astype(np.float32)
        packs[i]['probability']=scored[i][0]
        packs[i]['weight']=new_weights[i]
        packs[i]['physical_high_risk']=scored[i][1]
        table=pd.read_csv(out/(split+'_point_weights.csv'))
        table['old_failure_risk_score']=table['risk_probability']
        table['risk_probability']=scored[i][0] # compatibility column; not a probability
        table['physical_risk_score']=scored[i][0]
        table['point_weight']=new_weights[i]
        table['high_risk']=scored[i][1]
        table['nc_corner_degrees']=tables[i]['nc_corner_degrees'].to_numpy(float)
        table['nc_next_segment_length']=tables[i]['nc_next_segment_length'].to_numpy(float)
        table.to_csv(out/(split+'_point_weights.csv'),index=False,encoding='utf-8-sig')
    ratio=new_weights[0]/old_weights[0]
    weights=[w*ratio.reshape((-1,)+(1,)*(w.ndim-1)) for w in weights]
    all_df=d['point_df'];all_score,all_high=score(spec,all_df)
    all_weight=((1+mild*all_score+strength*all_high)/normalizer).astype(np.float32)
    all_path=out/'all_point_preprocessing_weights.csv'
    raw=pd.read_csv(all_path)
    keyed=all_df[POINT_KEYS].copy()
    keyed['physical_risk_score']=all_score
    keyed['physical_high_risk']=all_high
    keyed['physical_point_weight']=all_weight
    raw=raw.merge(keyed,on=POINT_KEYS,how='left',validate='one_to_one',sort=False)
    if raw.physical_risk_score.isna().any(): raise ValueError('Physical-risk raw point alignment failed')
    raw['old_failure_risk_score']=raw['risk_probability']
    raw['risk_probability']=raw['physical_risk_score']
    raw['point_weight']=raw['physical_point_weight']
    raw['high_risk']=raw['physical_high_risk']
    raw=raw.drop(columns=['physical_high_risk','physical_point_weight'])
    raw.to_csv(all_path,index=False,encoding='utf-8-sig')
    artifact['physical_risk']=spec
    artifact['channel_names'][:2]=['physical_risk_score','physical_risk_weight']
    if saved is None:
        summary={'definition':spec['definition'],'score_kind':spec['score_kind'],
            'training_programs':len(spec['programs']),'training_frame_quantiles':QUANTILES,
            'weight_strength':strength,'mild_score_strength':mild,
            'train_high_risk':int(scored[0][1].sum()),'train_points':len(scored[0][1]),
            'validation_high_risk':int(scored[1][1].sum()),'validation_points':len(scored[1][1]),
            'test_high_risk_unlabeled':int(scored[2][1].sum()),'test_points':len(scored[2][1]),
            'training_target_used_for_risk':False,'validation_target_used_for_risk':False,
            'test_target_used_for_risk':False}
        (out/'physical_risk_preprocessing.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Physical NC-motion risk applied: train %.2f%%, validation %.2f%%, test %.2f%% high risk.' %
          tuple(100*high.mean() for _,high in scored),flush=True)
    return xs[0],xs[1],xs[2],weights,artifact,packs


if __name__=='__main__':
    training.motion_risk.prepare=prepare
    training.main()
