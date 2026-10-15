"""Train-only difficulty preprocessor; no validation/test targets are accepted."""
import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

MOTION_COLS = ['risk_frame_displacement', 'risk_linear_speed', 'risk_linear_acceleration',
               'risk_rotation_speed', 'risk_rotation_acceleration', 'risk_velocity_change',
               'risk_pose_jump', 'risk_invalid_motion']


def motion_features(df, keys):
    """Use supplied frame v/a; sampled pose jumps are proxies, never called velocity."""
    def num(col):
        return pd.to_numeric(df.get(col, pd.Series(0., index=df.index)), errors='coerce').fillna(0).to_numpy(float)
    dt = num('motion_dt')
    speed = np.sqrt(sum(num('motion_velocity_v_'+a)**2 for a in 'XYZ'))
    acc = np.sqrt(sum(num('motion_acceleration_a_'+a)**2 for a in 'XYZ'))
    rot_v = np.sqrt(sum(num('motion_velocity_v_'+a)**2 for a in 'AC'))
    rot_a = np.sqrt(sum(num('motion_acceleration_a_'+a)**2 for a in 'AC'))
    speed = np.maximum(speed, np.abs(num('motion_linear_speed_xyz')))
    acc = np.maximum(acc, np.abs(num('motion_linear_acc_xyz')))
    f = pd.DataFrame(index=df.index)
    f[MOTION_COLS[0]] = speed * np.maximum(dt, 0)
    f[MOTION_COLS[1]] = speed
    f[MOTION_COLS[2]] = acc
    f[MOTION_COLS[3]] = rot_v
    f[MOTION_COLS[4]] = rot_a
    f[MOTION_COLS[5]] = acc * np.maximum(dt, 0)
    # Deduplicate mapped frames so multiple measurement points in a frame share a jump.
    unique = df.drop_duplicates(keys).sort_values(keys, kind='mergesort').copy()
    dxyz = unique.groupby('mapped_file_index', sort=False)[['pose_x','pose_y','pose_z']].diff().fillna(0)
    unique['risk_pose_jump'] = np.linalg.norm(dxyz.to_numpy(float), axis=1)
    aligned = df[keys].merge(unique[keys+['risk_pose_jump']], on=keys, how='left', validate='many_to_one', sort=False)
    f[MOTION_COLS[6]] = num('risk_pose_jump') if 'risk_pose_jump' in df else aligned['risk_pose_jump'].to_numpy()
    f[MOTION_COLS[7]] = ((dt <= 0) | ~np.isfinite(dt)).astype(float)
    return f.replace([np.inf, -np.inf], 0).fillna(0)


def known_features(df, cols, keys):
    f = motion_features(df, keys)
    for col in cols:
        if col not in f:
            if col not in df:
                raise ValueError('Missing risk input: '+col)
            f[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return f[cols].replace([np.inf, -np.inf], 0).fillna(0).to_numpy(np.float32)


def classifier(seed):
    return HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=12,
        min_samples_leaf=45, learning_rate=.045, l2_regularization=5., random_state=seed)


def probability(model, X):
    if len(model.classes_) == 1:
        return np.full(len(X), float(model.classes_[0]), dtype=np.float32)
    return model.predict_proba(X)[:, list(model.classes_).index(1)].astype(np.float32)


def fingerprint(api, d):
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    h.update(Path(api.__file__).read_bytes())
    h.update(str((api.RANDOM_SEED, os.getenv('TEE_RISK_TEACHER_EPOCHS','100'),
                  os.getenv('TEE_RISK_ALPHA','1.5'),os.getenv('TEE_RISK_BETA','.5'))).encode())
    for name in ['train_static_X','train_seq_X','train_y']:
        h.update(np.ascontiguousarray(d[name]).tobytes())
    h.update(d['train_meta'].to_csv(index=False).encode())
    return h.hexdigest()


def within_workpiece_blocks(meta, keys, block_count=10):
    """Contiguous NC/frame blocks inside each source, shared by repeated points."""
    unique=meta[['source_file_name']+keys].drop_duplicates().sort_values(['source_file_name']+keys,kind='mergesort').copy()
    unique['program_group']=unique['source_file_name']+'|'+unique['mapped_file_index'].astype(str)
    unique['frame_order']=unique.groupby('program_group',sort=False).cumcount()
    sizes=unique.groupby('program_group')['frame_order'].transform('max').to_numpy()+1
    unique['block']=(unique['frame_order'].to_numpy()*block_count/sizes).astype(int).clip(0,block_count-1)
    merged=meta[['source_file_name']+keys].merge(unique,on=['source_file_name']+keys,how='left',validate='many_to_one',sort=False)
    groups=(merged['program_group']+'|'+merged['block'].astype(str)).to_numpy()
    return groups,merged['frame_order'].to_numpy(),merged['program_group'].to_numpy()


def purge_near_held(fit_indices, hold_indices, source, order, radius):
    keep=np.ones(len(fit_indices),bool)
    for name in set(source[hold_indices]):
        held_order=np.unique(order[hold_indices[source[hold_indices]==name]])
        mask=source[fit_indices]==name
        values=order[fit_indices[mask]]
        at=np.searchsorted(held_order,values)
        left=np.maximum(at-1,0); right=np.minimum(at,len(held_order)-1)
        distance=np.minimum(abs(values-held_order[left]),abs(values-held_order[right]))
        keep[np.flatnonzero(mask)]=distance>radius
    return fit_indices[keep]


def teacher_oof(api, d):
    import tensorflow as tf
    groups,order,source=within_workpiece_blocks(d['train_meta'],api.FRAME_KEYS)
    yraw=d['target_scaler'].inverse_transform(d['train_y'])
    static_raw=d['static_scaler'].inverse_transform(d['train_static_X'])
    seq_raw=d['frame_scaler'].inverse_transform(d['train_seq_X'].reshape(-1,d['train_seq_X'].shape[-1])).reshape(d['train_seq_X'].shape)
    preds=np.full(yraw.shape,np.nan,dtype=np.float32)
    fold_id=np.zeros(len(yraw),dtype=int); summaries=[]
    for fold,(fit_all,hold) in enumerate(GroupKFold(n_splits=5).split(static_raw,groups=groups)):
        fit_all=purge_near_held(fit_all,hold,source,order,api.TIME_STEPS)
        fit_groups=sorted(set(groups[fit_all]))
        inner_groups=set(fit_groups[fold::10])
        fit=fit_all[np.array([g not in inner_groups for g in groups[fit_all]])]
        inner=fit_all[np.array([g in inner_groups for g in groups[fit_all]])]
        fit=purge_near_held(fit,inner,source,order,api.TIME_STEPS)
        assert not set(groups[fit]) & set(groups[hold])
        assert not set(groups[inner]) & set(groups[hold])
        ssc=StandardScaler().fit(static_raw[fit]); fsc=StandardScaler().fit(seq_raw[fit].reshape(-1,seq_raw.shape[-1]))
        ysc=StandardScaler().fit(yraw[fit]); edges=api.fit_error_class_edges(ysc)
        def arrays(idx):
            xs=ssc.transform(static_raw[idx]).astype(np.float32)
            sq=fsc.transform(seq_raw[idx].reshape(-1,seq_raw.shape[-1])).reshape((-1,api.TIME_STEPS,seq_raw.shape[-1])).astype(np.float32)
            yy=ysc.transform(yraw[idx]).astype(np.float32)
            cc=api.build_error_class_labels(yy,edges)
            dd,rr=api.build_auxiliary_targets(yy,ysc)
            return api.make_model_inputs(xs,sq),api.make_model_targets(yy,cc,dd,rr),api.make_model_sample_weights(api.build_high_value_regression_weights(cc),np.ones(cc.shape,np.float32),dd,rr)
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(api.RANDOM_SEED+fold)
        model=api.build_spatiotemporal_model(static_raw.shape[1],seq_raw.shape[2],api.TIME_STEPS,ysc.scale_)
        tx,ty,tw=arrays(fit); vx,vy,vw=arrays(inner); hx,_,_=arrays(hold)
        print(f'[NC risk teacher {fold+1}/5] fit={len(fit)}, inner_val={len(inner)}, held={len(hold)}',flush=True)
        hist=model.fit(tx,ty,sample_weight=tw,validation_data=(vx,vy,vw),
            epochs=int(os.getenv('TEE_RISK_TEACHER_EPOCHS','100')),batch_size=64,verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=6,factor=.5,min_lr=1e-6)])
        pred=api.split_model_predictions(model.predict(hx,verbose=0))[0]
        preds[hold]=ysc.inverse_transform(pred); fold_id[hold]=fold
        summaries.append({'fold':fold,'fit':len(fit),'inner_validation':len(inner),'held':len(hold),
                          'epochs':len(hist.history['loss']),'held_sources':sorted(set(source[hold]))})
        del model
    assert np.isfinite(preds).all()
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(api.RANDOM_SEED)
    return np.linalg.norm(yraw-preds,axis=1),fold_id,summaries


def risk_predict(artifact, df, keys):
    X=known_features(df,artifact['feature_cols'],keys)
    p=probability(artifact['model'],X)
    return risk_channels(artifact,X,p), p


def risk_channels(artifact, X, p):
    # Shock score is bounded using training-only 90th percentiles.
    shock=np.mean(np.clip(X[:,artifact['shock_indices']]/artifact['shock_scales'],0.,1.),axis=1)
    w=(1.+artifact['alpha']*p+artifact['beta']*shock)/artifact['weight_normalizer']
    w=np.maximum(w,.25).astype(np.float32)
    m=np.log1p(np.maximum(X[:,:len(MOTION_COLS)],0))
    raw=np.column_stack([p,w,m]).astype(np.float32)
    return {'weight':w,'probability':p,'shock':shock,'raw_channels':raw}


def prepare(g, d):
    api=SimpleNamespace(**g); out=Path(api.OUTPUT_DIR); out.mkdir(parents=True,exist_ok=True)
    keys=api.FRAME_KEYS
    saved=d['saved_artifacts']
    if saved is not None:
        artifact=saved.get('motion_risk')
        if artifact is None: raise ValueError('Weighted prediction requires matching motion_risk artifact.')
        tables=[d[x+'_diag_df'] for x in ['train','val','test']]
        packs=[risk_predict(artifact,t,keys)[0] for t in tables]
    else:
        key=fingerprint(api,d); cache=out/'motion_risk_cache.pkl'
        tables=[d[x+'_diag_df'] for x in ['train','val','test']]
        if cache.exists():
            cached=pickle.loads(cache.read_bytes())
        else: cached={}
        if cached.get('fingerprint')==key:
            artifact=cached['artifact']; packs=[cached['train_pack']]+[risk_predict(artifact,t,keys)[0] for t in tables[1:]]
            print('Reused verified motion risk cache.',flush=True)
        else:
            distance, fold_id, teacher_summary=teacher_oof(api,d)
            label=(distance >= api.POSITION_DISTANCE_THRESHOLD).astype(int)
            train_df=tables[0]
            feature_cols=MOTION_COLS+[c for c in train_df.columns if c.startswith(('pose_','sin_pose_','cos_pose_','temp_','source_line_progress','source_frame_progress','nc_')) or c in ['feed_speed','spindle_speed']]
            feature_cols=list(dict.fromkeys(feature_cols))
            assert not any(c in feature_cols for c in api.TARGET_COLS)
            X=known_features(train_df,feature_cols,keys); groups,order,source=within_workpiece_blocks(d['train_meta'],keys)
            oof_p=np.zeros(len(X),np.float32)
            for fold,(fit,hold) in enumerate(GroupKFold(5).split(X,groups=groups)):
                fit=purge_near_held(fit,hold,source,order,api.TIME_STEPS)
                m=classifier(api.RANDOM_SEED+fold).fit(X[fit],label[fit]); oof_p[hold]=probability(m,X[hold])
            model=classifier(api.RANDOM_SEED).fit(X,label)
            shock_indices=[0,2,3,4,5]
            scales=np.maximum(np.quantile(X[:,shock_indices],.90,axis=0),1e-6)
            alpha=float(os.getenv('TEE_RISK_ALPHA','1.5')); beta=float(os.getenv('TEE_RISK_BETA','.5'))
            artifact={'model':model,'feature_cols':feature_cols,'shock_indices':shock_indices,'shock_scales':scales,
                      'alpha':alpha,'beta':beta,'weight_normalizer':1.,'teacher_summary':teacher_summary,
                      'threshold_mm':api.POSITION_DISTANCE_THRESHOLD,'version':1}
            temp=risk_channels(artifact,X,oof_p); artifact['weight_normalizer']=float(temp['weight'].mean())
            train_pack=risk_channels(artifact,X,oof_p)
            artifact['channel_scaler']=StandardScaler().fit(train_pack['raw_channels'])
            artifact['high_risk_probability_threshold']=0.5  # replaced after training using validation recall only
            artifact['channel_names']=['risk_probability','risk_weight']+['log1p_'+c for c in MOTION_COLS]
            packs=[train_pack]+[risk_predict(artifact,t,keys)[0] for t in tables[1:]]
            report={'label_source':'5-fold contiguous within-workpiece phase-block held-out Hybrid NN, 10-frame embargo; train only',
                    'risk_probability_source':'5-fold phase-block held-out classifier for training; full-train classifier for validation/test',
                    'oof_failure_rate':float(label.mean()),'oof_auc':float(roc_auc_score(label,oof_p)) if len(np.unique(label))>1 else None,
                    'oof_average_precision':float(average_precision_score(label,oof_p)),
                    'oof_brier':float(brier_score_loss(label,oof_p)),
                    'training_weight_min':float(train_pack['weight'].min()),'training_weight_max':float(train_pack['weight'].max()),
                    'training_weight_mean':float(train_pack['weight'].mean()),'teacher_folds':teacher_summary,
                    'risk_feature_columns':feature_cols,'added_static_channels':artifact['channel_names']}
            (out/'risk_preprocessing_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
            diag=train_df[keys+['point_index','source_file_name']].copy()
            diag['teacher_oof_distance_mm']=distance; diag['teacher_oof_failure']=label; diag['teacher_fold']=fold_id
            diag['risk_probability_oof']=oof_p; diag['training_weight']=train_pack['weight']
            diag.to_csv(out/'training_oof_difficulty.csv',index=False,encoding='utf-8-sig')
            cache.write_bytes(pickle.dumps({'fingerprint':key,'artifact':artifact,'train_pack':train_pack}))
    arrays=[]
    for split,table,pack in zip(['train','val','test'],tables,packs):
        extra=artifact['channel_scaler'].transform(pack['raw_channels']).astype(np.float32)
        arrays.append(np.column_stack([d[split+'_static_X'],extra]).astype(np.float32))
        diag=table[keys+['point_index','source_file_name']+api.POSE_COLS].copy()
        diag['risk_probability']=pack['probability']; diag['point_weight']=pack['weight']; diag['motion_shock']=pack['shock']
        diag['high_risk']=pack['probability']>=artifact['high_risk_probability_threshold']
        for c in MOTION_COLS: diag[c]=motion_features(table,keys)[c].to_numpy()
        diag.to_csv(out/(split+'_point_weights.csv'),index=False,encoding='utf-8-sig')
    # Export every raw measurement point too; sequence warmup rows have no final prediction.
    all_df=d['point_df']; all_pack=risk_predict(artifact,all_df,keys)[0]
    all_export=all_df[keys+['point_index','source_file_name']+api.POSE_COLS].copy()
    all_export['risk_probability']=all_pack['probability']; all_export['point_weight']=all_pack['weight']
    all_export['split']=''; all_export['has_sequence']=False
    for split,table,pack in zip(['train','val','test'],tables,packs):
        lookup=table[keys+['point_index']].copy(); lookup['_p']=pack['probability']; lookup['_w']=pack['weight']
        merged=all_export.merge(lookup,on=keys+['point_index'],how='left',validate='one_to_one')
        mask=merged['_p'].notna().to_numpy()
        all_export.loc[mask,['risk_probability','point_weight']]=merged.loc[mask,['_p','_w']].to_numpy()
        all_export.loc[mask,'has_sequence']=True
    for split in ['train','val','test']:
        raw=d[split+'_points_raw'][keys+['point_index']].copy(); raw['_split']=split
        merged=all_export.merge(raw,on=keys+['point_index'],how='left',validate='one_to_one')
        mask=merged['_split'].notna().to_numpy(); all_export.loc[mask,'split']=split
    all_export.to_csv(out/'all_point_preprocessing_weights.csv',index=False,encoding='utf-8-sig')
    tw=[a.copy() for a in d['train_sample_weights']]
    for i in range(len(tw)): tw[i] *= packs[0]['weight'].reshape((-1,)+(1,)*(tw[i].ndim-1))
    print('Motion risk preprocessing complete; added channels:',artifact['channel_names'],flush=True)
    return arrays[0],arrays[1],arrays[2],tw,artifact,packs


def apply_calibration(spec, df, pred):
    out=pred.copy(); mode=spec['mode']
    if mode in ['global_xyz','xyz_tail']:
        out=out*np.asarray(spec['slopes'])+np.asarray(spec['intercepts'])
    elif mode in ['global_y','y_tail']:
        out[:,1]=out[:,1]*spec['slope']+spec['intercept']
    elif mode!='none':
        raise ValueError('Unknown frozen calibration: '+mode)
    if mode in ['xyz_tail','y_tail']:
        values=df[spec['col']].to_numpy()
        mask=values>=spec['threshold'] if spec['tail']=='high' else values<=spec['threshold']
        out[mask]+=np.asarray(spec['offsets'])
    return out
