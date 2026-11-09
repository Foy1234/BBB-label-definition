# verify_release.py — Validates archived frozen input hashes, molecule identities and dataset dimensions, reproduces XGBoost/GINE aggregate discrimination metrics from seed-level predictions, and independently checks PAMPA calibration and conditional-permutation summary statistics without fitting any model.
from pathlib import Path
import sys, json, hashlib, math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT=Path(__file__).resolve().parents[1]

def require(ok, message):
    if not ok:raise RuntimeError(message)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def evaluate_pampa(y, prediction):
    y=np.asarray(y,dtype=int)
    raw=np.asarray(prediction,dtype=float)
    clipped=np.clip(raw,0.0,1.0)
    ece=0.0
    for j in range(10):
        mask=((clipped>=j/10)&(clipped<(j+1)/10)) if j<9 else ((clipped>=.9)&(clipped<=1.0))
        if mask.any():ece+=mask.mean()*abs(y[mask].mean()-clipped[mask].mean())
    return {'auroc':roc_auc_score(y,raw),'auprc':average_precision_score(y,raw),'brier':brier_score_loss(y,clipped),'ece':ece}

def main():
    frozen={'consensus_dataset.csv':('51d09eea4e6a1659ea7820a93f5f63880161645c3bab8f7f9f93007bdbc9849a',4131),
            'labels_abc.csv':('ea6b31d500e1f04384952a096fabdf4e9ce3e0bd9533587442c542946f432c89',1234)}
    for fn,(digest,n) in frozen.items():
        path=ROOT/'data/frozen'/fn
        require(sha(path)==digest,f'Frozen source hash mismatch: {fn}')
        require(len(pd.read_csv(path))==n,f'Frozen source size mismatch: {fn}')
    nc=ROOT/'data/frozen/ncats_unique_molecule_audit.csv'
    require(sha(nc)=='e7a84978ff89433579f60a038041ffc2bf66817c466a94ec126a1feba82a915e','NCATS audit digest changed')
    ncat=pd.read_csv(nc)
    require(len(ncat)==437 and int(ncat.eligible_external.astype(bool).sum())==318,'NCATS source count mismatch')
    vec=pd.read_csv(ROOT/'data/derived/stage2/stage2_target_vectors_common1157.csv')
    perm=pd.read_csv(ROOT/'data/derived/stage2/P_99_permutation_targets_common1157.csv')
    require(len(vec)==1157 and len(perm)==1157,'Stage II target count mismatch')
    require(len([x for x in perm if x.startswith('P_')])==99,'Conditional permutation count mismatch')
    ref=pd.read_csv(ROOT/'results/stage3/analysis/metrics_from_seed_mean.csv')
    n_files=0
    for enc,seeds in [('xgb',list(range(42,62))),('gine',list(range(42,133,10)))]:
        for target in ('H','V','I','S','I_lambda_0.01','I_lambda_1.0'):
            frames=[]
            for seed in seeds:
                name=ROOT/f'results/stage3/predictions/{enc}/main/{target}__seed{seed}.csv'
                require(name.exists(),f'Missing {name}')
                frame=pd.read_csv(name)
                require(len(frame)==3035 and len(frame[['evaluation','parent_ik']].drop_duplicates())==3035,'Prediction molecule count or duplicated identity')
                require(frame.prediction.notna().all() and np.isfinite(frame.prediction).all(),'Invalid prediction values')
                frames.append(frame[['evaluation','parent_ik','gold','prediction']].rename(columns={'prediction':f'pred{seed}'}))
                n_files+=1
            merged=frames[0]
            for i,frame in enumerate(frames[1:],1):merged=merged.merge(frame,on=['evaluation','parent_ik','gold'],validate='one_to_one')
            require(len(merged)==3035,'Prediction file identity disagreement')
            pp=merged[[x for x in merged if x.startswith('pred')]].mean(axis=1)
            for ev,n in [('BBBP_test',291),('scaffold_test',137),('reference_out',2607)]:
                sub=merged[merged.evaluation==ev]
                y=sub.gold.astype(int).to_numpy();p=pp.loc[sub.index].to_numpy()
                require(len(sub)==n,'Evaluation size mismatch')
                row=ref[(ref.encoder==enc)&(ref.target==target)&(ref.evaluation==ev)]
                require(len(row)==1,'Missing source metric record')
                require(abs(roc_auc_score(y,p)-row.auroc.iloc[0])<1e-10,'AUROC mismatch')
                require(abs(average_precision_score(y,p)-row.auprc.iloc[0])<1e-10,'AUPRC mismatch')
    require(n_files==180,'Expected 180 main prediction files')
    pam=pd.read_csv(ROOT/'results/stage4/pampa/pampa_predictions_by_seed.csv')
    summary=pd.read_csv(ROOT/'results/stage4/pampa/pampa_mean_seed_metrics.csv')
    require(len(pam)==19020 and pam.groupby(['target','seed']).size().eq(317).all(),'Incomplete PAMPA seed predictions')
    for t in ('H','V','I'):
        sub=pam[pam.target==t].groupby(['parent_ik','gold'],sort=True).prediction.mean().reset_index()
        obs=evaluate_pampa(sub.gold,sub.prediction)
        src=summary[summary.target==t].iloc[0]
        for k in ('auroc','auprc','brier','ece'):require(abs(obs[k]-src[k])<1e-9,f'PAMPA {t} {k} mismatch')
    ps=pd.read_csv(ROOT/'results/stage3/analysis/conditional_permutation_summary.csv')
    require(ps.n_permutations.eq(99).all(),'Permutation count mismatch')
    aa=ps[(ps.evaluation=='BBBP_test')&(ps.metric=='auroc')].set_index('encoder')
    require(np.isclose(aa.loc['xgb','p_one_sided_I_higher'],.08) and np.isclose(aa.loc['gine','p_one_sided_I_higher'],.76),'Permutation anchor mismatch')
    rk=pd.read_csv(ROOT/'results/stage4/analysis/main_three_target_molecule_summary.csv')
    top=rk[rk.evaluation=='BBBP_test'].set_index('encoder').top10_switch_unique_molecules_HVI
    require(int(top['xgb'])==7 and int(top['gine'])==14,'BBBP Top-10 membership change count mismatch')
    return {'status':'PASS','main_prediction_files':n_files,'prediction_rows_per_file':3035,'PAMPA_seed_prediction_rows':len(pam),'conditional_permutations':99,'main_metrics_rows':len(ref),'ncats_eligible_rows':318}

if __name__=='__main__':
    try:print(json.dumps(main(),indent=2))
    except Exception as exc:print('REPRODUCTION CHECK FAILED:',exc,file=sys.stderr);raise
