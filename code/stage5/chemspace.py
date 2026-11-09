# chemspace.py — Analyzes BBBP test molecule-level training-target sensitivity across physicochemical descriptors, Morgan fingerprint neighborhoods, Murcko scaffolds and frozen SMARTS patterns while preserving Stage III evaluation identities.




from __future__ import annotations
import argparse, hashlib, json, os, sys, traceback, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RDLogger.DisableLog('rdApp.*')

ROOT=Path(__file__).resolve().parents[2]
STAGE4_PROJECT=ROOT/'code/stage4'
STAGE3_PROJECT=ROOT/'code/stage3'
EVAL_SIZES={'BBBP_test':291,'scaffold_test':137,'reference_out':2607}
TARGETS=('H','V','I')
METRICS=('score_range_HVI','rank_percentile_range_HVI','max_abs_prediction_deviation_from_H')
DESCRIPTORS=('MolWt','MolLogP','TPSA','HBD','HBA','RotB','RingCount','AromaticRingCount','FractionCSP3','HeavyAtomCount','FormalCharge')
EXPECTED_ENC={'xgb','gine'}
THRESHOLDS=(.3,.4,.5,.6)
NEIGHBORS=(3,5,10)
VERSION='BBB_Stage5_chemspace_final_v1'


def die(msg): raise RuntimeError('[STAGE5 ERROR] '+str(msg))
def jsonsave(p, obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def hashfile(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
    return h.hexdigest()
def resolve(path,default_project):
    if path:return Path(path).resolve()
    ptr=default_project/'CURRENT_RUN.txt'
    if not ptr.is_file():die('Missing CURRENT_RUN.txt: '+str(ptr))
    return Path(ptr.read_text(encoding='utf-8').strip()).resolve()
def bh(p):
    x=np.asarray(p,dtype=float); ans=np.full(len(x),np.nan); ix=np.flatnonzero(np.isfinite(x))
    if len(ix):
        order=np.argsort(x[ix]); n=len(ix); v=x[ix][order]
        q=np.minimum.accumulate((v*n/np.arange(1,n+1))[::-1])[::-1]
        ans[ix[order]]=np.clip(q,0,1)
    return ans

def load_and_validate(stage4,stage3,stage3_project,rules):
    if not (stage4/'STATS_READY.txt').is_file():die('Stage4 STATS_READY.txt missing')
    if not (stage4/'RESULT_STATUS.txt').is_file() or stage4.joinpath('RESULT_STATUS.txt').read_text().strip()!='COMPLETE_STATS_AND_PAMPA':
        die('Stage4 report is not COMPLETE_STATS_AND_PAMPA')
    sp=stage4/'analysis/stats_manifest.json'; ip=stage4/'stage4_info.json'
    if not sp.is_file() or not ip.is_file():die('Stage4 manifest/info missing')
    manifest=json.loads(sp.read_text()); info=json.loads(ip.read_text())
    if manifest.get('status')!='STATS_READY':die('Stage4 stats manifest is not ready')
    if Path(str(info['stage3_run'])).resolve()!=stage3 or Path(str(manifest['stage3_run'])).resolve()!=stage3:
        die('Stage3 run differs between Stage4 source manifest and requested path')
    cfgp=stage3/'run_config.json'
    if not cfgp.is_file():die('Stage3 run_config.json missing')
    h3=hashfile(cfgp)
    if info['stage3_run_config_sha256']!=h3 or manifest['stage3_run_config_sha256']!=h3:
        die('Stage3 run configuration hash mismatch against Stage4 report')
    cfg=json.loads(cfgp.read_text())
    if cfg.get('protocol')!='BBB_Stage3_controlled_v1':die('Wrong Stage3 protocol')
    if not (stage3/'PASS_PRECHECK.txt').is_file():die('Stage3 PASS_PRECHECK missing')
    if not (stage3_project/'stage3.py').is_file() or hashfile(stage3_project/'stage3.py') !=cfg['stage3_source_sha256']:
        die('Stage3 source SHA256 mismatch')
    conp=Path(cfg['source']['consensus']).resolve()
    if not conp.is_file() or hashfile(conp)!=cfg['source']['hashes']['consensus']:
        die('Frozen consensus missing or SHA256 mismatch')
    if not rules.is_file():die('Frozen SMARTS rules CSV missing')
    source_rules_hash=hashfile(rules)
    df=pd.read_csv(stage4/'analysis/molecule_target_sensitivity.csv',dtype={'parent_ik':str})
    required={'encoder','evaluation','parent_ik','prediction_H','prediction_V','prediction_I','rank_H','rank_V','rank_I',*METRICS,'rank_range_HVI','top10_membership_switch_HVI','top20_membership_switch_HVI'}
    if not required.issubset(df.columns):die('Stage4 molecule sensitivity CSV missing '+str(sorted(required-set(df.columns))))
    if df.duplicated(['encoder','evaluation','parent_ik']).any():die('Duplicate encoder/evaluation/parent rows')
    if set(df.encoder)!=EXPECTED_ENC or set(df.evaluation)!=set(EVAL_SIZES):die('Unexpected encoders or evaluations')
    for enc in sorted(EXPECTED_ENC):
        for ev,n in EVAL_SIZES.items():
            sub=df[(df.encoder==enc)&(df.evaluation==ev)]
            if len(sub)!=n:die(f'Stage4 molecule rows differ from frozen sample count: {enc} {ev} {len(sub)} not {n}')
            for k in ('prediction_H','prediction_V','prediction_I',*METRICS):
                if not np.isfinite(pd.to_numeric(sub[k],errors='coerce')).all():die(f'Nonfinite {k} in {enc} {ev}')
    sums=pd.read_csv(stage4/'analysis/main_three_target_molecule_summary.csv')
    if len(sums)!=6:die('Stage4 H/V/I summary must have six rows')
    for (_,r) in sums.iterrows():
        s=df[(df.encoder==r.encoder)&(df.evaluation==r.evaluation)]
        if len(s)!=int(r.n):die('Stage4 summary group sample size mismatch')
        for detail,overall in [('score_range_HVI','score_range_mean_HVI'),('rank_range_HVI','rank_range_mean_HVI')]:
            if not np.isclose(float(s[detail].mean()),float(r[overall]),atol=5e-9,rtol=0):
                die(f'Stage4 molecule CSV inconsistent with aggregate {r.encoder}/{r.evaluation}/{detail}')
        for detail,overall in [('top10_membership_switch_HVI','top10_switch_unique_molecules_HVI'),('top20_membership_switch_HVI','top20_switch_unique_molecules_HVI')]:
            if int(s[detail].astype(bool).sum())!=int(r[overall]):die('Stage4 Top-k checksum mismatch')
    b=df[df.evaluation.eq('BBBP_test')].copy()
    encs={e:set(b[b.encoder.eq(e)].parent_ik) for e in EXPECTED_ENC}
    if encs['xgb']!=encs['gine']:die('XGBoost/GINE BBBP molecule identities differ')
    con=pd.read_csv(conp,dtype=str,keep_default_na=False,low_memory=False)
    if not {'parent_ik','canon_smiles','split','in_reference_out'}.issubset(con):die('Frozen consensus is missing identity fields')
    if con.parent_ik.duplicated().any():die('Duplicate parent_ik in frozen consensus')
    
    
    
    
    test=con.loc[con.split.eq('test')].copy()
    excluded_by_old_filter=int(pd.to_numeric(test.in_reference_out,errors='coerce').fillna(0).ne(0).sum())
    if len(test)!=291 or set(test.parent_ik)!=encs['xgb']:
        missing=set(test.parent_ik)-encs['xgb']
        extra=encs['xgb']-set(test.parent_ik)
        die('Stage3 BBBP test identity still differs from Stage4 sensitivity: '
            f'frozen_split_test_n={len(test)}, stage4_unique_n={len(encs["xgb"])}, '
            f'only_frozen_n={len(missing)}, only_stage4_n={len(extra)}, '
            f'old_filter_excluded_n={excluded_by_old_filter}. '
            'No identity correction or molecule deletion was applied.')
    mapping=test.set_index('parent_ik')['canon_smiles']
    b['canon_smiles']=b.parent_ik.map(mapping)
    if b.canon_smiles.isna().any() or b.canon_smiles.eq('').any():die('Missing SMILES in frozen BBBP 291-parent set')
    
    
    meta={'protocol':VERSION,'stage4_run':str(stage4),'stage3_run':str(stage3),
          'stage4_molecule_sha256':hashfile(stage4/'analysis/molecule_target_sensitivity.csv'),
          'stage4_summary_sha256':hashfile(stage4/'analysis/main_three_target_molecule_summary.csv'),
          'stage3_run_config_sha256':h3,'frozen_consensus':str(conp),'frozen_consensus_sha256':hashfile(conp),
          'rules_file':str(rules),'rules_sha256':source_rules_hash,'bbbp_test_n':291,'encoder_n':{'xgb':291,'gine':291},
          'n_duplicate_canonical_smiles_in_bbbp':int(test.canon_smiles.duplicated().sum()),
          'bbbp_test_identity_selection':'split == test (exact Stage3 BBBP evaluation rule; no reference-out restriction)',
          'n_bbbp_test_rows_old_stage5_filter_would_exclude':excluded_by_old_filter,
          'bbbp_test_parent_identity_match_stage4':True,
          'targets':['H','V','I'],'scope':'Chemical-space endpoint BBBP_test; other endpoint summaries are unchanged from Stage4.'}
    return b,df,meta

def features(smiles):
    m=Chem.MolFromSmiles(str(smiles))
    if m is None:die('Invalid frozen canonical SMILES in BBBP test')
    return m,{
      'MolWt':Descriptors.MolWt(m),'MolLogP':Crippen.MolLogP(m),'TPSA':rdMolDescriptors.CalcTPSA(m),
      'HBD':Lipinski.NumHDonors(m),'HBA':Lipinski.NumHAcceptors(m),'RotB':Lipinski.NumRotatableBonds(m),
      'RingCount':rdMolDescriptors.CalcNumRings(m),'AromaticRingCount':rdMolDescriptors.CalcNumAromaticRings(m),
      'FractionCSP3':rdMolDescriptors.CalcFractionCSP3(m),'HeavyAtomCount':m.GetNumHeavyAtoms(),
      'FormalCharge':Chem.GetFormalCharge(m),
      'scaffold_smiles':MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False)
    }

def prepare_chemistry(bb):
    unique=bb.drop_duplicates('parent_ik').sort_values('parent_ik').copy().reset_index(drop=True)
    
    mols=[];dat=[]
    gen=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fps=[];X=np.zeros((len(unique),2048),dtype=np.uint8)
    for i,s in enumerate(unique.canon_smiles):
        m,attrs=features(s)
        mols.append(m);dat.append(attrs)
        fp=gen.GetFingerprint(m);fps.append(fp)
        DataStructs.ConvertToNumpyArray(fp,X[i])
    D=pd.concat([unique[['parent_ik','canon_smiles']],pd.DataFrame(dat)],axis=1)
    n=len(D);S=np.zeros((n,n),dtype=np.float64)
    for i,fp in enumerate(fps):
        if i+1<n:S[i,i+1:]=DataStructs.BulkTanimotoSimilarity(fp,fps[i+1:])
    S=S+S.T
    safe=S.copy();np.fill_diagonal(safe,-1)
    D['nearest_neighbor_tanimoto']=safe.max(axis=1)
    D['nearest_neighbor_parent_ik']=D.parent_ik.to_numpy()[safe.argmax(axis=1)]
    pca=PCA(n_components=2,random_state=20261004).fit_transform(X.astype(np.float64))
    D['morgan_pc1']=pca[:,0];D['morgan_pc2']=pca[:,1]
    return D,mols,S

def desc_permutation_boot(x,y,perms,boots):
    xx=np.asarray(x,float);yy=np.asarray(y,float)
    good=np.isfinite(xx)&np.isfinite(yy)
    if not np.all(good):die('Unexpected missing descriptor or target metric')
    n=len(xx)
    if np.ptp(xx)<1e-14 or np.ptp(yy)<1e-14:return np.nan,np.nan,np.nan,np.nan
    a=stats.rankdata(xx);b=stats.rankdata(yy)
    a=a-a.mean();b=b-b.mean();denom=np.sqrt(np.dot(a,a)*np.dot(b,b))
    obs=float(np.dot(a,b)/denom)
    pp=np.sum(np.abs((b[perms]@a)/denom)>=abs(obs)-1e-14)
    vals=[]
    for ix in boots:
        if len(np.unique(xx[ix]))<2 or len(np.unique(yy[ix]))<2:continue
        vals.append(stats.spearmanr(xx[ix],yy[ix]).statistic)
    lo,hi=np.percentile(vals,[2.5,97.5]) if vals else (np.nan,np.nan)
    return obs,float((pp+1)/(len(perms)+1)),float(lo),float(hi)

def adjacency_matrices(S):
    n=len(S);out={}
    for thr in THRESHOLDS:
        W=np.where(S>=thr,S,0.0);np.fill_diagonal(W,0)
        out[f'Moran_Tanimoto_ge_{thr:.1f}']=W
    safe=S.copy();np.fill_diagonal(safe,-np.inf)
    for k in NEIGHBORS:
        W=np.zeros_like(S)
        nearest=np.argsort(-safe,axis=1,kind='stable')[:,:k]
        rr=np.arange(n)[:,None]; W[rr,nearest]=S[rr,nearest]
        W=np.maximum(W,W.T);np.fill_diagonal(W,0)
        out[f'Moran_kNN_{k}']=W
    return out

def moran(y,W,perms,device):
    yy=np.asarray(y,float);z=yy-yy.mean(); s0=W.sum(); denom=float(z@z)
    if s0<=0 or denom==0:return np.nan,np.nan
    n=len(z);obs=float(n/s0*(z@W@z)/denom)
    if device=='cuda':
        import torch
        with torch.no_grad():
            tw=torch.as_tensor(W,dtype=torch.float64,device='cuda')
            tz=torch.as_tensor(z[perms].copy(),dtype=torch.float64,device='cuda')
            calc=((tz@tw)*tz).sum(dim=1)*(n/s0/denom)
            extremes=int(torch.count_nonzero(torch.abs(calc)>=abs(obs)-1e-13).item())
    else:
        zp=z[perms]
        if np.count_nonzero(W)<.55*len(W)**2:
            ws=csr_matrix(W)
            products=(ws@zp.T).T
        else:
            products=zp@W
        vals=(products*zp).sum(axis=1)*(n/s0/denom)
        extremes=int(np.count_nonzero(np.abs(vals)>=abs(obs)-1e-13))
    return obs,(extremes+1)/(len(perms)+1)

def scaffold_eta(y,g,perms):
    y=np.asarray(y,float);idx,groups=np.unique(np.asarray(g,str),return_inverse=True)
    sizes=np.bincount(groups);grand=float(np.mean(y));denom=float(np.sum((y-grand)**2))
    if denom<=0 or len(idx)<2:return np.nan,np.nan
    def f(vals):
        means=np.bincount(groups,weights=vals,minlength=len(idx))/sizes
        return float(np.sum(sizes*(means-grand)**2)/denom)
    obs=f(y);count=0
    for ix in perms:count+=f(y[ix])>=obs-1e-14
    return obs,(count+1)/(len(perms)+1)

def mean_diff_perm(y,mask,perms):
    n=len(y);nh=int(mask.sum())
    if nh<3 or n-nh<3:return np.nan,np.nan
    yy=np.asarray(y,float)
    obs=float(yy[mask].mean()-yy[~mask].mean())
    
    shuffled=yy[perms]
    vs=shuffled[:,mask].mean(axis=1)-shuffled[:,~mask].mean(axis=1)
    return obs,float((np.count_nonzero(abs(vs)>=abs(obs)-1e-14)+1)/(len(perms)+1))

def spearman_perm(x,y,perms):
    xx=stats.rankdata(x); yy=stats.rankdata(y)
    xx=xx-xx.mean();yy=yy-yy.mean();denom=np.sqrt((xx@xx)*(yy@yy))
    if denom<=0:return np.nan,np.nan
    obs=float(xx@yy/denom)
    vals=(yy[perms]@xx)/denom
    return obs,float((np.count_nonzero(np.abs(vals)>=abs(obs)-1e-14)+1)/(len(perms)+1))

def draw_maps(out,D):
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    for enc,g in D.groupby('encoder'):
        a=g.sort_values('parent_ik')
        plt.figure(figsize=(7.8,5.8))
        sc=plt.scatter(a.morgan_pc1,a.morgan_pc2,c=a.score_range_HVI,s=35,cmap='viridis')
        plt.xlabel('Morgan fingerprint PC1');plt.ylabel('Morgan fingerprint PC2')
        plt.title(f'{enc.upper()}: H/V/I prediction sensitivity on BBBP test (n=291)')
        plt.colorbar(sc,label='H/V/I score range');plt.tight_layout()
        plt.savefig(figdir/f'{enc}_BBBP_morgan_pca_sensitivity.png',dpi=200);plt.close()
        top=a.sort_values(['score_range_HVI','parent_ik'],ascending=[False,True]).head(20).iloc[::-1]
        plt.figure(figsize=(8.7,6.3));plt.barh(top.parent_ik,top.score_range_HVI)
        plt.xlabel('H/V/I score range');plt.ylabel('Parent InChIKey (first block)')
        plt.title(f'{enc.upper()}: top-20 target-sensitive BBBP molecules');plt.tight_layout()
        plt.savefig(figdir/f'{enc}_BBBP_top20_sensitivity.png',dpi=200);plt.close()
    a=D[D.encoder.eq('xgb')].set_index('parent_ik').sort_index()
    b=D[D.encoder.eq('gine')].set_index('parent_ik').sort_index()
    plt.figure(figsize=(6.5,5.7));plt.scatter(a.score_range_HVI,b.score_range_HVI,s=36,alpha=.8)
    plt.xlabel('XGBoost H/V/I score range');plt.ylabel('GINE H/V/I score range')
    plt.title('Cross-encoder molecular sensitivity concordance (BBBP test)');plt.tight_layout()
    plt.savefig(figdir/'encoder_concordance_sensitivity.png',dpi=200);plt.close()


def analyze(bb,allm,meta,rules_file,out,B,boot,seed,device):
    rng=np.random.default_rng(seed)
    
    chem,mols,S=prepare_chemistry(bb)
    D=bb.merge(chem,on=['parent_ik','canon_smiles'],how='left',validate='many_to_one').sort_values(['encoder','parent_ik']).reset_index(drop=True)
    if D[list(DESCRIPTORS)].isna().any().any() or D.shape[0]!=582:die('Chemistry join failed')
    D.to_csv(out/'bbbp_molecule_chemical_annotation.csv',index=False)
    rules=pd.read_csv(rules_file,dtype={'bit':str,'smarts':str})
    if not {'bit','smarts'}.issubset(rules):die('Frozen SMARTS rule columns missing')
    if rules.smarts.isna().any():die('Missing frozen SMARTS rule')
    hitgroups={}
    for _,r in rules.iterrows():
        p=Chem.MolFromSmarts(str(r.smarts))
        if p is None:die('Invalid frozen SMARTS: '+str(r.smarts))
        hit=np.array([m.HasSubstructMatch(p) for m in mols],dtype=bool)
        hitgroups.setdefault(hit.tobytes(),{'hit':hit,'items':[]})['items'].append((r.bit,r.smarts,str(r.get('direction',''))))
    frozen_groups=list(hitgroups.values());perms=rng.random((B,291)).argsort(axis=1)
    
    boots=rng.integers(0,291,size=(boot,291))
    W=adjacency_matrices(S)
    drows=[];nrows=[];srows=[];smrows=[];ctrows=[];grrows=[];crows=[]
    for enc in sorted(EXPECTED_ENC):
        print(f"[STAGE5] Encoder {enc}: chemistry analysis started",flush=True)
        g=D[D.encoder.eq(enc)].sort_values('parent_ik').reset_index(drop=True)
        if not np.array_equal(g.parent_ik.to_numpy(),chem.parent_ik.to_numpy()):die('SMILES/distance alignment mismatch')
        
        for name,group in g.groupby('scaffold_smiles',dropna=False):
            grrows.append({'encoder':enc,'scaffold_smiles':name,'n_molecules':len(group),
                  'mean_score_range_HVI':float(group.score_range_HVI.mean()),
                  'mean_rank_percentile_range_HVI':float(group.rank_percentile_range_HVI.mean()),
                  'top10_membership_switch_n':int(group.top10_membership_switch_HVI.astype(bool).sum()),
                  'mean_nearest_neighbor_tanimoto':float(group.nearest_neighbor_tanimoto.mean()),
                  'note': 'Acyclic molecules have empty Murcko scaffold' if not name else ''})
        for met in METRICS:
            print(f"[STAGE5] {enc} / {met}: 11 descriptors + 7 Moran graphs + scaffold/SMARTS",flush=True)
            y=g[met].to_numpy(float)
            for desc in DESCRIPTORS:
                rho,p,lo,hi=desc_permutation_boot(g[desc].to_numpy(float),y,perms,boots)
                drows.append(dict(encoder=enc,metric=met,descriptor=desc,n=len(g),spearman_rho=rho,
                                  ci_low=lo,ci_high=hi,perm_p=p))
            for wname,weights in W.items():
                obs,p=moran(y,weights,perms,device)
                nrows.append(dict(encoder=enc,metric=met,analysis=wname,n=len(g),
                        n_edges=int(np.count_nonzero(np.triu(weights,1))),statistic=obs,perm_p=p,device=device))
            sc=g[g.scaffold_smiles.ne('')].copy()
            cnt=sc.scaffold_smiles.value_counts();sc=sc[sc.scaffold_smiles.isin(cnt[cnt>=2].index)]
            if len(sc):
                locperms=rng.random((B,len(sc))).argsort(axis=1)
                eta,p=scaffold_eta(sc[met].to_numpy(float),sc.scaffold_smiles.to_numpy(),locperms)
                srows.append(dict(encoder=enc,metric=met,n_molecules=len(sc),n_scaffolds=sc.scaffold_smiles.nunique(),eta_squared=eta,perm_p=p))
            for gi,item in enumerate(frozen_groups,1):
                hit=item['hit']; items=item['items']; nh=int(hit.sum())
                if nh<3 or 291-nh<3:continue
                obs,p=mean_diff_perm(y,hit,perms)
                smrows.append(dict(encoder=enc,equivalence_group=f'G{gi:02d}',bits='|'.join(x[0] for x in items),
                      smarts=' || '.join(x[1] for x in items),n_rules=len(items),metric=met,n_hit=nh,n_nonhit=291-nh,
                      mean_hit=float(y[hit].mean()),mean_nonhit=float(y[~hit].mean()),mean_difference=obs,perm_p_two_sided=p,
                      inference='Exploratory frozen-rule overlay; not independently selected for target sensitivity'))
        
        for sw in ('top10_membership_switch_HVI','top20_membership_switch_HVI'):
            yes=g[sw].astype(bool)
            for desc in ('nearest_neighbor_tanimoto',*DESCRIPTORS):
                x=g.loc[yes,desc].to_numpy(float);z=g.loc[~yes,desc].to_numpy(float)
                if not len(x) or not len(z):continue
                p=stats.mannwhitneyu(x,z,alternative='two-sided').pvalue
                ctrows.append(dict(encoder=enc,switch=sw,descriptor=desc,n_switch=len(x),n_non_switch=len(z),
                        mean_switch=float(x.mean()),mean_non_switch=float(z.mean()),median_switch=float(np.median(x)),
                        median_non_switch=float(np.median(z)),mannwhitney_p=float(p)))
    a=D[D.encoder.eq('xgb')].set_index('parent_ik').sort_index()
    b=D[D.encoder.eq('gine')].set_index('parent_ik').sort_index()
    n=len(a);kk=math.ceil(.1*n)
    for met in METRICS+('rank_range_HVI',):
        x=a[met].to_numpy(float);y=b[met].to_numpy(float)
        rho,p=spearman_perm(x,y,perms)
        I=set(a.sort_values([met,'parent_ik'],ascending=[False,True]).head(kk).index)
        J=set(b.sort_values([met,'parent_ik'],ascending=[False,True]).head(kk).index)
        ov=len(I&J)
        crows.append(dict(encoder1='xgb',encoder2='gine',metric=met,n=n,spearman_rho=rho,perm_p=p,
                top10pct_sensitive_k=kk,top10pct_overlap=ov,top10pct_jaccard=ov/(2*kk-ov),
                hypergeom_p=float(stats.hypergeom.sf(ov-1,n,kk,kk))))
    tables={
       'descriptor_target_sensitivity_associations.csv':pd.DataFrame(drows),
       'neighborhood_target_sensitivity_autocorrelation.csv':pd.DataFrame(nrows),
       'scaffold_target_sensitivity_omnibus.csv':pd.DataFrame(srows),
       'scaffold_target_sensitivity_by_group.csv':pd.DataFrame(grrows),
       'frozen_smarts_target_sensitivity_overlay_unique_hitgroups.csv':pd.DataFrame(smrows),
       'topk_target_switch_chemical_context.csv':pd.DataFrame(ctrows),
       'encoder_concordance_target_sensitivity.csv':pd.DataFrame(crows),
    }
    for name,t in tables.items():
        if len(t):
            if name.startswith('descriptor_'):t['q_bh_within_encoder_metric']=t.groupby(['encoder','metric'])['perm_p'].transform(lambda x:bh(x.to_numpy()))
            if name.startswith('neighborhood_'):t['q_bh_within_encoder_metric']=t.groupby(['encoder','metric'])['perm_p'].transform(lambda x:bh(x.to_numpy()))
            if name=='scaffold_target_sensitivity_omnibus.csv':t['q_bh_within_encoder']=t.groupby('encoder')['perm_p'].transform(lambda x:bh(x.to_numpy()))
            if name.startswith('frozen_smarts_'):t['q_bh_unique_group_within_encoder_metric']=t.groupby(['encoder','metric'])['perm_p_two_sided'].transform(lambda x:bh(x.to_numpy()))
            if name.startswith('topk_'):t['q_bh_within_encoder_switch']=t.groupby(['encoder','switch'])['mannwhitney_p'].transform(lambda x:bh(x.to_numpy()))
            if name.startswith('encoder_'):
                t['q_bh_spearman']=bh(t.perm_p.to_numpy());t['q_bh_top_overlap']=bh(t.hypergeom_p.to_numpy())
        t.to_csv(out/name,index=False)
    draw_maps(out,D)
    
    stage4=Path(meta['stage4_run']);pd.read_csv(stage4/'analysis/main_three_target_molecule_summary.csv').to_csv(out/'stage4_three_target_endpoint_summary_LOCKED.csv',index=False)
    outmeta={**meta,'status':'PASS_STAGE5_ANALYSIS','n_perm':B,'n_bootstrap':boot,'seed':seed,
         'device_moran':device,'descriptor_columns':list(DESCRIPTORS),'metrics':list(METRICS),
         'moran_weighting':'symmetric weighted Morgan radius=2, 2048 bit chirality-aware Tanimoto; diagonal zero',
         'moran_permutation_test':'two-sided |I_perm| >= |I_obs|, B independent unconditional index permutations, unadjusted p; BH within encoder/metric',
         'scaffold_permutation':'Murcko nonempty, count >=2 only, eta-squared global label permutation; exploratory',
         'SMARTS_origin':'frozen historical stage-I rule set; exploratory overlays, deduplicated identical hit patterns',
         'input_semantics':'Stage4 H/V/I seed-means; only fixed BBBP test n=291 included in chemical-space inference',
         'notes':['No new molecular labels or predictors are fit.',
                  'BH adjustment limited to clearly named families; no global error-rate claim.',
                  'P-values from exploratory chemical-space analysis are associations, not causal evidence.',
                  'Moran permutation null presumes global exchangeability of sensitivity values; structural covariates may violate it.',
                  'Descriptor bootstrap treats molecules as sampling units and does not include retraining seed uncertainty.',
                  'Stage IV rank percentile is defined as rank range / n.']}
    jsonsave(out/'stage5_manifest.json',outmeta)
    pd.DataFrame([{'file':name,'sha256':hashfile(out/name),'n_rows':len(t)} for name,t in tables.items()]).to_csv(out/'table_inventory.csv',index=False)
    lines=['# BBB Stage V chemical-space analysis summary','',
           '- Status: PASS_STAGE5_ANALYSIS',
           '- Evaluation endpoint: BBBP test; 291 parent molecules per encoder for XGBoost and GINE',
           '- Training targets: Hard / Vote / Integrated (lambda=0.1), using Stage IV seed-averaged predictions',
           f'- Moran permutation device: {device}',
           f'- Permutations: {B}; descriptor bootstrap replicates: {boot}',
           '- Structural associations are exploratory associations with prediction sensitivity and are not interpreted as causal BBB mechanisms.','',
           '## Statistical tables']
    for filename,df in tables.items():
        qcols=[c for c in df if c.startswith('q_bh_')]
        qcount=int((pd.to_numeric(df[qcols[0]],errors='coerce')<.05).sum()) if qcols and len(df) else 0
        lines.append(f'- `{filename}`: {len(df)} rows; {qcount} rows have q < 0.05 in the first available BH-adjusted column.')
    lines.extend(['','## Interpretation scope','',
          '- Hard, Vote, and Integrated are alternative supervision representations constructed from the same training data.',
          '- Stage V statistics are computed from the locked Integrated definition with lambda=0.1.',
          '- SMARTS rules are frozen Stage I structural rules and are used as exploratory overlays.',
          '- Scaffold and reference-out molecule-level sensitivity summaries are retained from Stage IV.'])
    (out/'CHEMSPACE_SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'CHEMSPACE_READY.txt').write_text('PASS_STAGE5_ANALYSIS\n',encoding='utf-8')
    print('[PASS_STAGE5_ANALYSIS] BBBP n=291 per encoder; 7 Moran neighborhoods, 11 descriptors, frozen SMARTS groups',flush=True)
    print('[STAGE5] Moran device: ',device,flush=True)

def get_device(choice):
    if choice=='cpu':return 'cpu'
    try:
        import torch
        if torch.cuda.is_available():return 'cuda'
        if choice=='cuda':die('CUDA explicitly selected, but torch.cuda.is_available() is false')
    except ImportError:
        if choice=='cuda':die('CUDA selected, but torch is missing')
    return 'cpu'

def package(out):
    if not (out/'CHEMSPACE_READY.txt').exists():die('Analysis not finished; cannot pack')
    p=out/'BBB_STAGE5_CHEMSPACE_REPORT_TO_SEND.zip'
    tmp=out/'BBB_STAGE5_CHEMSPACE_REPORT_TO_SEND.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for f in sorted(out.iterdir()):
            if f.is_file() and f.name not in (p.name,tmp.name):
                if f.name.endswith(('.csv','.json','.md','.txt','.log')):z.write(f,f.name)
        for f in sorted((out/'figures').glob('*.png')):z.write(f,'figures/'+f.name)
    os.replace(tmp,p)
    print('[STAGE5] Report ready:',p, 'bytes',p.stat().st_size,flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['precheck','analyze','pack']);ap.add_argument('--out',required=True)
    ap.add_argument('--stage4',default='');ap.add_argument('--stage3',default='')
    ap.add_argument('--stage3-project',default=str(STAGE3_PROJECT));ap.add_argument('--rules',default=str(Path(__file__).with_name('reference')/'frozen_v1_significant_rules.csv'))
    ap.add_argument('--n-perm',type=int,default=2000);ap.add_argument('--n-boot',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=20261004);ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    if a.command=='pack':package(out);return
    s4=resolve(a.stage4,STAGE4_PROJECT)
    stage4info=s4/'stage4_info.json'
    s3=resolve(a.stage3,STAGE3_PROJECT) if a.stage3 else Path(json.loads(stage4info.read_text())['stage3_run']).resolve()
    bb,allm,meta=load_and_validate(s4,s3,Path(a.stage3_project),Path(a.rules))
    if a.command=='precheck':
        jsonsave(out/'precheck.json',{'status':'PASS_STAGE5_PRECHECK',**meta})
        (out/'PASS_STAGE5_PRECHECK.txt').write_text('PASS_STAGE5_PRECHECK\n',encoding='utf-8')
        print('[PASS_STAGE5_PRECHECK] 291 BBBP molecules x 2 encoders, Stage4/Stage3/frozen hashes match',flush=True)
        return
    if not (out/'PASS_STAGE5_PRECHECK.txt').exists():die('Run precheck first')
    previous=json.loads((out/'precheck.json').read_text())
    for f in ('stage4_molecule_sha256','frozen_consensus_sha256','stage3_run_config_sha256','rules_sha256'):
        if previous.get(f)!=meta.get(f):die('Inputs changed since precheck: '+f)
    if a.n_perm<99 or a.n_boot<100:die('n_perm>=99 and n_boot>=100 required')
    device=get_device(a.device)
    analyze(bb,allm,meta,Path(a.rules),out,a.n_perm,a.n_boot,a.seed,device)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[STAGE5 FAILED]',str(e),file=sys.stderr,flush=True)
        traceback.print_exc();sys.exit(2)
