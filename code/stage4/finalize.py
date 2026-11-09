# finalize.py — Recomputes paired statistical comparisons, molecular rank sensitivity, top-k membership and independent PAMPA-BBB evaluation from Stage III predictions, including strict identity auditing and exact-duplicate control.




from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time, traceback, importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[2]
PROJECT=BASE/'code/stage3'
MAIN=('H','V','I')
ALL=('H','V','I','S','I_lambda_0.01','I_lambda_1.0')
EVALS=('BBBP_test','scaffold_test','reference_out')
SEEDS={'xgb':list(range(42,62)),'gine':[42,52,62,72,82,92,102,112,122,132]}
PAIRS=(('H','V'),('H','I'),('V','I'))


def fail(msg): raise RuntimeError(msg)
def jsonwrite(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def stage3_run(path,stage3_project):
    if path:return Path(path).resolve()
    current=Path(stage3_project)/'CURRENT_RUN.txt'
    if not current.is_file():fail('Missing Stage3 CURRENT_RUN.txt; set STAGE3_RUN absolute path.')
    return Path(current.read_text().strip()).resolve()

def require_stage3(run,stage3_project,need_all=True):
    run=Path(run);p=Path(stage3_project)
    if not run.is_dir() or not (run/'PASS_PRECHECK.txt').is_file():fail('Missing verified Stage3 directory/PASS_PRECHECK.txt')
    cfgp=run/'run_config.json'
    if not cfgp.exists():fail('Missing Stage3 run_config.json')
    cfg=json.loads(cfgp.read_text())
    if cfg.get('protocol')!='BBB_Stage3_controlled_v1':fail('Unexpected Stage3 experiment protocol')
    if not (p/'stage3.py').is_file():fail('Missing original Stage3 stage3.py (used to preserve PAMPA model protocol)')
    if sha(p/'stage3.py')!=cfg.get('stage3_source_sha256'):fail('Stage3 source SHA mismatch; refusing to mix versions')
    if sha(p/'legacy_gine_cpu.py')!=cfg.get('gine_legacy_sha256'):fail('Stage3 GINE source SHA mismatch')
    for k,h in cfg['source']['hashes'].items():
        fname=(cfg['source']['consensus'] if k=='consensus' else cfg['source']['labels'] if k=='labels'
               else str(Path(cfg['source']['stage2'])/k))
        if not Path(fname).is_file() or sha(fname)!=h:fail('Stage3 frozen source missing or changed: '+fname)
    if need_all:
        cover=run/'analysis'/'coverage.json'
        if not cover.exists():fail('Run Stage3 summarize first: missing coverage.json')
        c=json.loads(cover.read_text())
        if c.get('status')!='COMPLETE' or any(c.get('missing_counts',{}).values()):fail('Stage3 coverage not COMPLETE')
    return cfg

def import_stage3(project):
    sys.path.insert(0,str(Path(project).resolve()))
    import stage3
    return stage3


def verify_published_stage3_metrics(run, enc, ev, y, means):
    
    fn=run/'analysis'/'metrics_from_seed_mean.csv'
    if not fn.is_file():fail('Stage3 source analysis missing metrics_from_seed_mean.csv')
    reference=pd.read_csv(fn)
    for target,pred in means.items():
        rows=reference[(reference.encoder==enc)&(reference.evaluation==ev)&(reference.target==target)]
        if len(rows)!=1:fail(f'Missing/duplicated Stage3 metric anchor for {enc} {ev} {target}')
        calculated=evaluate_metrics(y,pred)
        for met in ('auroc','auprc'):
            if not np.isclose(float(rows.iloc[0][met]), calculated[met], rtol=0, atol=2e-9):
                fail(f'Stage3 existing {met} differs from recovered per-seed predictions for {enc} {ev} {target}')


def load_predictions(run,enc,ev,targets=ALL):
    
    frames={};ys=None;ids=None
    for target in targets:
        vectors=[]
        for seed in SEEDS[enc]:
            fn=run/'predictions'/enc/'main'/f'{target}__seed{seed}.csv'
            if not fn.is_file():fail(f'Missing Stage3 per-seed prediction file: {fn}')
            df=pd.read_csv(fn,dtype={'parent_ik':str,'target':str,'evaluation':str})
            s=df[df.evaluation.eq(ev)].copy()
            if s.empty or s.parent_ik.duplicated().any():fail(f'Invalid evaluation subset in {fn}')
            iid=s.parent_ik.to_numpy(str)
            yy=s.gold.to_numpy(int);v=s.prediction.to_numpy(float)
            if not np.isfinite(v).all():fail(f'Non-finite prediction in {fn}')
            if ids is None:
                ids=iid;ys=yy
                if set(yy.tolist())!={0,1}:fail('Evaluation missing one class: '+ev)
            elif not np.array_equal(ids,iid) or not np.array_equal(ys,yy):
                fail(f'Evaluation parents/order/gold not aligned {enc} {ev} {target} seed {seed}')
            if df.seed.nunique()!=1 or int(df.seed.iloc[0])!=seed or df.target.nunique()!=1 or df.target.iloc[0]!=target:
                fail(f'Seed/target metadata mismatch {fn}')
            vectors.append(v)
        frames[target]=np.stack(vectors).mean(axis=0)
    target_n={'BBBP_test':291,'scaffold_test':137,'reference_out':2607}
    if len(ids)!=target_n[ev]:fail(f'Expected {target_n[ev]} {ev} molecules, found {len(ids)}')
    return ids,ys,frames


def gpu_score_batch(y,p):
    



    import torch
    ps,order=torch.sort(p,dim=1,descending=True,stable=True)
    ys=torch.gather(y,1,order)
    B,N=ys.shape
    cp=torch.cumsum(ys,dim=1)
    cn=torch.cumsum(1.0-ys,dim=1)
    end=torch.cat([ps[:,:-1].ne(ps[:,1:]),torch.ones((B,1),device=ps.device,dtype=torch.bool)],dim=1)
    posidx=torch.arange(N-1,device=ps.device,dtype=torch.int64).unsqueeze(0).expand(B,-1)
    prevmarkers=torch.cat([
        torch.full((B,1),-1,device=ps.device,dtype=torch.int64),
        torch.where(end[:,:-1],posidx,torch.full_like(posidx,-1))
    ],dim=1)
    previous=torch.cummax(prevmarkers,dim=1).values
    cp0=torch.where(previous>=0,cp.gather(1,previous.clamp_min(0)),torch.zeros_like(cp))
    cn0=torch.where(previous>=0,cn.gather(1,previous.clamp_min(0)),torch.zeros_like(cn))
    pg=(cp-cp0)*end
    ng=(cn-cn0)*end
    npos=cp[:,-1:];nneg=cn[:,-1:]
    auc=(pg*(nneg-cn+0.5*ng)).sum(dim=1)/(npos.squeeze(1)*nneg.squeeze(1)).clamp_min(1e-20)
    precision=cp/(cp+cn).clamp_min(1)
    ap=(pg*precision).sum(dim=1)/npos.squeeze(1).clamp_min(1)
    auc=torch.where((npos*nneg).squeeze(1)>0,auc,torch.nan)
    ap=torch.where(npos.squeeze(1)>0,ap,torch.nan)
    return torch.stack([auc,ap],dim=1)


def validate_torch_metric(device):
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    cases=[([0,1,1,0,1,0],[.2,.1,.9,.9,.1,.5]),([0,0,1,1,1],[.5,.5,.5,.5,.5]),
           ([1,0,1,0,1,1,0,0],[.1,.3,.3,.1,.8,.7,.7,.1])]
    for yy,xx in cases:
        y=torch.tensor([yy],dtype=torch.float64,device=device)
        p=torch.tensor([xx],dtype=torch.float64,device=device)
        got=gpu_score_batch(y,p).cpu().numpy()[0]
        exp=np.array([roc_auc_score(yy,xx),average_precision_score(yy,xx)])
        if not np.allclose(got,exp,atol=1e-12,rtol=0):fail(f'CUDA/torch rank metric differs from sklearn: {got} vs {exp}')


def resolve_device(requested):
    import torch
    requested=requested.lower()
    if requested=='cuda' and not torch.cuda.is_available():fail('CUDA explicitly requested but torch.cuda.is_available() is false')
    device='cuda' if requested=='cuda' or (requested=='auto' and torch.cuda.is_available()) else 'cpu'
    validate_torch_metric(device)
    meta={'requested':requested,'selected':device,'torch_version':torch.__version__,
          'gpu_name':torch.cuda.get_device_name(0) if device=='cuda' else None,
          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES','unset')}
    return device,meta


def ece_batch(y,p):
    
    import torch
    B,N=y.shape
    boundaries=torch.tensor(np.linspace(0,1,11)[1:-1],dtype=p.dtype,device=p.device)
    bins=torch.bucketize(p,boundaries,right=True)
    diff=torch.zeros((B,10),dtype=p.dtype,device=p.device)
    diff.scatter_add_(1,bins,p-y)
    return diff.abs().sum(dim=1)/N


def evaluate_metrics(y,ps,with_calibration=False):
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    yy=np.asarray(y,int); p=np.asarray(ps,float)
    if with_calibration:p=np.clip(p,0,1)
    out={'auroc':float(roc_auc_score(yy,p)),'auprc':float(average_precision_score(yy,p))}
    if with_calibration:
        edges=np.linspace(0,1,11)[1:-1]
        binidx=np.digitize(p,edges)
        out['brier']=float(brier_score_loss(yy,p))
        out['ece']=float(sum(abs((p[binidx==i]-yy[binidx==i]).sum()) for i in range(10))/len(yy))
    return out


def bootstrap_stats(y,model_predictions,B,seed,device,batch_size=64,with_calibration=False):
    import torch
    names=list(model_predictions)
    yy=np.asarray(y,dtype=int); n=len(yy)
    raws=np.stack([model_predictions[t] for t in names],axis=0).astype(np.float64)
    if with_calibration:raws=np.clip(raws,0,1)
    ytorch=torch.tensor(yy,device=device,dtype=torch.float64)
    ptorch=torch.tensor(raws,device=device,dtype=torch.float64)
    rng=np.random.default_rng(seed)
    mats=[[] for _ in names]
    count=0
    with torch.no_grad():
        for pos in range(0,B,batch_size):
            k=min(batch_size,B-pos)
            indices=torch.as_tensor(rng.integers(0,n,size=(k,n),dtype=np.int64),device=device,dtype=torch.long)
            yb=ytorch[indices]
            valid=((yb.sum(1)>0)&(yb.sum(1)<n)).cpu().numpy()
            for j in range(len(names)):
                pb=ptorch[j][indices]
                met=gpu_score_batch(yb,pb)
                if with_calibration:
                    brier=((pb-yb)**2).mean(dim=1)
                    ece=ece_batch(yb,pb)
                    met=torch.cat([met,brier[:,None],ece[:,None]],dim=1)
                mats[j].append(met.cpu().numpy())
            count+=k
    arrays={name:np.concatenate(parts,axis=0) for name,parts in zip(names,mats)}
    metrics=['auroc','auprc']+(['brier','ece'] if with_calibration else [])
    return arrays,metrics


def rank_metrics(ids,pred,include_three=True):
    from scipy.stats import spearmanr, rankdata
    n=len(ids);p={k:np.asarray(v,dtype=float) for k,v in pred.items()}
    ranks={k:rankdata(-v,method='average') for k,v in p.items()}
    def overlap(a,b,k):
        ia=set(np.argsort(-p[a],kind='stable')[:k]); ib=set(np.argsort(-p[b],kind='stable')[:k])
        return len(ia&ib),len(ia^ib)
    k10=max(1,int(math.ceil(.1*n)));k20=max(1,int(math.ceil(.2*n)))
    pairs=[('H','V'),('H','I'),('V','I'),('H','S'),('S','I'),('I_lambda_0.01','I'),('I','I_lambda_1.0'),('I_lambda_0.01','I_lambda_1.0')]
    rows=[]
    for a,b in pairs:
        o10,ch10=overlap(a,b,k10);o20,ch20=overlap(a,b,k20)
        rows.append({'comparison':b+'_minus_'+a,'n':n,'score_mean_abs_difference':float(np.mean(np.abs(p[b]-p[a]))),
                     'score_spearman':float(spearmanr(p[a],p[b]).statistic),
                     'rank_spearman':float(spearmanr(ranks[a],ranks[b]).statistic),
                     'mean_abs_rank_shift':float(np.mean(np.abs(ranks[b]-ranks[a]))),
                     'top10_k':k10,'top10_overlap':o10,'top10_membership_changed_unique':ch10,
                     'top20_k':k20,'top20_overlap':o20,'top20_membership_changed_unique':ch20})
    outputs=[]
    stack=np.stack([p[k] for k in MAIN]);rs=np.stack([ranks[k] for k in MAIN]);
    top10=[set(np.argsort(-p[t],kind='stable')[:k10]) for t in MAIN]
    top20=[set(np.argsort(-p[t],kind='stable')[:k20]) for t in MAIN]
    for i,id_ in enumerate(ids):
        outputs.append({'parent_ik':str(id_),**{f'prediction_{name}':float(p[name][i]) for name in MAIN},
            **{f'rank_{name}':float(ranks[name][i]) for name in MAIN},
            'score_range_HVI':float(np.ptp(stack[:,i])),
            'rank_range_HVI':float(np.ptp(rs[:,i])),
            'rank_percentile_range_HVI':float(np.ptp(rs[:,i])/n),
            'max_abs_prediction_deviation_from_H':float(max(abs(p['V'][i]-p['H'][i]),abs(p['I'][i]-p['H'][i]))),
            'top10_membership_switch_HVI':len(set(i in s for s in top10))>1,
            'top20_membership_switch_HVI':len(set(i in s for s in top20))>1})
    overall={'n':n,'score_range_mean_HVI':float(np.ptp(stack,axis=0).mean()),
             'score_range_median_HVI':float(np.median(np.ptp(stack,axis=0))),
             'score_range_max_HVI':float(np.ptp(stack,axis=0).max()),
             'rank_range_mean_HVI':float(np.ptp(rs,axis=0).mean()),
             'rank_range_median_HVI':float(np.median(np.ptp(rs,axis=0))),
             'rank_range_max_HVI':float(np.ptp(rs,axis=0).max()),
             'top10_switch_unique_molecules_HVI':int(sum(r['top10_membership_switch_HVI'] for r in outputs)),
             'top20_switch_unique_molecules_HVI':int(sum(r['top20_membership_switch_HVI'] for r in outputs))}
    return rows,outputs,overall


def analyze(stage3run,out,n_boot,device,batch):
    from scipy.stats import spearmanr
    report_dir=out/'analysis';report_dir.mkdir(parents=True,exist_ok=True)
    ci=[];ranks=[];molecules=[];three=[];metrics=[];lambda_rows=[]
    for enc in ('xgb','gine'):
        for ei,ev in enumerate(EVALS):
            ids,y,p=load_predictions(stage3run,enc,ev)
            for name,pred in p.items():
                metrics.append({'encoder':enc,'evaluation':ev,'target':name,'n':len(y),**evaluate_metrics(y,pred)})
            pairs,mol,overall=rank_metrics(ids,p)
            ranks.extend([{'encoder':enc,'evaluation':ev,**r} for r in pairs])
            molecules.extend([{'encoder':enc,'evaluation':ev,**m} for m in mol])
            three.append({'encoder':enc,'evaluation':ev,**overall})
            lambda_rows.extend([{'encoder':enc,'evaluation':ev,**r} for r in pairs if r['comparison'] in (
                 'I_minus_I_lambda_0.01','I_lambda_1.0_minus_I','I_lambda_1.0_minus_I_lambda_0.01')])
            prediction={t:p[t] for t in MAIN}
            draw,metnames=bootstrap_stats(y,prediction,n_boot,20261011+ei*1000+(0 if enc=='xgb' else 10000),device,batch)
            for a,b in PAIRS:
                for j,met in enumerate(metnames):
                    delta=draw[b][:,j]-draw[a][:,j]
                    v=delta[np.isfinite(delta)]
                    if len(v)<.95*n_boot:fail(f'Too many degenerate bootstrap resamples: {enc} {ev}')
                    true=evaluate_metrics(y,p[b])[met]-evaluate_metrics(y,p[a])[met]
                    ci.append({'encoder':enc,'evaluation':ev,'comparison':b+'_minus_'+a,'metric':met,
                        'n':len(y),'estimate_delta':true,'CI95_lower':float(np.quantile(v,.025)),
                        'CI95_upper':float(np.quantile(v,.975)), 'valid_bootstrap':len(v),
                        'requested_bootstrap':n_boot,'seed_mean_prediction':True,'bootstrap_seed':20261011+ei*1000+(0 if enc=='xgb' else 10000),
                        'bootstrap_method':'paired molecule resampling; percentile CI; fixed-seed mean; no multiplicity correction'})
            print('[BOOTSTRAP]',enc,ev,'n',len(y),'replicates',n_boot,flush=True)
    for filename,rows in [('main_paired_bootstrap.csv',ci),('main_target_metrics.csv',metrics),
          ('main_rank_topk_pairwise.csv',ranks),('main_three_target_molecule_summary.csv',three),
          ('molecule_target_sensitivity.csv',molecules),('lambda_prediction_rank_topk.csv',lambda_rows)]:
        pd.DataFrame(rows).to_csv(report_dir/filename,index=False)
    
    st=[]
    for target in ALL:
        for seed in SEEDS['gine']:
            meta=stage3run/'predictions'/'gine'/'main'/f'{target}__seed{seed}.json'
            if not meta.is_file():fail(f'Missing best-epoch record: {meta}')
            ob=json.loads(meta.read_text())
            st.append({'target':target,'seed':seed,'best_epoch':ob['best_epoch'],
                       'best_validation_mse':ob['best_validation_mse'],'n_train':ob['n_train'],'n_validation':ob['n_validation']})
    pd.DataFrame(st).to_csv(report_dir/'gine_early_stop_by_seed.csv',index=False)
    man={'status':'STATS_READY','n_bootstrap':n_boot,'device':device,'batch_size':batch,
         'stage3_run':str(stage3run),'stage3_run_config_sha256':sha(stage3run/'run_config.json'),
         'expected_xgb_seeds':SEEDS['xgb'],'expected_gine_seeds':SEEDS['gine'],
         'main_arms':list(MAIN),'additional_controls':['S','I_lambda_0.01','I_lambda_1.0'],
         'evaluation_sizes':{'BBBP_test':291,'scaffold_test':137,'reference_out':2607},
         'n_paired_contrasts':len(ci),'warnings':[
           'Percentile CIs are descriptive and do not include training-seed sampling uncertainty.',
           'Multiple comparisons are not BH-adjusted in this analysis; CI exclusion is not treated as family-wise significance.',
           'Candidate ranking analyses use score-based top-k and midrank ties.',
           'Existing chemical-space Moran/scaffold/descriptor/SMARTS results must be recomputed from new molecule-level sensitivity.',
           'No tests were used to select priors or model hyperparameters.']}
    jsonwrite(report_dir/'stats_manifest.json',man)
    (out/'STATS_READY.txt').write_text('PASS: Stage4 paired bootstrap and molecular sensitivity statistics\n')


def external_candidates(root,out):
    needed={'eligible_external','ncats_parent_smiles','ncats_connectivity_key','y_pubchem','PUBCHEM_CID'}
    records=[]
    for path in sorted(root.rglob('*.csv')):
        if not any(t in path.name.lower() for t in ('ncats','pampa','external')):continue
        try:
            header=set(pd.read_csv(path,nrows=0).columns)
            if needed<=header:
                df=pd.read_csv(path,usecols=['eligible_external','y_pubchem'],dtype=str,keep_default_na=False)
                elig=df.eligible_external.str.lower().isin(('true','1'))
                e=df.loc[elig]
                records.append({'path':str(path),'qualified_columns':True,'eligible_rows':len(e),
                                'eligible_pos':int((e.y_pubchem=='1').sum()),'eligible_neg':int((e.y_pubchem=='0').sum()),'sha256':sha(path)})
            else:
                records.append({'path':str(path),'qualified_columns':False,
                                'missing_columns':sorted(needed-header)})
        except Exception as e:
            records.append({'path':str(path),'read_error':str(e)[:200]})
    jsonwrite(out/'pampa_candidate_manifest.json',records)
    good=[r for r in records if r.get('eligible_rows')==318 and r.get('eligible_pos')==213 and r.get('eligible_neg')==105]
    print('[PAMPA CANDIDATES]',len(records),'searched; matching profile',len(good),flush=True)
    for r in good:print('  candidate:',r['path'],flush=True)
    print('Select the historical audited CSV path manually; do not select first match automatically.',flush=True)


def pampa_identity_audit(external_path, out, policy='strict', diagnostic_only=False):
    





    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit import DataStructs
    path=Path(external_path).resolve()
    if not path.is_file():fail('Audited qualified PAMPA CSV not found: '+str(path))
    raw=pd.read_csv(path,dtype=str,keep_default_na=False)
    required={'eligible_external','ncats_parent_smiles','ncats_connectivity_key','y_pubchem','PUBCHEM_CID'}
    if not required<=set(raw.columns):fail('PAMPA input lacks required fields: '+','.join(sorted(required-set(raw.columns))))
    ext=raw.loc[raw.eligible_external.str.strip().str.lower().isin(('true','1'))].copy().reset_index(drop=True)
    yy=ext.y_pubchem.str.strip()
    if len(ext)!=318 or yy.value_counts().to_dict()!={'1':213,'0':105}:
        fail(f'PAMPA historical qualifying assay row counts changed: {len(ext)}, {yy.value_counts().to_dict()}')
    ext['y_pubchem']=yy
    for col in ('PUBCHEM_CID','ncats_parent_smiles','ncats_connectivity_key'):
        vals=ext[col].str.strip()
        if vals.eq('').any() or vals.str.lower().isin(('nan','none','null')).any():
            fail(f'PAMPA has missing {col}; must correct source audit, no automatic fill')
        ext[col]=vals
    def connect(s):
        m=Chem.MolFromSmiles(s)
        if m is None:fail('PAMPA molecular structure cannot be parsed')
        m=rdMolStandardize.FragmentParent(m);m=rdMolStandardize.ChargeParent(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(m).split('-')[0]
    recomputed=[connect(s) for s in ext.ncats_parent_smiles]
    if not np.array_equal(np.asarray(recomputed),ext.ncats_connectivity_key.to_numpy(str)):
        bad=[int(i) for i in np.flatnonzero(np.asarray(recomputed)!=ext.ncats_connectivity_key.to_numpy(str))[:10]]
        fail(f'PAMPA stored parent connectivity differs from RDKit recomputation at source rows: {bad}')
    dup_cid=ext.PUBCHEM_CID.duplicated(keep=False)
    dup_parent=ext.ncats_connectivity_key.duplicated(keep=False)
    cid_groups=ext.loc[dup_cid].groupby('PUBCHEM_CID',sort=False)
    cid_multi_parent=[cid for cid,g in cid_groups if g.ncats_connectivity_key.nunique()!=1]
    gen=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    def mol_fp(s):
        m=Chem.MolFromSmiles(s)
        if m is None:fail('Invalid source PAMPA SMILES')
        x=np.zeros((2048,),np.uint8)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),x)
        return x
    duplicate_groups=[]; unsafe=[]
    for parent,g in ext.loc[dup_parent].groupby('ncats_connectivity_key',sort=False):
        structures=g.ncats_parent_smiles.tolist()
        fps=[mol_fp(v) for v in structures]
        same_fp=all(np.array_equal(fps[0],p) for p in fps[1:])
        same_label=g.y_pubchem.nunique()==1
        detail={'parent_key':parent,'n_rows':len(g),'row_indices':g.index.astype(int).tolist(),
                'cids':g.PUBCHEM_CID.tolist(),'labels':g.y_pubchem.tolist(),
                'same_fingerprint':bool(same_fp),'same_label':bool(same_label)}
        duplicate_groups.append(detail)
        if not same_fp or not same_label:unsafe.append(detail)
    eligible_unique=ext.drop_duplicates(subset=['ncats_connectivity_key'],keep='first').reset_index(drop=True)
    report={'status':'PASS_UNIQUE' if not dup_parent.any() and not dup_cid.any() else
             'UNSAFE_DUPLICATE' if unsafe or cid_multi_parent else 'SAFE_IDENTICAL_DUPLICATES',
            'source_path':str(path),'source_sha256':sha(path),
            'historical_eligible_rows':len(ext),'historical_positive_rows':int((yy=='1').sum()),
            'historical_negative_rows':int((yy=='0').sum()),
            'distinct_cids':int(ext.PUBCHEM_CID.nunique()),
            'distinct_parent_connectivities':int(ext.ncats_connectivity_key.nunique()),
            'cid_duplicate_row_count':int(dup_cid.sum()),'parent_duplicate_row_count':int(dup_parent.sum()),
            'duplicate_groups':duplicate_groups,'cids_reused_across_different_parent_keys':cid_multi_parent,
            'n_unsafe_groups':len(unsafe),
            'post_dedup_n':len(eligible_unique),'post_dedup_positive':int((eligible_unique.y_pubchem=='1').sum()),
            'post_dedup_negative':int((eligible_unique.y_pubchem=='0').sum()),
            'requested_policy':policy,'diagnostic_only':bool(diagnostic_only),
            'scientific_note':'318 is an assay-row count; after identical-only deduplication, the evaluation sample size is defined by the number of distinct parent identities.'}
    pout=Path(out)/'pampa';pout.mkdir(parents=True,exist_ok=True)
    jsonwrite(pout/'pampa_identity_audit.json',report)
    print('[PAMPA IDENTITY AUDIT] '+json.dumps({k:v for k,v in report.items() if k not in ('duplicate_groups','source_path','source_sha256','scientific_note')},ensure_ascii=False),flush=True)
    if unsafe or cid_multi_parent:
        fail('PAMPA duplicated parent has conflicting phenotype/fingerprint or CID maps to different parent. Audit report written; source curation needed.')
    if (dup_parent.any() or dup_cid.any()) and policy=='strict':
        fail('PAMPA contains repeated parent/CID; diagnostic written. If all duplicate groups have identical label AND model fingerprint, rerun with PAMPA_DUP_POLICY=identical-only. This changes effective sample size.')
    if policy not in ('strict','identical-only'):fail('Unknown PAMPA duplicate policy: '+policy)
    if dup_cid.any() and not dup_parent.any():
        fail('Same CID reused with distinct parent identities; never collapse')
    if eligible_unique.PUBCHEM_CID.duplicated().any() or eligible_unique.ncats_connectivity_key.duplicated().any():
        fail('PAMPA identity dedup invariant failed')
    if set(eligible_unique.y_pubchem)!={'0','1'}:fail('PAMPA eligible dataset lacks one phenotype after dedup')
    if dup_parent.any() and not diagnostic_only:
        collapsed=ext.loc[ext.duplicated('ncats_connectivity_key',keep='first')]
        collapsed[['PUBCHEM_CID','ncats_connectivity_key','y_pubchem','ncats_parent_smiles']].to_csv(pout/'excluded_exact_duplicate_assay_rows.csv',index=False)
        report['dedup_applied']=True
        jsonwrite(pout/'pampa_identity_audit.json',report)
        print(f'[PAMPA DEDUP] Historical {len(ext)} assay rows -> {len(eligible_unique)} unique molecular parents. Updated case counts: {report["post_dedup_negative"]} negative and {report["post_dedup_positive"]} positive.',flush=True)
    else:
        report['dedup_applied']=False
        jsonwrite(pout/'pampa_identity_audit.json',report)
    return eligible_unique,report


def pampa(stage3run,stage3_project,out,external_path,n_boot,device,batch,n_jobs,dup_policy='strict'):
    
    stage3=import_stage3(stage3_project)
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    path=Path(external_path).resolve()
    ext,identity=pampa_identity_audit(path,out,dup_policy)
    cfg=json.loads((stage3run/'run_config.json').read_text())
    con=stage3.read(cfg['source']['consensus'])
    def connect(s):
        m=Chem.MolFromSmiles(str(s))
        if m is None:fail('PAMPA molecular structure cannot be parsed')
        m=rdMolStandardize.FragmentParent(m);m=rdMolStandardize.ChargeParent(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(m).split('-')[0]
    allkeys=set(connect(s) for s in set(con.canon_smiles))
    extkeys=[connect(s) for s in ext.ncats_parent_smiles]
    if set(extkeys)&allkeys:fail('PAMPA structural overlap with consensus set')
    if not np.array_equal(np.asarray(extkeys),ext.ncats_connectivity_key.to_numpy()):fail('PAMPA stored connectivity-key mismatch')
    tr,common,targets,perms,evals=stage3.load_all(stage3run)
    pout=out/'pampa';pout.mkdir(exist_ok=True)
    csv=pout/'pampa_predictions_by_seed.csv'
    ids=ext.ncats_connectivity_key.to_numpy(str);gold=ext.y_pubchem.to_numpy(int)
    n=len(ext)
    expected=n*len(SEEDS['xgb'])*len(MAIN)
    if csv.exists():
        df=pd.read_csv(csv,dtype={'parent_ik':str,'PUBCHEM_CID':str,'target':str})
        if len(df)!=expected:fail(f'Existing final PAMPA prediction CSV row count disagrees with audited effective sample size: {len(df)} vs {expected}')
        if df.duplicated(['target','seed','parent_ik']).any():fail('Existing PAMPA predictions contain duplicate entries')
        if set(df.target)!=set(MAIN):fail('Existing PAMPA targets do not match H/V/I')
        for (t,seed),sub in df.groupby(['target','seed'],sort=False):
            if int(seed) not in SEEDS['xgb'] or len(sub)!=n:fail('Unexpected PAMPA seed or group size')
            if not np.array_equal(sub.parent_ik.to_numpy(str),ids) or not np.array_equal(sub.gold.to_numpy(int),gold):
                fail('Existing PAMPA output disagrees with audited identity/gold')
        if df.groupby(['target','seed']).ngroups!=len(MAIN)*len(SEEDS['xgb']):fail('Missing PAMPA seed in completed CSV')
        print('[PAMPA RESUME] complete prediction CSV verified, no refitting',flush=True)
    else:
        models_dir=pout/'model_predictions';models_dir.mkdir(exist_ok=True)
        todo=[(target,seed) for target in MAIN for seed in SEEDS['xgb']
              if not (models_dir/f'{target}__seed{seed}.csv').is_file()]
        print(f'[PAMPA MODEL CHECKPOINTS] {len(todo)} / 60 missing. Refitting uses identical Stage3 CPU learner.',flush=True)
        Xtr=Xte=None
        if todo:
            Xtr=stage3.molecular_fp(common.canon_smiles)
            Xte=stage3.molecular_fp(ext.ncats_parent_smiles)
        pieces=[]
        for target in MAIN:
            for seed in SEEDS['xgb']:
                one=models_dir/f'{target}__seed{seed}.csv'
                if not one.exists():
                    model=stage3.train_xgb(Xtr,targets[target],seed,n_jobs)
                    pred=np.asarray(model.predict(Xte),float)
                    if len(pred)!=n or not np.isfinite(pred).all():fail('PAMPA model returned invalid predictions')
                    sub=pd.DataFrame({'target':target,'seed':int(seed),
                        'PUBCHEM_CID':ext.PUBCHEM_CID.to_numpy(str),'parent_ik':ids,
                        'gold':gold,'prediction':pred})
                    tmp=one.with_suffix('.partial')
                    sub.to_csv(tmp,index=False);tmp.replace(one)
                    print('[PAMPA FIT]',target,'seed',seed,flush=True)
                else:
                    sub=pd.read_csv(one,dtype={'parent_ik':str,'PUBCHEM_CID':str,'target':str})
                    print('[PAMPA CHECKPOINT]',target,'seed',seed,flush=True)
                if (len(sub)!=n or not np.array_equal(sub.parent_ik.to_numpy(str),ids)
                    or not np.array_equal(sub.gold.to_numpy(int),gold)
                    or sub.target.nunique()!=1 or sub.target.iloc[0]!=target
                    or sub.seed.nunique()!=1 or int(sub.seed.iloc[0])!=seed
                    or not np.isfinite(sub.prediction.to_numpy(float)).all()):
                    fail(f'PAMPA checkpoint validation failed: {one}')
                pieces.append(sub)
        df=pd.concat(pieces,ignore_index=True)
        if len(df)!=expected:fail('PAMPA assembled result size differs from 60 * actual unique parent count')
        tmp=csv.with_suffix('.partial');df.to_csv(tmp,index=False);tmp.replace(csv)
    av={};ids=ext.ncats_connectivity_key.to_numpy(str);gold=ext.y_pubchem.to_numpy(int)
    for target in MAIN:
        frame=df[df.target.eq(target)].copy()
        parts=[]
        for s in SEEDS['xgb']:
            sub=frame[frame.seed.eq(s)]
            if not np.array_equal(sub.parent_ik.to_numpy(str),ids) or not np.array_equal(sub.gold.to_numpy(int),gold):
                fail(f'PAMPA stored predictions differ from frozen input: {target} seed{s}')
            parts.append(sub.prediction.to_numpy(float))
        av[target]=np.mean(parts,axis=0)
    stats=[];ci=[]
    for t in MAIN:stats.append({'target':t,'n':n,**evaluate_metrics(gold,av[t],with_calibration=True)})
    boot,metnames=bootstrap_stats(gold,av,n_boot,20261011+505050,device,batch,with_calibration=True)
    for a,b in PAIRS:
        for j,met in enumerate(metnames):
            d=boot[b][:,j]-boot[a][:,j]
            d=d[np.isfinite(d)]
            if len(d)<.95*n_boot:fail('Too many invalid PAMPA bootstrap replicates')
            ci.append({'evaluation':'PAMPA_BBB','comparison':b+'_minus_'+a,'metric':met,
                       'n':n,'estimate_delta':evaluate_metrics(gold,av[b],True)[met]-evaluate_metrics(gold,av[a],True)[met],
                       'CI95_lower':float(np.quantile(d,.025)),'CI95_upper':float(np.quantile(d,.975)),
                       'valid_bootstrap':len(d),'requested_bootstrap':n_boot,
                       'method':'paired molecule bootstrap on unique parents, fixed 20 seed means; percentile CI; no multiplicity adjustment'})
    pd.DataFrame(stats).to_csv(pout/'pampa_mean_seed_metrics.csv',index=False)
    pd.DataFrame(ci).to_csv(pout/'pampa_paired_bootstrap.csv',index=False)
    jsonwrite(pout/'pampa_provenance.json',{'qualified_external_path':str(path),'qualified_external_sha256':sha(path),
        'n_original_eligible_assay_rows':identity['historical_eligible_rows'],
        'n_unique_parent_molecules':n,'n_positive':int((gold==1).sum()),'n_negative':int((gold==0).sum()),
        'dedup_policy':dup_policy,'n_collapsed_records':identity['historical_eligible_rows']-n,
        'targets':list(MAIN),'xgb_seeds':SEEDS['xgb'],'model_protocol':'exact original Stage3 CPU XGBoost parameters',
        'bootstrap_device':device,'no_original_stage3_saved_model_weights':True,
        'notes':'PAMPA labels are held out from training; vote-derived soft, NOT simple channel average. If historical 318 included duplicate parents, updated unique-parent N supersedes historical N for independent evaluation.'})
    (out/'PAMPA_READY.txt').write_text(f'PASS: {n} unique PAMPA parents (from 318 eligible assay rows), H/V/I, 20 seeds, paired bootstrap\n')
    print(f'[PAMPA READY] {n} unique parents from 318 eligible assay rows; 60 paired 20-seed models',flush=True)

def pack(out,stage3run):
    from zipfile import ZipFile,ZIP_DEFLATED
    an=out/'analysis';pm=out/'pampa';stage4=out/'stage4_info.json'
    files=[out/'precheck.json',stage4,out/'STATS_READY.txt',out/'PAMPA_READY.txt']
    files+=list(an.glob('*.csv'))+list(an.glob('*.json'))
    files += [pm/'pampa_mean_seed_metrics.csv',pm/'pampa_paired_bootstrap.csv',pm/'pampa_provenance.json']
    files+=list(out.glob('*.log'))
    files+=list(pm.glob('*audit*.json'))
    files+=list(pm.glob('excluded_exact_duplicate_assay_rows.csv'))
    if (out/'pampa_candidate_manifest.json').exists():files.append(out/'pampa_candidate_manifest.json')
    (out/'RESULT_STATUS.txt').write_text('COMPLETE_STATS_AND_PAMPA\n' if (out/'STATS_READY.txt').exists() and (out/'PAMPA_READY.txt').exists()
                            else 'PARTIAL: see READY markers; no invented missing results\n',encoding='utf-8')
    files.append(out/'RESULT_STATUS.txt')
    result=out/'BBB_STAGE4_FINAL_REPORT_TO_SEND.zip'
    with ZipFile(result,'w',compression=ZIP_DEFLATED) as z:
        for f in files:
            if f.is_file():
                if f.stat().st_size>15_000_000:continue
                z.write(f,arcname=str(f.relative_to(out)))
    print('[REPORT]',result,'size',result.stat().st_size,flush=True)


def main():
    ap=argparse.ArgumentParser(description='BBB final analysis: Stage3 immutable inputs, GPU paired bootstrap, audited PAMPA')
    ap.add_argument('action',choices=['precheck','analyze','pampa','diagnose-pampa','discover','pack'])
    ap.add_argument('--stage3-run',default='')
    ap.add_argument('--stage3-project',default=str(PROJECT))
    ap.add_argument('--out',required=True)
    ap.add_argument('--external',default='')
    ap.add_argument('--duplicate-policy',choices=['strict','identical-only'],default='strict')
    ap.add_argument('--n-bootstrap',type=int,default=2000)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--n-jobs',type=int,default=4)
    ap.add_argument('--search-root',default=str(BASE))
    args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    stage3project=Path(args.stage3_project).resolve();srun=stage3_run(args.stage3_run,stage3project)
    try:
        if args.n_bootstrap<200 or args.batch_size<1:fail('Need >=200 bootstrap replicates and positive batch size')
        cfg=require_stage3(srun,stage3project,need_all=args.action!='discover')
        if args.action=='precheck':
            if (out/'precheck.json').exists():fail('Do not replace existing Stage4 precheck output')
            m={'status':'PASS_STAGE4_PRECHECK','stage3_run':str(srun),'stage3_run_config_sha256':sha(srun/'run_config.json'),
               'expected_eval_sizes':[291,137,2607], 'n_stage3_seeds':{k:len(v) for k,v in SEEDS.items()},
               'note':'No model trained; CUDA checked in analyze/pampa actions only.'}
            for enc in ('xgb','gine'):
                for ev in EVALS:
                    _ids,_y,_pred=load_predictions(srun,enc,ev)
                    verify_published_stage3_metrics(srun,enc,ev,_y,_pred)
            jsonwrite(out/'precheck.json',m)
            print('[PASS_STAGE4_PRECHECK]',flush=True)
        else:
            if args.action not in ('discover',) and not (out/'precheck.json').is_file():fail('Stage4 precheck not completed')
            if args.action=='analyze':
                if (out/'STATS_READY.txt').exists():fail('Stage4 analysis already completed; refuse overwrite')
                device,meta=resolve_device(args.device)
                jsonwrite(out/'stage4_info.json',{'device':meta,'n_bootstrap':args.n_bootstrap,'stage3_run':str(srun),
                 'stage3_run_config_sha256':sha(srun/'run_config.json')})
                print('[DEVICE]',meta,flush=True)
                analyze(srun,out,args.n_bootstrap,device,args.batch_size)
            elif args.action=='diagnose-pampa':
                if not args.external:fail('PAMPA diagnosis requires --external path')
                
                pampa_identity_audit(args.external,out,policy='identical-only',diagnostic_only=True)
                print('[PAMPA DIAGNOSIS READY] Review audit; this did not run any model',flush=True)
            elif args.action=='pampa':
                if not args.external:fail('PAMPA requires --external /absolute/path/audited_qualified_ncats.csv')
                if (out/'PAMPA_READY.txt').exists():fail('PAMPA report already completed; refuse overwrite')
                device,meta=resolve_device(args.device)
                pampa(srun,stage3project,out,args.external,args.n_bootstrap,device,args.batch_size,args.n_jobs,args.duplicate_policy)
            elif args.action=='discover':external_candidates(Path(args.search_root),out)
            elif args.action=='pack':pack(out,srun)
    except Exception as ex:
        print('[STAGE4 ERROR]',str(ex),file=sys.stderr,flush=True)
        traceback.print_exc()
        raise SystemExit(2)

if __name__=='__main__':main()
