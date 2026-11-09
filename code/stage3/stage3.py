# stage3.py — Validates immutable target vectors and frozen evaluation sets; performs paired XGBoost and GINE training across seeds and target definitions, evaluates predictions and conditional permutations, and exports molecule-level predictions and metrics.






from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback, math, platform
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
FROZEN = BASE/'data/frozen'
S2_PROJECT = BASE/'code/stage2'
CONS_SHA='51d09eea4e6a1659ea7820a93f5f63880161645c3bab8f7f9f93007bdbc9849a'
LAB_SHA='ea6b31d500e1f04384952a096fabdf4e9ce3e0bd9533587442c542946f432c89'
XGB_SEEDS=list(range(42,62))
GINE_SEEDS=[42,52,62,72,82,92,102,112,122,132]
PERM_SEEDS={'xgb':[42,52,61], 'gine':[42,52,62]}
TARGET_MAIN=('H','V','I','S','I_lambda_0.01','I_lambda_1.0')
PARAMS={'max_depth':5,'learning_rate':0.03,'n_estimators':300,
        'subsample':0.8,'colsample_bytree':0.8}
EVALS=('BBBP_test','scaffold_test','reference_out')


def die(message):raise RuntimeError(message)

def read(path):return pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig',low_memory=False)

def nums(s):return pd.to_numeric(s,errors='coerce')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1<<20),b''):h.update(buf)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def resolve_s2(explicit):
    if explicit:
        p=Path(explicit).resolve()
        if not (p/'READY_FOR_REVIEW.txt').exists():die(f'Stage2 QC marker missing: {p}')
        return p
    if not S2_PROJECT.exists():die(f'Stage2 project directory not found: {S2_PROJECT}')
    choices=[p.parent for p in S2_PROJECT.glob('results/stage2_targets_*/READY_FOR_REVIEW.txt')]
    if not choices:die('No READY_FOR_REVIEW.txt Stage2 run found. Specify --stage2 /ABS/PATH to the verified run folder.')
    choices.sort(key=lambda p:p.stat().st_mtime,reverse=True)
    return choices[0]


def get_sources(stage2,consensus,labels):
    con=Path(consensus).resolve(); lab=Path(labels).resolve()
    if not con.exists() or not lab.exists():die(f'Frozen input absent: {con}, {lab}')
    if sha256(con)!=CONS_SHA:die('Frozen consensus SHA256 mismatch')
    if sha256(lab)!=LAB_SHA:die('Frozen labels SHA256 mismatch')
    fnames=['stage2_target_vectors_common1157.csv','P_99_permutation_targets_common1157.csv','labels_abc_corrected_STAGE2_ONLY.csv']
    for f in fnames:
        if not (stage2/f).exists():die(f'Missing Stage2 export {stage2/f}')
    return {'consensus':str(con),'labels':str(lab),'stage2':str(stage2),
            'hashes':{**{f:sha256(stage2/f) for f in fnames},'consensus':CONS_SHA,'labels':LAB_SHA}}


def compute_strata(targets):
    return (nums(targets['label_A']).astype(int).astype(str)+'_'+nums(targets['has_bbbp']).astype(int).astype(str)
            +'_'+nums(targets['has_logbb']).astype(int).astype(str)).to_numpy()


def check_inputs(source,run):
    con=read(source['consensus']); lab=read(source['labels'])
    stage2=Path(source['stage2']); vec=read(stage2/'stage2_target_vectors_common1157.csv')
    pp=read(stage2/'P_99_permutation_targets_common1157.csv'); compat=read(stage2/'labels_abc_corrected_STAGE2_ONLY.csv')
    tr=con.loc[con.split.eq('train') & nums(con.in_reference_out).fillna(0).eq(0)].reset_index(drop=True)
    if (len(con),len(tr),int(con.split.eq('test').sum()))!=(4131,1234,291):die('Frozen population counts differ from preflight')
    if not np.array_equal(tr.parent_ik.to_numpy(),lab.parent_ik.to_numpy()):die('1234 frozen parent rows not aligned')
    Hfull=nums(lab['label_A']).to_numpy(float); mask=np.isfinite(Hfull)
    if int(mask.sum())!=1157:die('Common target mask != 1157')
    ids=tr.loc[mask,'parent_ik'].to_numpy(str)
    if (len(np.unique(ids))!=len(ids) or not np.array_equal(ids,vec.parent_ik.to_numpy())
            or not np.array_equal(ids,pp.parent_ik.to_numpy())):die('Stage2 target/permutation parent ordering mismatch')
    if not np.array_equal(lab.parent_ik.to_numpy(),compat.parent_ik.to_numpy()):die('Compatible target pool ordering mismatch')
    H=nums(vec.label_A).to_numpy(float);V=nums(vec.label_B).to_numpy(float)
    if not np.allclose(H,Hfull[mask],atol=1e-12,rtol=0):die('Stage2 Hard changed from frozen input')
    if not np.allclose(V,nums(lab.label_B).to_numpy(float)[mask],atol=1e-12,rtol=0):die('Stage2 Vote changed from frozen input')
    Y={'H':H,'V':V,'I':nums(vec.label_C_integrated).to_numpy(float),
       'S':nums(vec.label_S_class_smooth).to_numpy(float),
       'I_lambda_0.01':nums(vec['integrated_lambda_0.01']).to_numpy(float),
       'I_lambda_1.0':nums(vec['integrated_lambda_1']).to_numpy(float)}
    if not np.allclose(Y['I'],nums(vec['integrated_lambda_0.1']).to_numpy(float),atol=1e-12,rtol=0):die('Primary lambda 0.1 mismatch')
    if not np.allclose(Y['I'],nums(compat.loc[mask,'label_C_sigmoid_mu']).to_numpy(float),atol=1e-12,rtol=0):die('Stage2 compatibility I mismatch')
    if not np.allclose(Y['S'],nums(compat.loc[mask,'label_S_smooth']).to_numpy(float),atol=1e-12,rtol=0):die('Stage2 compatibility S mismatch')
    for name,x in Y.items():
        if not np.isfinite(x).all() or np.min(x)<-1e-10 or np.max(x)>1+1e-10:die(f'Target {name} invalid')
    if not np.array_equal(H,np.round(H)):die('Hard target not binary')
    permcols=[c for c in pp if c.startswith('P_')]
    if len(permcols)!=99:die('Require exactly 99 frozen conditional permutation target columns')
    if len({sha256_bytes(nums(pp[c]).to_numpy(float).tobytes()) for c in permcols})!=99:die('Nonunique conditional permutations')
    strata=compute_strata(vec); altered=[]
    for col in permcols:
        x=nums(pp[col]).to_numpy(float)
        if not np.isfinite(x).all() or not (0<=x.min() and x.max()<=1):die(f'Permuted target invalid {col}')
        for st in np.unique(strata):
            ix=np.flatnonzero(strata==st)
            if not np.allclose(np.sort(x[ix]),np.sort(Y['I'][ix]),atol=1e-12,rtol=0):
                die(f'Permutation {col} does not preserve per-stratum multiset: {st}')
        altered.append(int(np.sum(abs(x-Y['I'])>1e-10)))
    if min(altered)<100:die(f'Too few effective permuted values: {min(altered)}')
    if int((~np.isclose(V,np.round(V))).sum())!=8:die('Vote nonbinary anchor changed')
    
    bb=con.loc[con.split.eq('test'),'parent_ik']; ro=con.loc[nums(con.in_reference_out).fillna(0).eq(1),'parent_ik']
    if set(ids)&set(bb) or set(ids)&set(ro):die('Train/evaluation parent overlap')
    s2qc=stage2/'qc_report.json'
    if not s2qc.exists():die('Stage2 QC report missing')
    qc=json.loads(s2qc.read_text())
    if (stage2/'FAILURE.txt').exists():die('Stage2 failure marker present')
    out={'status':'PASS_PRECHECK','population':{'consensus':len(con),'train_pool':len(tr),
          'train_common':len(ids),'bbbp_test':len(bb),'reference_out':len(ro)},
          'n_vote_nonbinary':8,'n_perm':99,'n_perm_changed_min':min(altered),'n_perm_changed_max':max(altered),
          'target_stats':{k:{'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max())} for k,x in Y.items()},
          'source':source,'stage2_qc_status':qc.get('status','check qc report'),'note':'No model trained by precheck'}
    write_json(run/'precheck.json',out)
    (run/'PASS_PRECHECK.txt').write_text('PASS_PRECHECK\n',encoding='utf-8')
    print('[PASS_PRECHECK] 1157 parents, 99 valid permutations; changed target values',min(altered),'to',max(altered),flush=True)
    return out


def sha256_bytes(b):return hashlib.sha256(b).hexdigest()


def load_all(run,scaffold_path=None):
    cfg=json.loads((run/'run_config.json').read_text());source=cfg['source']
    for k,v in source['hashes'].items():
        p=source['consensus'] if k=='consensus' else source['labels'] if k=='labels' else str(Path(source['stage2'])/k)
        if sha256(p)!=v:die('Source changed after precheck: '+p)
    con=read(source['consensus']);lab=read(source['labels']); s2=Path(source['stage2'])
    vec=read(s2/'stage2_target_vectors_common1157.csv');pp=read(s2/'P_99_permutation_targets_common1157.csv')
    tr=con.loc[con.split.eq('train') & nums(con.in_reference_out).fillna(0).eq(0)].reset_index(drop=True)
    mask=nums(lab.label_A).notna().to_numpy(); common=tr.loc[mask].reset_index(drop=True)
    if not np.array_equal(common.parent_ik.to_numpy(),vec.parent_ik.to_numpy()):die('Training mask changed')
    targets={'H':nums(vec.label_A).to_numpy(float),'V':nums(vec.label_B).to_numpy(float),
         'I':nums(vec.label_C_integrated).to_numpy(float),'S':nums(vec.label_S_class_smooth).to_numpy(float),
         'I_lambda_0.01':nums(vec['integrated_lambda_0.01']).to_numpy(float),
         'I_lambda_1.0':nums(vec['integrated_lambda_1']).to_numpy(float)}
    if scaffold_path:
        sc=read(scaffold_path)
    else:
        scp=run/'scaffold_split_assignments.csv'
        if not scp.exists():
            from legacy_scaffold_split import scaffold_of, SEED, VAL_FRAC_WITHIN_TRAIN
            
            import subprocess
            subprocess.run([sys.executable,str(Path(__file__).with_name('legacy_scaffold_split.py')),
                '--consensus',source['consensus'],'--output',str(scp)],check=True)
        sc=read(scp)
    if 'scaffold_split' not in sc or 'b3db_label' not in sc:die('Invalid scaffold assignments')
    evals={'BBBP_test':(con.loc[con.split.eq('test')].reset_index(drop=True),'bbbp_label'),
           'scaffold_test':(sc.loc[sc.scaffold_split.eq('test')].reset_index(drop=True),'b3db_label'),
           'reference_out':(con.loc[nums(con.in_reference_out).fillna(0).eq(1)].reset_index(drop=True),'b3db_label')}
    if sc.scaffold_split.eq('test').sum()!=152:die(f'Scaffold assignment differs from original: {sc.scaffold_split.eq("test").sum()} not 152')
    eval_filtered={}
    for k,(frame,goldcol) in evals.items():
        mask_g=nums(frame[goldcol]).notna().to_numpy()
        sub=frame.loc[mask_g].reset_index(drop=True)
        gold=nums(sub[goldcol]).to_numpy(float)
        if not np.isin(gold,[0.,1.]).all():die(f'Invalid evaluation gold {k}')
        if len(set(sub.parent_ik)&set(common.parent_ik))>0:die(f'Train/eval overlap: {k}')
        if sub.parent_ik.duplicated().any():die(f'Duplicate eval parent IDs {k}')
        eval_filtered[k]=(sub,gold.astype(int))
    counts={k:len(v[1]) for k,v in eval_filtered.items()}
    if counts != {'BBBP_test':291,'scaffold_test':137,'reference_out':2607}:
        die(f'Evaluation sizes changed from frozen legacy analysis: {counts}')
    return tr,common,targets,pp,eval_filtered


def molecular_fp(smiles):
    from rdkit import Chem,DataStructs,RDLogger
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    RDLogger.DisableLog('rdApp.*')
    gen=GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    X=np.zeros((len(smiles),2048),dtype=np.uint8)
    for i,s in enumerate(smiles):
        m=Chem.MolFromSmiles(str(s))
        if m is None:die(f'Invalid canonical SMILES {i} {s}')
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(m),X[i])
    return X


def train_xgb(X,y,seed,n_jobs):
    import xgboost as xgb
    m=xgb.XGBRegressor(**PARAMS,random_state=int(seed),n_jobs=int(n_jobs))
    m.fit(X,y)
    return m


def metric(y,p):
    from sklearn.metrics import roc_auc_score,average_precision_score
    return {'auroc':float(roc_auc_score(y,p)), 'auprc':float(average_precision_score(y,p))}


def do_train(args,run):
    encoder=args.encoder; group=args.group
    if encoder not in ('xgb','gine') or group not in ('main','perm'):die('Invalid training mode')
    tr,common,targets,perm,evals=load_all(run,args.scaffold)
    if group=='main': jobs=[(k,k,y) for k,y in targets.items()]
    else:
        cols=[f'P_{n:03d}' for n in range(0,99)]
        if not all(c in perm.columns for c in cols):die('Permutation columns must be P_000..P_098')
        start=int(args.perm_start);stop=int(args.perm_stop)
        if not 0<=start<=stop<=98:die('Invalid permutation subset')
        jobs=[(f'P_{n:03d}',f'P_{n:03d}',nums(perm[f'P_{n:03d}']).to_numpy(float)) for n in range(start,stop+1)]
    seeds=(XGB_SEEDS if encoder=='xgb' else GINE_SEEDS) if group=='main' else PERM_SEEDS[encoder]
    out=run/'predictions'/encoder/group;out.mkdir(parents=True,exist_ok=True)
    if encoder=='xgb':
        Xtr=molecular_fp(common.canon_smiles)
        Xev={e:molecular_fp(sub.canon_smiles) for e,(sub,y) in evals.items()}
    else:
        import torch
        if torch.cuda.is_available():print('[GINE] CUDA present but frozen CPU protocol is used',flush=True)
        from legacy_gine_cpu import make_vocabularies, build_graph
        am,bm=make_vocabularies(tr.canon_smiles.tolist())
        gtr=[build_graph(s,am,bm) for s in tr.canon_smiles]
        gev={e:[build_graph(s,am,bm) for s in sub.canon_smiles] for e,(sub,y) in evals.items()}
        mask=nums(read(json.loads((run/'run_config.json').read_text())['source']['labels']).label_A).notna().to_numpy()
        
        index_to_common=np.full(len(tr),-1,dtype=int);index_to_common[np.flatnonzero(mask)]=np.arange(len(common))
    for target_id,public_name,y in jobs:
        if not np.isfinite(y).all() or len(y)!=len(common):die(f'Target invalid: {target_id}')
        for seed in seeds:
            name=f'{public_name}__seed{seed}.csv'
            dest=out/name
            if dest.exists():
                validate_single_prediction(dest,evals,public_name,seed)
                print('[RESUME]',encoder,group,name,flush=True);continue
            t0=time.time()
            if encoder=='xgb':
                model=train_xgb(Xtr,y,seed,args.n_jobs)
                pred={e:model.predict(Xev[e]).astype(float) for e in EVALS}
                extra={}
            else:
                from legacy_gine_cpu import train_model,predict
                rng=np.random.RandomState(seed)
                order=rng.permutation(len(tr));nv=int(.15*len(tr))
                vi,ti=order[:nv],order[nv:]
                
                graphs=[g.clone() for g in gtr]
                for ii in np.concatenate([ti,vi]):
                    ci=index_to_common[ii]
                    if ci>=0:graphs[ii].y=torch.tensor([y[ci]],dtype=torch.float32)
                tg=[graphs[i] for i in ti if index_to_common[i]>=0]
                vg=[graphs[i] for i in vi if index_to_common[i]>=0]
                if not tg or not vg:die(f'Empty GINE train/val split seed {seed}')
                model,best_epoch,best_val=train_model(tg,vg,seed)
                pred={e:predict(model,gev[e]).astype(float) for e in EVALS}
                extra={'best_epoch':int(best_epoch),'best_validation_mse':float(best_val),'n_train':len(tg),'n_validation':len(vg)}
            result=[]
            for e in EVALS:
                sub,g=evals[e];ps=pred[e]
                if len(ps)!=len(g) or not np.isfinite(ps).all():die(f'Bad predictions {encoder} {target_id} {e}')
                for iid,gold,p in zip(sub.parent_ik,g,ps):
                    result.append((encoder,group,public_name,int(seed),e,str(iid),int(gold),float(p)))
            tmp=dest.with_suffix('.tmp')
            pd.DataFrame(result,columns=['encoder','group','target','seed','evaluation','parent_ik','gold','prediction']).to_csv(tmp,index=False)
            tmp.replace(dest)
            if extra:
                write_json(out/f'{public_name}__seed{seed}.json',{**extra,'elapsed_seconds':time.time()-t0})
            print('[DONE]',encoder,group,public_name,'seed',seed,'elapsed',round(time.time()-t0,2),'s',flush=True)
    print('[TRAIN_PHASE_DONE]',encoder,group,len(jobs),'targets',flush=True)


def validate_single_prediction(path,evals,target,seed):
    df=pd.read_csv(path)
    for e,(sub,g) in evals.items():
        s=df[df.evaluation.eq(e)]
        if len(s)!=len(sub) or not np.array_equal(s.parent_ik.astype(str).to_numpy(),sub.parent_ik.to_numpy()):die(f'Existing prediction corrupted: {path}')
        if not np.array_equal(s.gold.to_numpy(int),g):die(f'Existing gold labels mismatch: {path}')
        if not np.isfinite(s.prediction.to_numpy(float)).all():die(f'Existing prediction has NaN: {path}')
    if df.target.nunique()!=1 or df.target.iloc[0]!=target or df.seed.nunique()!=1 or int(df.seed.iloc[0])!=int(seed):die(f'Wrong target/seed in {path}')


def all_predictions(run,encoder,group,expected_targets=None,expected_seeds=None):
    parent=run/'predictions'/encoder/group
    if not parent.exists():
        missing=([f'{name}__seed{seed}.csv' for name in expected_targets for seed in expected_seeds] if expected_targets is not None else [])
        return None,missing
    paths=sorted(parent.glob('*.csv'))
    if expected_targets is not None:
        missing=[f'{name}__seed{seed}.csv' for name in expected_targets for seed in expected_seeds if not (parent/f'{name}__seed{seed}.csv').exists()]
    else:missing=[]
    dfs=[pd.read_csv(p) for p in paths]
    return (pd.concat(dfs,ignore_index=True) if dfs else None),missing


def pampa_eval(args,run):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from sklearn.metrics import brier_score_loss
    fn=Path(args.qualified_external)
    if not fn.is_file():die('Specify existing --qualified-external path for audited NCATS table')
    raw=pd.read_csv(fn,dtype=str,keep_default_na=False)
    needed={'eligible_external','ncats_parent_smiles','ncats_connectivity_key','y_pubchem','PUBCHEM_CID'}
    if not needed<=set(raw.columns):die('Qualified PAMPA file missing expected fields')
    mask=raw.eligible_external.astype(str).str.lower().isin(('true','1'))
    ext=raw.loc[mask].reset_index(drop=True)
    if len(ext)!=318 or ext.y_pubchem.value_counts().to_dict()!={'1':213,'0':105}:die('PAMPA eligible counts unexpected')
    if ext.PUBCHEM_CID.duplicated().any() or ext.ncats_connectivity_key.duplicated().any():die('PAMPA duplicate identity')
    def connect(s):
        m=Chem.MolFromSmiles(str(s))
        if m is None:die('Invalid PAMPA SMILES')
        m=rdMolStandardize.FragmentParent(m);m=rdMolStandardize.ChargeParent(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(m).split('-')[0]
    cfg=json.loads((run/'run_config.json').read_text()); con=read(cfg['source']['consensus'])
    all_keys={connect(s) for s in con.canon_smiles}
    ext_keys=[connect(s) for s in ext.ncats_parent_smiles]
    if set(ext_keys)&all_keys:die('PAMPA parent connectivity leakage')
    if not np.array_equal(np.asarray(ext_keys),ext.ncats_connectivity_key.to_numpy()):die('PAMPA reference parent keys differ')
    tr,common,targets,perm,evals=load_all(run,args.scaffold)
    xtr=molecular_fp(common.canon_smiles);xte=molecular_fp(ext.ncats_parent_smiles)
    out=run/'pampa';out.mkdir(exist_ok=True)
    preds=out/'pampa_predictions_by_seed.csv'
    rows=[]
    
    for t in ('H','V','I'):
        for seed in XGB_SEEDS:
            model=train_xgb(xtr,targets[t],seed,args.n_jobs)
            p=model.predict(xte)
            for cid,parent,g,z in zip(ext.PUBCHEM_CID,ext.ncats_connectivity_key,ext.y_pubchem,p):
                rows.append((t,seed,str(cid),str(parent),int(g),float(z)))
    df=pd.DataFrame(rows,columns=['target','seed','PUBCHEM_CID','parent_ik','gold','prediction'])
    df.to_csv(preds,index=False)
    mean=df.groupby(['target','parent_ik','gold'],sort=False)['prediction'].mean().reset_index()
    results=[]
    for t in ('H','V','I'):
        sub=mean[mean.target.eq(t)];y=sub.gold.to_numpy(int);p=np.clip(sub.prediction.to_numpy(float),0,1)
        vals=metric(y,p)
        bins=np.clip(np.digitize(p,np.linspace(0,1,11)[1:-1]),0,9)
        ece=float(sum(np.mean(bins==i)*abs(np.mean(p[bins==i])-np.mean(y[bins==i])) for i in range(10) if np.any(bins==i)))
        results.append({'target':t,'n':len(y),**vals,'brier':float(brier_score_loss(y,p)),'ece':ece})
    pd.DataFrame(results).to_csv(out/'pampa_mean_seed_metrics.csv',index=False)
    write_json(out/'pampa_provenance.json',{'qualified_external_path':str(fn),'sha256':sha256(fn),'n_eligible':len(ext),
        'targets':['H','V','I'],'different_from_legacy_simple_channel_average':True,'xgb_seeds':XGB_SEEDS})
    print('[PAMPA DONE]',results,flush=True)


def summarize(args,run):
    from scipy.stats import spearmanr
    allmetric=[]; meanrows=[]; completed={}; missing={}; permtest=[]; pairrows=[]; rankrows=[]
    for encoder in ('xgb','gine'):
        df,miss=all_predictions(run,encoder,'main',TARGET_MAIN,XGB_SEEDS if encoder=='xgb' else GINE_SEEDS)
        missing[encoder+'_main']=miss
        if df is None:continue
        if df.duplicated(['target','seed','evaluation','parent_ik']).any():die(f'Duplicate model predictions {encoder}')
        for (target,seed,ev),sub in df.groupby(['target','seed','evaluation']):
            allmetric.append({'encoder':encoder,'target':target,'seed':int(seed),'evaluation':ev,'n':len(sub),**metric(sub.gold.to_numpy(int),sub.prediction.to_numpy(float))})
        mean=df.groupby(['target','evaluation','parent_ik','gold'],sort=False)['prediction'].mean().reset_index()
        for (target,ev),sub in mean.groupby(['target','evaluation']):
            meanrows.append({'encoder':encoder,'target':target,'evaluation':ev,'n':len(sub),**metric(sub.gold.to_numpy(int),sub.prediction.to_numpy(float))})
        for ev in EVALS:
            sub=mean[mean.evaluation.eq(ev)];wide=sub.pivot(index='parent_ik',columns='target',values='prediction')
            if 'H' not in wide or 'I' not in wide:continue
            ygold=sub[sub.target.eq('H')].set_index('parent_ik').gold.loc[wide.index].to_numpy(int)
            for t in [x for x in TARGET_MAIN if x in wide and x!='H']:
                a=wide['H'].to_numpy(float);b=wide[t].to_numpy(float)
                if np.isnan(a).any() or np.isnan(b).any():continue
                r1=pd.Series(a).rank(method='average',ascending=False).to_numpy();r2=pd.Series(b).rank(method='average',ascending=False).to_numpy()
                k10=max(1,int(math.ceil(.1*len(a)))); k20=max(1,int(math.ceil(.2*len(a))))
                setA=set(np.argsort(-a,kind='stable')[:k10]);setB=set(np.argsort(-b,kind='stable')[:k10]);
                setA20=set(np.argsort(-a,kind='stable')[:k20]);setB20=set(np.argsort(-b,kind='stable')[:k20])
                rankrows.append({'encoder':encoder,'evaluation':ev,'comparison':'H_vs_'+t,'n':len(a),
                       'prediction_MAD':float(np.mean(abs(a-b))), 'score_spearman':float(spearmanr(a,b).statistic),
                       'mean_abs_rank_shift':float(np.mean(abs(r1-r2))),
                       'top10_overlap':len(setA&setB),'top10_size':k10,'top20_overlap':len(setA20&setB20),'top20_size':k20,
                       'top10_membership_changed_unique':len(setA^setB)})
        per,missP=all_predictions(run,encoder,'perm', [f'P_{i:03d}' for i in range(0,99)],PERM_SEEDS[encoder])
        missing[encoder+'_perm']=missP
        if per is None or missP or any(f'I__seed{s}.csv' in miss for s in PERM_SEEDS[encoder]):continue
        if per.duplicated(['target','seed','evaluation','parent_ik']).any():die(f'Duplicate permutation predictions: {encoder}')
        
        i_sub=df[df.target.eq('I') & df.seed.isin(PERM_SEEDS[encoder])]
        for ev in EVALS:
            i_eval=i_sub[i_sub.evaluation.eq(ev)]
            
            i_mean=i_eval.groupby(['parent_ik','gold'])['prediction'].mean().reset_index()
            i_stat=metric(i_mean.gold.to_numpy(int),i_mean.prediction.to_numpy(float))
            null=[]
            for n in range(0,99):
                name=f'P_{n:03d}';sub=per[(per.target.eq(name))&(per.evaluation.eq(ev))]
                vv=sub.groupby(['parent_ik','gold'])['prediction'].mean().reset_index()
                if not np.array_equal(i_mean.parent_ik.to_numpy(),vv.parent_ik.to_numpy()) or not np.array_equal(i_mean.gold.to_numpy(),vv.gold.to_numpy()):die('Permutation evaluation parent set changed')
                mm=metric(vv.gold.to_numpy(int),vv.prediction.to_numpy(float))
                null.append({'replicate':n,**mm})
            for met in ('auroc','auprc'):
                z=np.array([r[met] for r in null]); obs=i_stat[met]
                p_one=(1+int((z>=obs-1e-12).sum()))/(len(z)+1)
                permtest.append({'encoder':encoder,'evaluation':ev,'metric':met,'observed_I':obs,
                    'null_mean':float(z.mean()),'null_sd':float(z.std(ddof=1)),
                    'null_q025':float(np.quantile(z,.025)),'null_q975':float(np.quantile(z,.975)),
                    'p_one_sided_I_higher':p_one,'n_permutations':99,'paired_seeds':','.join(map(str,PERM_SEEDS[encoder]))})
                for r in null:pairrows.append({'encoder':encoder,'evaluation':ev,'metric':met,'replicate':r['replicate'],'value':r[met]})
    out=run/'analysis';out.mkdir(exist_ok=True)
    for name,rows in [('metrics_by_seed.csv',allmetric),('metrics_from_seed_mean.csv',meanrows),
         ('rank_and_topk_comparisons.csv',rankrows),('conditional_permutation_summary.csv',permtest),
         ('conditional_permutation_null_distribution.csv',pairrows)]:
        if rows:pd.DataFrame(rows).to_csv(out/name,index=False)
    write_json(out/'coverage.json',{'status':'PARTIAL' if any(missing.values()) else 'COMPLETE',
         'missing_counts':{k:len(v) for k,v in missing.items()},
         'examples':{k:v[:8] for k,v in missing.items()},
         'permutation_test_complete_encoders':sorted(set(r['encoder'] for r in permtest)),
         'note':'Permutation P-values conditional on selected within-stratum randomizations and matched seeds; no causal proof.'})
    print('[ANALYSIS]', 'metrics',len(meanrows),'perm-inference',len(permtest), 'missing', {k:len(v) for k,v in missing.items()},flush=True)


def pack(run):
    from zipfile import ZipFile,ZIP_DEFLATED
    sources=['run_config.json','precheck.json','analysis/coverage.json','analysis/metrics_by_seed.csv','analysis/metrics_from_seed_mean.csv',
             'analysis/rank_and_topk_comparisons.csv','analysis/conditional_permutation_summary.csv',
             'analysis/conditional_permutation_null_distribution.csv',
             'pampa/pampa_mean_seed_metrics.csv','pampa/pampa_provenance.json']
    for enc in ('xgb','gine'):
        f=run/'predictions'/enc/'main'
        if f.exists():
            for p in f.glob('*.json'):sources.append(str(p.relative_to(run)))
    archive=run/'BBB_STAGE3_REPORT_TO_SEND.zip'
    with ZipFile(archive,'w',compression=ZIP_DEFLATED) as z:
        for name in sources:
            path=run/name
            if path.exists():z.write(path,arcname=name)
    print('[REPORT]',archive,archive.stat().st_size,'bytes',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['precheck','train','pampa','summarize','pack'])
    ap.add_argument('--run',required=True)
    ap.add_argument('--stage2',default='')
    ap.add_argument('--consensus',default=str(FROZEN/'consensus_dataset.csv'))
    ap.add_argument('--labels',default=str(FROZEN/'labels_abc.csv'))
    ap.add_argument('--encoder',choices=['xgb','gine'],default='xgb')
    ap.add_argument('--group',choices=['main','perm'],default='main')
    ap.add_argument('--n-jobs',type=int,default=4)
    ap.add_argument('--perm-start',type=int,default=0)
    ap.add_argument('--perm-stop',type=int,default=98)
    ap.add_argument('--scaffold',default='')
    ap.add_argument('--qualified-external',default='')
    args=ap.parse_args();run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    try:
        if args.action=='precheck':
            if (run/'PASS_PRECHECK.txt').exists():die('Precheck already done in this folder. Choose another timestamped run directory.')
            stage2=resolve_s2(args.stage2)
            source=get_sources(stage2,args.consensus,args.labels)
            write_json(run/'run_config.json',{'protocol':'BBB_Stage3_controlled_v1','source':source,
                'xgb_seeds':XGB_SEEDS,'gine_seeds':GINE_SEEDS,'perm_seeds':PERM_SEEDS,
                'target_names':TARGET_MAIN,'params_xgb':PARAMS,'env_python':platform.python_version(),
                'stage3_source_sha256':sha256(__file__),
                'gine_legacy_sha256':sha256(Path(__file__).with_name('legacy_gine_cpu.py'))})
            check_inputs(source,run)
        else:
            if not (run/'PASS_PRECHECK.txt').exists():die('PASS_PRECHECK marker absent. Refusing to train.')
            if args.action=='train':do_train(args,run)
            elif args.action=='pampa':pampa_eval(args,run)
            elif args.action=='summarize':summarize(args,run)
            elif args.action=='pack':pack(run)
    except Exception as ex:
        print('[STAGE3 ERROR]',str(ex),file=sys.stderr,flush=True)
        traceback.print_exc()
        raise SystemExit(2)

if __name__=='__main__':main()
