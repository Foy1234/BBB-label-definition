# fit_corrected_targets.py — Fits molecule-balanced, regularized multi-channel evidence targets from frozen B3DB, BBBP and logBB observations; generates Hard, Vote, Integrated, smoothing and shrinkage sensitivity targets together with 99 conditional permutations and optimization diagnostics.





from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

CONS_SHA = '51d09eea4e6a1659ea7820a93f5f63880161645c3bab8f7f9f93007bdbc9849a'
LAB_SHA = 'ea6b31d500e1f04384952a096fabdf4e9ce3e0bd9533587442c542946f432c89'
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'data/frozen'
DEFAULT_LAMBDAS=(0.01,0.1,1.0)
COMMON_N=1157
SEED_P=20261011
EXPECTED_COVERAGE={'bbbp_n':18,'logbb_n':186,'b3db_distinct_state_n_gt1':8,'b3db_conflict_n':8,
                   'b3db_distinct_state_entries':1165,'logbb_effective_rows':200}
STDS = {'delta':2.0,'a':2.0,'b':2.0,'logsigma':2.0}



GLOBAL_BOUNDS=[(-4.0,4.0), (0.05,4.0), (-4.0,4.0), (0.0,0.0)]


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def sigmoid(x):
    x=np.asarray(x,dtype=float)
    return np.exp(-np.logaddexp(0.,-x))


def as_num(x):
    try:
        s=str(x).strip()
        if not s:return np.nan
        v=float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError,ValueError):return np.nan


def get_b3db(row):
    raw=str(row['b3db_raw_labels']).strip()
    if raw:
        tokens=[as_num(v) for v in raw.split(';')]
        nums=[float(v) for v in tokens if np.isfinite(v) and v in (0.,1.)]
        if nums:return nums
    single=as_num(row['b3db_label'])
    return [float(single)] if single in (0.,1.) else []


def get_logbb(row):
    return [float(n) for n in (as_num(x) for x in str(row['b3db_logbb_list']).split(';')) if np.isfinite(n)]


def collect(df, shuffled=False, rng=None):
    n=len(df)
    has_b=np.zeros(n,bool);qb=np.zeros(n);nb=np.zeros(n,int)
    has_p=np.zeros(n,bool);qp=np.zeros(n)
    has_z=np.zeros(n,bool); nz=np.zeros(n,int)
    raw_z=[]; raw_b=[]; z_rows=[]
    for i,r in enumerate(df.to_dict('records')):
        b=get_b3db(r); z=get_logbb(r)
        if shuffled:
            b=list(rng.permutation(b)) if len(b)>1 else b
            z=list(rng.permutation(z)) if len(z)>1 else z
        raw_b.append(b); raw_z.append(z)
        if b:
            has_b[i]=True; qb[i]=float(np.mean(b)); nb[i]=len(b)
        p=as_num(r['bbbp_label'])
        if p in (0.,1.): has_p[i]=True;qp[i]=p
        if z:
            has_z[i]=True;nz[i]=len(z);z_rows.append((i,z))
    if not np.any(has_z):
        raise ValueError('No training-side logBB observations; expected a three-channel model.')
    if not np.any(has_p):
        raise ValueError('No training-side BBBP observations; delta unidentifiable.')
    
    z_means=[float(np.mean(z)) for _,z in z_rows]
    z_location=float(np.mean(z_means))
    z_scale=float(np.std(z_means,ddof=0))
    if not (np.isfinite(z_scale) and z_scale>1e-8):raise ValueError('logBB scale is degenerate')
    zmean=np.zeros(n);zsq=np.zeros(n)
    for i,z in z_rows:
        zv=(np.asarray(z,dtype=float)-z_location)/z_scale
        zmean[i]=float(np.mean(zv))
        zsq[i]=float(np.mean(zv*zv))
    return dict(has_b=has_b,qb=qb,nb=nb,has_p=has_p,qp=qp,
        has_z=has_z,nz=nz,zmean=zmean,zsq=zsq,
        z_location=z_location,z_scale=z_scale,raw_b=raw_b,raw_z=raw_z)


class MultiChannel:
    
    def __init__(self,evidence,lam):
        self.ev=evidence;self.lam=float(lam)
        self.n=len(evidence['qb'])
        assert self.lam>0

    def penalty(self,params):
        delta,a,b,ls=params
        return .5*(delta/STDS['delta'])**2 + .5*((a-1)/STDS['a'])**2 + .5*(b/STDS['b'])**2 + .5*(ls/STDS['logsigma'])**2

    def value(self,L,params):
        e=self.ev;d,a,b,ls=params
        if L.shape!=(self.n,):raise ValueError('Invalid latent shape')
        val=.5*self.lam*np.dot(L,L)
        be=e['has_b']; pe=e['has_p']; ze=e['has_z']
        val+=np.sum(np.logaddexp(0,L[be])-e['qb'][be]*L[be])
        val+=np.sum(np.logaddexp(0,L[pe]+d)-e['qp'][pe]*(L[pe]+d))
        p=a*L[ze]+b
        sse=e['zsq'][ze]-2*p*e['zmean'][ze]+p*p
        val+=np.sum(.5*np.exp(-2*ls)*sse+ls)
        return float(val+self.penalty(params))

    def grad_latent_hess(self,L,params):
        e=self.ev;d,a,b,ls=params
        g=self.lam*L.copy();h=np.full(self.n,self.lam)
        be=e['has_b'];pe=e['has_p'];ze=e['has_z']
        sb=sigmoid(L[be]);g[be]+=sb-e['qb'][be];h[be]+=sb*(1-sb)
        sp=sigmoid(L[pe]+d);g[pe]+=sp-e['qp'][pe];h[pe]+=sp*(1-sp)
        sc=np.exp(-2*ls)
        g[ze]+=sc*a*(a*L[ze]+b-e['zmean'][ze]);h[ze]+=sc*a*a
        return g,h

    def global_value_jac(self,params,L):
        e=self.ev;d,a,b,ls=params; ze=e['has_z'];pe=e['has_p']
        sp=sigmoid(L[pe]+d)
        pv=a*L[ze]+b
        residual=pv-e['zmean'][ze]
        sse=e['zsq'][ze]-2*pv*e['zmean'][ze]+pv*pv
        inv=np.exp(-2*ls)
        
        v=np.sum(np.logaddexp(0,L[pe]+d)-e['qp'][pe]*(L[pe]+d))
        v+=np.sum(.5*inv*sse+ls)
        v+=self.penalty(params)
        g=np.array([
            np.sum(sp-e['qp'][pe])+d/STDS['delta']**2,
            inv*np.dot(residual,L[ze])+(a-1)/STDS['a']**2,
            inv*np.sum(residual)+b/STDS['b']**2,
            np.sum(1-inv*sse)+ls/STDS['logsigma']**2,
        ],dtype=float)
        return float(v),g

    def optimize_latent(self,L,params,max_steps=120):
        
        L=np.asarray(L,float).copy()
        for _ in range(max_steps):
            g,h=self.grad_latent_hess(L,params)
            step=np.clip(g/h,-3.,3.)
            nxt=np.clip(L-step,-12.,12.)
            if np.max(np.abs(nxt-L))<1e-12:
                L=nxt;break
            L=nxt
        gg,_=self.grad_latent_hess(L,params)
        return L,float(np.max(np.abs(gg)))

    def fit(self,max_outer=80,trace=False):
        




        L=np.zeros(self.n,float)
        par=np.array([0.,1.,0.,0.],float)
        history=[];last=np.inf
        for it in range(1,max_outer+1):
            prevL=L.copy();prevpar=par.copy()
            L,local_grad=self.optimize_latent(L,par)
            r=minimize(lambda p:self.global_value_jac(p,L),par,method='L-BFGS-B',
                       jac=True,bounds=GLOBAL_BOUNDS,
                       options={'maxiter':500,'ftol':1e-13,'gtol':1e-9,'maxls':60})
            if not r.success and r.jac is not None and np.max(np.abs(r.jac))>1e-5:
                raise RuntimeError(f'Global optimizer failed on lambda={self.lam}: {r.message}')
            par=np.asarray(r.x,float)
            L,local_grad=self.optimize_latent(L,par)
            val=self.value(L,par)
            rel=abs(val-last)/(1+abs(last)) if np.isfinite(last) else np.inf
            deltaL=float(np.max(np.abs(L-prevL)))
            deltap=float(np.max(np.abs(par-prevpar)))
            if np.isfinite(last) and val>last+1e-7*(1+abs(last)):
                raise RuntimeError(f'Composite objective increased unexpectedly: {last} => {val}')
            history.append({'iteration':it,'objective':val,'rel_obj_change':rel,
                            'max_latent_change':deltaL,'max_global_change':deltap,
                            'max_latent_gradient':local_grad,'stage':'alternating'})
            if trace and (it==1 or it%10==0):
                print(f'    alt_iter={it:03d} obj={val:.8f} rel={rel:.2g} dL={deltaL:.2g} dpar={deltap:.2g}',flush=True)
            if rel<=1e-10 and deltaL<=1e-6 and deltap<=1e-6 and local_grad<=1e-6:
                break
            last=val
        def whole(x):
            latent=x[:self.n];params=x[self.n:]
            val=self.value(latent,params)
            gl=self.grad_latent_hess(latent,params)[0]
            gp=self.global_value_jac(params,latent)[1]
            return val,np.concatenate([gl,gp])
        x0=np.concatenate([L,par])
        bounds=[(-12.,12.)]*self.n+GLOBAL_BOUNDS
        sol=minimize(whole,x0,method='L-BFGS-B',jac=True,bounds=bounds,
                     options={'maxiter':6000,'maxls':80,'maxcor':30,'ftol':1e-13,'gtol':1e-7})
        final=np.asarray(sol.x,float);L=final[:self.n];par=final[self.n:]
        fin_val=self.value(L,par)
        if fin_val>history[-1]['objective']+1e-6:
            raise RuntimeError(f'Joint optimizer worsened objective: {fin_val} > {history[-1]["objective"]}')
        g=self.grad_latent_hess(L,par)[0]
        unconstrained_latent_grad=float(np.max(np.abs(g)))
        
        
        gradient=np.asarray(whole(final)[1]);pg=gradient.copy()
        for j,(lower,upper) in enumerate(bounds):
            if (final[j]<=lower+1e-7 and pg[j]>0) or (final[j]>=upper-1e-7 and pg[j]<0):pg[j]=0.
        kkt=float(np.max(np.abs(pg)))
        conv=bool(np.isfinite(fin_val) and sol.success and kkt<=5e-4 and unconstrained_latent_grad<=5e-4)
        history.append({'iteration':len(history)+1,'objective':fin_val,
                        'rel_obj_change':abs(history[-1]['objective']-fin_val)/(1+abs(fin_val)),
                        'max_latent_change':float(np.max(np.abs(L-x0[:self.n]))),
                        'max_global_change':float(np.max(np.abs(par-x0[self.n:]))),
                        'max_latent_gradient':unconstrained_latent_grad,
                        'stage':'joint_polish'})
        if trace:
            print(f'    joint polish: success={sol.success} nit={sol.nit} '
                  f'objective={fin_val:.8f} KKT={kkt:.3g}',flush=True)
        return {'L':L,'params':par,'target':sigmoid(L),
                'history':history,'converged':conv,'iterations':len(history),
                'final_objective':float(fin_val),
                'max_latent_gradient':unconstrained_latent_grad,
                'projected_gradient_max':kkt, 'joint_solver_message':str(sol.message)}


def finite_difference(fun,v,j,step=1e-6):
    x=v.copy();x[j]+=step; f1=fun(x)
    x[j]-=2*step;f0=fun(x)
    return (f1-f0)/(2*step)


def gradient_check(model,fit):
    e=model.ev;n=model.n; L=fit['L'].copy(); p=fit['params'].copy()
    candidates=[]
    for mask in [e['has_b'] & ~e['has_z'] & ~e['has_p'],e['has_z'] & ~e['has_p'],e['has_p']]:
        a=np.flatnonzero(mask)
        if len(a):candidates.append(int(a[0]))
    if not candidates:raise RuntimeError('No representative molecule for gradient check')
    
    rng=np.random.default_rng(601);L+=rng.normal(0,.07,len(L));p=np.array([.14,1.2,.15,.05])
    g,h=model.grad_latent_hess(L,p);errs=[]
    for j in candidates:
        fd=finite_difference(lambda arr:model.value(arr,p),L,j)
        errs.append({'kind':'latent_gradient','row_index':j,'analytic':float(g[j]),'numeric':float(fd), 'abs_error':abs(g[j]-fd)})
        hfd=(model.grad_latent_hess(L+np.eye(1,n,j).flatten()*1e-5,p)[0][j]-model.grad_latent_hess(L-np.eye(1,n,j).flatten()*1e-5,p)[0][j])/(2e-5)
        errs.append({'kind':'latent_hessian','row_index':j,'analytic':float(h[j]),'numeric':float(hfd),'abs_error':abs(h[j]-hfd)})
    ag=model.global_value_jac(p,L)[1]
    for j in range(4):
        fd=finite_difference(lambda x:model.global_value_jac(x,L)[0],p,j)
        errs.append({'kind':'global_gradient','parameter':['delta','a','b','log_sigma'][j],'analytic':float(ag[j]),'numeric':float(fd),'abs_error':abs(ag[j]-fd)})
    error=max(x['abs_error'] for x in errs)
    return {'pass':bool(error<1e-5),'max_abs_error':float(error),'checks':errs}


def load_frozen(consensus,labels,strict_hash=True):
    hc=sha256(consensus); hl=sha256(labels)
    if strict_hash and (hc!=CONS_SHA or hl!=LAB_SHA):
        raise RuntimeError('Frozen file SHA256 mismatch: refusing to fit on possibly different inputs. Check exact selected paths.')
    con=pd.read_csv(consensus,dtype=str,keep_default_na=False,encoding='utf-8-sig',low_memory=False)
    labs=pd.read_csv(labels,dtype=str,keep_default_na=False,encoding='utf-8-sig',low_memory=False)
    require={'parent_ik','split','in_reference_out','b3db_label','bbbp_label','b3db_raw_labels','b3db_logbb_list'}
    if not require.issubset(con.columns):raise RuntimeError(f'Missing consensus columns {require-set(con.columns)}')
    require_lab={'parent_ik','label_A','label_B','label_C_sigmoid_mu'}
    if not require_lab.issubset(labs.columns):raise RuntimeError(f'Missing label columns {require_lab-set(labs.columns)}')
    ins=pd.to_numeric(con['in_reference_out'],errors='coerce').fillna(0).eq(0)
    tr=con.loc[con['split'].eq('train') & ins].reset_index(drop=True)
    if len(tr)!=len(labs) or not np.array_equal(tr['parent_ik'].to_numpy(),labs['parent_ik'].to_numpy()):
        raise RuntimeError('Frozen Stage-II train pool ID/order mismatch')
    if tr['parent_ik'].duplicated().any() or con['parent_ik'].duplicated().any():raise RuntimeError('Duplicated parent IDs')
    hard=pd.to_numeric(labs['label_A'],errors='coerce');vote=pd.to_numeric(labs['label_B'],errors='coerce')
    if np.any(~np.isfinite(hard.dropna().to_numpy(float))) or not np.isin(hard.dropna().to_numpy(float),[0.,1.]).all():raise RuntimeError('Hard target not binary')
    mask=hard.notna().to_numpy()
    if int(mask.sum())!=COMMON_N:raise RuntimeError(f'Expected common n=1157, got {int(mask.sum())}')
    if not np.isfinite(vote[mask]).all():raise RuntimeError('Vote has missing values on common mask')
    cv=pd.to_numeric(labs['label_C_sigmoid_mu'],errors='coerce')
    if not np.isfinite(cv[mask]).all():raise RuntimeError('Legacy integrated missing on common mask')
    if (pd.to_numeric(con['in_reference_out'],errors='coerce').fillna(0).eq(1) & con['parent_ik'].isin(tr.loc[mask,'parent_ik'])).any():
        raise RuntimeError('Reference-out parent leakage')
    eval_=con['split'].eq('test')
    if tr.loc[mask,'parent_ik'].isin(con.loc[eval_,'parent_ik']).any():raise RuntimeError('Test overlap')
    if int(eval_.sum())!=291:raise RuntimeError('BBBP test size mismatch')
    cvote=vote[mask].to_numpy(float)
    if not np.all((cvote>=0)&(cvote<=1)):raise RuntimeError('Vote out of range')
    return con,tr,labs,mask,{'consensus_sha256':hc,'labels_sha256':hl,'n_train_pool':len(tr),'n_common_train':int(mask.sum()), 'n_test':int(eval_.sum())}


def strata_permutations(ids,H,I,has_p,has_z,B=99,seed=SEED_P):
    groups={}
    for i,(h,p,z) in enumerate(zip(H,has_p,has_z)):
        k=f'H{int(h)}_BBBP{int(p)}_LOGBB{int(z)}'
        groups.setdefault(k,[]).append(i)
    rng=np.random.default_rng(seed)
    permuted=np.empty((len(ids),B),dtype=float)
    diagnostics=[]
    movable=0
    for key,idxlist in sorted(groups.items()):
        ii=np.asarray(idxlist,dtype=int)
        uniq=int(len(np.unique(np.round(I[ii],12))))
        if len(ii)>=2 and uniq>1: movable+=len(ii)
        diagnostics.append({'stratum':key,'n':int(len(ii)),'n_unique_values':uniq,
                            'potentially_swappable':bool(len(ii)>=2 and uniq>1)})
    for b in range(B):
        p=I.copy()
        for key,idxlist in groups.items():
            ix=np.asarray(idxlist,int)
            if len(ix)>=2:
                p[ix]=rng.permutation(I[ix])
                if not np.array_equal(np.sort(p[ix]),np.sort(I[ix])):
                    raise RuntimeError('Failed per-stratum exact multiset preservation')
        permuted[:,b]=p
    changed=np.sum(np.abs(permuted-I[:,None])>1e-12,axis=0)
    if np.min(changed)==0:raise RuntimeError('At least one conditional permutation is identity; abort')
    if movable<10:raise RuntimeError(f'Too few permutable molecules: {movable}')
    return permuted,{'strata':diagnostics,'n_movable':movable,'min_n_changed':int(changed.min()),
                     'max_n_changed':int(changed.max()),'mean_n_changed':float(changed.mean()),
                     'n_permutations':B,'seed':seed}


def summary_stat(name,lam,v,H):
    v=np.asarray(v,float)
    return {'target':name,'lambda':lam,'n':len(v),'mean':float(v.mean()),
       'sd':float(v.std(ddof=1)), 'min':float(v.min()),'q05':float(np.quantile(v,.05)),
       'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'max':float(v.max()),
       'mean_abs_vs_hard':float(np.mean(np.abs(v-H))),
       'n_nonbinary':int(np.sum((v>1e-10)&(v<1-1e-10)))}


def write_json(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main(args):
    out=Path(args.outdir).expanduser().resolve()
    
    if out.exists() and any(x.name!='run.log' for x in out.iterdir()):
        raise RuntimeError('Output directory contains prior result files; refusing overwrite')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'session.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
               'consensus':str(args.consensus),'labels':str(args.labels),
               'model':'molecule-balanced composite likelihood v2; q_B from frozen original-row Vote; logBB standardized-noise-sigma FIXED=1',
               'permutation_seed':SEED_P,'lambda_values':list(DEFAULT_LAMBDAS),
               'status':'STARTED'})
    con,tr,labs,mask,guard=load_frozen(args.consensus,args.labels,strict_hash=not args.allow_hash_mismatch)
    train=tr.loc[mask].reset_index(drop=True)
    ids=train['parent_ik'].to_numpy(str)
    hard=pd.to_numeric(labs.loc[mask,'label_A'],errors='raise').to_numpy(float)
    vote=pd.to_numeric(labs.loc[mask,'label_B'],errors='raise').to_numpy(float)
    ev=collect(train)
    if not np.all(ev['has_b']):raise RuntimeError('All common training molecules should have valid B3DB classification')
    
    
    
    
    
    
    
    q_consensus_distinct=ev['qb'].copy()
    support_issue=[]
    for j,(states,q) in enumerate(zip(ev['raw_b'],vote)):
        st=set(states)
        consistent=((st=={0.0} and abs(q)<1e-10) or
                    (st=={1.0} and abs(q-1)<1e-10) or
                    (st=={0.0,1.0} and 0<q<1))
        if not consistent:
            support_issue.append(j)
    if support_issue:
        raise RuntimeError(f'Frozen Vote conflicts with B3DB consensus distinct-state support for {len(support_issue)} molecules; first train row index={support_issue[0]}')
    
    
    
    n_nonbinary=int(np.sum((vote>1e-10)&(vote<1-1e-10)))
    if n_nonbinary != 8:
        raise RuntimeError(f'Expected eight nonbinary frozen Vote targets; got {n_nonbinary}')
    vote_diff=np.abs(vote-q_consensus_distinct)
    vote_audit={'source':'frozen label_B from legacy stage7_main.py vote_soft_label_map() on original B3DB TSV',
                'source_file_sha256_verified':True,
                'b3db_consensus_raw_labels_are_distinct_states_NOT_full_rows':True,
                'n_frozen_vote_nonbinary':n_nonbinary,
                'n_frozen_vote_vs_deduplicated_states_disagree':int(np.sum(vote_diff>1e-9)),
                'max_abs_difference_vs_deduplicated_states':float(np.max(vote_diff)),
                'n_support_contradictions':len(support_issue),
                'raw_B3DB_TSV_independently_rechecked':False,
                'note':'Actual raw classification record counts cannot be inferred from frozen consensus b3db_raw_labels. No duplicate-row weight is used.'}
    ev['qb']=vote.copy()
    coverage={'b3db_n':int(np.sum(ev['has_b'])), 'bbbp_n':int(np.sum(ev['has_p'])),
              'logbb_n':int(np.sum(ev['has_z'])), 'b3db_distinct_state_n_gt1':int(np.sum(ev['nb']>1)),
              'b3db_conflict_n':int(np.sum((ev['qb']>0)&(ev['qb']<1))),
              'b3db_distinct_state_entries':int(np.sum(ev['nb'])),'logbb_effective_rows':int(np.sum(ev['nz'])),
              'standardization_logbb_mean':ev['z_location'],
              'standardization_logbb_sd':ev['z_scale']}
    for key,expected in EXPECTED_COVERAGE.items():
        if coverage[key]!=expected:
            raise RuntimeError(f'Channel coverage anchor mismatch: {key}={coverage[key]} expected={expected}; the selected input does not match the locked coverage anchors.')
    print('[frozen] n_pool',guard['n_train_pool'],'common',guard['n_common_train'], 'coverage',coverage,flush=True)
    fit_by_lam={};stats=[];diagnostics=[];grad_report=None
    for lam in DEFAULT_LAMBDAS:
        print(f'[fit] lambda={lam:g}',flush=True)
        model=MultiChannel(ev,lam); f=model.fit(trace=True);fit_by_lam[lam]=f
        if not f['converged']:raise RuntimeError(f'lambda={lam:g} did not meet convergence criteria in {f["iterations"]} cycles')
        if grad_report is None and lam==0.1:
            grad_report=gradient_check(model,f)
            if not grad_report['pass']:raise RuntimeError(f'Numerical gradient verification failed: {grad_report["max_abs_error"]}')
        v=f['target'];stats.append(summary_stat('integrated',lam,v,hard))
        diagnostics.append({'lambda':lam,'iterations':f['iterations'],'objective':f['final_objective'],
          'projected_grad_max':f['projected_gradient_max'],
          'max_latent_gradient':f['max_latent_gradient'], 'delta':float(f['params'][0]),
          'a':float(f['params'][1]),'b':float(f['params'][2]),
          'sigma_standardized':float(np.exp(f['params'][3])),
          'converged':f['converged']})
        print(f'    completed in {f["iterations"]} cycles, target_mean={v.mean():.6f}',flush=True)
    
    shuffled=collect(train,shuffled=True,rng=np.random.default_rng(27011))
    
    
    shuffled['qb']=vote.copy()
    order_input_error=max(float(np.max(np.abs(ev[k]-shuffled[k]))) for k in ['qb','qp','zmean','zsq'])
    reordered=MultiChannel(shuffled,0.1).fit(trace=False)
    order_target_error=float(np.max(np.abs(reordered['target']-fit_by_lam[0.1]['target'])))
    order_check={'input_aggregate_max_abs_diff':order_input_error,'refit_target_max_abs_diff':order_target_error,
                 'refit_converged':reordered['converged'],
                 'pass':bool(order_input_error<1e-10 and order_target_error<1e-7 and reordered['converged'])}
    if not order_check['pass']:raise RuntimeError(f'Order invariance failed: {order_check}')
    integ=fit_by_lam[0.1]['target']
    means={int(h):float(integ[hard==h].mean()) for h in [0.,1.]}
    smooth=np.where(hard==0.,means[0],means[1])
    if not all(np.isfinite(x).all() and (x>0).all() and (x<1).all() for x in [integ,smooth]):
        raise RuntimeError('Integrated/Smooth targets out of (0,1)')
    per,per_diag=strata_permutations(ids,hard,integ,ev['has_p'],ev['has_z'],B=99)
    stats+= [summary_stat('hard',None,hard,hard), summary_stat('vote',None,vote,hard),
             summary_stat('class_conditional_smooth',0.1,smooth,hard)]
    
    compatible=pd.DataFrame({'parent_ik':labs['parent_ik'].astype(str),
              'label_A':labs['label_A'].astype(str), 'label_B':labs['label_B'].astype(str),
              'label_C_sigmoid_mu':['']*len(labs),'label_S_smooth':['']*len(labs)})
    maskpos=np.flatnonzero(mask)
    compatible.loc[maskpos,'label_C_sigmoid_mu']=[format(v,'.17g') for v in integ]
    compatible.loc[maskpos,'label_S_smooth']=[format(v,'.17g') for v in smooth]
    if not np.array_equal(compatible['parent_ik'].to_numpy(),tr['parent_ik'].to_numpy()):
        raise RuntimeError('Compatibility export lost frozen pool row ordering')
    compatible.to_csv(out/'labels_abc_corrected_STAGE2_ONLY.csv',index=False,encoding='utf-8-sig')
    local_hessian=MultiChannel(ev,0.1).grad_latent_hess(fit_by_lam[0.1]['L'],fit_by_lam[0.1]['params'])[1]
    sd_local=1/np.sqrt(local_hessian)
    full=pd.DataFrame({'parent_ik':ids,'label_A':hard,'label_B':vote,
                       'label_C_integrated':integ,'label_S_class_smooth':smooth,
                       'has_bbbp':ev['has_p'].astype(int),'has_logbb':ev['has_z'].astype(int),
                       'b3db_n_distinct_states_in_consensus':ev['nb'], 'b3db_positive_fraction_from_frozen_vote':ev['qb'],
                       'latent_local_curvature_sd':sd_local,
                       'logbb_within_sd_std':np.asarray([
                            (np.std((np.asarray(z)-ev['z_location'])/ev['z_scale'],ddof=0) if len(z)>=2 else np.nan)
                            for z in ev['raw_z']],float)})
    for lam,f in fit_by_lam.items():full[f'integrated_lambda_{lam:g}']=f['target']
    full.to_csv(out/'stage2_target_vectors_common1157.csv',index=False,encoding='utf-8-sig')
    permframe=pd.DataFrame({'parent_ik':ids})
    for j in range(per.shape[1]):permframe[f'P_{j:03d}']=per[:,j]
    permframe.to_csv(out/'P_99_permutation_targets_common1157.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(out/'target_distributions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diagnostics).to_csv(out/'fit_parameters.csv',index=False,encoding='utf-8-sig')
    pairs=[]
    for a,b in ((.01,.1),(.01,1.),(.1,1.)):
        aa=fit_by_lam[a]['target'];bb=fit_by_lam[b]['target']
        pairs.append({'lambda_a':a,'lambda_b':b,
          'mean_absolute_target_difference':float(np.mean(np.abs(aa-bb))),
          'max_absolute_target_difference':float(np.max(np.abs(aa-bb))),
          'spearman_target':float(spearmanr(aa,bb).statistic)})
    pd.DataFrame(pairs).to_csv(out/'lambda_pairwise_target_sensitivity.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(fit_by_lam[0.1]['history']).to_csv(out/'optimization_trace_primary.csv',index=False,encoding='utf-8-sig')
    bound_alerts=[]
    for d in diagnostics:
        for k,bnd in [('delta',GLOBAL_BOUNDS[0]),('a',GLOBAL_BOUNDS[1]),('b',GLOBAL_BOUNDS[2])]:
            if min(abs(d[k]-bnd[0]),abs(d[k]-bnd[1]))<1e-4:
                bound_alerts.append(f'lambda={d["lambda"]} parameter {k} is near a bound')
    qc={'status':'PASS_TARGETS_READY_FOR_INDEPENDENT_REVIEW','guard':guard,
        'parameter_boundary_warnings':bound_alerts,
        'fitting_only_common_1157':True,'old_target_preserved_in_input':True,
        'common_train_no_test_evidence':True,'frozen_vote_provenance':vote_audit,
        'coverage':coverage,'n_train_common':len(ids),'order_invariance':order_check,
        'gradient_check':grad_report,'fit_diagnostics':diagnostics,
        'permutation_diagnostics':per_diag,'class_conditional_means':means,
        'permutations_do_not_modify_frozen_inputs':True,
        'details':'Target fitting only. No XGB/GINE, no performance or ranking claims can be drawn from this run.'}
    write_json(out/'qc_report.json',qc)
    with (out/'qc_summary.md').open('w',encoding='utf-8') as f:
        f.write('# Stage II target construction — QC summary\n\n')
        f.write('**PASS: target construction and numerical quality-control checks completed.**\n\n')
        f.write(f"- Source consensus hash: `{guard['consensus_sha256']}`\n- Source labels hash: `{guard['labels_sha256']}`\n")
        f.write(f'- Common training n: {len(ids)}; full train pool n: {len(labs)}\n')
        for k,v in coverage.items():f.write(f'- {k}: {v}\n')
        f.write('- B3DB q_B source: frozen label_B reconstructed from original B3DB_classification.tsv by legacy stage7; original row multiplicities NOT recoverable from consensus raw_labels\n')
        f.write(f"- Frozen Vote vs deduplicated consensus states: {vote_audit['n_frozen_vote_vs_deduplicated_states_disagree']} discordant fractions; max difference {vote_audit['max_abs_difference_vs_deduplicated_states']:.10g}; support contradictions {vote_audit['n_support_contradictions']}\n")
        f.write('- Full original B3DB TSV was not independently re-read in this run; sample count of original duplicate records remains unknown.\n')
        for d in diagnostics:f.write(f"- Lambda={d['lambda']}: {d['iterations']} cycles; objective={d['objective']:.6f}; δ={d['delta']:.5f}, a={d['a']:.5f}, b={d['b']:.5f}, σ={d['sigma_standardized']:.5f}\n")
        f.write(f"- Gradient max error: {grad_report['max_abs_error']:.3g}\n")
        if bound_alerts:f.write('- Parameter-boundary flags: '+ '; '.join(bound_alerts)+'\n')
        f.write(f"- Order invariance max target diff: {order_target_error:.3g}\n")
        f.write(f"- Permutable training molecules: {per_diag['n_movable']}; changed per permutation min/avg/max: {per_diag['min_n_changed']}/{per_diag['mean_n_changed']:.1f}/{per_diag['max_n_changed']}\n")
        f.write('\n## Important interpretation\nThe 99 conditional permutations preserve the exact Integrated target multiset within each frozen (Hard, BBBP-present, logBB-present) stratum. They do not form independent observations and do not test model performance in this stage.\n')
        f.write('The Integrated target is a penalized, molecule-balanced composite-likelihood target, not a physiological BBB probability or an exact Bayesian posterior.\n')
        f.write('logBB values are standardized using only the 1157 train molecules (equal weight by molecule); logBB observation-noise SD is FIXED to 1 on this scale to prevent nonidentifiable variance collapse when most measured molecules have only one observation. The data do not justify interpreting this as an estimated assay noise SD.\n')
        f.write('\n## Generated files\n- labels_abc_corrected_STAGE2_ONLY.csv (1234 pool rows, compatible with downstream masks)\n- stage2_target_vectors_common1157.csv\n- P_99_permutation_targets_common1157.csv\n- target_distributions.csv; fit_parameters.csv; lambda_pairwise_target_sensitivity.csv\n')
        f.write('\nThe original frozen labels_abc.csv remains unchanged; XGBoost/GINE evaluation is performed in the subsequent modeling stage.\n')
    (out/'READY_FOR_REVIEW.txt').write_text('TARGET CONSTRUCTION QC PASSED; NOT A COMPLETED DOWNSTREAM EXPERIMENT\n',encoding='utf-8')
    write_json(out/'session.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
          'status':'PASS_TARGETS_READY_FOR_INDEPENDENT_REVIEW', 'data_input_hashes':guard,
          'model':'molecule-balanced penalized composite likelihood v2; original-row q_B from frozen Vote; standardized logBB noise SD fixed=1',
          'permutation_seed':SEED_P})
    print('[PASS] Saved new targets and numerical QC; no downstream model training performed.',flush=True)
    print('[REPORT] ',out/'qc_summary.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="BBB controlled training-target and label evidence computation")
    parser.add_argument('--consensus',default=str(DEFAULT_ROOT/'consensus_dataset.csv'))
    parser.add_argument('--labels',default=str(DEFAULT_ROOT/'labels_abc.csv'))
    parser.add_argument('--outdir',required=True)
    parser.add_argument('--allow-hash-mismatch',action='store_true',help='Allow execution when locked input hashes differ from the archived hashes')
    a=parser.parse_args()
    try:main(a)
    except Exception as exc:
        o=Path(a.outdir);o.mkdir(parents=True,exist_ok=True)
        (o/'FAILURE.txt').write_text(type(exc).__name__+': '+str(exc)+'\n',encoding='utf-8')
        traceback.print_exc();sys.exit(2)
