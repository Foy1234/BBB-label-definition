# legacy_scaffold_split.py — Builds and audits Bemis–Murcko scaffold partitions using fixed molecular identities for the independent scaffold evaluation set.
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
SEED=42; VAL_FRAC_WITHIN_TRAIN=0.20

def scaffold_of(smi):
    try: return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(str(smi))
    except Exception: return 'INVALID'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--consensus',required=True); ap.add_argument('--output',required=True); args=ap.parse_args()
    df=pd.read_csv(args.consensus,dtype=str).fillna('')
    req=['parent_ik','canon_smiles','split','in_reference_out','bbbp_label','b3db_label','n_references']
    miss=[c for c in req if c not in df.columns]
    if miss: raise SystemExit(f'Missing consensus columns: {miss}')
    keep=df[pd.to_numeric(df['in_reference_out'],errors='coerce').fillna(0).astype(int)==0].copy()
    train_pool=keep[keep['split'].eq('train')].reset_index(drop=True)
    test_pool=keep[~keep['split'].eq('train')].reset_index(drop=True)
    train_pool['scaffold']=train_pool['canon_smiles'].map(scaffold_of)
    test_pool['scaffold']=test_pool['canon_smiles'].map(scaffold_of)
    train_scaffolds=set(train_pool['scaffold'])
    test_mask=~test_pool['scaffold'].isin(train_scaffolds)
    test_assignment=test_pool.loc[test_mask].copy()
    train_scaf_mols=defaultdict(list)
    for i,row in train_pool.iterrows(): train_scaf_mols[row['scaffold']].append(i)
    rng=np.random.RandomState(SEED); scaffolds=sorted(train_scaf_mols); rng.shuffle(scaffolds)
    target=int(round(len(train_pool)*VAL_FRAC_WITHIN_TRAIN)); val_scafs=set(); n=0
    for s in scaffolds:
        if n>=target: break
        val_scafs.add(s); n+=len(train_scaf_mols[s])
    train_pool['scaffold_split']=np.where(train_pool['scaffold'].isin(val_scafs),'validation','train')
    test_assignment['scaffold_split']='test'
    out=pd.concat([train_pool,test_assignment],ignore_index=True)
    cols=['parent_ik','canon_smiles','scaffold','scaffold_split','bbbp_label','b3db_label','n_references']
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out[cols].to_csv(args.output,index=False,encoding='utf-8-sig')
    info={'seed':SEED,'train_pool':len(train_pool),'test_pool_candidates':len(test_pool),'scaffold_test':len(test_assignment),
          'split_counts':out['scaffold_split'].value_counts().to_dict(),
          'parent_overlap_train_test':len(set(out.loc[out.scaffold_split.eq('train'),'parent_ik']) & set(out.loc[out.scaffold_split.eq('test'),'parent_ik'])),
          'scaffold_overlap_train_test':len(set(out.loc[out.scaffold_split.eq('train'),'scaffold']) & set(out.loc[out.scaffold_split.eq('test'),'scaffold']))}
    print(json.dumps(info,ensure_ascii=False,indent=2)); print('SCAFFOLD SPLIT PREPARED:',args.output)
if __name__=='__main__': main()
