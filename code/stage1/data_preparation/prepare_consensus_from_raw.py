# prepare_consensus_from_raw.py — Builds an auxiliary consensus table from original BBBP and B3DB classifications using standardized parent connectivity, fixed source exclusions and seeded data partitioning; the immutable published frozen consensus remains the authoritative modeling input.






from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")
SREMOVER = SaltRemover.SaltRemover()
SEED=42; TEST_FRAC=.15; VAL_FRAC=.15; REF_OUT_REFS={"R27"}

def std(s):
    try:
        m=Chem.MolFromSmiles(str(s));
        if m is None: return None,None
        m=SREMOVER.StripMol(m); Chem.RemoveStereochemistry(m)
        ik=MolToInchiKey(m); canon=Chem.MolToSmiles(m,isomericSmiles=False,canonical=True)
        return canon, ik.split("-")[0] if ik else None
    except Exception: return None,None

def refs(x):
    return sorted({z.strip() for z in str(x).split("|") if z.strip()})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(Path(__file__).resolve().parent)); ap.add_argument('--non-strict',action='store_true'); a=ap.parse_args()
    root=Path(a.root); bbbp=pd.read_csv(root/'data/raw/BBBP/BBBP.csv',dtype=str).fillna(''); b3=pd.read_csv(root/'data/raw/B3DB/B3DB_classification.tsv',sep='\t',dtype=str).fillna('')
    bbbp[['canon','parent_ik']]=pd.DataFrame([std(x) for x in bbbp.smiles],index=bbbp.index)
    b3[['canon','parent_ik']]=pd.DataFrame([std(x) for x in b3.SMILES],index=b3.index)
    bbbp['label']=pd.to_numeric(bbbp.p_np,errors='coerce'); b3['label']=b3['BBB+/BBB-'].str.strip().map({'BBB+':1,'BBB-':0})
    b3rows=[]
    for ik,g in b3.dropna(subset=['parent_ik','label']).groupby('parent_ik',sort=False):
        labs=sorted(set(g.label.astype(int))); allrefs=sorted({r for x in g.get('reference',pd.Series('',index=g.index)) for r in refs(x)})
        b3rows.append({'parent_ik':ik,'canon_smiles_b3db':g.canon.dropna().iloc[0], 'b3db_raw_labels':';'.join(map(str,labs)), 'b3db_label':int(g.label.iloc[0]), 'b3db_label_conflict':int(len(labs)>1), 'reference_list':'|'.join(allrefs), 'n_references':len(allrefs)})
    bbrows=[]
    for ik,g in bbbp.dropna(subset=['parent_ik','label']).groupby('parent_ik',sort=False):
        labs=sorted(set(g.label.astype(int))); bbrows.append({'parent_ik':ik,'canon_smiles_bbbp':g.canon.dropna().iloc[0], 'bbbp_raw_labels':';'.join(map(str,labs)), 'bbbp_label':int(g.label.iloc[0]), 'bbbp_label_conflict':int(len(labs)>1)})
    m=pd.DataFrame(bbrows).merge(pd.DataFrame(b3rows),on='parent_ik',how='outer')
    m['canon_smiles']=m['canon_smiles_bbbp'].fillna(m['canon_smiles_b3db']); m['reference_list']=m['reference_list'].fillna(''); m['n_references']=m['n_references'].fillna(0).astype(int)
    for c in ['bbbp_label','b3db_label']: m[c]=m[c].fillna('')
    m['overlap']=(m.bbbp_label.ne('')&m.b3db_label.ne('')).astype(int)
    m['source']=np.where(m.overlap.eq(1),'overlap',np.where(m.bbbp_label.ne(''),'bbbp_only','b3db_only'))
    m['divergence']=m.apply(lambda r: int(r.bbbp_label!='' and r.b3db_label!='' and int(r.bbbp_label)!=int(r.b3db_label)),axis=1)
    rng=np.random.RandomState(SEED); m['split']=''; m['in_reference_out']=0
    idx=m.index[m.bbbp_label.ne('')].to_numpy().copy(); rng.shuffle(idx); ntest=int(round(len(idx)*TEST_FRAC)); m.loc[idx[:ntest],'split']='test'
    m['in_reference_out']=m.reference_list.map(lambda x:int(bool(set(str(x).split('|'))&REF_OUT_REFS)))
    pool=m.index[(m.split!='test')&m.in_reference_out.eq(0)].to_numpy().copy(); rng.shuffle(pool); nval=int(round(len(pool)*VAL_FRAC/(1-TEST_FRAC))); m.loc[pool[:nval],'split']='validation'; m.loc[pool[nval:],'split']='train'
    out=root/'data/frozen/consensus_dataset.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Frozen source inputs cannot be overwritten by an auxiliary reconstruction')
    m.to_csv(out,index=False,encoding='utf-8-sig')
    report={'total':len(m),'test':int(m.split.eq('test').sum()),'train':int(m.split.eq('train').sum()),'validation':int(m.split.eq('validation').sum()),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    if not a.non_strict:
        assert report['total']==4131, report; assert report['test']==291, report; assert report['train']==1234, report
    print(json.dumps(report,ensure_ascii=False,indent=2)); print(out)
if __name__=='__main__': main()
