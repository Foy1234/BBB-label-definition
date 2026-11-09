# prepare_ncats_audit_from_raw.py — Reconstructs an auxiliary NCATS PAMPA identity and eligibility audit by joining original PubChem assay observations and compound properties, standardizing connectivity, detecting duplicate CIDs and comparing structures against frozen consensus identities.





from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey
RDLogger.DisableLog('rdApp.*')

def norm(s):
    out={'canonical_smiles':None,'full_inchikey':None,'parent_smiles':None,'parent_inchikey':None,'connectivity_key':None,'had_multiple_fragments':False,'normalization_ok':False}
    try:
        m=Chem.MolFromSmiles(str(s));
        if m is None:return out
        out['had_multiple_fragments']=len(Chem.GetMolFrags(m))>1; out['canonical_smiles']=Chem.MolToSmiles(m,isomericSmiles=False); out['full_inchikey']=MolToInchiKey(m)
        p=rdMolStandardize.FragmentParent(m)
        try:p=rdMolStandardize.ChargeParent(p)
        except Exception:pass
        out['parent_smiles']=Chem.MolToSmiles(p,isomericSmiles=False); out['parent_inchikey']=MolToInchiKey(p); out['connectivity_key']=out['parent_inchikey'].split('-')[0]; out['normalization_ok']=True
    except Exception:pass
    return out

def nf(series,prefix): return pd.DataFrame([norm(x) for x in series]).add_prefix(prefix)

def choose_smiles(df):
    for c in ['SMILES','CanonicalSMILES','ConnectivitySMILES','IsomericSMILES']:
        if c in df.columns:return c
    raise ValueError('No SMILES-like column found in NCATS assay/properties merge')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default=str(Path(__file__).resolve().parent)); ap.add_argument('--non-strict',action='store_true'); a=ap.parse_args(); root=Path(a.root)
    assay_path=root/'data/raw/NCATS_PAMPA/NCATS_PAMPA_BBB_AID1845228.csv'; prop_path=root/'data/raw/NCATS_PAMPA/NCATS_PAMPA_BBB_AID1845228_properties.csv'; cons_path=root/'data/frozen/consensus_dataset.csv'
    assay_raw=pd.read_csv(assay_path,dtype=str); assay=assay_raw[pd.to_numeric(assay_raw['PUBCHEM_CID'],errors='coerce').notna()].copy(); assay['PUBCHEM_CID']=pd.to_numeric(assay.PUBCHEM_CID).astype(int); assay['Phenotype']=assay.Phenotype.str.strip(); assay['permeability_numeric']=pd.to_numeric(assay.Permeability.str.replace('>','',regex=False).str.strip(),errors='coerce'); assay['exact_10']=assay.permeability_numeric.eq(10); assay['y_pubchem']=assay.Phenotype.isin(['Moderate','High']).astype(int)
    props=pd.read_csv(prop_path,dtype=str); cidcol='CID' if 'CID' in props.columns else 'PUBCHEM_CID'; props[cidcol]=pd.to_numeric(props[cidcol],errors='coerce').astype('Int64'); assay=assay.merge(props,left_on='PUBCHEM_CID',right_on=cidcol,how='left',suffixes=('','_prop'))
    smi=choose_smiles(assay); assay=pd.concat([assay.reset_index(drop=True),nf(assay[smi],'ncats_')],axis=1)
    cons=pd.read_csv(cons_path,dtype=str).fillna(''); cons2=pd.concat([cons.reset_index(drop=True),nf(cons.canon_smiles,'cons_')],axis=1); cset=set(cons2.cons_connectivity_key.dropna())
    assay['exclude_any_consensus_overlap']=assay.ncats_connectivity_key.isin(cset)
    
    rows=[]
    for cid,g in assay.groupby('PUBCHEM_CID',sort=False):
        ph=g.Phenotype.astype(str).tolist(); pv=g.permeability_numeric.astype(float).tolist(); conflict=len(set(ph))>1 or len(set(pv))>1
        r=g.iloc[0].copy(); r['ambiguous_duplicate']=conflict; r['duplicate_measurements']='|'.join(f'{p}:{v:g}' for p,v in zip(ph,pv)); rows.append(r)
    u=pd.DataFrame(rows).reset_index(drop=True); u['eligible_external']=~u.exclude_any_consensus_overlap.astype(bool)&~u.ambiguous_duplicate.astype(bool)
    out=root/'data/frozen/ncats_unique_molecule_audit.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('Frozen source inputs cannot be overwritten by an auxiliary reconstruction')
    u.to_csv(out,index=False,encoding='utf-8-sig')
    rep={'assay_rows':len(assay),'unique_cids':u.PUBCHEM_CID.nunique(),'eligible_external':int(u.eligible_external.sum()),'ambiguous_duplicate_cids':u.loc[u.ambiguous_duplicate.astype(bool),'PUBCHEM_CID'].astype(int).tolist()}
    if not a.non_strict:
        assert len(assay)==438,rep; assert u.PUBCHEM_CID.nunique()==437,rep; assert rep['eligible_external']==318,rep; assert rep['ambiguous_duplicate_cids']==[457193],rep
    print(json.dumps(rep,ensure_ascii=False,indent=2)); print(out)
if __name__=='__main__':main()
