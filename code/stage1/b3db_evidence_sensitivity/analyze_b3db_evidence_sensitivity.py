# analyze_b3db_evidence_sensitivity.py — Calculates B3DB source evidence coverage, conflicting classification indicators, molecule-level hard-label prediction sensitivity, and prespecified group comparisons.





























from __future__ import annotations
import argparse, json, math, shutil, zipfile, hashlib
from pathlib import Path
from typing import Optional, Iterable, Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy import stats

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import SaltRemover
    from rdkit.Chem.inchi import MolToInchiKey
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False

PRIMARY_VIEW = "parent_first_consistent"
EXPECTED_VIEWS = [
    "parent_first_all",
    "parent_first_consistent",
    "parent_majority_all",
    "parent_majority_consistent",
    "canonical_first_all",
    "canonical_first_consistent",
    "canonical_majority_all",
    "canonical_majority_consistent",
]
RNG_SEED = 20260929

def parent_inchikey(smiles):
    if not HAVE_RDKIT or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        m = SaltRemover.SaltRemover().StripMol(m)
        Chem.RemoveStereochemistry(m)
        ik = MolToInchiKey(m)
        return ik.split("-")[0] if ik else None
    except Exception:
        return None

def find_path(explicit: Optional[str], candidates: Iterable[Path], what: str, required=True):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"{what} not found: {p}")
    tried = []
    for p in candidates:
        p = p.expanduser().resolve()
        tried.append(str(p))
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(f"Could not locate {what}. Tried:\n" + "\n".join(tried))
    return None

def extract_results(p: Path, out: Path) -> Path:
    if p.is_dir():
        return p
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "r") as z:
        z.extractall(out)
    return out

def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    tmp = np.empty_like(vals)
    tmp[order] = q
    out[np.where(ok)[0]] = tmp
    return out

def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a)==0 or len(b)==0: return np.nan
    
    z = np.concatenate([a,b])
    r = stats.rankdata(z, method="average")
    ra = r[:len(a)].sum()
    u = ra - len(a)*(len(a)+1)/2
    return float((2*u)/(len(a)*len(b)) - 1)

def bootstrap_mean_diff(a, b, n_boot, rng):
    a=np.asarray(a,float); b=np.asarray(b,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)<3 or len(b)<3:
        return np.nan,np.nan,np.nan
    point=float(a.mean()-b.mean())
    vals=np.empty(n_boot,float)
    for i in range(n_boot):
        vals[i]=a[rng.integers(0,len(a),len(a))].mean()-b[rng.integers(0,len(b),len(b))].mean()
    lo,hi=np.quantile(vals,[.025,.975])
    return point,float(lo),float(hi)

def bootstrap_spearman(x,y,n_boot,rng):
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    if len(x)<10 or len(np.unique(x))<2 or len(np.unique(y))<2:
        return np.nan,np.nan,np.nan,np.nan
    rho,p=stats.spearmanr(x,y)
    vals=[]
    n=len(x)
    for _ in range(n_boot):
        ix=rng.integers(0,n,n)
        if len(np.unique(x[ix]))<2 or len(np.unique(y[ix]))<2: continue
        vals.append(stats.spearmanr(x[ix],y[ix]).statistic)
    if len(vals)<100:
        return float(rho),float(p),np.nan,np.nan
    lo,hi=np.quantile(vals,[.025,.975])
    return float(rho),float(p),float(lo),float(hi)

def fisher_enrichment(exposure, outcome):
    x=np.asarray(exposure).astype(bool); y=np.asarray(outcome).astype(bool)
    a=int(np.sum(x&y)); b=int(np.sum(x&~y)); c=int(np.sum(~x&y)); d=int(np.sum(~x&~y))
    OR,p=stats.fisher_exact([[a,b],[c,d]], alternative="two-sided")
    re=a/(a+b) if a+b else np.nan
    ru=c/(c+d) if c+d else np.nan
    rr=re/ru if np.isfinite(re) and np.isfinite(ru) and ru>0 else np.nan
    return dict(exposed_outcome=a,exposed_no_outcome=b,unexposed_outcome=c,unexposed_no_outcome=d,
                risk_exposed=re,risk_unexposed=ru,risk_ratio=rr,odds_ratio=OR,fisher_p=p)

def infer_b3db_raw_counts(b3db_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if b3db_path is None:
        return None
    if not HAVE_RDKIT:
        raise RuntimeError("RDKit required to reconstruct B3DB classification record counts.")
    b3=pd.read_csv(b3db_path, sep="\t", dtype=str)
    b3.columns=[c.strip() for c in b3.columns]
    smi = "SMILES" if "SMILES" in b3.columns else ("smiles" if "smiles" in b3.columns else None)
    if smi is None:
        raise ValueError("B3DB classification file requires SMILES/smiles column.")
    b3["parent_ik"]=b3[smi].map(parent_inchikey)
    g=b3.dropna(subset=["parent_ik"]).groupby("parent_ik").size().rename("b3db_class_raw_row_count")
    return g.reset_index()

def numeric_consensus(df, col, default=0):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v3-results", default=None)
    ap.add_argument("--consensus", default=None)
    ap.add_argument("--b3db", default=None, help="B3DB_classification.tsv; used only to reconstruct repeated classification row counts")
    ap.add_argument("--outdir", default="results/b3db_evidence_sensitivity")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--primary-view", default=PRIMARY_VIEW)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    args=ap.parse_args()

    root=Path(__file__).resolve().parent
    out=(root/args.outdir).resolve()
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    rng=np.random.default_rng(RNG_SEED)

    v3=find_path(args.v3_results,[
        root/"input"/"RESULTS_TO_SEND_FOR_ANALYSIS.zip",
        root.parent/"BBB_multiview_hardlabel_v3"/"results"/"multiview_hardlabel_v3"/"RESULTS_TO_SEND_FOR_ANALYSIS.zip",
        Path(__file__).resolve().parents[3] / 'results/stage1/hard_label_audit',
        Path(__file__).resolve().parents[3] / 'results/stage1/hard_label_audit',
    ],"v3 results")
    v3dir=extract_results(v3,out/"_v3_extracted")

    cons_path=find_path(args.consensus,[
        root/"data"/"frozen"/"consensus_dataset.csv",
        root.parent/"BBB_multiview_hardlabel_v3"/"data"/"frozen"/"consensus_dataset.csv",
        Path(__file__).resolve().parents[3] / 'data/frozen/consensus_dataset.csv',
        Path(__file__).resolve().parents[3] / 'data/frozen/consensus_dataset.csv',
    ],"consensus_dataset.csv")

    b3_path=find_path(args.b3db,[
        root/"data"/"raw"/"B3DB_数据"/"B3DB_classification.tsv",
        root.parent/"BBB_multiview_hardlabel_v3"/"data"/"raw"/"B3DB_数据"/"B3DB_classification.tsv",
        Path(__file__).resolve().parents[3] / 'data/raw/B3DB/B3DB_classification.tsv',
        Path(__file__).resolve().parents[3] / 'data/raw/B3DB/B3DB_classification.tsv',
    ],"B3DB_classification.tsv",required=False)

    mp=pd.read_csv(v3dir/"mean_predictions_long.csv")
    sp=pd.read_csv(v3dir/"predictions_by_seed.csv")
    tk=pd.read_csv(v3dir/"topk_membership.csv")
    cons=pd.read_csv(cons_path,dtype=str); cons.columns=[c.strip() for c in cons.columns]

    mp=mp[mp["eval"].eq("BBBP_test")].copy()
    sp=sp[sp["eval"].eq("BBBP_test")].copy()
    tk=tk[tk["eval"].eq("BBBP_test")].copy()

    views=sorted(mp["view"].unique().tolist())
    seeds=sorted(pd.to_numeric(sp["seed"],errors="coerce").dropna().astype(int).unique().tolist())
    mols=sorted(mp["molecule_id"].astype(str).unique().tolist())

    audit={"n_molecules":len(mols),"n_views":len(views),"views":views,"n_seeds":len(seeds),"seeds":seeds}
    if args.strict:
        assert len(mols)==291, audit
        assert set(views)==set(EXPECTED_VIEWS), audit
        assert len(seeds)==20 and seeds==list(range(42,62)), audit

    
    if "parent_ik" not in cons.columns:
        raise ValueError("consensus_dataset.csv missing parent_ik")
    cc=cons.drop_duplicates("parent_ik").copy()
    base=pd.DataFrame({"molecule_id":mols}).merge(cc,left_on="molecule_id",right_on="parent_ik",how="left",indicator=True)
    audit["consensus_linked"]=int((base["_merge"]=="both").sum())
    audit["consensus_unlinked"]=int((base["_merge"]!="both").sum())
    if args.strict and audit["consensus_unlinked"]!=0:
        raise RuntimeError(f"Consensus linkage failed: {audit}")
    base=base.drop(columns=["_merge"])

    
    base["b3db_label_conflict"]=numeric_consensus(base,"b3db_label_conflict",0).astype(int)
    base["n_references"]=numeric_consensus(base,"n_references",0).astype(int)
    base["b3db_logbb_n"]=numeric_consensus(base,"b3db_logbb_n",0).astype(int)

    raw_counts=infer_b3db_raw_counts(b3_path)
    if raw_counts is not None:
        base=base.merge(raw_counts,left_on="molecule_id",right_on="parent_ik",how="left",suffixes=("","_raw"))
        if "parent_ik_raw" in base.columns: base=base.drop(columns=["parent_ik_raw"])
        base["b3db_class_raw_row_count"]=pd.to_numeric(base["b3db_class_raw_row_count"],errors="coerce").fillna(0).astype(int)
        audit["b3db_class_count_source"]="raw B3DB_classification.tsv"
    elif "b3db_n_rows" in base.columns:
        base["b3db_class_raw_row_count"]=numeric_consensus(base,"b3db_n_rows",0).astype(int)
        audit["b3db_class_count_source"]="consensus b3db_n_rows"
    else:
        base["b3db_class_raw_row_count"]=0
        audit["b3db_class_count_source"]="unavailable"
        if args.strict:
            raise RuntimeError("Cannot obtain B3DB classification row counts; provide B3DB_classification.tsv.")

    
    base["b3db_total_record_count"]=(base["b3db_class_raw_row_count"]+base["b3db_logbb_n"]).astype(int)
    base["b3db_evidence_channel_count"]=(
        (base["b3db_class_raw_row_count"]>0).astype(int)+
        (base["n_references"]>0).astype(int)+
        (base["b3db_logbb_n"]>0).astype(int)
    )
    base["b3db_conflict"]=(base["b3db_label_conflict"]>0).astype(int)
    base["low_reference_membership"]=(base["n_references"]<=1).astype(int)
    base["multi_reference_membership"]=(base["n_references"]>=2).astype(int)
    base["single_or_no_class_record"]=(base["b3db_class_raw_row_count"]<=1).astype(int)
    base["repeated_class_records"]=(base["b3db_class_raw_row_count"]>=2).astype(int)
    base["no_logbb"]=(base["b3db_logbb_n"]==0).astype(int)
    base["has_logbb"]=(base["b3db_logbb_n"]>=1).astype(int)

    
    seedstat=(sp.groupby(["view","molecule_id"],as_index=False)
               .agg(seed_mean=("prediction","mean"),seed_sd=("prediction","std"),
                    seed_min=("prediction","min"),seed_max=("prediction","max")))
    seedstat["seed_range"]=seedstat["seed_max"]-seedstat["seed_min"]
    seedwide=seedstat.pivot(index="molecule_id",columns="view",values="seed_sd")
    seeddiag=pd.DataFrame(index=seedwide.index)
    seeddiag["primary_seed_sd"]=seedwide[args.primary_view]
    seeddiag["mean_seed_sd_across_views"]=seedwide.mean(axis=1)
    seeddiag["max_seed_sd_across_views"]=seedwide.max(axis=1)
    seeddiag=seeddiag.reset_index()

    
    pred=mp.pivot(index="molecule_id",columns="view",values="mean_prediction")
    diag=pd.DataFrame(index=pred.index)
    diag["view_prediction_mean"]=pred.mean(axis=1)
    diag["view_prediction_sd"]=pred.std(axis=1,ddof=0)
    diag["view_prediction_min"]=pred.min(axis=1)
    diag["view_prediction_max"]=pred.max(axis=1)
    diag["view_prediction_range"]=diag["view_prediction_max"]-diag["view_prediction_min"]
    diag["max_abs_dev_from_primary"]=pred.sub(pred[args.primary_view],axis=0).abs().max(axis=1)

    
    ranks=pred.rank(axis=0,method="average",ascending=False)
    n=len(ranks)
    pct=(ranks-1)/max(1,n-1)
    diag["rank_min"]=ranks.min(axis=1)
    diag["rank_max"]=ranks.max(axis=1)
    diag["rank_range"]=diag["rank_max"]-diag["rank_min"]
    diag["rank_sd"]=ranks.std(axis=1,ddof=0)
    diag["rank_percentile_min"]=pct.min(axis=1)
    diag["rank_percentile_max"]=pct.max(axis=1)
    diag["rank_percentile_range"]=diag["rank_percentile_max"]-diag["rank_percentile_min"]
    diag["rank_percentile_sd"]=pct.std(axis=1,ddof=0)
    diag=diag.reset_index()

    
    t10=tk.pivot(index="molecule_id",columns="view",values="top10pct")
    t20=tk.pivot(index="molecule_id",columns="view",values="top20pct")
    topdiag=pd.DataFrame(index=t10.index)
    topdiag["top10_membership_views"]=t10.sum(axis=1).astype(int)
    topdiag["top10_switch"]=((t10.min(axis=1)!=t10.max(axis=1))).astype(int)
    topdiag["top20_membership_views"]=t20.sum(axis=1).astype(int)
    topdiag["top20_switch"]=((t20.min(axis=1)!=t20.max(axis=1))).astype(int)
    topdiag=topdiag.reset_index()

    mol=base.merge(diag,on="molecule_id",how="left").merge(seeddiag,on="molecule_id",how="left").merge(topdiag,on="molecule_id",how="left")

    
    def add_contrast(name,a,b):
        if a in pred.columns and b in pred.columns:
            d=(pred[a]-pred[b]).abs()
            mol[name]=mol["molecule_id"].map(d)
    add_contrast("absdiff_first_all_vs_consistent","parent_first_all","parent_first_consistent")
    add_contrast("absdiff_first_vs_majority_all","parent_first_all","parent_majority_all")
    add_contrast("absdiff_majority_all_vs_consistent","parent_majority_all","parent_majority_consistent")
    
    add_contrast("absdiff_canonical_first_all_vs_consistent","canonical_first_all","canonical_first_consistent")
    add_contrast("absdiff_canonical_first_vs_majority_all","canonical_first_all","canonical_majority_all")

    
    core_metrics=[
        "primary_seed_sd","mean_seed_sd_across_views","view_prediction_range",
        "max_abs_dev_from_primary","rank_range","rank_percentile_range"
    ]
    for m in core_metrics:
        thr=float(mol[m].quantile(.90))
        mol[f"tail10_{m}"]=(mol[m]>=thr).astype(int)
        audit[f"tail10_threshold_{m}"]=thr

    mol.to_csv(out/"molecule_level_b3db_evidence_sensitivity.csv",index=False)

    
    evid_cols=["b3db_conflict","n_references","b3db_class_raw_row_count","b3db_logbb_n",
               "b3db_total_record_count","b3db_evidence_channel_count"]
    desc=[]
    for c in evid_cols:
        s=pd.to_numeric(mol[c],errors="coerce")
        desc.append(dict(variable=c,n=int(s.notna().sum()),mean=float(s.mean()),sd=float(s.std()),
                         median=float(s.median()),q25=float(s.quantile(.25)),q75=float(s.quantile(.75)),
                         min=float(s.min()),max=float(s.max()),zeros=int((s==0).sum())))
    pd.DataFrame(desc).to_csv(out/"b3db_evidence_distribution.csv",index=False)

    
    cont_evidence=["n_references","b3db_class_raw_row_count","b3db_logbb_n","b3db_total_record_count","b3db_evidence_channel_count"]
    corr=[]
    for e in cont_evidence:
        for m in core_metrics:
            rho,p,lo,hi=bootstrap_spearman(mol[e],mol[m],args.n_boot,rng)
            corr.append(dict(evidence_variable=e,sensitivity_metric=m,n=int(mol[[e,m]].dropna().shape[0]),
                             spearman_rho=rho,p_value=p,bootstrap95_lo=lo,bootstrap95_hi=hi))
    corr=pd.DataFrame(corr)
    corr["q_bh"]=bh_adjust(corr["p_value"])
    corr.to_csv(out/"continuous_evidence_sensitivity_correlations.csv",index=False)

    
    contrasts=[
        ("b3db_conflict","conflict","nonconflict",1,0),
        ("low_reference_membership","n_references<=1","n_references>=2",1,0),
        ("single_or_no_class_record","class_rows<=1","class_rows>=2",1,0),
        ("no_logbb","no_logBB","has_logBB",1,0),
    ]
    rows=[]
    for var,g1,g0,v1,v0 in contrasts:
        for m in core_metrics + ["top10_switch","top20_switch"]:
            a=pd.to_numeric(mol.loc[mol[var]==v1,m],errors="coerce").dropna().to_numpy()
            b=pd.to_numeric(mol.loc[mol[var]==v0,m],errors="coerce").dropna().to_numpy()
            if m.startswith("top"):
                
                point=float(a.mean()-b.mean()) if len(a) and len(b) else np.nan
                lo=hi=np.nan
                U_p=np.nan; delta=np.nan
            else:
                point,lo,hi=bootstrap_mean_diff(a,b,args.n_boot,rng)
                try: U_p=float(stats.mannwhitneyu(a,b,alternative="two-sided").pvalue)
                except Exception: U_p=np.nan
                delta=cliffs_delta(a,b)
            rows.append(dict(group_variable=var,group_1=g1,group_0=g0,metric=m,n1=len(a),n0=len(b),
                             mean1=float(np.mean(a)) if len(a) else np.nan,mean0=float(np.mean(b)) if len(b) else np.nan,
                             mean_difference_1_minus_0=point,bootstrap95_lo=lo,bootstrap95_hi=hi,
                             cliffs_delta=delta,mannwhitney_p=U_p))
    comp=pd.DataFrame(rows)
    comp["q_bh_within_analysis"]=bh_adjust(comp["mannwhitney_p"])
    comp.to_csv(out/"prespecified_group_sensitivity_contrasts.csv",index=False)

    
    enrich=[]
    exposures=["b3db_conflict","low_reference_membership","single_or_no_class_record","no_logbb"]
    outcomes=[f"tail10_{m}" for m in core_metrics]+["top10_switch","top20_switch"]
    for e in exposures:
        for o in outcomes:
            r=fisher_enrichment(mol[e],mol[o]); r.update(exposure=e,outcome=o,n=len(mol))
            enrich.append(r)
    enr=pd.DataFrame(enrich)
    enr["q_bh"]=bh_adjust(enr["fisher_p"])
    enr.to_csv(out/"tail_and_topk_enrichment_b3db_side.csv",index=False)

    
    pair=[]
    for i,a in enumerate(views):
        for b in views[i+1:]:
            da=pred[a]; db=pred[b]
            ra=ranks[a]; rb=ranks[b]
            topa=t10[a].astype(bool); topb=t10[b].astype(bool)
            pair.append(dict(view_a=a,view_b=b,
                             mean_abs_prediction_difference=float(np.mean(np.abs(da-db))),
                             max_abs_prediction_difference=float(np.max(np.abs(da-db))),
                             spearman_rank=float(stats.spearmanr(ra,rb).statistic),
                             mean_abs_rank_difference=float(np.mean(np.abs(ra-rb))),
                             top10_switch_molecules=int(np.sum(topa!=topb))))
    pd.DataFrame(pair).to_csv(out/"pairwise_view_instability_summary.csv",index=False)

    
    identity_pairs=[
      ("parent_first_all","canonical_first_all"),
      ("parent_first_consistent","canonical_first_consistent"),
      ("parent_majority_all","canonical_majority_all"),
      ("parent_majority_consistent","canonical_majority_consistent"),
    ]
    ident=[]
    for a,b in identity_pairs:
        d=np.abs(pred[a]-pred[b])
        ident.append(dict(view_a=a,view_b=b,max_abs_diff=float(d.max()),mean_abs_diff=float(d.mean()),exact_equal=bool(np.all(d.to_numpy()<=1e-12))))
    ident=pd.DataFrame(ident)
    ident.to_csv(out/"v3_identity_control_recheck.csv",index=False)
    if args.strict and not ident["exact_equal"].all():
        raise RuntimeError("v3 identity-axis prediction control failed; do not interpret downstream evidence sensitivity.")

    with open(out/"input_audit.json","w",encoding="utf-8") as f:
        json.dump(audit,f,indent=2,ensure_ascii=False)

    
    summary={
        "scientific_scope":"B3DB-side evidence quantity/conflict versus model-output sensitivity; no BBBP probability calibration claim",
        "n_BBBP_test_molecules":len(mol),
        "n_views":len(views),"n_seeds":len(seeds),
        "b3db_conflict_n":int(mol["b3db_conflict"].sum()),
        "n_references_le1_n":int(mol["low_reference_membership"].sum()),
        "top10_switch_n":int(mol["top10_switch"].sum()),
        "top20_switch_n":int(mol["top20_switch"].sum()),
        "interpretation_boundary":"n_references and record counts are evidence coverage/quantity proxies, not experimental-quality scores."
    }
    with open(out/"analysis_summary.json","w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)

    text = f"""B3DB evidence-stratified construction-sensitivity analysis COMPLETE

Scope:
- Fixed BBBP test molecules: {len(mol)}
- Construction views: {len(views)}
- XGBoost seeds: {len(seeds)}
- Uses B3DB-side evidence metadata only.
- Does NOT interpret raw B3DB-trained scores as calibrated BBBP probabilities.

Primary outputs:
1. molecule_level_b3db_evidence_sensitivity.csv
2. continuous_evidence_sensitivity_correlations.csv
3. prespecified_group_sensitivity_contrasts.csv
4. tail_and_topk_enrichment_b3db_side.csv
5. pairwise_view_instability_summary.csv
6. v3_identity_control_recheck.csv

Interpretation rule:
Evidence of a hidden local-sensitivity effect requires concordant signals such as:
- negative association between B3DB evidence quantity and seed/view/rank instability; and/or
- conflict/low-coverage groups showing larger instability with CI/effect-size support; and/or
- enrichment of those groups in instability tails or top-k switching.
No result should be described as 'label quality causes error' because the available metadata quantify
B3DB evidence coverage/conflict, not experimental truth or causal label quality.
"""
    (out/"SUMMARY_README.txt").write_text(text,encoding="utf-8")

    
    zpath=out/"RESULTS_TO_SEND_FOR_ANALYSIS.zip"
    with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p.name!="RESULTS_TO_SEND_FOR_ANALYSIS.zip":
                z.write(p,arcname=p.name)

    print("ANALYSIS COMPLETE")
    print(f"Output: {out}")
    print(f"Send back: {zpath}")

if __name__=="__main__":
    main()
