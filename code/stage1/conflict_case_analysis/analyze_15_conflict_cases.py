# analyze_15_conflict_cases.py — Matches 15 molecules with contradictory B3DB classification records to nonconflicting controls and calculates paired prediction, rank-sensitivity, uncertainty and trajectory summaries.
































from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import textwrap
import zipfile
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import SaltRemover, rdFingerprintGenerator
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")

PRIMARY_VIEW = "parent_first_consistent"
VIEW_ORDER = [
    "parent_first_all",
    "canonical_first_all",
    "parent_first_consistent",
    "canonical_first_consistent",
    "parent_majority_all",
    "canonical_majority_all",
    "parent_majority_consistent",
    "canonical_majority_consistent",
]
VIEW_LABELS = {
    "parent_first_all": "P-first-all",
    "canonical_first_all": "C-first-all",
    "parent_first_consistent": "P-first-cons",
    "canonical_first_consistent": "C-first-cons",
    "parent_majority_all": "P-majority-all",
    "canonical_majority_all": "C-majority-all",
    "parent_majority_consistent": "P-majority-cons",
    "canonical_majority_consistent": "C-majority-cons",
}
EXPECTED_N_CASES = 15
RNG_SEED = 20260929

def parent_inchikey(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
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
        p = Path(p).expanduser().resolve()
        tried.append(str(p))
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(f"Could not locate {what}. Tried:\n" + "\n".join(tried))
    return None

def extract_if_zip(path: Path, dest: Path) -> Path:
    if path.is_dir():
        return path
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(dest)
    return dest

def locate_inside(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    if len(hits) > 1:
        
        hits = sorted(hits, key=lambda p: (len(p.parts), str(p)))
    return hits[0]

def read_b3db(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "SMILES" not in df.columns:
        raise ValueError("B3DB_classification.tsv must contain SMILES.")
    label_col = "BBB+/BBB-" if "BBB+/BBB-" in df.columns else None
    if label_col is None:
        for c in df.columns:
            if c.lower() in {"label", "bbb", "class", "classification"}:
                label_col = c
                break
    if label_col is None:
        raise ValueError("Cannot identify B3DB classification label column.")
    ref_col = "reference" if "reference" in df.columns else None
    if ref_col is None:
        for c in df.columns:
            if "ref" in c.lower():
                ref_col = c
                break

    def parse_label(x):
        s = str(x).strip().upper()
        if s in {"BBB+", "1", "TRUE", "POS", "POSITIVE"}:
            return 1
        if s in {"BBB-", "0", "FALSE", "NEG", "NEGATIVE"}:
            return 0
        return np.nan

    df = df.copy()
    df["_raw_row_index"] = np.arange(len(df))
    df["parent_ik"] = df["SMILES"].map(parent_inchikey)
    df["raw_label_text"] = df[label_col]
    df["label01"] = df[label_col].map(parse_label)
    df["raw_reference_text"] = df[ref_col].fillna("") if ref_col else ""
    return df

def parse_ref_codes(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x)
    
    codes = re.findall(r"\bR\d+\b", s, flags=re.I)
    if codes:
        return [c.upper() for c in codes]
    return [t.strip() for t in re.split(r"[|;,]+", s) if t.strip()]

def load_reference_map(path: Optional[Path]):
    if path is None:
        return {}, "code_only"
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return {}, "code_only"
    cols = {c.lower(): c for c in df.columns}
    refc = cols.get("reference") or cols.get("ref") or cols.get("code")
    if not refc:
        return {}, "code_only"
    text_cols = [c for c in df.columns if c != refc]
    mp = {}
    placeholder = True
    for _, r in df.iterrows():
        code = str(r[refc]).strip()
        bits = []
        for c in text_cols:
            val = str(r.get(c, "")).strip()
            if val and val.lower() != "nan":
                bits.append(f"{c}={val}")
                if "待人工" not in val and "placeholder" not in val.lower():
                    placeholder = False
        mp[code] = "; ".join(bits)
    return mp, ("mapped" if not placeholder else "placeholder_map")

def fingerprint(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    return gen.GetFingerprint(m)

def tanimoto(a, b):
    if a is None or b is None:
        return np.nan
    return float(DataStructs.TanimotoSimilarity(a, b))

def safe_z(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / sd

def exact_signflip_p(diffs):
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan
    obs = abs(float(np.mean(d)))
    if n <= 20:
        vals = []
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            vals.append(abs(float(np.mean(d * np.asarray(signs)))))
        return float(np.mean(np.asarray(vals) >= obs - 1e-15))
    rng = np.random.default_rng(RNG_SEED)
    vals = []
    for _ in range(100000):
        signs = rng.choice([-1.0, 1.0], size=n)
        vals.append(abs(float(np.mean(d * signs))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))

def paired_bootstrap_ci(diffs, n_boot, rng):
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return np.nan, np.nan
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = d[rng.integers(0, len(d), len(d))].mean()
    return tuple(map(float, np.quantile(vals, [0.025, 0.975])))

def wrap(s, width=95):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False, break_on_hyphens=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-results", default=None,
                    help="RESULTS_TO_SEND_FOR_ANALYSIS.zip from BBB_B3DB_evidence_sensitivity_v1, or extracted directory.")
    ap.add_argument("--v3-results", default=None,
                    help="RESULTS_TO_SEND_FOR_ANALYSIS.zip from BBB_multiview_hardlabel_v3, or extracted directory.")
    ap.add_argument("--b3db", default=None, help="Raw B3DB_classification.tsv")
    ap.add_argument("--reference-map", default=None,
                    help="Optional CSV mapping R-codes to literature metadata. If absent, provenance remains code-level only.")
    ap.add_argument("--outdir", default="results/conflict_case_analysis")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--strict", action="store_true", default=True)
    ap.add_argument("--no-strict", action="store_false", dest="strict")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    out = (here / args.outdir).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "figures").mkdir()

    evidence_path = find_path(args.evidence_results, [
        here / "input" / "RESULTS_TO_SEND_FOR_ANALYSIS_evidence.zip",
        here.parent / "BBB_B3DB_evidence_sensitivity_v1" / "results" / "b3db_evidence_sensitivity" / "RESULTS_TO_SEND_FOR_ANALYSIS.zip",
        Path(__file__).resolve().parents[3] / 'results/stage1/b3db_evidence_sensitivity',
        Path(__file__).resolve().parents[3] / 'results/stage1/b3db_evidence_sensitivity',
    ], "B3DB evidence-sensitivity results")
    evdir = extract_if_zip(evidence_path, out / "_evidence_extracted")
    evfile = locate_inside(evdir, "molecule_level_b3db_evidence_sensitivity.csv")
    ev = pd.read_csv(evfile)

    v3_path = find_path(args.v3_results, [
        here / "input" / "RESULTS_TO_SEND_FOR_ANALYSIS_v3.zip",
        here.parent / "BBB_multiview_hardlabel_v3" / "results" / "multiview_hardlabel_v3" / "RESULTS_TO_SEND_FOR_ANALYSIS.zip",
        Path(__file__).resolve().parents[3] / 'results/stage1/hard_label_audit',
        Path(__file__).resolve().parents[3] / 'results/stage1/hard_label_audit',
    ], "v3 multiview results")
    v3dir = extract_if_zip(v3_path, out / "_v3_extracted")
    mp = pd.read_csv(locate_inside(v3dir, "mean_predictions_long.csv"))
    seed = pd.read_csv(locate_inside(v3dir, "predictions_by_seed.csv"))
    topk = pd.read_csv(locate_inside(v3dir, "topk_membership.csv"))

    b3db_path = find_path(args.b3db, [
        here / "data" / "raw" / "B3DB_数据" / "B3DB_classification.tsv",
        here.parent / "BBB_multiview_hardlabel_v3" / "data" / "raw" / "B3DB_数据" / "B3DB_classification.tsv",
        Path(__file__).resolve().parents[3] / 'data/raw/B3DB/B3DB_classification.tsv',
        Path(__file__).resolve().parents[3] / 'data/raw/B3DB/B3DB_classification.tsv',
    ], "B3DB_classification.tsv")
    b3 = read_b3db(b3db_path)

    refmap_path = find_path(args.reference_map, [
        here / "data" / "frozen" / "reference_map.csv",
        here.parent / "BBB_multiview_hardlabel_v3" / "data" / "frozen" / "reference_map.csv",
        Path(__file__).resolve().parents[3] / 'data/frozen/reference_map.csv',
    ], "reference_map.csv", required=False)
    refmap, refmap_status = load_reference_map(refmap_path)

    
    mp = mp[mp["eval"].eq("BBBP_test")].copy()
    seed = seed[seed["eval"].eq("BBBP_test")].copy()
    topk = topk[topk["eval"].eq("BBBP_test")].copy()

    if args.strict:
        if mp["molecule_id"].nunique() != 291:
            raise RuntimeError("Expected frozen BBBP test n=291.")
        if set(mp["view"].unique()) != set(VIEW_ORDER):
            raise RuntimeError(f"Unexpected view set: {sorted(mp['view'].unique())}")
        seeds = sorted(pd.to_numeric(seed["seed"], errors="coerce").dropna().astype(int).unique())
        if seeds != list(range(42, 62)):
            raise RuntimeError(f"Expected seeds 42-61, got {seeds}")

    cases = ev[pd.to_numeric(ev["b3db_conflict"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    cases = cases.sort_values(["rank_range", "molecule_id"], ascending=[False, True]).reset_index(drop=True)
    if args.strict and len(cases) != EXPECTED_N_CASES:
        raise RuntimeError(f"Expected {EXPECTED_N_CASES} conflict molecules, found {len(cases)}.")

    
    raw = b3[b3["parent_ik"].isin(cases["molecule_id"])].copy()
    raw = raw.sort_values("_raw_row_index")
    raw.to_csv(out / "b3db_raw_rows_15_conflict_cases.csv", index=False)

    raw_summ = []
    for mid, g in raw.groupby("parent_ik", sort=False):
        glab = g["label01"].dropna().astype(int).tolist()
        refs = []
        for x in g["raw_reference_text"]:
            refs.extend(parse_ref_codes(x))
        refs_unique = list(dict.fromkeys(refs))
        n0, n1 = glab.count(0), glab.count(1)
        majority = "tie" if n0 == n1 else (1 if n1 > n0 else 0)
        first = glab[0] if glab else np.nan
        annotations = []
        for r in refs_unique:
            if r in refmap and refmap[r]:
                annotations.append(f"{r}: {refmap[r]}")
        raw_summ.append({
            "molecule_id": mid,
            "raw_row_count_reconstructed": len(g),
            "raw_label_sequence": "|".join(map(str, glab)),
            "raw_label_count_0": n0,
            "raw_label_count_1": n1,
            "first_label_from_raw": first,
            "majority_label_from_raw": majority,
            "raw_row_reference_sequence": " || ".join(g["raw_reference_text"].fillna("").astype(str).tolist()),
            "unique_reference_codes_from_raw": "|".join(refs_unique),
            "reference_annotations": " || ".join(annotations),
            "reference_metadata_status": refmap_status,
        })
    raw_summ = pd.DataFrame(raw_summ)
    cases = cases.merge(raw_summ, on="molecule_id", how="left")
    if args.strict and cases["raw_label_sequence"].isna().any():
        missing = cases.loc[cases["raw_label_sequence"].isna(), "molecule_id"].tolist()
        raise RuntimeError(f"Raw B3DB rows missing for conflict cases: {missing}")

    
    pred = mp.pivot(index="molecule_id", columns="view", values="mean_prediction")[VIEW_ORDER]
    ranks = pred.rank(axis=0, ascending=False, method="average")
    rankpct = (ranks - 1.0) / max(1, len(ranks) - 1)

    top10 = topk.pivot(index="molecule_id", columns="view", values="top10pct")[VIEW_ORDER]
    top20 = topk.pivot(index="molecule_id", columns="view", values="top20pct")[VIEW_ORDER]

    primary_rankpct = rankpct[PRIMARY_VIEW]
    primary_pred = pred[PRIMARY_VIEW]

    
    controls = ev[
        pd.to_numeric(ev["b3db_conflict"], errors="coerce").fillna(0).astype(int).eq(0)
        & (pd.to_numeric(ev["b3db_class_raw_row_count"], errors="coerce").fillna(0) > 0)
    ].copy()
    controls = controls[~controls["molecule_id"].isin(cases["molecule_id"])].copy()

    
    combined = pd.concat([
        cases.assign(_role="case"),
        controls.assign(_role="control")
    ], ignore_index=True)
    combined["match_log_refs"] = np.log1p(pd.to_numeric(combined["n_references"], errors="coerce").fillna(0))
    combined["match_log_class_rows"] = np.log1p(pd.to_numeric(combined["b3db_class_raw_row_count"], errors="coerce").fillna(0))
    combined["match_log_logbb_n"] = np.log1p(pd.to_numeric(combined["b3db_logbb_n"], errors="coerce").fillna(0))
    combined["primary_rank_percentile"] = combined["molecule_id"].map(primary_rankpct)
    combined["primary_prediction"] = combined["molecule_id"].map(primary_pred)

    zcols = ["match_log_refs", "match_log_class_rows", "match_log_logbb_n", "primary_rank_percentile"]
    for c in zcols:
        combined["_z_" + c] = safe_z(combined[c])

    case_m = combined[combined["_role"].eq("case")].set_index("molecule_id")
    ctl_m = combined[combined["_role"].eq("control")].set_index("molecule_id")
    case_ids = cases["molecule_id"].tolist()
    ctl_ids = controls["molecule_id"].tolist()

    
    smiles_map = {}
    for _, r in ev.iterrows():
        smi = r.get("canon_smiles_b3db")
        if not isinstance(smi, str) or not smi:
            smi = r.get("canon_smiles")
        smiles_map[str(r["molecule_id"])] = smi
    fps = {m: fingerprint(smiles_map.get(m)) for m in case_ids + ctl_ids}

    cost = np.zeros((len(case_ids), len(ctl_ids)), dtype=float)
    comp_rows = []
    for i, cid in enumerate(case_ids):
        cr = case_m.loc[cid]
        for j, xid in enumerate(ctl_ids):
            xr = ctl_m.loc[xid]
            numeric = 0.0
            for c in zcols:
                numeric += float((cr["_z_" + c] - xr["_z_" + c]) ** 2)
            numeric = math.sqrt(numeric)

            
            label_case = pd.to_numeric(pd.Series([cr.get("b3db_label")]), errors="coerce").iloc[0]
            label_ctl = pd.to_numeric(pd.Series([xr.get("b3db_label")]), errors="coerce").iloc[0]
            label_mismatch = int(np.isfinite(label_case) and np.isfinite(label_ctl) and label_case != label_ctl)

            refout_case = int(pd.to_numeric(pd.Series([cr.get("in_reference_out", 0)]), errors="coerce").fillna(0).iloc[0])
            refout_ctl = int(pd.to_numeric(pd.Series([xr.get("in_reference_out", 0)]), errors="coerce").fillna(0).iloc[0])
            refout_mismatch = int(refout_case != refout_ctl)

            sim = tanimoto(fps.get(cid), fps.get(xid))
            structural_penalty = (1.0 - sim) if np.isfinite(sim) else 1.0

            total = numeric + 100.0 * label_mismatch + 20.0 * refout_mismatch + 0.5 * structural_penalty
            cost[i, j] = total

    rr, cc = linear_sum_assignment(cost)
    match_rows = []
    for i, j in zip(rr, cc):
        cid, xid = case_ids[i], ctl_ids[j]
        cr, xr = case_m.loc[cid], ctl_m.loc[xid]
        sim = tanimoto(fps.get(cid), fps.get(xid))
        label_case = pd.to_numeric(pd.Series([cr.get("b3db_label")]), errors="coerce").iloc[0]
        label_ctl = pd.to_numeric(pd.Series([xr.get("b3db_label")]), errors="coerce").iloc[0]
        refout_case = int(pd.to_numeric(pd.Series([cr.get("in_reference_out", 0)]), errors="coerce").fillna(0).iloc[0])
        refout_ctl = int(pd.to_numeric(pd.Series([xr.get("in_reference_out", 0)]), errors="coerce").fillna(0).iloc[0])
        match_rows.append({
            "conflict_molecule_id": cid,
            "control_molecule_id": xid,
            "matching_cost": float(cost[i, j]),
            "same_primary_b3db_label": int((not np.isfinite(label_case) or not np.isfinite(label_ctl)) or label_case == label_ctl),
            "same_reference_out_status": int(refout_case == refout_ctl),
            "morgan_tanimoto": sim,
            "case_n_references": cr.get("n_references"),
            "control_n_references": xr.get("n_references"),
            "case_class_rows": cr.get("b3db_class_raw_row_count"),
            "control_class_rows": xr.get("b3db_class_raw_row_count"),
            "case_logbb_n": cr.get("b3db_logbb_n"),
            "control_logbb_n": xr.get("b3db_logbb_n"),
            "case_primary_rank_percentile": cr.get("primary_rank_percentile"),
            "control_primary_rank_percentile": xr.get("primary_rank_percentile"),
            "case_primary_prediction": cr.get("primary_prediction"),
            "control_primary_prediction": xr.get("primary_prediction"),
        })
    matches = pd.DataFrame(match_rows)
    if args.strict:
        if len(matches) != EXPECTED_N_CASES or matches["control_molecule_id"].nunique() != EXPECTED_N_CASES:
            raise RuntimeError("1:1 without-replacement matching failed.")
        if not matches["same_primary_b3db_label"].all():
            raise RuntimeError("Strict matching failed to preserve primary B3DB label for all pairs.")
        if not matches["same_reference_out_status"].all():
            raise RuntimeError("Strict matching failed to preserve reference-out status for all pairs.")
    matches.to_csv(out / "matched_nonconflict_controls.csv", index=False)

    
    ev_idx = ev.set_index("molecule_id")
    casebook_rows = []
    for _, m in matches.iterrows():
        cid, xid = m["conflict_molecule_id"], m["control_molecule_id"]
        c = cases[cases["molecule_id"].eq(cid)].iloc[0].to_dict()
        x = ev_idx.loc[xid].to_dict()
        row = {}
        for k, v in c.items():
            row["case_" + k] = v
        for k, v in x.items():
            row["control_" + k] = v
        for k, v in m.items():
            row["match_" + k] = v
        casebook_rows.append(row)
    casebook = pd.DataFrame(casebook_rows)
    casebook.to_csv(out / "conflict_cases_with_matched_controls.csv", index=False)

    
    long_rows = []
    for pair_id, m in matches.reset_index(drop=True).iterrows():
        cid, xid = m["conflict_molecule_id"], m["control_molecule_id"]
        for role, mid in [("conflict", cid), ("matched_nonconflict", xid)]:
            for order, v in enumerate(VIEW_ORDER):
                long_rows.append({
                    "pair_id": pair_id + 1,
                    "role": role,
                    "molecule_id": mid,
                    "view_order": order + 1,
                    "view": v,
                    "view_label": VIEW_LABELS[v],
                    "mean_prediction": float(pred.loc[mid, v]),
                    "rank": float(ranks.loc[mid, v]),
                    "rank_percentile": float(rankpct.loc[mid, v]),
                    "top10pct": int(top10.loc[mid, v]),
                    "top20pct": int(top20.loc[mid, v]),
                    "delta_prediction_from_primary": float(pred.loc[mid, v] - pred.loc[mid, PRIMARY_VIEW]),
                    "delta_rank_from_primary": float(ranks.loc[mid, v] - ranks.loc[mid, PRIMARY_VIEW]),
                })
    traj = pd.DataFrame(long_rows)
    traj.to_csv(out / "case_and_control_8view_trajectories_long.csv", index=False)

    
    metrics = [
        "view_prediction_range",
        "max_abs_dev_from_primary",
        "rank_range",
        "rank_percentile_range",
        "primary_seed_sd",
        "mean_seed_sd_across_views",
    ]
    rng = np.random.default_rng(RNG_SEED)
    pair_stats = []
    for metric in metrics:
        a = np.array([float(ev_idx.loc[x, metric]) for x in matches["conflict_molecule_id"]])
        b = np.array([float(ev_idx.loc[x, metric]) for x in matches["control_molecule_id"]])
        d = a - b
        lo, hi = paired_bootstrap_ci(d, args.n_boot, rng)
        try:
            wp = float(stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)
        except Exception:
            wp = np.nan
        pair_stats.append({
            "metric": metric,
            "n_pairs": len(d),
            "conflict_mean": float(np.mean(a)),
            "control_mean": float(np.mean(b)),
            "paired_mean_difference": float(np.mean(d)),
            "paired_median_difference": float(np.median(d)),
            "bootstrap95_lo": lo,
            "bootstrap95_hi": hi,
            "exact_signflip_p": exact_signflip_p(d),
            "wilcoxon_p": wp,
            "n_conflict_gt_control": int(np.sum(d > 0)),
            "n_equal": int(np.sum(d == 0)),
            "n_conflict_lt_control": int(np.sum(d < 0)),
        })
    pd.DataFrame(pair_stats).to_csv(out / "matched_pair_sensitivity_summary.csv", index=False)

    
    readable = []
    for _, m in matches.iterrows():
        cid, xid = m["conflict_molecule_id"], m["control_molecule_id"]
        c = cases[cases["molecule_id"].eq(cid)].iloc[0]
        x = ev_idx.loc[xid]
        readable.append({
            "conflict_molecule_id": cid,
            "conflict_name": c.get("bbbp_name", ""),
            "conflict_smiles": c.get("canon_smiles_b3db", c.get("canon_smiles", "")),
            "raw_label_sequence": c.get("raw_label_sequence", ""),
            "raw_label_counts_0_1": f"{c.get('raw_label_count_0','')}:{c.get('raw_label_count_1','')}",
            "first_label": c.get("first_label_from_raw", ""),
            "majority_label": c.get("majority_label_from_raw", ""),
            "reference_codes": c.get("unique_reference_codes_from_raw", c.get("reference_list", "")),
            "reference_annotations": c.get("reference_annotations", ""),
            "n_references": c.get("n_references", ""),
            "classification_rows": c.get("b3db_class_raw_row_count", ""),
            "logbb_n": c.get("b3db_logbb_n", ""),
            "view_prediction_range": c.get("view_prediction_range", ""),
            "rank_range": c.get("rank_range", ""),
            "rank_percentile_range": c.get("rank_percentile_range", ""),
            "primary_seed_sd": c.get("primary_seed_sd", ""),
            "matched_control_id": xid,
            "matched_control_name": x.get("bbbp_name", ""),
            "control_n_references": x.get("n_references", ""),
            "control_classification_rows": x.get("b3db_class_raw_row_count", ""),
            "control_logbb_n": x.get("b3db_logbb_n", ""),
            "control_view_prediction_range": x.get("view_prediction_range", ""),
            "control_rank_range": x.get("rank_range", ""),
            "control_rank_percentile_range": x.get("rank_percentile_range", ""),
            "control_primary_seed_sd": x.get("primary_seed_sd", ""),
            "morgan_tanimoto": m["morgan_tanimoto"],
            "matching_cost": m["matching_cost"],
        })
    pd.DataFrame(readable).to_csv(out / "CASE_TABLE_FOR_MANUSCRIPT_REVIEW.csv", index=False)

    
    pdf_path = out / "CONFLICT_CASEBOOK_15_MOLECULES.pdf"
    with PdfPages(pdf_path) as pdf:
        for pair_no, m in enumerate(matches.itertuples(index=False), start=1):
            cid = m.conflict_molecule_id
            xid = m.control_molecule_id
            crow = cases[cases["molecule_id"].eq(cid)].iloc[0]
            xrow = ev_idx.loc[xid]
            labels = [VIEW_LABELS[v] for v in VIEW_ORDER]
            xs = np.arange(len(labels))

            
            fig = plt.figure(figsize=(11.5, 8.0))
            ax = fig.add_axes([0.10, 0.25, 0.84, 0.58])
            ax.plot(xs, pred.loc[cid, VIEW_ORDER].values, marker="o", label=f"Conflict: {crow.get('bbbp_name','')} ({cid})")
            ax.plot(xs, pred.loc[xid, VIEW_ORDER].values, marker="o", label=f"Matched non-conflict: {xrow.get('bbbp_name','')} ({xid})")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("Mean XGBoost output")
            ax.set_xlabel("Prespecified construction view (categorical; not temporal)")
            ax.set_title(f"Case {pair_no}/15 — 8-view prediction trajectory")
            ax.legend(loc="best")
            info = (
                f"Conflict raw labels: {crow.get('raw_label_sequence','NA')} | "
                f"first={crow.get('first_label_from_raw','NA')} | majority={crow.get('majority_label_from_raw','NA')}\n"
                f"B3DB refs: {crow.get('unique_reference_codes_from_raw', crow.get('reference_list',''))}\n"
                f"Case rank range={crow.get('rank_range', np.nan):.1f}; "
                f"control rank range={float(xrow.get('rank_range', np.nan)):.1f}; "
                f"match Tanimoto={float(m.morgan_tanimoto) if np.isfinite(m.morgan_tanimoto) else np.nan:.3f}"
            )
            fig.text(0.08, 0.06, wrap(info, 125), fontsize=9, va="bottom")
            p = out / "figures" / f"{pair_no:02d}_{cid}_prediction_trajectory.png"
            fig.savefig(p, dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            
            fig = plt.figure(figsize=(11.5, 8.0))
            ax = fig.add_axes([0.10, 0.25, 0.84, 0.58])
            ax.plot(xs, rankpct.loc[cid, VIEW_ORDER].values, marker="o", label=f"Conflict: {crow.get('bbbp_name','')} ({cid})")
            ax.plot(xs, rankpct.loc[xid, VIEW_ORDER].values, marker="o", label=f"Matched non-conflict: {xrow.get('bbbp_name','')} ({xid})")
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("Rank percentile (0 = highest-ranked)")
            ax.set_xlabel("Prespecified construction view (categorical; not temporal)")
            ax.set_title(f"Case {pair_no}/15 — 8-view rank trajectory")
            ax.invert_yaxis()
            ax.legend(loc="best")
            info = (
                f"Case view prediction range={crow.get('view_prediction_range', np.nan):.4f}; "
                f"rank-percentile range={crow.get('rank_percentile_range', np.nan):.4f}; "
                f"primary seed SD={crow.get('primary_seed_sd', np.nan):.4f}\n"
                f"Control view prediction range={float(xrow.get('view_prediction_range', np.nan)):.4f}; "
                f"rank-percentile range={float(xrow.get('rank_percentile_range', np.nan)):.4f}; "
                f"primary seed SD={float(xrow.get('primary_seed_sd', np.nan)):.4f}"
            )
            fig.text(0.08, 0.07, wrap(info, 125), fontsize=9, va="bottom")
            p = out / "figures" / f"{pair_no:02d}_{cid}_rank_trajectory.png"
            fig.savefig(p, dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    
    interpretation = f"""# Interpretation boundary for the 15-molecule case analysis

- Conflict cases identified in frozen BBBP test: {len(cases)}
- Matching: 1:1 without replacement to non-conflict BBBP-test molecules with B3DB classification evidence.
- Matching prioritizes the same primary B3DB representative label and the same reference-out status;
  it additionally balances B3DB reference coverage, classification-row count, logBB count,
  primary-view rank percentile, and mildly favors Morgan-fingerprint similarity.
- Matching NEVER uses the downstream instability outcomes themselves.

## What this analysis can support
It can show whether molecules *characterized by internally conflicting B3DB evidence* display
larger construction-dependent movement in model output/rank than otherwise comparable
non-conflict molecules.

## What this analysis cannot support
These 15 cases are frozen TEST molecules. Their own B3DB conflicts were not used as training labels
in the v3 models. Therefore the analysis cannot prove that each molecule's own conflicting rows
caused its downstream rank drift. The appropriate language is association / localized sensitivity,
not molecule-specific causal effect.

## Provenance
Reference metadata status: {refmap_status}.
If no curated R-code-to-publication map is supplied, the package reports B3DB reference codes and
row-level reference memberships only; it does not invent literature identities.
"""
    (out / "INTERPRETATION_BOUNDARY.md").write_text(interpretation, encoding="utf-8")

    audit = {
        "status": "complete",
        "n_conflict_cases": int(len(cases)),
        "n_matched_controls": int(matches["control_molecule_id"].nunique()),
        "n_views": int(mp["view"].nunique()),
        "n_BBBP_test_molecules": int(mp["molecule_id"].nunique()),
        "reference_metadata_status": refmap_status,
        "primary_view": PRIMARY_VIEW,
        "view_order": VIEW_ORDER,
        "strict": bool(args.strict),
    }
    with open(out / "analysis_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    
    zpath = out / "RESULTS_TO_SEND_FOR_ANALYSIS.zip"
    keep_files = [
        "b3db_raw_rows_15_conflict_cases.csv",
        "matched_nonconflict_controls.csv",
        "conflict_cases_with_matched_controls.csv",
        "case_and_control_8view_trajectories_long.csv",
        "matched_pair_sensitivity_summary.csv",
        "CASE_TABLE_FOR_MANUSCRIPT_REVIEW.csv",
        "CONFLICT_CASEBOOK_15_MOLECULES.pdf",
        "INTERPRETATION_BOUNDARY.md",
        "analysis_audit.json",
    ]
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name in keep_files:
            p = out / name
            if p.exists():
                z.write(p, arcname=name)
        for p in sorted((out / "figures").glob("*.png")):
            z.write(p, arcname=f"figures/{p.name}")

    print("CASE ANALYSIS COMPLETE")
    print(f"Conflict molecules: {len(cases)}")
    print(f"Matched controls: {matches['control_molecule_id'].nunique()}")
    print(f"Casebook: {pdf_path}")
    print(f"Send back: {zpath}")

if __name__ == "__main__":
    main()
