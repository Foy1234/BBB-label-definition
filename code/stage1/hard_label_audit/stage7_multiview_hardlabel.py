# stage7_multiview_hardlabel.py — Audits eight prespecified molecular-identity, label-aggregation and conflict-handling schemes; trains controlled XGBoost hard-label models and exports paired performance, molecule-level sensitivity and top-ranked candidate-set comparisons.


















from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

RDLogger.DisableLog("rdApp.*")

RADIUS = 2
N_BITS = 2048
DEFAULT_SEEDS = list(range(42, 62))
PRIMARY_VIEW = "parent_first_consistent"
FEATURE_REPRESENTATION = "frozen consensus canon_smiles keyed by parent_ik"
BOOTSTRAP_SEED = 20260928
XGB_PARAMS = dict(
    max_depth=5,
    learning_rate=0.03,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
)
VIEW_DEFS = [
    ("parent", "first", "all"),
    ("parent", "first", "consistent"),
    ("parent", "majority", "all"),
    ("parent", "majority", "consistent"),
    ("canonical", "first", "all"),
    ("canonical", "first", "consistent"),
    ("canonical", "majority", "all"),
    ("canonical", "majority", "consistent"),
]
SREMOVER = SaltRemover.SaltRemover()


@dataclass
class Paths:
    root: Path
    bbbp: Path
    b3db: Path
    consensus: Path
    ncats_audit: Path
    out: Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def standardize_smiles(smiles: str) -> Tuple[str | None, str | None]:
    
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        mol = SREMOVER.StripMol(mol)
        Chem.RemoveStereochemistry(mol)
        canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        ik = MolToInchiKey(mol)
        parent = ik.split("-")[0] if ik else None
        return canon, parent
    except Exception:
        return None, None


def standardize_raw(df: pd.DataFrame, smiles_col: str, source: str) -> pd.DataFrame:
    out = df.copy()
    out["_source_row"] = np.arange(len(out), dtype=int)
    vals = [standardize_smiles(x) for x in out[smiles_col].astype(str)]
    out["_canon"] = [v[0] for v in vals]
    out["_parent"] = [v[1] for v in vals]
    out["_parse_ok"] = out["_parent"].notna() & out["_canon"].notna()
    out["_source"] = source
    return out


def normalize_b3db_label(x: str) -> float:
    s = str(x).strip()
    if s == "BBB+":
        return 1.0
    if s == "BBB-":
        return 0.0
    if s in {"0", "1", "0.0", "1.0"}:
        return float(s)
    return np.nan


def normalize_binary(x) -> float:
    try:
        v = float(str(x).strip())
        return v if v in (0.0, 1.0) else np.nan
    except Exception:
        return np.nan


def parse_seeds(text: str | None) -> List[int]:
    if not text:
        return DEFAULT_SEEDS.copy()
    vals = []
    for z in text.split(","):
        z = z.strip()
        if z:
            vals.append(int(z))
    if not vals:
        raise ValueError("--seeds produced an empty seed list")
    return vals


def load_inputs(paths: Paths, strict: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    missing = [p for p in [paths.bbbp, paths.b3db, paths.consensus, paths.ncats_audit] if not p.exists()]
    if missing:
        msg = "Missing required inputs:\n" + "\n".join(f"  - {p}" for p in missing)
        msg += "\nSee README_CN.md. Frozen consensus and frozen NCATS audit are strongly recommended for the formal run."
        raise FileNotFoundError(msg)

    bbbp0 = pd.read_csv(paths.bbbp, dtype=str).fillna("")
    b3db0 = pd.read_csv(paths.b3db, sep="\t", dtype=str).fillna("")
    cons = pd.read_csv(paths.consensus, dtype=str).fillna("")
    ncats = pd.read_csv(paths.ncats_audit, dtype=str).fillna("")

    req_bbbp = {"smiles", "p_np"}
    req_b3db = {"SMILES", "BBB+/BBB-"}
    req_cons = {"parent_ik", "canon_smiles", "split", "in_reference_out", "bbbp_label", "b3db_label"}
    req_ncats = {"eligible_external", "y_pubchem", "ncats_parent_smiles"}
    for name, df, req in [
        ("BBBP.csv", bbbp0, req_bbbp),
        ("B3DB_classification.tsv", b3db0, req_b3db),
        ("consensus_dataset.csv", cons, req_cons),
        ("ncats_unique_molecule_audit.csv", ncats, req_ncats),
    ]:
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"{name} missing columns: {sorted(miss)}")

    bbbp = standardize_raw(bbbp0, "smiles", "BBBP")
    bbbp["_label"] = bbbp["p_np"].map(normalize_binary)
    b3db = standardize_raw(b3db0, "SMILES", "B3DB")
    b3db["_label"] = b3db["BBB+/BBB-"].map(normalize_b3db_label)

    train_mask = cons["split"].eq("train") & pd.to_numeric(cons["in_reference_out"], errors="coerce").fillna(0).eq(0)
    test_mask = cons["split"].eq("test")
    train_cons = cons.loc[train_mask].copy().reset_index(drop=True)
    test_cons = cons.loc[test_mask].copy().reset_index(drop=True)
    eligible = ncats.loc[bool_series(ncats["eligible_external"])].copy().reset_index(drop=True)

    audit = {
        "input_files": {
            "BBBP.csv": {"path": str(paths.bbbp), "sha256": sha256(paths.bbbp), "rows": len(bbbp0)},
            "B3DB_classification.tsv": {"path": str(paths.b3db), "sha256": sha256(paths.b3db), "rows": len(b3db0)},
            "consensus_dataset.csv": {"path": str(paths.consensus), "sha256": sha256(paths.consensus), "rows": len(cons)},
            "ncats_unique_molecule_audit.csv": {"path": str(paths.ncats_audit), "sha256": sha256(paths.ncats_audit), "rows": len(ncats)},
        },
        "bbbp_parse_failures": int((~bbbp["_parse_ok"]).sum()),
        "b3db_parse_failures": int((~b3db["_parse_ok"]).sum()),
        "consensus_total": len(cons),
        "frozen_train_parent_n": len(train_cons),
        "frozen_bbbp_test_n": len(test_cons),
        "frozen_pampa_eligible_n": len(eligible),
    }

    if strict:
        expected = {
            "BBBP raw rows": (len(bbbp0), 2050),
            "B3DB classification raw rows": (len(b3db0), 7807),
            "consensus parent molecules": (len(cons), 4131),
            "frozen train parents": (len(train_cons), 1234),
            "frozen BBBP test": (len(test_cons), 291),
            "frozen PAMPA eligible": (len(eligible), 318),
        }
        bad = [f"{k}: observed {a}, expected {b}" for k, (a, b) in expected.items() if a != b]
        if bad:
            raise AssertionError("Strict frozen-input checks failed:\n" + "\n".join(bad))

    return bbbp, b3db, cons, ncats, audit


def group_label_info(df: pd.DataFrame, key_col: str) -> Dict[str, dict]:
    info: Dict[str, dict] = {}
    valid = df[df[key_col].notna() & np.isfinite(df["_label"])].copy()
    for key, g in valid.groupby(key_col, sort=False):
        labels = g["_label"].astype(int).tolist()
        unique = sorted(set(labels))
        n1 = int(sum(labels))
        n0 = len(labels) - n1
        majority = None if n1 == n0 else int(n1 > n0)
        parents = sorted({str(x) for x in g["_parent"].dropna().astype(str) if str(x)})
        info[str(key)] = {
            "first": int(labels[0]),
            "majority": majority,
            "tie": bool(n1 == n0),
            "conflict": bool(len(unique) > 1),
            "n_rows": len(labels),
            "n0": n0,
            "n1": n1,
            "first_row": int(g["_source_row"].iloc[0]),
            "parent": str(g["_parent"].iloc[0]),
            "parents": parents,
            "n_parents": len(parents),
            "canon": str(g["_canon"].iloc[0]),
        }
    return info


def build_views(bbbp: pd.DataFrame, b3db: pd.DataFrame, cons: pd.DataFrame, strict: bool) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    









    train_cons = cons[
        cons["split"].eq("train") & pd.to_numeric(cons["in_reference_out"], errors="coerce").fillna(0).eq(0)
    ].copy()
    if train_cons["parent_ik"].astype(str).duplicated().any():
        dup = train_cons.loc[train_cons["parent_ik"].astype(str).duplicated(keep=False), "parent_ik"].astype(str).unique()[:10]
        raise AssertionError(f"Frozen training consensus has duplicate parent_ik values, e.g. {dup.tolist()}")
    train_parents = set(train_cons["parent_ik"].astype(str))

    
    feature_lookup = train_cons.set_index(train_cons["parent_ik"].astype(str))["canon_smiles"].astype(str).to_dict()
    frozen_order_lookup = {str(parent): int(i) for i, parent in enumerate(train_cons["parent_ik"].astype(str).tolist())}
    empty_feature = [p for p, smi in feature_lookup.items() if not str(smi).strip()]
    if empty_feature:
        raise AssertionError(f"Frozen consensus has empty canon_smiles for training parents, e.g. {empty_feature[:10]}")

    
    
    b3tr = b3db[b3db["_parse_ok"] & b3db["_parent"].astype(str).isin(train_parents)].copy()
    bbtr = bbbp[bbbp["_parse_ok"] & bbbp["_parent"].astype(str).isin(train_parents)].copy()

    views: Dict[str, pd.DataFrame] = {}
    summaries = []
    definitions = []

    for identity, aggregation, filt in VIEW_DEFS:
        name = f"{identity}_{aggregation}_{filt}"
        key_col = "_parent" if identity == "parent" else "_canon"
        b3info = group_label_info(b3tr, key_col)
        bbinfo = group_label_info(bbtr, key_col)

        
        
        
        multi_parent_groups = [k for k, rec in b3info.items() if rec.get("n_parents", 0) != 1]
        bb_multi_parent_groups = [k for k, rec in bbinfo.items() if rec.get("n_parents", 0) != 1]
        if strict and (multi_parent_groups or bb_multi_parent_groups):
            ex_b3 = [(k, b3info[k].get("parents")) for k in multi_parent_groups[:5]]
            ex_bb = [(k, bbinfo[k].get("parents")) for k in bb_multi_parent_groups[:5]]
            raise AssertionError(
                f"{name}: construction identity groups map to !=1 frozen parent; shared-X control undefined. "
                f"B3DB examples={ex_b3}; BBBP examples={ex_bb}"
            )

        rows = []
        n_ties = 0
        n_consistency_drop = 0
        n_b3_conflict_groups = 0
        n_bb_conflict_groups = 0
        n_multi_parent_groups = len(multi_parent_groups)
        n_bb_multi_parent_groups = len(bb_multi_parent_groups)

        for key, rec in b3info.items():
            if rec["conflict"]:
                n_b3_conflict_groups += 1
            bbrec = bbinfo.get(key)
            bb_conflict = bool(bbrec and bbrec["conflict"])
            if bb_conflict:
                n_bb_conflict_groups += 1
            if aggregation == "majority" and rec["majority"] is None:
                n_ties += 1
                continue
            if filt == "consistent" and (rec["conflict"] or bb_conflict):
                n_consistency_drop += 1
                continue
            label = rec[aggregation]
            if label is None:
                continue

            parents = rec.get("parents", [])
            if len(parents) != 1:
                
                
                continue
            parent = parents[0]
            if parent not in feature_lookup:
                continue
            feature_smiles = str(feature_lookup[parent])

            rows.append({
                "view": name,
                "identity": identity,
                "aggregation": aggregation,
                "conflict_filter": filt,
                "train_key": key,
                "parent_ik": parent,
                
                "canon_smiles": feature_smiles,
                "feature_smiles": feature_smiles,
                
                "construction_identity_value": key,
                "construction_canon_smiles": key if identity == "canonical" else rec.get("canon", ""),
                "label": int(label),
                "b3db_n_rows": rec["n_rows"],
                "b3db_n0": rec["n0"],
                "b3db_n1": rec["n1"],
                "b3db_conflict": int(rec["conflict"]),
                "bbbp_conflict": int(bb_conflict),
                "source_first_row": rec["first_row"],
                "frozen_train_order": frozen_order_lookup[parent],
            })

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"Training view {name} is empty")
        if frame["train_key"].duplicated().any():
            raise AssertionError(f"Duplicate train_key in {name}")
        if frame["parent_ik"].duplicated().any():
            d = frame.loc[frame["parent_ik"].duplicated(keep=False), "parent_ik"].astype(str).unique()[:10]
            raise AssertionError(f"Duplicate parent_ik in downstream training view {name}: {d.tolist()}")
        if frame["label"].nunique() < 2:
            raise AssertionError(f"Training view {name} has <2 classes")
        
        
        frame = frame.sort_values("frozen_train_order", kind="mergesort").reset_index(drop=True)
        views[name] = frame
        summaries.append({
            "view": name,
            "identity": identity,
            "aggregation": aggregation,
            "conflict_filter": filt,
            "raw_b3db_rows_in_frozen_train_universe": len(b3tr),
            "identity_groups_with_b3db_labels": len(b3info),
            "identity_groups_mapping_to_multiple_parents": n_multi_parent_groups,
            "bbbp_identity_groups_mapping_to_multiple_parents": n_bb_multi_parent_groups,
            "majority_tie_groups_excluded": n_ties,
            "b3db_conflict_groups_before_filter": n_b3_conflict_groups,
            "bbbp_conflict_groups_before_filter": n_bb_conflict_groups,
            "consistent_filter_groups_excluded": n_consistency_drop,
            "training_n": len(frame),
            "positive_n": int(frame["label"].sum()),
            "negative_n": int((frame["label"] == 0).sum()),
            "positive_rate": float(frame["label"].mean()),
            "unique_parent_n": int(frame["parent_ik"].nunique()),
            "feature_representation": FEATURE_REPRESENTATION,
        })
        definitions.append({
            "view": name,
            "identity_rule": "14-character parent InChIKey connectivity block" if identity == "parent" else "canonical non-isomeric SMILES after primary standardization",
            "b3db_aggregation": "first valid label in original B3DB file order" if aggregation == "first" else "hard majority of all B3DB rows; exact ties unresolved/excluded",
            "conflict_rule": "retain resolvable labels" if filt == "all" else "exclude identities with internal binary conflict in B3DB or BBBP, computed inside frozen training parent universe only",
            "training_target": "view-specific B3DB hard binary label (0/1)",
            "downstream_feature_representation": FEATURE_REPRESENTATION,
            "construction_identity_used_as_feature": False,
        })

    summary = pd.DataFrame(summaries)
    defs = pd.DataFrame(definitions)

    
    
    if strict:
        expected = {
            "parent_first_all": (1157, 425, 732),
            "parent_first_consistent": (1149, 422, 727),
            "parent_majority_all": (1152, 423, 729),
            "parent_majority_consistent": (1149, 422, 727),
            "canonical_first_all": (1157, 425, 732),
            "canonical_first_consistent": (1149, 422, 727),
            "canonical_majority_all": (1152, 423, 729),
            "canonical_majority_consistent": (1149, 422, 727),
        }
        bad = []
        for view, (n, pos, neg) in expected.items():
            f = views[view]
            got = (len(f), int(f["label"].sum()), int((f["label"] == 0).sum()))
            if got != (n, pos, neg):
                bad.append(f"{view}: observed n/pos/neg={got}, expected {(n, pos, neg)}")
        if bad:
            raise AssertionError("V3 target/membership integrity checks failed; construction logic changed:\n" + "\n".join(bad))

    
    pfa = views["parent_first_all"].copy()
    hard = pd.to_numeric(train_cons["b3db_label"], errors="coerce")
    old = train_cons.loc[np.isfinite(hard), ["parent_ik"]].copy()
    old["old_label"] = hard[np.isfinite(hard)].astype(int).to_numpy()
    chk = pfa.merge(old, on="parent_ik", how="outer", indicator=True)
    chk["label_match"] = np.where(chk["_merge"].eq("both"), chk["label"].eq(chk["old_label"]), False)
    if strict:
        if len(pfa) != 1157:
            raise AssertionError(f"parent_first_all training_n={len(pfa)}; expected historical 1157")
        if not (chk["_merge"].eq("both").all() and chk["label_match"].all()):
            raise AssertionError("parent_first_all does not reproduce frozen Stage-7 B3DB hard labels")
        if pfa["parent_ik"].astype(str).tolist() != old["parent_ik"].astype(str).tolist():
            raise AssertionError("parent_first_all parent order does not reproduce historical Stage-7 consensus-row order")

    
    eq_rows = []
    names = list(views)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = views[left][["parent_ik", "label", "feature_smiles"]].rename(
                columns={"label": "left_label", "feature_smiles": "left_feature_smiles"}
            )
            b = views[right][["parent_ik", "label", "feature_smiles"]].rename(
                columns={"label": "right_label", "feature_smiles": "right_feature_smiles"}
            )
            m = a.merge(b, on="parent_ik", how="outer", indicator=True)
            both = m["_merge"].eq("both")
            label_mismatch = int((both & m["left_label"].ne(m["right_label"])).sum())
            feature_mismatch = int((both & m["left_feature_smiles"].ne(m["right_feature_smiles"])).sum())
            eq_rows.append({
                "left": left,
                "right": right,
                "left_n": len(a),
                "right_n": len(b),
                "shared_parent_n": int(both.sum()),
                "left_only_parent_n": int(m["_merge"].eq("left_only").sum()),
                "right_only_parent_n": int(m["_merge"].eq("right_only").sum()),
                "shared_label_mismatch_n": label_mismatch,
                "shared_feature_smiles_mismatch_n": feature_mismatch,
                "same_parent_set": bool(m["_merge"].eq("both").all()),
                "same_labels_on_shared": bool(label_mismatch == 0),
                "same_features_on_shared": bool(feature_mismatch == 0),
            })
    equivalence = pd.DataFrame(eq_rows)
    if strict and int(equivalence["shared_feature_smiles_mismatch_n"].sum()) != 0:
        raise AssertionError("Shared-X control failed: shared parents have different feature_smiles across views")

    
    
    
    control_rows = []
    for aggregation in ["first", "majority"]:
        for filt in ["all", "consistent"]:
            left = f"parent_{aggregation}_{filt}"
            right = f"canonical_{aggregation}_{filt}"
            row = equivalence[(equivalence["left"].eq(left) & equivalence["right"].eq(right)) |
                              (equivalence["left"].eq(right) & equivalence["right"].eq(left))]
            if row.empty:
                raise AssertionError(f"Missing identity-axis equivalence row for {left} vs {right}")
            r = row.iloc[0].to_dict()
            r.update({"aggregation": aggregation, "conflict_filter": filt})
            control_rows.append(r)
    representation_control = pd.DataFrame(control_rows)
    if strict:
        bad = representation_control[
            (~representation_control["same_parent_set"]) |
            (~representation_control["same_labels_on_shared"]) |
            (~representation_control["same_features_on_shared"])
        ]
        if len(bad):
            raise AssertionError(
                "Formal v3 identity-axis control failed: corresponding parent/canonical views are not identical in "
                "membership/labels/shared-X.\n" + bad.to_string(index=False)
            )

    return views, summary, chk, equivalence, representation_control

def make_fingerprints(smiles: Iterable[str]) -> np.ndarray:
    smiles = list(smiles)
    gen = GetMorganGenerator(radius=RADIUS, fpSize=N_BITS, includeChirality=True)
    X = np.zeros((len(smiles), N_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid standardized SMILES at feature row {i}: {smi!r}")
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), X[i])
    return X


def load_eval_sets(cons: pd.DataFrame, ncats: pd.DataFrame, strict: bool) -> Dict[str, pd.DataFrame]:
    test = cons[cons["split"].eq("test")].copy().reset_index(drop=True)
    test["gold"] = pd.to_numeric(test["bbbp_label"], errors="coerce")
    if test["gold"].isna().any():
        raise AssertionError("Frozen BBBP test contains missing bbbp_label")
    test["gold"] = test["gold"].astype(int)
    test["molecule_id"] = test["parent_ik"].astype(str)
    test["eval_smiles"] = test["canon_smiles"].astype(str)
    test["endpoint"] = "BBBP_test"

    ext = ncats.loc[bool_series(ncats["eligible_external"])].copy().reset_index(drop=True)
    ext["gold"] = pd.to_numeric(ext["y_pubchem"], errors="coerce")
    if ext["gold"].isna().any():
        raise AssertionError("Eligible NCATS rows contain missing y_pubchem")
    ext["gold"] = ext["gold"].astype(int)
    if "ncats_parent_inchikey" in ext.columns:
        ext["molecule_id"] = ext["ncats_parent_inchikey"].astype(str)
    elif "PUBCHEM_CID" in ext.columns:
        ext["molecule_id"] = "CID:" + ext["PUBCHEM_CID"].astype(str)
    else:
        ext["molecule_id"] = [f"PAMPA:{i}" for i in range(len(ext))]
    ext["eval_smiles"] = ext["ncats_parent_smiles"].astype(str)
    ext["endpoint"] = "PAMPA_BBB"

    if strict:
        assert len(test) == 291
        assert len(ext) == 318
    return {"BBBP_test": test, "PAMPA_BBB": ext}


def safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    try:
        if method == "pearson":
            return float(pearsonr(a, b).statistic)
        return float(spearmanr(a, b).statistic)
    except Exception:
        return float("nan")


def metric_pair(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def topk_set(ids: List[str], p: np.ndarray, frac: float) -> set:
    n = len(ids)
    k = max(1, int(math.ceil(n * frac)))
    
    order = sorted(range(n), key=lambda i: (-float(p[i]), str(ids[i])))[:k]
    return {str(ids[i]) for i in order}


def run_models(views: Dict[str, pd.DataFrame], eval_sets: Dict[str, pd.DataFrame], seeds: List[int], n_jobs: int, out: Path):
    view_features = {}
    for name, frame in views.items():
        view_features[name] = make_fingerprints(frame["feature_smiles"])
    eval_features = {name: make_fingerprints(frame["eval_smiles"]) for name, frame in eval_sets.items()}

    per_seed_rows = []
    pred_rows = []
    arrays: Dict[str, Dict[str, List[np.ndarray]]] = {
        ev: {view: [] for view in views} for ev in eval_sets
    }
    start = time.time()
    total = len(seeds) * len(views)
    done = 0

    for seed in seeds:
        for view_name, train in views.items():
            params = dict(XGB_PARAMS)
            params.update(random_state=seed, n_jobs=n_jobs)
            model = xgb.XGBRegressor(**params)
            model.fit(view_features[view_name], train["label"].to_numpy(float))
            for ev_name, ev in eval_sets.items():
                p = np.clip(model.predict(eval_features[ev_name]), 0.0, 1.0)
                y = ev["gold"].to_numpy(int)
                auc, ap = metric_pair(y, p)
                per_seed_rows.append({
                    "seed": seed, "view": view_name, "eval": ev_name,
                    "n": len(y), "positive_n": int(y.sum()),
                    "roc_auc": auc, "pr_auc": ap,
                })
                arrays[ev_name][view_name].append(p)
                for i, pred in enumerate(p):
                    pred_rows.append({
                        "seed": seed, "view": view_name, "eval": ev_name,
                        "molecule_index": i,
                        "molecule_id": str(ev.iloc[i]["molecule_id"]),
                        "gold": int(y[i]), "prediction": float(pred),
                    })
            done += 1
            elapsed = time.time() - start
            print(f"[{done:3d}/{total}] seed={seed} view={view_name} elapsed={elapsed/60:.1f} min", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    pred_by_seed = pd.DataFrame(pred_rows)
    per_seed.to_csv(out / "metrics_per_seed.csv", index=False, encoding="utf-8-sig")
    pred_by_seed.to_csv(out / "predictions_by_seed.csv", index=False, encoding="utf-8-sig")

    mean_arrays: Dict[str, Dict[str, np.ndarray]] = {ev: {} for ev in eval_sets}
    mean_rows = []
    metric_rows = []
    for ev_name, ev in eval_sets.items():
        y = ev["gold"].to_numpy(int)
        ids = ev["molecule_id"].astype(str).tolist()
        for view_name in views:
            A = np.vstack(arrays[ev_name][view_name])
            pmean = A.mean(axis=0)
            mean_arrays[ev_name][view_name] = pmean
            auc, ap = metric_pair(y, pmean)
            s = per_seed[(per_seed["eval"] == ev_name) & (per_seed["view"] == view_name)]
            metric_rows.append({
                "view": view_name, "eval": ev_name, "n": len(y), "positive_n": int(y.sum()),
                "roc_auc_mean_prediction": auc, "pr_auc_mean_prediction": ap,
                "roc_auc_seed_mean": float(s["roc_auc"].mean()), "roc_auc_seed_sd": float(s["roc_auc"].std(ddof=1)) if len(s) > 1 else 0.0,
                "pr_auc_seed_mean": float(s["pr_auc"].mean()), "pr_auc_seed_sd": float(s["pr_auc"].std(ddof=1)) if len(s) > 1 else 0.0,
                "seed_n": len(s),
            })
            for i, pred in enumerate(pmean):
                mean_rows.append({
                    "view": view_name, "eval": ev_name, "molecule_index": i,
                    "molecule_id": ids[i], "gold": int(y[i]), "mean_prediction": float(pred),
                })
    metrics_summary = pd.DataFrame(metric_rows)
    mean_long = pd.DataFrame(mean_rows)
    metrics_summary.to_csv(out / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    mean_long.to_csv(out / "mean_predictions_long.csv", index=False, encoding="utf-8-sig")
    return per_seed, pred_by_seed, mean_arrays, mean_long, metrics_summary


def identity_axis_prediction_control(pred_by_seed: pd.DataFrame, out: Path, strict: bool) -> pd.DataFrame:
    
    rows = []
    for aggregation in ["first", "majority"]:
        for filt in ["all", "consistent"]:
            left = f"parent_{aggregation}_{filt}"
            right = f"canonical_{aggregation}_{filt}"
            for ev in sorted(pred_by_seed["eval"].unique()):
                for seed in sorted(pred_by_seed["seed"].unique()):
                    a = pred_by_seed[(pred_by_seed["view"].eq(left)) & (pred_by_seed["eval"].eq(ev)) & (pred_by_seed["seed"].eq(seed))][["molecule_id", "prediction"]].rename(columns={"prediction": "left_prediction"})
                    b = pred_by_seed[(pred_by_seed["view"].eq(right)) & (pred_by_seed["eval"].eq(ev)) & (pred_by_seed["seed"].eq(seed))][["molecule_id", "prediction"]].rename(columns={"prediction": "right_prediction"})
                    m = a.merge(b, on="molecule_id", how="outer", indicator=True)
                    both = m["_merge"].eq("both")
                    if not m["_merge"].eq("both").all():
                        max_diff = float("nan")
                        mean_diff = float("nan")
                    else:
                        d = np.abs(m.loc[both, "left_prediction"].to_numpy(float) - m.loc[both, "right_prediction"].to_numpy(float))
                        max_diff = float(np.max(d)) if len(d) else 0.0
                        mean_diff = float(np.mean(d)) if len(d) else 0.0
                    rows.append({
                        "aggregation": aggregation,
                        "conflict_filter": filt,
                        "left": left,
                        "right": right,
                        "eval": ev,
                        "seed": int(seed),
                        "n_left": len(a),
                        "n_right": len(b),
                        "all_molecule_ids_match": bool(m["_merge"].eq("both").all()),
                        "mean_abs_prediction_difference": mean_diff,
                        "max_abs_prediction_difference": max_diff,
                    })
    outdf = pd.DataFrame(rows)
    outdf.to_csv(out / "identity_axis_prediction_control.csv", index=False, encoding="utf-8-sig")
    if strict:
        bad = outdf[(~outdf["all_molecule_ids_match"]) | (outdf["max_abs_prediction_difference"].fillna(np.inf) > 1e-12)]
        if len(bad):
            raise AssertionError(
                "V3 identity-axis prediction control failed: corresponding parent/canonical models differ despite "
                "identical membership/labels/shared-X/order.\n" + bad.head(20).to_string(index=False)
            )
    return outdf


def pairwise_sensitivity(mean_arrays: Dict[str, Dict[str, np.ndarray]], eval_sets: Dict[str, pd.DataFrame], out: Path):
    rows = []
    top_rows = []
    view_names = list(next(iter(mean_arrays.values())).keys())
    for ev_name, ev in eval_sets.items():
        ids = ev["molecule_id"].astype(str).tolist()
        
        for view in view_names:
            p = mean_arrays[ev_name][view]
            top10 = topk_set(ids, p, 0.10)
            top20 = topk_set(ids, p, 0.20)
            for i, mol_id in enumerate(ids):
                top_rows.append({
                    "eval": ev_name, "view": view, "molecule_id": mol_id,
                    "mean_prediction": float(p[i]),
                    "top10pct": int(mol_id in top10), "top20pct": int(mol_id in top20),
                })
        for i, left in enumerate(view_names):
            for right in view_names[i + 1:]:
                a, b = mean_arrays[ev_name][left], mean_arrays[ev_name][right]
                diff = np.abs(a - b)
                row = {
                    "eval": ev_name, "left": left, "right": right, "n": len(ids),
                    "pearson_prediction": safe_corr(a, b, "pearson"),
                    "spearman_prediction": safe_corr(a, b, "spearman"),
                    "mean_abs_prediction_difference": float(np.mean(diff)),
                    "rmse_prediction_difference": float(np.sqrt(np.mean((a - b) ** 2))),
                    "max_abs_prediction_difference": float(np.max(diff)),
                }
                for frac, label in [(0.10, "top10pct"), (0.20, "top20pct")]:
                    sa, sb = topk_set(ids, a, frac), topk_set(ids, b, frac)
                    inter = len(sa & sb)
                    union = len(sa | sb)
                    k = len(sa)
                    row[f"{label}_k"] = k
                    row[f"{label}_overlap_n"] = inter
                    row[f"{label}_overlap_fraction"] = inter / k if k else np.nan
                    row[f"{label}_jaccard"] = inter / union if union else np.nan
                rows.append(row)
    pair = pd.DataFrame(rows)
    top = pd.DataFrame(top_rows)
    pair.to_csv(out / "pairwise_prediction_sensitivity.csv", index=False, encoding="utf-8-sig")
    top.to_csv(out / "topk_membership.csv", index=False, encoding="utf-8-sig")
    return pair, top


def stable_seed(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return (BOOTSTRAP_SEED + int(h, 16)) % (2**32 - 1)


def bootstrap_vs_primary(mean_arrays: Dict[str, Dict[str, np.ndarray]], eval_sets: Dict[str, pd.DataFrame], n_boot: int, out: Path):
    summary_rows = []
    dist_rows = []
    for ev_name, ev in eval_sets.items():
        y = ev["gold"].to_numpy(int)
        p0 = mean_arrays[ev_name][PRIMARY_VIEW]
        auc0, ap0 = metric_pair(y, p0)
        for view, p in mean_arrays[ev_name].items():
            if view == PRIMARY_VIEW:
                continue
            auc1, ap1 = metric_pair(y, p)
            obs = {
                "delta_roc_auc": auc1 - auc0,
                "delta_pr_auc": ap1 - ap0,
                "mean_abs_prediction_difference": float(np.mean(np.abs(p - p0))),
            }
            rng = np.random.default_rng(stable_seed(f"{ev_name}|{view}|{n_boot}"))
            vals = {k: [] for k in obs}
            valid = 0
            attempts = 0
            while valid < n_boot:
                attempts += 1
                if attempts > max(10000, n_boot * 50):
                    raise RuntimeError(f"Could not obtain {n_boot} valid bootstrap samples for {ev_name}/{view}")
                idx = rng.integers(0, len(y), len(y))
                yy = y[idx]
                if len(np.unique(yy)) < 2:
                    continue
                b0, b1 = p0[idx], p[idx]
                a0, r0 = metric_pair(yy, b0)
                a1, r1 = metric_pair(yy, b1)
                v = {
                    "delta_roc_auc": a1 - a0,
                    "delta_pr_auc": r1 - r0,
                    "mean_abs_prediction_difference": float(np.mean(np.abs(b1 - b0))),
                }
                for metric, value in v.items():
                    vals[metric].append(value)
                    dist_rows.append({
                        "bootstrap": valid + 1, "eval": ev_name, "view": view,
                        "primary_view": PRIMARY_VIEW, "metric": metric, "value": value,
                    })
                valid += 1
            for metric, arr in vals.items():
                arr = np.asarray(arr, float)
                summary_rows.append({
                    "eval": ev_name, "view": view, "primary_view": PRIMARY_VIEW,
                    "metric": metric, "observed": obs[metric],
                    "bootstrap_mean": float(arr.mean()),
                    "ci95_lo": float(np.quantile(arr, 0.025)),
                    "ci95_hi": float(np.quantile(arr, 0.975)),
                    "n_boot_valid": len(arr),
                })
    summary = pd.DataFrame(summary_rows)
    dist = pd.DataFrame(dist_rows)
    summary.to_csv(out / "primary_view_bootstrap.csv", index=False, encoding="utf-8-sig")
    dist.to_csv(out / "primary_view_bootstrap_distribution.csv", index=False, encoding="utf-8-sig")
    return summary, dist


def package_results(out: Path) -> Path:
    keep = [
        "analysis_config.json", "input_audit.json", "run_manifest.json", "rule_definitions.csv",
        "training_view_summary.csv", "training_views_long.csv", "historical_anchor_check.csv",
        "training_view_equivalence.csv", "representation_control_check.csv", "identity_axis_prediction_control.csv",
        "metrics_per_seed.csv", "metrics_summary.csv", "predictions_by_seed.csv",
        "mean_predictions_long.csv", "pairwise_prediction_sensitivity.csv", "topk_membership.csv",
        "primary_view_bootstrap.csv", "primary_view_bootstrap_distribution.csv", "run.log",
    ]
    zpath = out / "RESULTS_TO_SEND_FOR_ANALYSIS.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in keep:
            p = out / name
            if p.exists():
                z.write(p, arcname=name)
    return zpath


def check_only(paths: Paths, strict: bool) -> dict:
    bbbp, b3db, cons, ncats, audit = load_inputs(paths, strict)
    views, summary, anchor, equivalence, representation_control = build_views(bbbp, b3db, cons, strict)
    eval_sets = load_eval_sets(cons, ncats, strict)
    audit["view_training_summary"] = summary.to_dict("records")
    audit["shared_feature_representation"] = FEATURE_REPRESENTATION
    audit["training_view_equivalence"] = equivalence.to_dict("records")
    audit["representation_control_check"] = representation_control.to_dict("records")
    audit["historical_anchor_parent_first_all_match_rate"] = float(anchor["label_match"].mean()) if len(anchor) else None
    audit["evaluation_sets"] = {k: {"n": len(v), "positive_n": int(v["gold"].sum())} for k, v in eval_sets.items()}
    return audit


def main() -> int:
    ap = argparse.ArgumentParser(description="BBB controlled training-target and label evidence computation")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Package root")
    ap.add_argument("--output-dir", default=None, help="Output directory; default ROOT/results/multiview_hardlabel_v3")
    ap.add_argument("--seeds", default=None, help="Comma-separated seeds; default 42..61")
    ap.add_argument("--n-jobs", type=int, default=max(1, min(16, os.cpu_count() or 1)), help="XGBoost threads per model")
    ap.add_argument("--n-boot", type=int, default=2000, help="Paired molecule bootstrap replicates")
    ap.add_argument("--check-inputs", action="store_true", help="Validate inputs/views and exit")
    ap.add_argument("--non-strict", action="store_true", help="Disable frozen-size assertions (for synthetic/debug only)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.output_dir).resolve() if args.output_dir else root / "results" / "multiview_hardlabel_v3"
    out.mkdir(parents=True, exist_ok=True)
    paths = Paths(
        root=root,
        bbbp=root / "data" / "raw" / "BBBP" / "BBBP.csv",
        b3db=root / "data" / "raw" / "B3DB" / "B3DB_classification.tsv",
        consensus=root / "data" / "frozen" / "consensus_dataset.csv",
        ncats_audit=root / "data" / "frozen" / "ncats_unique_molecule_audit.csv",
        out=out,
    )
    strict = not args.non_strict

    if args.check_inputs:
        audit = check_only(paths, strict)
        write_json(out / "input_audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print("INPUT CHECK: PASS")
        return 0

    seeds = parse_seeds(args.seeds)
    bbbp, b3db, cons, ncats, audit = load_inputs(paths, strict)
    views, view_summary, anchor, equivalence, representation_control = build_views(bbbp, b3db, cons, strict)
    eval_sets = load_eval_sets(cons, ncats, strict)

    config = {
        "analysis": "8 construction views x hard binary label x fixed parent-level X x fixed XGBoost protocol x BBBP/PAMPA evaluation",
        "version": "v3_shared_parent_feature_control",
        "primary_view": PRIMARY_VIEW,
        "views": [f"{a}_{b}_{c}" for a, b, c in VIEW_DEFS],
        "seeds": seeds,
        "n_boot": args.n_boot,
        "morgan_radius": RADIUS,
        "morgan_bits": N_BITS,
        "morgan_include_chirality": True,
        "shared_training_feature_representation": FEATURE_REPRESENTATION,
        "construction_identity_used_as_downstream_feature": False,
        "xgboost": {**XGB_PARAMS, "n_jobs": args.n_jobs},
        "main_prediction": "mean prediction across prespecified seeds",
        "evaluation_sets": ["BBBP_test", "PAMPA_BBB"],
        "pampa_binary_rule": "Low=0; Moderate+High=1 from frozen ncats audit",
        "no_soft_or_probabilistic_targets_in_this_primary_experiment": True,
    }
    audit["view_training_summary"] = view_summary.to_dict("records")
    audit["shared_feature_representation"] = FEATURE_REPRESENTATION
    audit["training_view_equivalence"] = equivalence.to_dict("records")
    audit["representation_control_check"] = representation_control.to_dict("records")
    audit["historical_anchor_parent_first_all_match_rate"] = float(anchor["label_match"].mean()) if len(anchor) else None
    audit["evaluation_sets"] = {k: {"n": len(v), "positive_n": int(v["gold"].sum())} for k, v in eval_sets.items()}
    write_json(out / "analysis_config.json", config)
    write_json(out / "input_audit.json", audit)
    view_summary.to_csv(out / "training_view_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(views.values(), ignore_index=True).to_csv(out / "training_views_long.csv", index=False, encoding="utf-8-sig")
    anchor.to_csv(out / "historical_anchor_check.csv", index=False, encoding="utf-8-sig")
    equivalence.to_csv(out / "training_view_equivalence.csv", index=False, encoding="utf-8-sig")
    representation_control.to_csv(out / "representation_control_check.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {
            "view": f"{a}_{b}_{c}", "identity": a, "aggregation": b, "conflict_filter": c,
            "primary": int(f"{a}_{b}_{c}" == PRIMARY_VIEW),
            "downstream_feature_representation": FEATURE_REPRESENTATION,
            "construction_identity_used_as_feature": False,
        }
        for a, b, c in VIEW_DEFS
    ]).to_csv(out / "rule_definitions.csv", index=False, encoding="utf-8-sig")

    print("=== Training views ===")
    print(view_summary.to_string(index=False))
    print("=== V3 shared-X representation control ===")
    print(representation_control.to_string(index=False))
    print("=== Evaluation sets ===")
    for name, frame in eval_sets.items():
        print(name, len(frame), "positive=", int(frame["gold"].sum()))

    t0 = time.time()
    _, pred_by_seed, mean_arrays, _, metrics_summary = run_models(views, eval_sets, seeds, args.n_jobs, out)
    identity_axis_prediction_control(pred_by_seed, out, strict)
    pairwise_sensitivity(mean_arrays, eval_sets, out)
    bootstrap_vs_primary(mean_arrays, eval_sets, args.n_boot, out)
    elapsed = time.time() - t0

    manifest = {
        "status": "complete",
        "analysis_version": "v3_shared_parent_feature_control",
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion,
            "xgboost": xgb.__version__,
        },
        "input_sha256": audit["input_files"],
        "output_files": {},
    }
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name not in {"run_manifest.json", "RESULTS_TO_SEND_FOR_ANALYSIS.zip"}:
            manifest["output_files"][p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    write_json(out / "run_manifest.json", manifest)
    zpath = package_results(out)

    print("\n=== Main metrics on mean predictions ===")
    print(metrics_summary.to_string(index=False))
    print(f"\nCOMPLETE. Send this file for analysis:\n{zpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
