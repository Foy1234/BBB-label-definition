#!/usr/bin/env python3
"""Audit full stereochemistry-preserving InChIKeys for the 15 connectivity-level mixed-class parent groups while reproducing the modeling parent identity through the stereochemistry-removed branch. The audit is read-only with respect to model inputs, labels, partitions, and predictions."""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey

EXPECTED_PARENT_KEYS = {
    "AOJJSUZBOXZQNB", "AQHHHDLHHXJYJD", "IMSSROKUHAOUJS", "LOUPRKONTZGTKE",
    "METKIMKYRPQLGS", "ORFOPKXBNMVMKC", "PMATZTZNYRCHOR", "RWTNPBWLLIMQHL",
    "SGDBTWWWUNNDEQ", "SGTNSNPWRIOYBX", "SZLZWPPUNLXJEA", "ULGZDMOVFRHVEP",
    "UREBDLICKHMUKA", "VAAUVRVFOQPIGI", "VWPOSFSPZNDTMJ",
}
EXPECTED_ROWS = 65
EXPECTED_FULL_KEYS = 65


def parse_label(value):
    s = str(value).strip().upper()
    if s in {"BBB+", "1", "TRUE", "POS", "POSITIVE"}:
        return 1
    if s in {"BBB-", "0", "FALSE", "NEG", "NEGATIVE"}:
        return 0
    return np.nan


def structure_keys(smiles: str, remover: SaltRemover.SaltRemover):
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    mol = remover.StripMol(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None, None

    full_ik = MolToInchiKey(Chem.Mol(mol))
    parent_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(parent_mol)
    parent_full_ik = MolToInchiKey(parent_mol)
    parent_ik = parent_full_ik.split("-")[0] if parent_full_ik else None
    iso_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return parent_ik, full_ik, iso_smiles


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[3]
    ap.add_argument("--b3db", default=str(root / "data" / "raw" / "B3DB" / "B3DB_classification.tsv"), help="B3DB classification TSV")
    ap.add_argument("--outdir", default=str(root / "results" / "audits" / "structure_identity"), help="Output directory")
    ap.add_argument("--strict", action="store_true", help="Fail if expected 65-row/65-key audit does not reproduce")
    args = ap.parse_args()

    if rdkit.__version__ != "2026.3.5":
        warnings.warn(
            f"RDKit {rdkit.__version__} detected; reproducibility package pins RDKit 2026.3.5. "
            "Re-run in the pinned environment before final publication."
        )

    src = Path(args.b3db)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src, sep="\t", dtype=str)
    required = {"NO.", "compound_name", "SMILES", "BBB+/BBB-", "reference"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required B3DB columns: {sorted(missing)}")

    remover = SaltRemover.SaltRemover()
    df = df.copy()
    df["_raw_row_index"] = np.arange(len(df))
    df["label01"] = df["BBB+/BBB-"].map(parse_label)
    keys = df["SMILES"].map(lambda s: structure_keys(s, remover))
    df["parent_ik"] = [x[0] for x in keys]
    df["full_isomeric_InChIKey"] = [x[1] for x in keys]
    df["isomeric_SMILES_after_salt_removal"] = [x[2] for x in keys]

    audit = df[
        df["parent_ik"].isin(EXPECTED_PARENT_KEYS)
        & df["label01"].isin([0, 1])
        & df["full_isomeric_InChIKey"].notna()
    ].copy()
    audit = audit.sort_values("_raw_row_index", kind="stable")

    if set(audit["parent_ik"].unique()) != EXPECTED_PARENT_KEYS:
        missing_parents = sorted(EXPECTED_PARENT_KEYS - set(audit["parent_ik"].unique()))
        extra_parents = sorted(set(audit["parent_ik"].unique()) - EXPECTED_PARENT_KEYS)
        raise RuntimeError(f"Parent-key set mismatch; missing={missing_parents}, extra={extra_parents}")

    by_key = audit.groupby("full_isomeric_InChIKey")["label01"].nunique()
    n_conflicting_full_keys = int((by_key > 1).sum())
    summary = (
        audit.groupby("parent_ik", sort=True)
        .agg(
            compound_names=("compound_name", lambda s: " | ".join(dict.fromkeys(s.astype(str)))),
            n_raw_records=("NO.", "size"),
            raw_record_numbers=("NO.", lambda s: "; ".join(s.astype(str))),
            n_BBB_positive=("label01", lambda s: int((s == 1).sum())),
            n_BBB_negative=("label01", lambda s: int((s == 0).sum())),
            n_distinct_full_InChIKey=("full_isomeric_InChIKey", "nunique"),
        )
        .reset_index()
    )
    summary["same_full_InChIKey_with_both_labels"] = summary["parent_ik"].map(
        audit.groupby("parent_ik").apply(
            lambda g: bool((g.groupby("full_isomeric_InChIKey")["label01"].nunique() > 1).any()),
            include_groups=False,
        ).to_dict()
    )

    n_rows = len(audit)
    n_full = int(audit["full_isomeric_InChIKey"].nunique())
    if args.strict and (n_rows != EXPECTED_ROWS or n_full != EXPECTED_FULL_KEYS or n_conflicting_full_keys != 0):
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, {EXPECTED_FULL_KEYS} distinct full keys, and 0 conflicting full keys; "
            f"observed rows={n_rows}, unique_full_keys={n_full}, conflicting_full_keys={n_conflicting_full_keys}."
        )

    records_out = outdir / "mixed15_full_inchikey_records.csv"
    summary_out = outdir / "mixed15_full_inchikey_summary.csv"
    metadata_out = outdir / "mixed15_full_inchikey_audit_metadata.json"
    keep = [
        "NO.", "_raw_row_index", "compound_name", "parent_ik", "BBB+/BBB-", "label01",
        "full_isomeric_InChIKey", "SMILES", "isomeric_SMILES_after_salt_removal",
        "reference", "threshold", "logBB", "group", "comments"
    ]
    audit[keep].to_csv(records_out, index=False)
    summary.to_csv(summary_out, index=False)
    metadata = {
        "rdkit_version": rdkit.__version__,
        "input_file": str(src),
        "n_parent_groups": int(audit["parent_ik"].nunique()),
        "n_raw_records": int(n_rows),
        "n_distinct_full_isomeric_InChIKey": int(n_full),
        "n_full_InChIKey_with_both_labels": int(n_conflicting_full_keys),
        "expected_parent_groups": 15,
        "expected_raw_records": EXPECTED_ROWS,
        "expected_distinct_full_keys": EXPECTED_FULL_KEYS,
        "stereochemistry_retained_for_full_key": True,
        "stereochemistry_removed_only_for_parent_key": True,
        "model_inputs_or_predictions_modified": False,
    }
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Wrote: {records_out}\nWrote: {summary_out}\nWrote: {metadata_out}")


if __name__ == "__main__":
    main()
