"""Reconstruct original-record B3DB vote proportions.

Reads the complete B3DB classification TSV in its frozen row order, groups
valid BBB+/BBB- observations by standardized parent connectivity InChIKey,
and computes the positive fraction using all original classification rows.
Retains positive/negative record counts, first valid class, conflict flags,
and rejected source records. It never derives vote proportions from the
consensus table's deduplicated class-state list.
"""

from __future__ import annotations

import pandas as pd

from common import parse_parent


def reconstruct(path, output):
    cls = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"SMILES", "BBB+/BBB-"}
    if not required.issubset(cls.columns):
        raise ValueError(f"B3DB classification fields missing: {sorted(required - set(cls.columns))}")
    identities = [parse_parent(s) for s in cls["SMILES"]]
    cls["parent_ik"] = [t[0] for t in identities]
    cls["canon_smiles"] = [t[1] for t in identities]
    cls["identity_status"] = [t[2] for t in identities]
    cls["row_number"] = cls.index + 1
    classes = cls["BBB+/BBB-"].str.strip()
    cls["classification"] = classes.map({"BBB+": 1, "BBB-": 0})
    usable = cls.loc[cls["parent_ik"].notna() & cls["classification"].notna()].copy()
    usable["classification"] = usable["classification"].astype(int)
    group = usable.groupby("parent_ik", sort=True)["classification"]
    agg = group.agg(n_records="size", n_positive="sum", first_valid_label="first").reset_index()
    agg["n_negative"] = agg["n_records"] - agg["n_positive"]
    agg["vote_fraction"] = agg["n_positive"] / agg["n_records"]
    agg["internal_conflict"] = ((agg["n_positive"] > 0) & (agg["n_negative"] > 0)).astype(int)
    agg = agg[["parent_ik", "n_records", "n_positive", "n_negative", "first_valid_label", "vote_fraction", "internal_conflict"]]
    agg.to_csv(output / "vote_all_b3db_parents.csv", index=False)
    unusable = cls.loc[~(cls["parent_ik"].notna() & cls["classification"].notna()),
                      ["row_number", "NO.", "SMILES", "BBB+/BBB-", "identity_status"]].copy() if "NO." in cls else cls.loc[~(cls["parent_ik"].notna() & cls["classification"].notna()), ["row_number", "SMILES", "BBB+/BBB-", "identity_status"]].copy()
    unusable.to_csv(output / "vote_excluded_source_records.csv", index=False)
    stats = {
        "raw_classification_rows": int(len(cls)),
        "valid_classification_rows_with_parent_identity": int(len(usable)),
        "unusable_classification_rows": int(len(unusable)),
        "unique_parents_with_valid_classification": int(len(agg)),
        "conflicting_parents_all_b3db": int(agg["internal_conflict"].sum()),
    }
    return agg, stats
