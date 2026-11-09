"""Reconstruct parent-level B3DB continuous logBB evidence.

Reads the original regression TSV, applies the historical parent-identity
processing, retains only finite numeric measurements with valid identities,
then reconstructs semicolon-separated three-decimal measurement lists,
minimum/maximum values, counts, and sorted unique source references. Each
valid original observation is also exported as an independent row. The
frozen consensus is not modified by this reconstruction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import parse_parent, references


def reconstruct(path, output):
    reg = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = {"SMILES", "logBB", "reference"}
    if not required.issubset(reg.columns):
        raise ValueError(f"B3DB regression fields missing: {sorted(required - set(reg.columns))}")
    identities = [parse_parent(s) for s in reg["SMILES"]]
    reg["parent_ik"] = [t[0] for t in identities]
    reg["canon_smiles"] = [t[1] for t in identities]
    reg["identity_status"] = [t[2] for t in identities]
    reg["row_number"] = reg.index + 1
    reg["logbb_numeric"] = pd.to_numeric(reg["logBB"], errors="coerce")
    is_finite = np.isfinite(reg["logbb_numeric"].to_numpy(dtype=float))
    usable = reg.loc[reg["parent_ik"].notna() & is_finite].copy()
    unusable = reg.loc[~(reg["parent_ik"].notna() & is_finite)].copy()
    unusable[["row_number", "SMILES", "logBB", "identity_status"]].to_csv(output / "logbb_excluded_source_records.csv", index=False)
    long_rows = []
    for row in usable.itertuples(index=False):
        refs = references(row.reference)
        long_rows.append({
            "source_row_number": int(row.row_number),
            "parent_ik": row.parent_ik,
            "canon_smiles": row.canon_smiles,
            "logbb": f"{row.logbb_numeric:.3f}",
            "logbb_reference_list": "|".join(refs),
            "n_logbb_references": len(refs),
        })
    pd.DataFrame(long_rows, columns=["source_row_number", "parent_ik", "canon_smiles", "logbb", "logbb_reference_list", "n_logbb_references"]).to_csv(output / "logbb_measurements_long.csv", index=False)
    parents = []
    for parent, group in usable.groupby("parent_ik", sort=True):
        nums = sorted(float(v) for v in group["logbb_numeric"])
        refset = sorted({ref for raw in group["reference"] for ref in references(raw)})
        parents.append({
            "parent_ik": parent,
            "b3db_logbb_list": ";".join(f"{v:.3f}" for v in nums),
            "b3db_logbb_min": f"{min(nums):.3f}",
            "b3db_logbb_max": f"{max(nums):.3f}",
            "b3db_logbb_n": len(nums),
            "b3db_logbb_refs": "|".join(refset),
        })
    agg = pd.DataFrame(parents, columns=["parent_ik", "b3db_logbb_list", "b3db_logbb_min", "b3db_logbb_max", "b3db_logbb_n", "b3db_logbb_refs"])
    agg.to_csv(output / "logbb_all_b3db_parents.csv", index=False)
    stats = {
        "raw_regression_rows": int(len(reg)),
        "valid_regression_rows_with_parent_identity": int(len(usable)),
        "unusable_regression_rows": int(len(unusable)),
        "unique_parents_with_valid_logbb": int(len(agg)),
    }
    return agg, stats
