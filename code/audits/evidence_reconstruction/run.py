"""Audit and package raw B3DB Vote and logBB evidence reconstruction.

Executes both raw-record reconstruction modules, validates the reconstructed
vote proportions against frozen label_B for all 1,157 common training
parents, and compares each reconstructed logBB evidence field with the
4,131-row frozen consensus. Exports molecule- and record-level diagnostics,
input checksums, versions, and a machine-readable pass/fail report. No
training target, model prediction, or frozen source file is modified.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit

from common import sha256
from reconstruct_logbb import reconstruct as reconstruct_logbb
from reconstruct_vote import reconstruct as reconstruct_vote


def comparable_count(value):
    if pd.isna(value) or str(value).strip() == "":
        return "0"
    return str(int(float(value)))


def string_field(value):
    if pd.isna(value):
        return ""
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct Vote and logBB from raw B3DB and validate frozen evidence")
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw" / "B3DB")
    parser.add_argument("--frozen-dir", type=Path, default=root / "data" / "frozen")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "audits" / "evidence_reconstruction")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw = args.raw_dir.resolve()
    frozen = args.frozen_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    files = {
        "B3DB_classification.tsv": raw / "B3DB_classification.tsv",
        "B3DB_regression.tsv": raw / "B3DB_regression.tsv",
        "consensus_dataset.csv": frozen / "consensus_dataset.csv",
        "labels_abc.csv": frozen / "labels_abc.csv",
    }
    missing = [k for k, v in files.items() if not v.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(missing))

    vote, vote_stats = reconstruct_vote(files["B3DB_classification.tsv"], output)
    logbb, logbb_stats = reconstruct_logbb(files["B3DB_regression.tsv"], output)
    frozen_labels = pd.read_csv(files["labels_abc.csv"], dtype=str)
    frozen_consensus = pd.read_csv(files["consensus_dataset.csv"], dtype=str)
    for d, name in [(frozen_labels, "labels_abc.csv"), (frozen_consensus, "consensus_dataset.csv")]:
        if "parent_ik" not in d.columns or d["parent_ik"].isna().any() or d["parent_ik"].duplicated().any():
            raise ValueError(name + " must contain a nonempty unique parent_ik column")
    required_labels = {"label_A", "label_B", "parent_ik"}
    if not required_labels.issubset(frozen_labels.columns):
        raise ValueError("Frozen labels missing: " + ", ".join(sorted(required_labels - set(frozen_labels.columns))))
    expected_logbb_fields = ["b3db_logbb_list", "b3db_logbb_min", "b3db_logbb_max", "b3db_logbb_n", "b3db_logbb_refs"]
    if any(k not in frozen_consensus.columns for k in expected_logbb_fields):
        raise ValueError("Frozen consensus missing one or more logBB evidence fields")

    labels_common = frozen_labels.loc[pd.to_numeric(frozen_labels["label_A"], errors="coerce").notna()].copy()
    vote_check = labels_common[["parent_ik", "label_A", "label_B"]].merge(vote, on="parent_ik", how="left", validate="one_to_one")
    vote_check["vote_frozen"] = pd.to_numeric(vote_check["label_B"], errors="coerce")
    vote_check["hard_frozen"] = pd.to_numeric(vote_check["label_A"], errors="coerce")
    vote_check["vote_abs_difference"] = (vote_check["vote_frozen"] - vote_check["vote_fraction"]).abs()
    vote_check["vote_matches"] = (vote_check["vote_abs_difference"] <= 1e-12) & vote_check["vote_frozen"].notna() & vote_check["vote_fraction"].notna()
    vote_check["first_label_matches_hard"] = (vote_check["first_valid_label"] == vote_check["hard_frozen"]) & vote_check["first_valid_label"].notna()
    vote_check.to_csv(output / "vote_common1157_frozen_comparison.csv", index=False)
    vote_check.loc[~vote_check["vote_matches"] | ~vote_check["first_label_matches_hard"]].to_csv(output / "vote_mismatches.csv", index=False)

    frozen_keys = set(frozen_consensus["parent_ik"])
    logbb.loc[~logbb["parent_ik"].isin(frozen_keys)].to_csv(output / "logbb_parents_outside_frozen_consensus.csv", index=False)
    logbb_check = frozen_consensus[["parent_ik"] + expected_logbb_fields].merge(logbb, on="parent_ik", how="left", suffixes=("_frozen", "_reconstructed"), validate="one_to_one")
    comparisons = {}
    for col in expected_logbb_fields:
        lhs = logbb_check[col + "_frozen"].map(comparable_count if col == "b3db_logbb_n" else string_field)
        rhs = logbb_check[col + "_reconstructed"].map(comparable_count if col == "b3db_logbb_n" else string_field)
        logbb_check[col + "_matches"] = lhs == rhs
        comparisons[col] = int((~logbb_check[col + "_matches"]).sum())
    match_fields = [x + "_matches" for x in expected_logbb_fields]
    logbb_check["all_evidence_fields_match"] = logbb_check[match_fields].all(axis=1)
    logbb_check.to_csv(output / "logbb_consensus_frozen_comparison.csv", index=False)
    logbb_check.loc[~logbb_check["all_evidence_fields_match"]].to_csv(output / "logbb_mismatches.csv", index=False)

    raw_in_common = int((vote_check["n_records"] > 0).sum())
    expected = {"classification_raw": 7807, "regression_raw": 1058, "frozen_consensus": 4131, "frozen_labels": 1234, "common_training": 1157}
    observed = {
        "classification_raw": vote_stats["raw_classification_rows"],
        "regression_raw": logbb_stats["raw_regression_rows"],
        "frozen_consensus": int(len(frozen_consensus)),
        "frozen_labels": int(len(frozen_labels)),
        "common_training": int(len(labels_common)),
    }
    count_checks = {k: observed[k] == v for k, v in expected.items()}
    vote_mismatch = int((~vote_check["vote_matches"]).sum())
    hard_mismatch = int((~vote_check["first_label_matches_hard"]).sum())
    logbb_mismatch = int((~logbb_check["all_evidence_fields_match"]).sum())
    succeeded = all(count_checks.values()) and vote_mismatch == 0 and hard_mismatch == 0 and logbb_mismatch == 0 and raw_in_common == len(labels_common)

    report = {
        "status": "PASS" if succeeded else "FAIL",
        "scope": "Raw B3DB classification/regression evidence reconstruction against frozen data; no model fitting",
        "environment": {"python": platform.python_version(), "rdkit": rdkit.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "input_sha256": {key: sha256(path) for key, path in files.items()},
        "expected_row_counts": expected,
        "observed_row_counts": observed,
        "expected_row_count_checks": count_checks,
        "classification": vote_stats,
        "regression": logbb_stats,
        "training_parents_with_raw_vote": raw_in_common,
        "training_parents_with_nonbinary_vote": int(((vote_check["vote_fraction"] > 0) & (vote_check["vote_fraction"] < 1)).sum()),
        "vote_mismatched_training_parents": vote_mismatch,
        "hard_first_label_mismatched_training_parents": hard_mismatch,
        "max_vote_absolute_difference": None if vote_check["vote_abs_difference"].isna().all() else float(vote_check["vote_abs_difference"].max()),
        "logbb_mismatched_consensus_parents": logbb_mismatch,
        "logbb_field_mismatch_counts": comparisons,
        "logbb_parents_in_frozen_consensus": int(logbb["parent_ik"].isin(frozen_keys).sum()),
        "logbb_parents_outside_frozen_consensus": int((~logbb["parent_ik"].isin(frozen_keys)).sum()),
    }
    (output / "qc_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [
        "# B3DB original-record evidence reconstruction and frozen-input validation",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Classification: {vote_stats['raw_classification_rows']} source rows; {vote_stats['valid_classification_rows_with_parent_identity']} valid rows; {vote_stats['unique_parents_with_valid_classification']} unique parents.",
        f"Continuous logBB: {logbb_stats['raw_regression_rows']} source rows; {logbb_stats['valid_regression_rows_with_parent_identity']} valid measurements; {logbb_stats['unique_parents_with_valid_logbb']} unique parents.",
        f"Vote: {len(labels_common)} common training parents compared; {vote_mismatch} vote mismatches; {hard_mismatch} first-record hard-label mismatches; {report['training_parents_with_nonbinary_vote']} nonbinary vote values.",
        f"logBB: {len(frozen_consensus)} frozen consensus parents compared; {logbb_mismatch} mismatched parent rows; {report['logbb_parents_outside_frozen_consensus']} valid regression-only parent identities not present in the frozen consensus.",
        "",
        "Reconstruction does not regenerate or overwrite the frozen consensus, frozen labels, Stage II targets, predictions, or manuscript statistics.",
        "",
        "For detailed field-level comparisons see logbb_consensus_frozen_comparison.csv and vote_common1157_frozen_comparison.csv.",
        "The recorded software versions are those of the execution environment; compatibility is verified by the molecule-level comparisons, not inferred from version equality.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("B3DB evidence reconstruction:", report["status"])
    print("Source classification:", vote_stats)
    print("Source regression:", logbb_stats)
    print("Common-training Vote mismatches:", vote_mismatch)
    print("Common-training Hard/first-label mismatches:", hard_mismatch)
    print("Frozen consensus logBB mismatches:", logbb_mismatch)
    print("QC report:", output / "qc_report.json")
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
