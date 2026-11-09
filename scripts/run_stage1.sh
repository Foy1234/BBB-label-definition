#!/usr/bin/env bash
# run_stage1.sh — Recomputes the controlled hard-label construction audit from the original BBBP and B3DB source files and regenerates the B3DB evidence and 15-pair conflict sensitivity analyses using frozen parent identities.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/reproduction_runs/full_stage1"
mkdir -p "$OUT"
python code/stage1/hard_label_audit/stage7_multiview_hardlabel.py --root "$ROOT" --output-dir "$OUT/hard_label_audit" --n-jobs 4
python code/stage1/b3db_evidence_sensitivity/analyze_b3db_evidence_sensitivity.py --v3-results "$OUT/hard_label_audit" --consensus "$ROOT/data/frozen/consensus_dataset.csv" --b3db "$ROOT/data/raw/B3DB/B3DB_classification.tsv" --outdir "$OUT/b3db_evidence_sensitivity"
python code/stage1/conflict_case_analysis/analyze_15_conflict_cases.py --evidence-results "$OUT/b3db_evidence_sensitivity" --v3-results "$OUT/hard_label_audit" --b3db "$ROOT/data/raw/B3DB/B3DB_classification.tsv" --outdir "$OUT/conflict_case_analysis"
