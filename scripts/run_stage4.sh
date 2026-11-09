#!/usr/bin/env bash
# run_stage4.sh — Performs frozen-prediction statistical analysis and exact-duplicate-audited independent PAMPA evaluation on the archived or recomputed Stage III run without changing model parameters.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
S3="$ROOT/reproduction_runs/stage3_analysis_input"
if [ ! -d "$S3" ]; then mkdir -p "$S3"; cp -a "$ROOT/results/stage3/." "$S3/"; fi
OUT="$ROOT/reproduction_runs/analysis_stage4"
if [ -f "$ROOT/reproduction_runs/full_stage3/analysis/coverage.json" ]; then S3="$ROOT/reproduction_runs/full_stage3"; OUT="$ROOT/reproduction_runs/full_stage4"; fi
mkdir -p "$OUT"
python code/stage4/finalize.py precheck --out "$OUT" --stage3-run "$S3" --stage3-project "$ROOT/code/stage3"
python code/stage4/finalize.py analyze --out "$OUT" --stage3-run "$S3" --stage3-project "$ROOT/code/stage3" --device auto --n-bootstrap 2000
python code/stage4/finalize.py diagnose-pampa --out "$OUT" --stage3-run "$S3" --stage3-project "$ROOT/code/stage3" --external "$ROOT/data/frozen/ncats_unique_molecule_audit.csv"
python code/stage4/finalize.py pampa --out "$OUT" --stage3-run "$S3" --stage3-project "$ROOT/code/stage3" --external "$ROOT/data/frozen/ncats_unique_molecule_audit.csv" --duplicate-policy identical-only --device auto --n-bootstrap 2000 --n-jobs 4
python code/stage4/finalize.py pack --out "$OUT" --stage3-run "$S3" --stage3-project "$ROOT/code/stage3"
