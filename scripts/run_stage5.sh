#!/usr/bin/env bash
# run_stage5.sh — Recomputes all chemical-space descriptor, scaffold, Morgan-neighborhood and fixed SMARTS sensitivity analyses using the independently generated Stage IV outputs and frozen BBBP identities.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
S3="$ROOT/reproduction_runs/stage3_analysis_input"
S4="$ROOT/reproduction_runs/analysis_stage4"
OUT="$ROOT/reproduction_runs/analysis_stage5"
if [ -f "$ROOT/reproduction_runs/full_stage4/RESULT_STATUS.txt" ]; then S3="$ROOT/reproduction_runs/full_stage3"; S4="$ROOT/reproduction_runs/full_stage4"; OUT="$ROOT/reproduction_runs/full_stage5"; fi
mkdir -p "$OUT"
python code/stage5/chemspace.py precheck --out "$OUT" --stage3 "$S3" --stage4 "$S4" --stage3-project "$ROOT/code/stage3"
python code/stage5/chemspace.py analyze --out "$OUT" --stage3 "$S3" --stage4 "$S4" --stage3-project "$ROOT/code/stage3" --device auto --n-perm 2000 --n-boot 2000
