#!/usr/bin/env bash
# run_stage6.sh — Performs three repeated five-fold cross-fitted Platt calibrations on frozen or recomputed Stage IV PAMPA predictions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="$ROOT/results/stage4/pampa/pampa_predictions_by_seed.csv"
REFERENCE="$ROOT/results/stage4/pampa/pampa_mean_seed_metrics.csv"
OUT="$ROOT/reproduction_runs/stage6_pampa_calibration"

if [ -f "$ROOT/reproduction_runs/full_stage4/pampa/pampa_predictions_by_seed.csv" ]; then
  INPUT="$ROOT/reproduction_runs/full_stage4/pampa/pampa_predictions_by_seed.csv"
  REFERENCE="$ROOT/reproduction_runs/full_stage4/pampa/pampa_mean_seed_metrics.csv"
  OUT="$ROOT/reproduction_runs/full_stage6"
elif [ -f "$ROOT/reproduction_runs/analysis_stage4/pampa/pampa_predictions_by_seed.csv" ]; then
  INPUT="$ROOT/reproduction_runs/analysis_stage4/pampa/pampa_predictions_by_seed.csv"
  REFERENCE="$ROOT/reproduction_runs/analysis_stage4/pampa/pampa_mean_seed_metrics.csv"
fi

mkdir -p "$OUT"
python code/stage6/pampa_calibration.py --input "$INPUT" --reference "$REFERENCE" --out "$OUT" --seeds 42,43,44 --folds 5
