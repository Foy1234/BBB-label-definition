#!/usr/bin/env bash
# run_stage2.sh — Builds the six molecular supervision target arrays and 99 stratified conditional permutations from byte-verified frozen training evidence without modifying baseline model inputs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/reproduction_runs/full_stage2"
mkdir -p "$OUT"
python code/stage2/fit_corrected_targets.py --consensus "$ROOT/data/frozen/consensus_dataset.csv" --labels "$ROOT/data/frozen/labels_abc.csv" --outdir "$OUT"
