#!/usr/bin/env bash
# run_stage3.sh — Repeats Stage III prechecks and all six target definitions for 20 XGBoost and 10 GINE seeds, including 99 conditional permutations under the original matched three-seed controls.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/reproduction_runs/full_stage3"
S2="$ROOT/reproduction_runs/full_stage2"
mkdir -p "$OUT"
python code/stage3/stage3.py precheck --run "$OUT" --stage2 "$S2" --consensus "$ROOT/data/frozen/consensus_dataset.csv" --labels "$ROOT/data/frozen/labels_abc.csv" --scaffold "$ROOT/data/frozen/scaffold_split_assignments.csv"
for enc in xgb gine; do
  python code/stage3/stage3.py train --run "$OUT" --encoder "$enc" --group main --n-jobs 4
  python code/stage3/stage3.py train --run "$OUT" --encoder "$enc" --group perm --perm-start 0 --perm-stop 98 --n-jobs 4
done
python code/stage3/stage3.py summarize --run "$OUT"
