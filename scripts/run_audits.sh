#!/usr/bin/env bash
# run_audits.sh — Reconstructs raw B3DB Vote/logBB evidence and executes the full-InChIKey identity audit for the 15 mixed-class parent groups.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python code/audits/evidence_reconstruction/run.py   --raw-dir "$ROOT/data/raw/B3DB"   --frozen-dir "$ROOT/data/frozen"   --output-dir "$ROOT/reproduction_runs/audits/evidence_reconstruction"

python code/audits/structure_identity/full_inchikey_audit.py   --b3db "$ROOT/data/raw/B3DB/B3DB_classification.tsv"   --outdir "$ROOT/reproduction_runs/audits/structure_identity"   --strict
