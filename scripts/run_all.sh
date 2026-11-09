#!/usr/bin/env bash
# run_all.sh — Dispatches verification, analysis-only recomputation, audit-only execution, or the complete Stage I–VI reproducibility workflow.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-verify}"
case "$MODE" in
  verify)
    python scripts/verify_release.py
    python scripts/verify_extended_results.py
    ;;
  analysis)
    bash scripts/run_stage4.sh
    bash scripts/run_stage5.sh
    bash scripts/run_stage6.sh
    ;;
  audits)
    bash scripts/run_audits.sh
    ;;
  full)
    bash scripts/run_stage1.sh
    bash scripts/run_stage2.sh
    bash scripts/run_stage3.sh
    bash scripts/run_stage4.sh
    bash scripts/run_stage5.sh
    bash scripts/run_stage6.sh
    bash scripts/run_audits.sh
    ;;
  *)
    echo 'Usage: bash scripts/run_all.sh {verify|analysis|audits|full}' >&2
    exit 2
    ;;
esac
