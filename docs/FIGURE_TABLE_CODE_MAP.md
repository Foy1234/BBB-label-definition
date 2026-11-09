# Manuscript Figure and Table Mapping

| Manuscript item | Primary implementation | Primary output |
|---|---|---|
| Fig. 1 Overall study framework | Computational workflow summary | `docs/ANALYSIS_WORKFLOW.md` |
| Fig. 2 Data architecture, frozen partitions, and evaluation sets | `code/validation/eval_audit.py`; Stage I data-preparation code | `results/audits/overlap_matrix.csv`; `data/frozen/consensus_dataset.csv` |
| Fig. 3 Stage I hard-label construction and molecular sensitivity | `code/stage1/hard_label_audit/stage7_multiview_hardlabel.py`; `code/stage1/conflict_case_analysis/analyze_15_conflict_cases.py` | `results/stage1/hard_label_audit/`; `results/stage1/conflict_case_analysis/` |
| Fig. 4 Stage II evidence-retention targets and controls | `code/stage2/fit_corrected_targets.py`; `code/stage3/stage3.py`; `code/stage4/finalize.py` | `data/derived/stage2/`; `results/stage3/analysis/conditional_permutation_summary.csv`; `results/stage4/analysis/` |
| Fig. 5 Encoder dependence and chemical-space organization | `code/stage5/chemspace.py` | `results/stage5/` |
| Fig. 6 Database generalization and PAMPA score scale | `code/stage4/finalize.py`; `code/stage6/pampa_calibration.py` | `results/stage4/analysis/`; `results/stage4/pampa/`; `results/stage6/pampa_calibration/` |
| Table 1 Hard-label construction schemes | Stage I data construction | `results/stage1/hard_label_audit/training_view_summary.csv` |
| Table 2 Matched 15-pair sensitivity analysis | `code/stage1/conflict_case_analysis/analyze_15_conflict_cases.py` | `results/stage1/conflict_case_analysis/matched_pair_sensitivity_summary.csv` |
| Table 3 Database evaluation performance | `code/stage3/stage3.py`; `code/stage4/finalize.py` | `results/stage4/analysis/main_target_metrics.csv`; `results/stage4/analysis/main_paired_bootstrap.csv` |
| Table 4 Raw PAMPA evaluation | `code/stage4/finalize.py` | `results/stage4/pampa/pampa_mean_seed_metrics.csv` |
| Supplementary Figs. S1-S2 | Stage I and Stage IV outputs | `results/stage1/`; `results/stage4/analysis/` |
| Supplementary Fig. S3 | `code/stage5/chemspace.py` | `results/stage5/descriptor_target_sensitivity_associations.csv` |
| Supplementary Fig. S4 | `code/stage4/finalize.py`; `code/stage6/pampa_calibration.py` | `results/stage4/pampa/`; `results/stage6/pampa_calibration/` |
| Supplementary Section S10 / Tables S16-S18 | `code/audits/structure_identity/full_inchikey_audit.py` and source-trace outputs | `results/audits/structure_identity/` |
