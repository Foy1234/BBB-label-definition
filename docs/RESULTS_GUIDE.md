# Results File Guide

## Stage I

- `results/stage1/hard_label_audit/metrics_summary.csv`: overall performance for the three effective hard-label states.
- `results/stage1/hard_label_audit/mean_predictions_long.csv`: seed-averaged predictions for BBBP test molecules.
- `results/stage1/conflict_case_analysis/matched_pair_sensitivity_summary.csv`: paired effects for 15 mixed-class parents and matched controls.
- `results/stage1/conflict_case_analysis/case_and_control_8view_trajectories_long.csv`: eight-view trajectories for the matched 15 pairs.

## Stage II

- `data/derived/stage2/stage2_target_vectors_common1157.csv`: Hard, Vote, Integrated, Smooth, and λ-sensitivity targets.
- `data/derived/stage2/P_99_permutation_targets_common1157.csv`: 99 conditional-permutation target arrays.

## Stage III

- `results/stage3/predictions/xgb/main/`: per-seed XGBoost predictions for the primary targets.
- `results/stage3/predictions/gine/main/`: per-seed GINE predictions for the primary targets.
- `results/stage3/analysis/metrics_from_seed_mean.csv`: principal metrics computed after seed averaging.
- `results/stage3/analysis/conditional_permutation_summary.csv`: conditional-permutation statistics.

## Stage IV

- `results/stage4/analysis/main_target_metrics.csv`: database evaluation-set performance.
- `results/stage4/analysis/main_rank_topk_pairwise.csv`: rank and Top-k membership changes.
- `results/stage4/analysis/molecule_target_sensitivity.csv`: molecule-level target sensitivity.
- `results/stage4/analysis/main_paired_bootstrap.csv`: paired-bootstrap target differences and confidence intervals.
- `results/stage4/pampa/pampa_predictions_by_seed.csv`: per-seed PAMPA predictions.
- `results/stage4/pampa/pampa_mean_seed_metrics.csv`: raw-score PAMPA metrics.

## Stage V

- `results/stage5/bbbp_molecule_chemical_annotation.csv`: descriptors, PCA coordinates, and sensitivity values for BBBP test parents.
- `results/stage5/encoder_concordance_target_sensitivity.csv`: cross-encoder sensitivity concordance.
- `results/stage5/neighborhood_target_sensitivity_autocorrelation.csv`: Moran's I in Morgan neighborhoods.
- `results/stage5/descriptor_target_sensitivity_associations.csv`: physicochemical descriptor associations.
- `results/stage5/frozen_smarts_target_sensitivity_overlay_unique_hitgroups.csv`: frozen SMARTS-pattern associations.

## Stage VI

- `results/stage6/pampa_calibration/locked_external_metrics.csv`: locked raw PAMPA score metrics.
- `results/stage6/pampa_calibration/crossfit_metrics_by_repeat.csv`: target-specific metrics for each cross-fitting repeat.
- `results/stage6/pampa_calibration/crossfit_summary_mean_sd.csv`: cross-fitting mean and standard deviation.
- `results/stage6/pampa_calibration/crossfit_oof_predictions.csv`: out-of-fold calibrated scores for 317 parents.
- `results/stage6/pampa_calibration/calibrator_coefficients.csv`: fold-specific Platt coefficients.
- `results/stage6/pampa_calibration/calibration_bins_seed42.csv`: binned calibration statistics for split seed 42.

## Audits

- `results/audits/overlap_matrix.csv`: identity overlap among database evaluation sets.
- `results/audits/evidence_reconstruction/qc_report.json`: aggregate QC for raw B3DB Vote/logBB evidence reconstruction.
- `results/audits/structure_identity/mixed15_full_inchikey_summary.csv`: full-InChIKey summary for the 15 mixed-class parent groups.
- `results/audits/structure_identity/mixed15_full_inchikey_records.csv`: 65 original B3DB classification records used in the identity audit.
- `results/audits/structure_identity/BBB_structure_identity_source_trace.xlsx`: structural identity, source tracing, and molecule-level DMPK evidence tables.

## Main figures

`results/figures/main/` contains the final image files for manuscript Figs. 1-6.
