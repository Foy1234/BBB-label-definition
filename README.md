# BBB Training-Target Definition and Molecular-Ranking Reproducibility Package

This package contains the computational workflow, frozen datasets, model predictions, statistical results, structural identity audits, raw-evidence reconstruction, and PAMPA-BBB calibration analyses used in the study. The directory structure follows the analysis sequence from molecular identity processing and hard-label construction through evidence-retention targets, model training, molecular ranking, chemical-space analysis, and PAMPA-BBB endpoint evaluation.

## Directory structure

```text
code/
  stage1/        Data preparation, hard-label construction, and matched mixed-class analysis
  stage2/        Hard/Vote/Integrated/Smooth target construction and conditional-permutation targets
  stage3/        XGBoost and GINE training, database evaluation, and permutation training
  stage4/        Molecular score/rank/Top-k statistics, paired bootstrap, and raw PAMPA evaluation
  stage5/        Chemical-space, physicochemical, Morgan-neighborhood, Murcko-scaffold, and SMARTS analyses
  stage6/        PAMPA-domain cross-fitted Platt calibration
  audits/        Raw B3DB evidence reconstruction and full-InChIKey structural identity audit
  validation/    Evaluation-set identity and overlap audit

data/
  raw/           BBBP, B3DB, and NCATS/PubChem source snapshots
  frozen/        Immutable analysis inputs used by the manuscript workflow
  derived/       Training targets and conditional-permutation targets

results/
  stage1/ ... stage6/
  audits/
  figures/main/

docs/
  ANALYSIS_WORKFLOW.md
  METHODS_AND_PARAMETERS.md
  DATA_DICTIONARY.md
  DATA_PROVENANCE.md
  FIGURE_TABLE_CODE_MAP.md
  RESULTS_GUIDE.md

environment/
  requirements.lock
  platform_information.md
  verify_cuda.py

scripts/
  run_all.sh
  run_stage1.sh ... run_stage6.sh
  run_audits.sh
  verify_release.py
  verify_extended_results.py
```

## Execution modes

From the package root:

```bash
bash scripts/run_all.sh verify
bash scripts/run_all.sh analysis
bash scripts/run_all.sh audits
bash scripts/run_all.sh full
```

- `verify` checks locked inputs, archived model predictions, conditional-permutation outputs, raw PAMPA evaluation, cross-fitted PAMPA calibration, evidence reconstruction, and structural identity audit anchors.
- `analysis` recomputes Stage IV, Stage V, and Stage VI analyses from archived Stage III predictions without retraining the primary models.
- `audits` reconstructs raw B3DB Vote/logBB evidence and runs the full-InChIKey audit for the 15 connectivity-level mixed-class parent groups.
- `full` executes Stage I through Stage VI in sequence and then runs the evidence and identity audits.

Recomputed outputs are written under `reproduction_runs/`; archived manuscript results under `results/` remain unchanged.

## Reference environment

- Ubuntu 22.04
- Python 3.12.3
- CUDA 12.8
- NVIDIA V100 32 GB
- RDKit 2026.3.5
- XGBoost 3.2.0
- PyTorch 2.8.0+cu128
- PyTorch Geometric 2.8.0.post1
- scikit-learn 1.9.0

The complete dependency set is recorded in `environment/requirements.lock`.

## Locked dataset sizes

- BBBP parents with valid classification: 1,941
- B3DB parents with valid classification: 4,020
- Harmonized parent identity table: 4,131
- BBBP test set: 291
- R27-associated reference-out set: 2,607
- Scaffold test set: 137
- Stage II common supervised training set: 1,157
- PAMPA-BBB evaluation parents: 317 (213 positive, 104 negative)

## Result navigation

`docs/FIGURE_TABLE_CODE_MAP.md` maps manuscript figures and tables to implementations and result files. `docs/RESULTS_GUIDE.md` indexes the principal numerical outputs. Field definitions and dataset provenance are documented in `docs/DATA_DICTIONARY.md` and `docs/DATA_PROVENANCE.md`.
