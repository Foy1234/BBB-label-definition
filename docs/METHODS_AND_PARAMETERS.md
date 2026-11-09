# Methods and Fixed Parameters

## Molecular identity

- Salt handling: RDKit default `SaltRemover`
- Stereochemistry in the primary modeling identity branch: `Chem.RemoveStereochemistry`
- Canonical SMILES: `MolToSmiles(isomericSmiles=False, canonical=True)`
- Parent identity: first 14 characters of the full InChIKey generated from the processed molecule
- Structural identity audit: a parallel full-InChIKey branch retains source stereochemical annotations

## Data partitions

- Random split seed: 42
- BBBP test: 291
- Reference-out: 2,607
- BBBP test/reference-out overlap: 265
- Frozen validation split: 264
- Split-train pool: 1,234
- Common supervised training set: 1,157
- Scaffold test: 137

## Stage I XGBoost

- Morgan radius: 2
- Fingerprint bits: 2,048
- Seeds: 42–61
- `max_depth`: 5
- `learning_rate`: 0.03
- `n_estimators`: 300
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- Objective: squared error

## Stage II Integrated target

- Primary regularization parameter: λ = 0.1
- Sensitivity values: λ = 0.01 and 1.0
- Additional BBBP classification coverage: 18/1,157
- logBB coverage: 186/1,157
- Conditional permutations: 99
- Permutation strata: Hard class, BBBP-evidence availability, and logBB-evidence availability

## GINE

- Graph-convolution layers: 3
- Hidden width: 64
- Batch normalization + ReLU
- Dropout: 0.2
- Global mean pooling
- Prediction head: 64 → 32 → 1
- Sigmoid output
- Batch size: 64
- Adam learning rate: 0.001
- Weight decay: 0.0001
- Maximum epochs: 120
- Early-stopping patience: 12
- Seeds: 42, 52, 62, 72, 82, 92, 102, 112, 122, 132

## Statistical analysis

- Seed-averaged prediction is the primary molecule-level score.
- Paired bootstrap: 2,000 replicates.
- Stage I matched 15-pair analysis: exact sign-flip test plus paired-bootstrap confidence interval.
- Chemical-space permutation tests: 2,000 permutations.
- Benjamini-Hochberg adjustment is applied within prespecified multiple-testing families.

## Raw PAMPA evaluation

- Independent parent identities: 317
- Positive: 213
- Negative: 104
- AUROC/AUPRC: calculated from raw model scores.
- Brier/ECE: calculated after clipping raw scores to [0, 1].
- ECE: 10 fixed-width bins.

## Cross-fitted PAMPA Platt calibration

- 5-fold `StratifiedKFold`
- Repeat seeds: 42, 43, 44
- `LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)`
- Each parent receives an out-of-fold calibrated probability.
- Reported metrics: Brier score, ECE10, log loss, AUROC, and AUPRC.
