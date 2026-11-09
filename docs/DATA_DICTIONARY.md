# Data Dictionary

Dataset-specific fields are listed below. The original field names and spelling are preserved. Fields not assigned a specialized meaning are source attributes or intermediate computation fields retained as provided.

## BBBP raw
`data/raw/BBBP/BBBP.csv`

| Field | Meaning |
|---|---|
| `num` | Source or intermediate analytical field retained under its original name |
| `name` | Source or intermediate analytical field retained under its original name |
| `p_np` | Source or intermediate analytical field retained under its original name |
| `smiles` | Source or intermediate analytical field retained under its original name |

## B3DB classifications
`data/raw/B3DB/B3DB_classification.tsv`

| Field | Meaning |
|---|---|
| `NO.` | Source or intermediate analytical field retained under its original name |
| `compound_name` | Source or intermediate analytical field retained under its original name |
| `IUPAC_name` | Source or intermediate analytical field retained under its original name |
| `SMILES` | Source or intermediate analytical field retained under its original name |
| `CID` | Source or intermediate analytical field retained under its original name |
| `logBB` | Source or intermediate analytical field retained under its original name |
| `BBB+/BBB-` | Source or intermediate analytical field retained under its original name |
| `Inchi` | Source or intermediate analytical field retained under its original name |
| `threshold` | Source or intermediate analytical field retained under its original name |
| `reference` | Source or intermediate analytical field retained under its original name |
| `group` | Source or intermediate analytical field retained under its original name |
| `comments` | Source or intermediate analytical field retained under its original name |

## B3DB measurements
`data/raw/B3DB/B3DB_regression.tsv`

| Field | Meaning |
|---|---|
| `NO.` | Source or intermediate analytical field retained under its original name |
| `compound_name` | Source or intermediate analytical field retained under its original name |
| `IUPAC_name` | Source or intermediate analytical field retained under its original name |
| `SMILES` | Source or intermediate analytical field retained under its original name |
| `CID` | Source or intermediate analytical field retained under its original name |
| `logBB` | Source or intermediate analytical field retained under its original name |
| `Inchi` | Source or intermediate analytical field retained under its original name |
| `reference` | Source or intermediate analytical field retained under its original name |
| `group` | Source or intermediate analytical field retained under its original name |
| `comments` | Source or intermediate analytical field retained under its original name |

## NCATS assay raw
`data/raw/NCATS_PAMPA/NCATS_PAMPA_BBB_AID1845228.csv`

| Field | Meaning |
|---|---|
| `PUBCHEM_RESULT_TAG` | Source or intermediate analytical field retained under its original name |
| `PUBCHEM_SID` | PubChem substance identifier |
| `PUBCHEM_CID` | PubChem compound identifier |
| `PUBCHEM_EXT_DATASOURCE_SMILES` | Source or intermediate analytical field retained under its original name |
| `PUBCHEM_ACTIVITY_OUTCOME` | Source or intermediate analytical field retained under its original name |
| `PUBCHEM_ACTIVITY_SCORE` | Source or intermediate analytical field retained under its original name |
| `PUBCHEM_ACTIVITY_URL` | Source or intermediate analytical field retained under its original name |
| `PUBCHEM_ASSAYDATA_COMMENT` | Source or intermediate analytical field retained under its original name |
| `Phenotype` | Original PAMPA phenotype category |
| `Permeability` | Original PAMPA assay value (may include qualifiers) |
| `Analysis Comment` | Source or intermediate analytical field retained under its original name |

## NCATS PubChem properties
`data/raw/NCATS_PAMPA/NCATS_PAMPA_BBB_AID1845228_properties.csv`

| Field | Meaning |
|---|---|
| `CID` | Source or intermediate analytical field retained under its original name |
| `ConnectivitySMILES` | Source or intermediate analytical field retained under its original name |
| `SMILES` | Source or intermediate analytical field retained under its original name |
| `InChIKey` | Source or intermediate analytical field retained under its original name |
| `Title` | Source or intermediate analytical field retained under its original name |

## Frozen consensus
`data/frozen/consensus_dataset.csv`

| Field | Meaning |
|---|---|
| `parent_ik` | Parent-connectivity InChIKey grouping identifier |
| `canon_smiles_bbbp` | Source or intermediate analytical field retained under its original name |
| `bbbp_name` | Source or intermediate analytical field retained under its original name |
| `bbbp_raw_labels` | Source or intermediate analytical field retained under its original name |
| `bbbp_label` | Source or intermediate analytical field retained under its original name |
| `bbbp_label_conflict` | Source or intermediate analytical field retained under its original name |
| `canon_smiles_b3db` | Source or intermediate analytical field retained under its original name |
| `b3db_raw_labels` | Source or intermediate analytical field retained under its original name |
| `b3db_label` | Source or intermediate analytical field retained under its original name |
| `b3db_label_conflict` | Source or intermediate analytical field retained under its original name |
| `reference_list` | Source or intermediate analytical field retained under its original name |
| `n_references` | Source or intermediate analytical field retained under its original name |
| `b3db_logbb_list` | Source or intermediate analytical field retained under its original name |
| `b3db_logbb_min` | Source or intermediate analytical field retained under its original name |
| `b3db_logbb_max` | Source or intermediate analytical field retained under its original name |
| `b3db_logbb_n` | Source or intermediate analytical field retained under its original name |
| `b3db_logbb_refs` | Source or intermediate analytical field retained under its original name |
| `canon_smiles` | Standardized canonical SMILES |
| `overlap` | Source or intermediate analytical field retained under its original name |
| `divergence` | Source or intermediate analytical field retained under its original name |
| `source` | Source or intermediate analytical field retained under its original name |
| `split` | Frozen dataset partition |
| `in_reference_out` | Membership in reference-out holdout |

## Frozen labels
`data/frozen/labels_abc.csv`

| Field | Meaning |
|---|---|
| `parent_ik` | Parent-connectivity InChIKey grouping identifier |
| `canon_smiles` | Standardized canonical SMILES |
| `b3db_label` | Source or intermediate analytical field retained under its original name |
| `bbbp_label` | Source or intermediate analytical field retained under its original name |
| `label_A` | Frozen hard binary target |
| `label_B` | Proportion-based Vote target |
| `label_C_sigmoid_mu` | Source or intermediate analytical field retained under its original name |

## Frozen scaffold
`data/frozen/scaffold_split_assignments.csv`

| Field | Meaning |
|---|---|
| `parent_ik` | Parent-connectivity InChIKey grouping identifier |
| `canon_smiles` | Standardized canonical SMILES |
| `scaffold` | Source or intermediate analytical field retained under its original name |
| `scaffold_split` | Source or intermediate analytical field retained under its original name |
| `bbbp_label` | Source or intermediate analytical field retained under its original name |
| `b3db_label` | Source or intermediate analytical field retained under its original name |
| `n_references` | Source or intermediate analytical field retained under its original name |

## Frozen NCATS
`data/frozen/ncats_unique_molecule_audit.csv`

| Field | Meaning |
|---|---|
| `PUBCHEM_CID` | PubChem compound identifier |
| `Phenotype` | Original PAMPA phenotype category |
| `Permeability` | Original PAMPA assay value (may include qualifiers) |
| `permeability_numeric` | Numeric parse of assay observation |
| `exact_10` | Source or intermediate analytical field retained under its original name |
| `y_pubchem` | Binary study mapping of assay phenotype |
| `y_10_low` | Source or intermediate analytical field retained under its original name |
| `y_10_moderate` | Source or intermediate analytical field retained under its original name |
| `SMILES` | Source or intermediate analytical field retained under its original name |
| `Title` | Source or intermediate analytical field retained under its original name |
| `ncats_canonical_smiles` | Source or intermediate analytical field retained under its original name |
| `ncats_isomeric_smiles` | Source or intermediate analytical field retained under its original name |
| `ncats_full_inchikey` | Source or intermediate analytical field retained under its original name |
| `ncats_parent_smiles` | Source or intermediate analytical field retained under its original name |
| `ncats_parent_inchikey` | Source or intermediate analytical field retained under its original name |
| `ncats_connectivity_key` | NCATS standardized parent-connectivity identity |
| `ncats_had_multiple_fragments` | Source or intermediate analytical field retained under its original name |
| `overlap_cid_b3db` | Source or intermediate analytical field retained under its original name |
| `overlap_canonical_consensus` | Source or intermediate analytical field retained under its original name |
| `overlap_full_inchikey_consensus` | Source or intermediate analytical field retained under its original name |
| `overlap_parent_full_inchikey_consensus` | Source or intermediate analytical field retained under its original name |
| `overlap_parent_connectivity_consensus` | Source or intermediate analytical field retained under its original name |
| `overlap_parent_connectivity_bbbp` | Source or intermediate analytical field retained under its original name |
| `overlap_parent_connectivity_b3db` | Source or intermediate analytical field retained under its original name |
| `salt_parent_only_overlap` | Source or intermediate analytical field retained under its original name |
| `stereo_or_protonation_only_overlap` | Source or intermediate analytical field retained under its original name |
| `exclude_any_consensus_overlap` | Source or intermediate analytical field retained under its original name |
| `ambiguous_duplicate` | Indicator of conflicting duplicate CID measurements |
| `duplicate_measurements` | Source or intermediate analytical field retained under its original name |
| `eligible_external` | Eligibility indicator for NCATS external endpoint |

## Three target vectors
`data/derived/stage2/stage2_target_vectors_common1157.csv`

| Field | Meaning |
|---|---|
| `parent_ik` | Parent-connectivity InChIKey grouping identifier |
| `label_A` | Frozen hard binary target |
| `label_B` | Proportion-based Vote target |
| `label_C_integrated` | Integrated target at primary shrinkage λ = 0.1 |
| `label_S_class_smooth` | Class-conditional smoothing-control target |
| `has_bbbp` | Availability flag for training-side BBBP classification evidence |
| `has_logbb` | Availability flag for training-side logBB evidence |
| `b3db_n_distinct_states_in_consensus` | Source or intermediate analytical field retained under its original name |
| `b3db_positive_fraction_from_frozen_vote` | Source or intermediate analytical field retained under its original name |
| `latent_local_curvature_sd` | Source or intermediate analytical field retained under its original name |
| `logbb_within_sd_std` | Source or intermediate analytical field retained under its original name |
| `integrated_lambda_0.01` | Source or intermediate analytical field retained under its original name |
| `integrated_lambda_0.1` | Source or intermediate analytical field retained under its original name |
| `integrated_lambda_1` | Source or intermediate analytical field retained under its original name |

## Permutation vectors
`data/derived/stage2/P_99_permutation_targets_common1157.csv`

| Field | Meaning |
|---|---|
| `parent_ik` | Parent-connectivity InChIKey grouping identifier |
| `P_000` | Conditional permutation replicate of the continuous Integrated target |
| `P_001` | Conditional permutation replicate of the continuous Integrated target |
| `P_002` | Conditional permutation replicate of the continuous Integrated target |
| `P_003` | Conditional permutation replicate of the continuous Integrated target |
| `P_004` | Conditional permutation replicate of the continuous Integrated target |
| `P_005` | Conditional permutation replicate of the continuous Integrated target |
| `P_006` | Conditional permutation replicate of the continuous Integrated target |
| `P_007` | Conditional permutation replicate of the continuous Integrated target |
| `P_008` | Conditional permutation replicate of the continuous Integrated target |
| `P_009` | Conditional permutation replicate of the continuous Integrated target |
| `P_010` | Conditional permutation replicate of the continuous Integrated target |
| `P_011` | Conditional permutation replicate of the continuous Integrated target |
| `P_012` | Conditional permutation replicate of the continuous Integrated target |
| `P_013` | Conditional permutation replicate of the continuous Integrated target |
| `P_014` | Conditional permutation replicate of the continuous Integrated target |
| `P_015` | Conditional permutation replicate of the continuous Integrated target |
| `P_016` | Conditional permutation replicate of the continuous Integrated target |
| `P_017` | Conditional permutation replicate of the continuous Integrated target |
| `P_018` | Conditional permutation replicate of the continuous Integrated target |
| `P_019` | Conditional permutation replicate of the continuous Integrated target |
| `P_020` | Conditional permutation replicate of the continuous Integrated target |
| `P_021` | Conditional permutation replicate of the continuous Integrated target |
| `P_022` | Conditional permutation replicate of the continuous Integrated target |
| `P_023` | Conditional permutation replicate of the continuous Integrated target |
| `P_024` | Conditional permutation replicate of the continuous Integrated target |
| `P_025` | Conditional permutation replicate of the continuous Integrated target |
| `P_026` | Conditional permutation replicate of the continuous Integrated target |
| `P_027` | Conditional permutation replicate of the continuous Integrated target |
| `P_028` | Conditional permutation replicate of the continuous Integrated target |
| `P_029` | Conditional permutation replicate of the continuous Integrated target |
| `P_030` | Conditional permutation replicate of the continuous Integrated target |
| `P_031` | Conditional permutation replicate of the continuous Integrated target |
| `P_032` | Conditional permutation replicate of the continuous Integrated target |
| `P_033` | Conditional permutation replicate of the continuous Integrated target |
| `P_034` | Conditional permutation replicate of the continuous Integrated target |
| `P_035` | Conditional permutation replicate of the continuous Integrated target |
| `P_036` | Conditional permutation replicate of the continuous Integrated target |
| `P_037` | Conditional permutation replicate of the continuous Integrated target |
| `P_038` | Conditional permutation replicate of the continuous Integrated target |
| `P_039` | Conditional permutation replicate of the continuous Integrated target |
| `P_040` | Conditional permutation replicate of the continuous Integrated target |
| `P_041` | Conditional permutation replicate of the continuous Integrated target |
| `P_042` | Conditional permutation replicate of the continuous Integrated target |
| `P_043` | Conditional permutation replicate of the continuous Integrated target |
| `P_044` | Conditional permutation replicate of the continuous Integrated target |
| `P_045` | Conditional permutation replicate of the continuous Integrated target |
| `P_046` | Conditional permutation replicate of the continuous Integrated target |
| `P_047` | Conditional permutation replicate of the continuous Integrated target |
| `P_048` | Conditional permutation replicate of the continuous Integrated target |
| `P_049` | Conditional permutation replicate of the continuous Integrated target |
| `P_050` | Conditional permutation replicate of the continuous Integrated target |
| `P_051` | Conditional permutation replicate of the continuous Integrated target |
| `P_052` | Conditional permutation replicate of the continuous Integrated target |
| `P_053` | Conditional permutation replicate of the continuous Integrated target |
| `P_054` | Conditional permutation replicate of the continuous Integrated target |
| `P_055` | Conditional permutation replicate of the continuous Integrated target |
| `P_056` | Conditional permutation replicate of the continuous Integrated target |
| `P_057` | Conditional permutation replicate of the continuous Integrated target |
| `P_058` | Conditional permutation replicate of the continuous Integrated target |
| `P_059` | Conditional permutation replicate of the continuous Integrated target |
| `P_060` | Conditional permutation replicate of the continuous Integrated target |
| `P_061` | Conditional permutation replicate of the continuous Integrated target |
| `P_062` | Conditional permutation replicate of the continuous Integrated target |
| `P_063` | Conditional permutation replicate of the continuous Integrated target |
| `P_064` | Conditional permutation replicate of the continuous Integrated target |
| `P_065` | Conditional permutation replicate of the continuous Integrated target |
| `P_066` | Conditional permutation replicate of the continuous Integrated target |
| `P_067` | Conditional permutation replicate of the continuous Integrated target |
| `P_068` | Conditional permutation replicate of the continuous Integrated target |
| `P_069` | Conditional permutation replicate of the continuous Integrated target |
| `P_070` | Conditional permutation replicate of the continuous Integrated target |
| `P_071` | Conditional permutation replicate of the continuous Integrated target |
| `P_072` | Conditional permutation replicate of the continuous Integrated target |
| `P_073` | Conditional permutation replicate of the continuous Integrated target |
| `P_074` | Conditional permutation replicate of the continuous Integrated target |
| `P_075` | Conditional permutation replicate of the continuous Integrated target |
| `P_076` | Conditional permutation replicate of the continuous Integrated target |
| `P_077` | Conditional permutation replicate of the continuous Integrated target |
| `P_078` | Conditional permutation replicate of the continuous Integrated target |
| `P_079` | Conditional permutation replicate of the continuous Integrated target |
| `P_080` | Conditional permutation replicate of the continuous Integrated target |
| `P_081` | Conditional permutation replicate of the continuous Integrated target |
| `P_082` | Conditional permutation replicate of the continuous Integrated target |
| `P_083` | Conditional permutation replicate of the continuous Integrated target |
| `P_084` | Conditional permutation replicate of the continuous Integrated target |
| `P_085` | Conditional permutation replicate of the continuous Integrated target |
| `P_086` | Conditional permutation replicate of the continuous Integrated target |
| `P_087` | Conditional permutation replicate of the continuous Integrated target |
| `P_088` | Conditional permutation replicate of the continuous Integrated target |
| `P_089` | Conditional permutation replicate of the continuous Integrated target |
| `P_090` | Conditional permutation replicate of the continuous Integrated target |
| `P_091` | Conditional permutation replicate of the continuous Integrated target |
| `P_092` | Conditional permutation replicate of the continuous Integrated target |
| `P_093` | Conditional permutation replicate of the continuous Integrated target |
| `P_094` | Conditional permutation replicate of the continuous Integrated target |
| `P_095` | Conditional permutation replicate of the continuous Integrated target |
| `P_096` | Conditional permutation replicate of the continuous Integrated target |
| `P_097` | Conditional permutation replicate of the continuous Integrated target |
| `P_098` | Conditional permutation replicate of the continuous Integrated target |

## Compatible targets
`data/derived/stage2/labels_abc_corrected_STAGE2_ONLY.csv`

| Field | Meaning |
|---|---|
| `parent_ik` | Parent-connectivity InChIKey grouping identifier |
| `label_A` | Frozen hard binary target |
| `label_B` | Proportion-based Vote target |
| `label_C_sigmoid_mu` | Source or intermediate analytical field retained under its original name |
| `label_S_smooth` | Source or intermediate analytical field retained under its original name |

## PAMPA seed-level predictions
`results/stage4/pampa/pampa_predictions_by_seed.csv`

| Field | Meaning |
|---|---|
| `target` | Training-target name |
| `seed` | Pseudorandom training seed |
| `PUBCHEM_CID` | PubChem compound identifier |
| `parent_ik` | Parent-connectivity InChIKey grouping identifier |
| `gold` | Frozen binary evaluation label |
| `prediction` | Per-molecule model score |
