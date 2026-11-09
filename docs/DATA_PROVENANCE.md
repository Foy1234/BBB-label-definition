# Dataset Provenance and Frozen Snapshots

The data files are identified by their content hashes, record counts and original sources. Dataset retrieval dates are not recorded. Source repositories may contain newer records than those used in this study.

| Release file | Records | SHA-256 | Source |
|---|---:|---|---|
| `data/raw/BBBP/BBBP.csv` | 2,050 | `d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662` | https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv |
| `data/raw/B3DB/B3DB_classification.tsv` | 7,807 | `47a160aea1551423ead2aab70b301fb1646aea03cf151fd614eb37c78fa55b82` | https://github.com/theochem/B3DB |
| `data/raw/B3DB/B3DB_regression.tsv` | 1,058 | `1be4e33ab1fa1d99897541b6e0a9a00cd6061cf970a36dc838e640407735eba1` | https://github.com/theochem/B3DB |
| `data/raw/NCATS_PAMPA/NCATS_PAMPA_BBB_AID1845228.csv` | 440 | `e9d9b0c471d2c05be0a5df58be0028b3dc389478474aea67784c8eb38a25a16f` | https://pubchem.ncbi.nlm.nih.gov/bioassay/1845228 |
| `data/raw/NCATS_PAMPA/NCATS_PAMPA_BBB_AID1845228_properties.csv` | 437 | `f4d8fbd322279b5a1773c0695b8fce227338ffe3b8f3615d5224ce67a3ed4d22` | https://pubchem.ncbi.nlm.nih.gov/bioassay/1845228 |
| `data/frozen/ncats_unique_molecule_audit.csv` | 437 | `e7a84978ff89433579f60a038041ffc2bf66817c466a94ec126a1feba82a915e` | https://pubchem.ncbi.nlm.nih.gov/bioassay/1845228 |

The NCATS assay CSV contains 440 physical CSV rows after the header, including two PubChem descriptive metadata rows. The study preprocessing identifies 438 actual assay observations, aggregates them to 437 unique CIDs, and flags 318 eligible records for external evaluation. These records represent 317 unique parent connectivity identities after exact duplicate reconciliation (213 positives and 104 negatives). B3DB source classification records are from the earlier 7,807-row dataset snapshot, not later B3DB updates.

The study maps PubChem `Phenotype` values Moderate or High to 1 and Low to 0. The frozen NCATS audit contains additional historical identity-overlap and alternative cutoff columns; the upstream preparation script reconstructs the primary CID and eligibility screening, but not the exact full historical audit columns and byte representation. The authoritative input to the final Stage IV analysis is the immutable `data/frozen/ncats_unique_molecule_audit.csv`.

Reference papers: Clark (1999), https://doi.org/10.1021/js980402t; Abraham et al. (2006), https://doi.org/10.1002/jps.20595; Meng et al. (2021), https://doi.org/10.1038/s41597-021-01069-5.

Rights and attribution: `DATA_LICENSES.md`.
