"""B3DB raw-evidence reconstruction utilities.

Normalizes SMILES with RDKit's default SaltRemover and stereochemistry
removal, then obtains parent identities from the first InChIKey block. The
functions implement the structural-identity convention used by the frozen
BBBP/B3DB consensus. Record-level identity failures remain auditable and
are never converted to artificial negative measurements.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover
from rdkit.Chem.inchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")
REMOVER = SaltRemover.SaltRemover()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_parent(smiles: object) -> tuple[str | None, str | None, str]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "blank_smiles"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, "rdkit_parse_failed"
        mol = REMOVER.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None, None, "empty_after_salt_removal"
        Chem.RemoveStereochemistry(mol)
        canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        ik = MolToInchiKey(mol)
        if not canon or not ik or not ik.split("-")[0]:
            return None, canon or None, "identity_generation_failed"
        parent = ik.split("-")[0]
        if len(parent) != 14:
            return None, canon, "invalid_connectivity_block"
        return parent, canon, "ok"
    except Exception as ex:
        return None, None, "rdkit_exception:" + type(ex).__name__


def references(field: object) -> list[str]:
    if not isinstance(field, str):
        return []
    return sorted({r.strip() for r in field.split("|") if r.strip()})
