"""
Structure-based docking scoring pillar for drug repurposing.

Uses AlphaFold predicted protein structures to estimate binding potential.
Supports two modes:

1. **Real docking** (if AutoDock Vina is installed): Generates 3D drug
   conformers, fetches AlphaFold structures, runs Vina docking to get
   actual binding energies. Slow but accurate (~1-5 min per compound).

2. **RDKit proxy** (fallback): Uses molecular properties (Lipinski,
   LogP, TPSA, rotatable bonds) as a drug-likeness proxy for binding
   compatibility. Fast but approximate.

The proxy is always used for initial filtering; real docking (when
available) is applied to top candidates only.
"""
from __future__ import annotations

import time
import requests
import numpy as np
from pathlib import Path
from typing import Optional

# Module-level caches
_uniprot_cache: dict[str, Optional[str]] = {}
_structure_cache: dict[str, bool] = {}
_docking_scores_cache: dict[str, dict] = {}

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

STRUCTURE_CACHE_DIR = Path("data/alphafold")


def gene_to_uniprot(gene_symbol: str) -> Optional[str]:
    """Map gene symbol to UniProt ID via UniProt API."""
    if gene_symbol in _uniprot_cache:
        return _uniprot_cache[gene_symbol]

    try:
        resp = requests.get(
            f"{UNIPROT_API}/search",
            params={
                "query": f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "size": 1,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                uniprot_id = results[0].get("primaryAccession")
                _uniprot_cache[gene_symbol] = uniprot_id
                return uniprot_id

        time.sleep(0.1)
    except Exception:
        pass

    _uniprot_cache[gene_symbol] = None
    return None


def fetch_alphafold_structure(uniprot_id: str) -> Optional[str]:
    """
    Fetch AlphaFold predicted structure for a protein.

    Downloads the CIF file and caches it locally.
    Returns path to the cached structure file, or None.
    """
    STRUCTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = STRUCTURE_CACHE_DIR / f"AF-{uniprot_id}-F1-model_v4.cif"

    if cache_path.exists():
        return str(cache_path)

    try:
        # Get structure URL from AlphaFold API
        resp = requests.get(f"{ALPHAFOLD_API}/{uniprot_id}", timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                cif_url = data[0].get("cifUrl")
                if cif_url:
                    # Download CIF file
                    cif_resp = requests.get(cif_url, timeout=30)
                    if cif_resp.status_code == 200:
                        cache_path.write_bytes(cif_resp.content)
                        return str(cache_path)

        time.sleep(0.2)
    except Exception:
        pass

    return None


def compute_binding_score(smiles: str, target_gene: str) -> Optional[float]:
    """
    Estimate binding potential between a drug (SMILES) and target protein.

    Uses RDKit molecular properties as a proxy for binding:
    - Drug-likeness (Lipinski's Rule of 5 compliance)
    - Molecular weight in binding range
    - LogP in favorable range
    - Number of rotatable bonds (flexibility)
    - Pharmacophore features matching

    This is NOT real docking — it's a drug-target compatibility estimate.
    Real docking (Vina/GNINA) would be more accurate but 100x slower.

    Returns: 0-1 score (1 = highly compatible) or None if can't compute
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Calculate drug-likeness features
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        rings = Lipinski.RingCount(mol)

        # Score components (each 0-1)
        scores = []

        # MW: ideal 150-500 for binding
        if 150 <= mw <= 500:
            scores.append(1.0)
        elif mw < 150:
            scores.append(mw / 150)
        elif mw <= 800:
            scores.append(max(0, 1.0 - (mw - 500) / 300))
        else:
            scores.append(0.1)

        # LogP: ideal 0-5 for cell permeability + binding
        if 0 <= logp <= 5:
            scores.append(1.0)
        elif -2 <= logp < 0:
            scores.append(0.7)
        elif logp > 5:
            scores.append(max(0, 1.0 - (logp - 5) / 5))
        else:
            scores.append(0.3)

        # HBA + HBD: can form hydrogen bonds with target
        hb_total = hba + hbd
        if 2 <= hb_total <= 10:
            scores.append(1.0)
        elif hb_total < 2:
            scores.append(0.4)
        else:
            scores.append(max(0.2, 1.0 - (hb_total - 10) / 10))

        # Rotatable bonds: some flexibility helps binding
        if 2 <= rotatable <= 8:
            scores.append(1.0)
        elif rotatable < 2:
            scores.append(0.6)
        else:
            scores.append(max(0.2, 1.0 - (rotatable - 8) / 10))

        # TPSA: polar surface area for binding interactions
        if 20 <= tpsa <= 130:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Rings: drug-like molecules typically have 1-4 rings
        if 1 <= rings <= 4:
            scores.append(1.0)
        elif rings == 0:
            scores.append(0.4)
        else:
            scores.append(0.6)

        return np.mean(scores)

    except Exception:
        return None


DOCKING_CACHE_DIR = Path("data/docking_cache")

_vina_available = None


def is_vina_available() -> bool:
    """Check if AutoDock Vina is available (binary or Python package)."""
    global _vina_available
    if _vina_available is not None:
        return _vina_available

    # Try Python package
    try:
        from vina import Vina
        _vina_available = True
        return True
    except ImportError:
        pass

    # Try binary
    import shutil
    if shutil.which("vina"):
        _vina_available = True
        return True

    _vina_available = False
    return False


def prepare_ligand_pdbqt(smiles: str) -> Optional[str]:
    """Convert SMILES to 3D conformer and write as PDBQT for Vina.

    Returns path to temporary PDBQT file, or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import tempfile

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                return None

        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        # Write as PDB first, then convert to PDBQT via meeko
        pdb_path = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
        Chem.MolToPDBFile(mol, pdb_path)

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)[0]
            pdbqt_string, _, _ = PDBQTWriterLegacy.write_string(mol_setup)
            pdbqt_path = pdb_path.replace(".pdb", ".pdbqt")
            with open(pdbqt_path, "w") as f:
                f.write(pdbqt_string)
            return pdbqt_path
        except ImportError:
            return pdb_path  # Return PDB as fallback

    except Exception:
        return None


def dock_with_vina(
    ligand_path: str,
    receptor_path: str,
    center: tuple = (0, 0, 0),
    box_size: tuple = (30, 30, 30),
) -> Optional[float]:
    """Run AutoDock Vina docking. Returns binding energy (kcal/mol) or None."""
    try:
        from vina import Vina
        v = Vina(sf_name="vina")
        v.set_receptor(receptor_path)
        v.set_ligand_from_file(ligand_path)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))
        v.dock(exhaustiveness=8, n_poses=1)
        energies = v.energies()
        if energies is not None and len(energies) > 0:
            return float(energies[0][0])  # Best pose energy
    except ImportError:
        pass
    except Exception:
        pass
    return None


def dock_compound_cached(
    drug_id: str,
    smiles: str,
    uniprot_id: str,
    structure_path: str,
) -> Optional[float]:
    """Dock a compound against a target, with caching."""
    DOCKING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{drug_id}_{uniprot_id}"
    cache_file = DOCKING_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        import json
        data = json.loads(cache_file.read_text())
        return data.get("binding_energy")

    ligand_path = prepare_ligand_pdbqt(smiles)
    if ligand_path is None:
        return None

    energy = dock_with_vina(ligand_path, structure_path)

    # Cache result
    import json
    cache_file.write_text(json.dumps({
        "drug_id": drug_id,
        "uniprot_id": uniprot_id,
        "binding_energy": energy,
    }))

    return energy


def score_drugs_for_disease_docking(
    disease_name: str,
    compound_entities: list[str],
    triplets,
    smiles_map: dict,
    drug_names: dict,
) -> dict:
    """
    Score drugs by structural binding potential to disease targets.

    For each drug:
    1. Get SMILES structure
    2. Check if drug targets have AlphaFold structures
    3. Compute binding compatibility score

    Returns: dict[compound_entity] -> (docking_score, best_target, has_structure)
    """
    # Get disease-relevant gene targets from DRKG
    disease_genes = set()
    for _, row in triplets[
        triplets["head"].str.startswith("Disease::")
    ].iterrows():
        if disease_name.lower() in str(row["head"]).lower():
            if str(row["tail"]).startswith("Gene::"):
                disease_genes.add(row["tail"])

    # Also reverse
    for _, row in triplets[
        triplets["tail"].str.startswith("Disease::")
    ].iterrows():
        if disease_name.lower() in str(row["tail"]).lower():
            if str(row["head"]).startswith("Gene::"):
                disease_genes.add(row["head"])

    if not disease_genes:
        return {}

    # Load gene symbol mapping
    try:
        from opencure.scoring.mendelian_randomization import _load_entrez_to_symbol
        entrez_map = _load_entrez_to_symbol()
    except Exception:
        entrez_map = {}

    # Map disease genes to symbols
    disease_gene_symbols = set()
    for g in disease_genes:
        gene_id = g.split("::")[1] if "::" in g else g
        for sub_id in gene_id.split(";"):
            if sub_id.strip() in entrez_map:
                disease_gene_symbols.add(entrez_map[sub_id.strip()])

    docking_scores = {}

    for compound in compound_entities:
        # Get SMILES
        db_id = compound.split("::")[1] if "::" in compound else compound
        smiles = smiles_map.get(db_id)
        if not smiles:
            continue

        # Compute binding score (drug-likeness proxy)
        binding_score = compute_binding_score(smiles, "")
        if binding_score is None or binding_score < 0.3:
            continue

        # Get this drug's targets
        drug_genes = set()
        mask = (triplets["head"] == compound) & (triplets["tail"].str.startswith("Gene::"))
        for _, row in triplets[mask].head(15).iterrows():
            gene_id = row["tail"].split("::")[1]
            for sub_id in gene_id.split(";"):
                if sub_id.strip() in entrez_map:
                    drug_genes.add(entrez_map[sub_id.strip()])

        # Check if drug targets overlap with disease targets
        target_overlap = drug_genes & disease_gene_symbols
        if target_overlap:
            overlap_bonus = min(0.3, 0.1 * len(target_overlap))
            final_score = min(1.0, binding_score + overlap_bonus)
            best_target = next(iter(target_overlap))

            # Try real Vina docking for high-scoring candidates
            if is_vina_available() and final_score > 0.5:
                uniprot_id = gene_to_uniprot(best_target)
                if uniprot_id:
                    structure_path = fetch_alphafold_structure(uniprot_id)
                    if structure_path:
                        energy = dock_compound_cached(db_id, smiles, uniprot_id, structure_path)
                        if energy is not None:
                            # Convert Vina energy to 0-1 score
                            # Typical range: -12 (very good) to 0 (no binding)
                            vina_score = min(1.0, max(0.0, -energy / 12.0))
                            # Blend proxy and Vina (Vina dominates when available)
                            final_score = 0.3 * final_score + 0.7 * vina_score

            docking_scores[compound] = (final_score, best_target, True)

    return docking_scores
