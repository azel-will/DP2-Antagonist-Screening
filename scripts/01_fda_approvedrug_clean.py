"""
SDF → clean FDA-approved small-molecule drug list
(remove toxic-metal-containing compounds)
"""

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, PandasTools

# ---------------- CONFIG ---------------- #
# Use relative paths; run this script from the project root
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "data"
SDF_FILE  = ROOT_DIR / "00_raw" / "drugbank_fda.sdf"
CLEAN_CSV = ROOT_DIR / "01_intermediate" / "fda_approvedrug_clean.csv"

NAME_COLS = ['GENERIC_NAME', 'NAME', 'DRUG_NAME', 'TRADE_NAME']
MAX_MW    = 800

# Toxic metals (atomic numbers) commonly restricted in drug discovery
# Ti V Cr Mn Co Ni Cu Zn As Se Br Ag Cd Sn Sb Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr
TOXIC_METALS = {
    22, 23, 24, 25, 27, 28, 29, 30, 33, 34, 35, 47, 48, 50, 51, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99, 100, 101, 102, 103
}

# ---------------- UTILS ---------------- #
def canonical(smiles: str) -> str | None:
    """Return canonical SMILES without stereochemistry."""
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol, isomericSmiles=True)


def is_small_drug(smiles: str) -> bool:
    """Check if molecule is a small drug (MW ≤ 800 and ≥ 1 ring)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return Descriptors.MolWt(mol) <= MAX_MW and rdMolDescriptors.CalcNumRings(mol) >= 1


def contains_toxic_metal(mol: Chem.Mol) -> bool:
    """Detect if molecule contains any toxic metal atom."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in TOXIC_METALS:
            return True
    return False


# ---------------- PIPELINE ---------------- #
# 1. Load SDF
df = PandasTools.LoadSDF(str(SDF_FILE))

# 2. Extract SMILES and name
df['smiles'] = df['ROMol'].apply(
    lambda m: None if m is None else Chem.MolToSmiles(m, isomericSmiles=True)
)
name_col = next((c for c in NAME_COLS if c in df.columns), None)
if name_col is None:
    raise KeyError('No name field found in SDF')
df = df.rename(columns={name_col: 'Name'})[['smiles', 'Name']].dropna()

# 3. Canonicalize and deduplicate
df['SMILES'] = df['smiles'].apply(canonical)
df = df.dropna(subset=['SMILES']).drop_duplicates('SMILES')

# 4. Remove toxic-metal-containing molecules
df = df[
    df['SMILES'].apply(
        lambda smi: not contains_toxic_metal(Chem.MolFromSmiles(smi))
    )
]

# 5. Filter small-molecule drugs
df = df[df['SMILES'].apply(is_small_drug)]

# 6. Save clean list
df[['SMILES', 'Name']].to_csv(CLEAN_CSV, index=False)
print(f'Clean small-molecule drugs (no toxic metals): {len(df)} rows → {CLEAN_CSV}')