"""
Three-file merge → Standardized SMILES → IC50 sorting → Deduplication by SMILES → pIC50
Ultimately, only the ID, pIC50, SMILES, will be retained
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from pathlib import Path

# ----------root ----------
ROOT = Path(__file__).resolve().parent.parent 
files = [
    ROOT / "data/01_intermediate/bindingdb_clean.csv",
    ROOT / "data/01_intermediate/pubchem_clean.csv",
    ROOT / "data/01_intermediate/chembl_clean.csv",
]
out_file = ROOT / "data/01_intermediate/merged_clean.csv"

# ---------- function ----------
def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

# ---------- main ----------
dfs = []
must_cols = {"IC50(nM)", "SMILES", "ID"}

for f in files:
    if not f.exists():
        raise FileNotFoundError(f"The file cannot be found: {f}")
    df_tmp = pd.read_csv(f)
    miss = must_cols - set(df_tmp.columns)
    if miss:
        raise ValueError(f"{f} Missing required columns: {miss}")
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

# ---------- clean ----------
df["SMILES"] = df["SMILES"].apply(canonical)
df = df.dropna(subset=["SMILES"])
before = len(df)
df = df.groupby("SMILES", as_index=False).agg({
    "IC50(nM)": "median",
    "ID": lambda x: "|".join(sorted(set(x.astype(str))))
})
print(f" Before deduplication {before} rows，After deduplication {len(df)} rows，deal {before - len(df)} rows")
# IC50 → pIC50
df["pIC50"] = 9 - np.log10(df["IC50(nM)"])
# save
df = df[["ID", "pIC50", "SMILES", "IC50(nM)"]]
out_file.parent.mkdir(parents=True, exist_ok=True) 
df.to_csv(out_file, index=False)
print(f" Merge Completed → {out_file}  all {len(df)} rows")