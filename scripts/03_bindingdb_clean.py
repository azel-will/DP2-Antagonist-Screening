#!/usr/bin/env python
"""
BindingDB IC50 cleaning from SDF
- Input:  SDF
- Output:  ID, SMILES, IC50(nM)  (only final clean)
- Rule:  InChI→SMILES, drop ><≤≥, sort by IC50, dedup by SMILES, keep smallest
- Relative paths, English comments
"""
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import PandasTools

# ---------- 1. SDF → DataFrame ----------
in_sdf = Path(__file__).parent.parent / "data/00_raw/bindingdb_raw.sdf"
df = PandasTools.LoadSDF(str(in_sdf))

# ---------- 2. Only leave needed ----------
keep = ["BindingDB MonomerID", "Ligand InChI", "IC50 (nM)"]
df = df[keep].dropna(subset=keep)

# ---------- 3. ID→InChI to SMILES ----------
id2inchi = df.set_index("BindingDB MonomerID")["Ligand InChI"].to_dict()

# unique InChI 
inchi2smiles = {}
for inchi in set(id2inchi.values()):
    mol = Chem.MolFromInchi(inchi)
    inchi2smiles[inchi] = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

df["SMILES"] = df["BindingDB MonomerID"].map(
    lambda mid: inchi2smiles[id2inchi[mid]]
)
df = df.dropna(subset=["SMILES"])
df = df.rename(columns={
    "BindingDB MonomerID": "ID",
    "IC50 (nM)": "IC50(nM)"
})

# ---------- 5. clean ><≤≥  ----------
mask = df["IC50(nM)"].astype(str).str.contains(r"[><≤≥]", na=False)
df = df[~mask].copy()
df["IC50(nM)"] = (
    df["IC50(nM)"]
    .astype(str)
    .str.replace(r"[,\s]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)
df = df[df["IC50(nM)"] > 0]
df['ID'] = 'BDBM' + df['ID'].astype(str)
df = df.groupby("SMILES", as_index=False).agg({
    "IC50(nM)": "median",
    "ID": lambda x: "|".join(sorted(set(x.astype(str))))
})
# ---------- 8. output ----------
out_file = Path(__file__).parent.parent / "data/01_intermediate/bindingdb_clean.csv"
df[["ID", "SMILES", "IC50(nM)"]].to_csv(out_file, index=False)
print(f"✅ Clean → {out_file}  {len(df)} rows")