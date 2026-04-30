#!/usr/bin/env python
"""
ChEMBL IC50 cleaning from semi-colon CSV
- Input:  CSV
- Output:  ID, SMILES, IC50(nM)
- Rule: IC50 & '=' & units=nM only
"""
import pandas as pd
from pathlib import Path

# ---------- 1. read (semicolon, has header) ----------
in_file = Path(__file__).parent.parent / "data/00_raw/chembl_raw.csv"
df = pd.read_csv(in_file, sep=";", header=0)

# ---------- 2. keep & filter ----------
keep = ["Molecule ChEMBL ID", "Smiles", "Standard Type", "Standard Relation", "Standard Value", "Standard Units"]
df = df[keep]

df = df[df["Standard Type"] == "IC50"]
df = df[df["Standard Relation"].str.strip() == "'='"]

# ---------- 3. units → nM ----------
def to_nM(val, unit):
    val, unit = float(val), str(unit).strip()
    if unit == "uM":  return val * 1000
    if unit == "pM":  return val * 0.001
    return val  # already nM

df["IC50(nM)"] = df.apply(lambda r: to_nM(r["Standard Value"], r["Standard Units"]), axis=1)

# ---------- 4. final columns & dropna last ----------
out = df[["Molecule ChEMBL ID", "Smiles", "IC50(nM)"]].rename(
    columns={"Molecule ChEMBL ID": "ID", "Smiles": "SMILES"}
).dropna()

# ---------- 5. sort by IC50(nM) and dedup by SMILES (keep smallest) ----------
out = out.groupby("SMILES", as_index=False).agg({
    "IC50(nM)": "median",
    "ID": lambda x: "|".join(sorted(set(x.astype(str))))
})

# ---------- 6. output ----------
out_file = Path(__file__).parent.parent / "data/01_intermediate/chembl_clean.csv"
out.to_csv(out_file, index=False)
print(f"✅ Clean → {out_file}  {len(out)} rows")