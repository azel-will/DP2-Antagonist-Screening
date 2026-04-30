"""
pubchem Q9Y5Y4 IC50 data-cleaning & SMILES fetching script.
"""
import pandas as pd
import requests
import io
from pathlib import Path

# ---------- 1.read ----------
in_file = Path(__file__).parent.parent / "data/00_raw/pubchem_raw.csv"
df = pd.read_csv(in_file)  # Change to your path

# ---------- 2.clean ----------
df = df[['Activity_Value', 'Compound_CID', 'Activity_Type', 'Activity']]
df = df[df['Activity_Type'] == 'IC50']
df['Activity_Value'] *= 1000 # Convert μM to nM
df = df[df['Activity'].isin(['Active', 'Inactive'])]
df = df.rename(columns={'Activity_Value': 'IC50(nM)', 'Compound_CID': 'ID'})
df = df[['ID', 'IC50(nM)']].dropna()
df = df.drop_duplicates(subset='ID')
# ---------- 3.get SMILES ----------
cids = df['ID'].tolist()
cids_str = ','.join(map(str, cids))

url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES,Title/CSV'
headers = {"Content-Type": "application/x-www-form-urlencoded"}
data = {"cid": cids_str}

try:
    response = requests.post(url, data=data, headers=headers)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"请求出错: {e}")
    raise
# ---------- 4.merge ----------
smiles_df = pd.read_csv(io.StringIO(response.text))
df_merge = pd.merge(
    df,
    smiles_df,
    left_on='ID',
    right_on='CID',
    how='left'
).drop(columns='CID')

df_merge = df_merge.rename(columns={'ConnectivitySMILES': 'SMILES'})
df_merge = df_merge.groupby("SMILES", as_index=False).agg({
    "IC50(nM)": "median",
    "ID": lambda x: "|".join(sorted(set(x.astype(str))))
})
# ---------- 5.save ----------
out_file = Path(__file__).parent.parent / "data/01_intermediate/pubchem_clean.csv"
df_merge[['ID', 'IC50(nM)', 'SMILES']].to_csv(out_file, index=False)
print(f'✅ dataclean → {out_file}  all {len(df_merge)} rows')