#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. Clustering - Hierarchical extraction of 10% of the external blind test set
2. The remaining 90% should be done respectively:
-Random 80/10
-Skeleton 80/10
-Fingerprint clustering 80/10
"""
from pathlib import Path
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp
from collections import defaultdict

# ---------- root ----------
ROOT = Path(__file__).resolve().parent.parent         
IN_FILE = ROOT / "data/01_intermediate/merged_clean.csv"               
SPLITS  = ROOT / "data/split"                              

for d in [SPLITS / "external_split",SPLITS / "random_split", SPLITS / "scaffold_split", SPLITS / "fingerprint_split"]:
    d.mkdir(parents=True, exist_ok=True)

# ---------- global parameter ----------
SMILES_COL = 'SMILES'
LABEL_COL  = 'pIC50'
EXT_RATIO  = 0.10
MAX_TRIAL  = 50
RAND_SEED  = 42

# 1. External clustering - Hierarchical extraction of 10%
def external_cluster_split():
    df = pd.read_csv(IN_FILE)
    mols = [Chem.MolFromSmiles(s) for s in df[SMILES_COL]]
    df['MW']   = [Descriptors.MolWt(m) for m in mols]
    df['logP'] = [Descriptors.MolLogP(m) for m in mols]
    df['TPSA'] = [Descriptors.TPSA(m) for m in mols]
    df['strata'] = pd.cut(df[LABEL_COL], bins=3, labels=['L', 'M', 'H'])

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
    n_clu = max(5, int(len(df) / 100))
    df['cluster'] = KMeans(n_clusters=n_clu, random_state=42).fit_predict(np.array(fps))

    min_cnt = 2
    while True:
        cnt = df.groupby(['strata', 'cluster']).size()
        small = cnt[cnt < min_cnt].index
        if small.empty:
            break
        for (st, cl) in small:
            same_st_max = cnt.xs(st, level=0).idxmax()
            df.loc[df['cluster'] == cl, 'cluster'] = same_st_max
        df['cluster'] = df['cluster'].astype(int)

    def check_dist(ext, rem):
        for c in [LABEL_COL, 'MW', 'logP', 'TPSA']:
            if ks_2samp(ext[c], rem[c])[1] <= 0.05:
                return False
        return True

    seed = RAND_SEED
    for trial in range(MAX_TRIAL):
        ext, rem = train_test_split(
            df, test_size=1 - EXT_RATIO, random_state=seed,
            stratify=df[['strata', 'cluster']]
        )
        if check_dist(ext, rem):
            print(f' The distribution test of the external set has passed  seed={seed}')
            break
        seed += 1
    else:
        raise RuntimeError(' The distribution test was still not passed after 50 re-draws')

    ext.drop(columns=['strata', 'cluster','TPSA','logP','MW']).to_csv(SPLITS / "external_split/external_test.csv", index=False)
    rem.drop(columns=['strata', 'cluster','TPSA','logP','MW']).to_csv(SPLITS / "external_split/remain_90.csv", index=False)
    print('external_test done：', len(ext), f'({len(ext)/len(df):.1%})')

# 2. random 80/10
def random_split():
    df = pd.read_csv(SPLITS / "external_split/remain_90.csv")
    train, val = train_test_split(df, test_size=0.111, random_state=42, shuffle=True)
    train.to_csv(SPLITS / "random_split/random_train.csv", index=False)
    val.to_csv(SPLITS / "random_split/random_val.csv", index=False)
    print('Random train/val done:', len(train), len(val))

# 3. skeleton 80/10
def scaffold_split():
    def generate_scaffold(smiles, include_chirality=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    df = pd.read_csv(SPLITS / "external_split/remain_90.csv")
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(df['SMILES']):
        scaf = generate_scaffold(smi)
        scaffolds[scaf].append(idx)

    scaffold_sets = sorted(list(scaffolds.values()), key=len, reverse=True)

    train_indices, val_indices = [], []
    for s_set in scaffold_sets:

        if len(train_indices) < 0.888 * len(df):
            train_indices.extend(s_set)
        else:
            val_indices.extend(s_set)

    train = df.iloc[train_indices]
    val   = df.iloc[val_indices]
    train.to_csv(SPLITS / "scaffold_split/scaffold_train.csv", index=False)
    val.to_csv(SPLITS / "scaffold_split/scaffold_val.csv", index=False)
    print('Scaffold train/val done:', len(train), len(val))

# 4. Fingerprint clustering 80/10
def fingerprint_split():
    df = pd.read_csv(SPLITS / "external_split/remain_90.csv")
    mols = [Chem.MolFromSmiles(s) for s in df[SMILES_COL]]
    fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]
    X = np.array(fps)
    n_clu = max(5, int(len(df) / 100))
    df['cluster'] = KMeans(n_clusters=n_clu, random_state=42).fit_predict(X)

    train_idx, val_idx = [], []
    for c in range(n_clu):
        idx = df.index[df['cluster'] == c].tolist()
        tr, va = train_test_split(idx, test_size=0.111, random_state=42)
        train_idx.extend(tr)
        val_idx.extend(va)
    train = df.loc[train_idx]
    val   = df.loc[val_idx]
    train.drop(columns=['cluster']).to_csv(SPLITS / "fingerprint_split/fingerprint_train.csv", index=False)
    val.drop(columns=['cluster']).to_csv(SPLITS / "fingerprint_split/fingerprint_val.csv", index=False)
    print('Fingerprint train/val done:', len(train), len(val))

if __name__ == '__main__':
    external_cluster_split()
    random_split()
    scaffold_split()
    fingerprint_split()
    print('>>> all split done！')