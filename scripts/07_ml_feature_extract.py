import os
import sys
import joblib
import numpy as np
import pandas as pd
import argparse
import random
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, MolFromSmiles, rdReducedGraphs, AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
from multiprocessing import Pool

# Path configuration
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "data"
SPLIT_BASE_DIR = ROOT_DIR / "split"
EXTERNAL_CSV = SPLIT_BASE_DIR / "external_split" / "external_test.csv"
EXTERNAL_FEAT_DIR = SPLIT_BASE_DIR / "external_split" / "feature"
BASE_SPLITS = ["fingerprint", "random", "scaffold"]
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check training dependencies existence
def check_train_deps(split_type):
    feat_dir = SPLIT_BASE_DIR / f"{split_type}_split" / "feature"
    return (feat_dir / f"{split_type}_scaler.pkl").exists() and (feat_dir / f"{split_type}_selector.pkl").exists()

# Get existing files for current mode only
def get_all_exist_files(split_type, mode, save_path=None):
    feat_dir = SPLIT_BASE_DIR / f"{split_type}_split" / "feature"
    files = {
        "train": [feat_dir / f"{split_type}_{s}.pkl" for s in ["scaler", "selector", "var_selector", "1200_train", "1200_val"]],
        "test": [EXTERNAL_FEAT_DIR / f"{split_type}_1200_external.pkl"],
        "predict": [save_path] if save_path else []
    }
    return [f for f in files.get(mode, []) if f and f.exists()]

# Batch delete old files for cleanup
def batch_delete_files(file_list):
    for f in file_list:
        if f.exists():
            f.unlink()
    print(f"Deleted {len(file_list)} old files, preparing to re-extract")

# Substructure patterns for feature extraction
_substructs = {k: Chem.MolFromSmarts(v) for k, v in {
    "oh": "[OH]", "nh": "[NH]", "cooh": "C(=O)O", "ester": "C(=O)O[C;!$(OC=O)]"
}.items()}

# Physical descriptors list
PHY_DESC = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
            Descriptors.NumHDonors, Descriptors.NumHAcceptors,
            Descriptors.NumRotatableBonds, Descriptors.RingCount,
            Descriptors.NumAromaticRings, Descriptors.FractionCSP3,
            Descriptors.NHOHCount, Descriptors.MolMR, Descriptors.NumValenceElectrons,
            Descriptors.MaxPartialCharge, Descriptors.MinPartialCharge,
            Descriptors.NumAliphaticRings, Descriptors.NumSaturatedRings,
            Descriptors.NumHeteroatoms, Descriptors.HeavyAtomCount,
            Descriptors.NumSaturatedCarbocycles, Descriptors.BalabanJ, Descriptors.Kappa1]

# Topological torsion fingerprint generator
_ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)

# Extract fingerprint and substructure features
def fp_substruct_feat(mol):
    if mol is None:
        return np.zeros(2220, dtype=np.uint8)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True)
    maccs = AllChem.GetMACCSKeysFingerprint(mol)
    sub5 = [len(mol.GetSubstructMatches(_substructs[k])) for k in ["oh", "nh", "cooh"]] + [Descriptors.NumAromaticRings(mol), len(mol.GetSubstructMatches(_substructs["ester"]))]
    return np.array(list(ecfp) + list(fcfp) + list(maccs) + sub5, dtype=np.uint8)

# Extract 21 physical descriptors
def phy_21(mol):
    return np.zeros(21, dtype=np.float32) if mol is None else np.array([d(mol) for d in PHY_DESC], dtype=np.float32)

# Extract 4506 orthogonal features (atom pairs, torsions, descriptors, pharmacophore)
def ortho_4506(mol):
    if mol is None:
        return np.zeros(4506, dtype=np.float32)
    ap = Pairs.GetAtomPairFingerprintAsBitVect(mol)
    tt = np.array(_ttgen.GetFingerprint(mol), dtype=np.uint8)
    desc200 = np.array([Descriptors.descList[i][1](mol) for i in range(1, 201)], dtype=np.float32)
    phar210 = np.array(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)
    return np.hstack([list(ap), tt, desc200, phar210])

# Process single molecule and extract all features
def process_single_mol(smiles):
    mol = MolFromSmiles(smiles)
    return fp_substruct_feat(mol), phy_21(mol), ortho_4506(mol)

# Build feature selector to reduce dimensions to 1200
def build_selector_1200(X_full, y, target_dim=1200):
    y = y.ravel() if y.ndim == 2 else y
    X_ortho_orig = X_full[:, 2241:6747].copy()
    var_selector = VarianceThreshold(threshold=1e-9)
    try:
        X_ortho = var_selector.fit_transform(X_ortho_orig.copy())
    except ValueError:
        X_ortho, var_selector = X_ortho_orig, None
    n_samples = X_ortho.shape[0]
    pca_ortho = PCA(n_components=min(1500, n_samples, X_ortho.shape[1]), random_state=42, svd_solver='randomized')
    with tqdm(total=100, desc="Starting feature selection", bar_format="{desc}: {bar}|") as pbar:
        X_ortho_pca = pca_ortho.fit_transform(X_ortho)
        pbar.update(50)
        X_merged = np.hstack([X_full[:, :2241], X_ortho_pca])
        must_keep = np.union1d(np.union1d(np.arange(2215, 2220), np.arange(2048, 2214)), np.arange(2220, 2241))
        n_must_keep = len(must_keep)
        if (remaining := target_dim - n_must_keep) < 0:
            raise ValueError(f"Must keep {n_must_keep} dims, exceeds target {target_dim}")
        non_core = np.setdiff1d(np.arange(X_merged.shape[1]), must_keep)
        rf = RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_merged, y)
        importances = rf.feature_importances_
        pbar.update(30)
        top_non_core = non_core if len(non_core) < remaining else non_core[np.argsort(importances[non_core])[-remaining:]]
        top_k_indices = np.union1d(must_keep, top_non_core)
        pbar.update(20)
    print(f"Final selected feature dimensions: {len(top_k_indices)} (target {target_dim})")
    return {"pca_ortho": pca_ortho, "top_k_indices": top_k_indices, "var_selector": var_selector}, pca_ortho, X_merged[:, top_k_indices]

# Extract base features with multiprocessing
def extract_base_features(df, feat_dir, split_type, is_train=False):
    smiles_list = df["SMILES"].tolist()
    print(f"Total {len(smiles_list)} SMILES, starting feature extraction...")
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_single_mol, df.SMILES), total=len(smiles_list), bar_format="{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"))
    X_fp = np.array([r[0] for r in results], dtype=np.uint8)
    X_phy = np.array([r[1] for r in results], dtype=np.float32)
    X_ortho = np.array([r[2] for r in results], dtype=np.float32)
    scaler_path = feat_dir / f"{split_type}_scaler.pkl"
    if not is_train:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler {scaler_path.name}, please run --mode train --split {split_type} first")
        scaler_phy = joblib.load(scaler_path)
        print(f"Loaded scaler: {scaler_path.name}")
    else:
        scaler_phy = StandardScaler().fit(X_phy)
        joblib.dump(scaler_phy, scaler_path)
        print(f"Saved scaler: {scaler_path.name}")
    X_phy = np.nan_to_num(scaler_phy.transform(X_phy), nan=0.0, posinf=0.0, neginf=0.0)
    return np.hstack([X_fp, X_phy, X_ortho]), smiles_list

# Process features to 1200 dimensions
def process_1200_feat(X_full, df, feat_dir, split_type, is_train=False):
    y = df["pIC50"].values.reshape(-1, 1) if "pIC50" in df.columns else None
    selector_path = feat_dir / f"{split_type}_selector.pkl"
    if is_train:
        if y is None:
            raise ValueError("Training set must contain pIC50 column")
        selector, _, X_clean1200 = build_selector_1200(X_full, y)
        joblib.dump(selector, selector_path)
        print(f"Saved selector: {selector_path.name}")
        if selector["var_selector"] is not None:
            joblib.dump(selector["var_selector"], feat_dir / f"{split_type}_var_selector.pkl")
        return np.nan_to_num(X_clean1200, nan=0.0, posinf=0.0, neginf=0.0), y
    if not selector_path.exists():
        raise FileNotFoundError(f"Selector not found {selector_path}, please run training set first")
    selector = joblib.load(selector_path)
    X_ortho = X_full[:, 2241:6747]
    if selector["var_selector"] is not None:
        X_ortho = selector["var_selector"].transform(X_ortho)
    X_merged = np.hstack([X_full[:, :2241], selector["pca_ortho"].transform(X_ortho)])
    return np.nan_to_num(X_merged[:, selector["top_k_indices"]], nan=0.0, posinf=0.0, neginf=0.0), y

# Verify data consistency between saved features and CSV
def check_data_consistency(save_path, csv_path, has_label=True):
    data = joblib.load(save_path)
    X, smiles, y = data['X'], data['SMILES'], data.get('y') if has_label else None
    csv = pd.read_csv(csv_path)
    n_smi = len(smiles)
    with tqdm(total=100, desc="Data consistency check", bar_format="{desc}: {bar}|") as pbar:
        len_check = (X.shape[0] == (y.shape[0] if has_label else n_smi) == n_smi) and (n_smi == len(csv))
        pbar.update(30)
        spot_check = True
        if n_smi == 0:
            print("No samples to check, data is empty")
            spot_check = False
        else:
            for i in random.sample(range(n_smi), min(5, n_smi)):
                print(f"Randomly sampled records for check, indices: {[i]}")
                smi_match = (smiles[i] == csv.loc[i, 'SMILES'])
                label_match = (abs(y[i, 0] - csv.loc[i, 'pIC50']) < 1e-6) if has_label else True
                if not (smi_match and label_match):
                    spot_check = False
                    print(f"Sample {i} check failed: SMILES[{smi_match}], Label[{label_match}]")
                    break
        pbar.update(70)
    if len_check and spot_check:
        print("Consistency check passed: data fully matches")
    else:
        print("Consistency check failed: data has errors, please check")
        sys.exit(1)

# Process train/val split
def process_single_split_train(split_type, test_mode=False):
    split_dir = SPLIT_BASE_DIR / f"{split_type}_split"
    feat_dir = split_dir / "feature"
    feat_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print(f"Starting processing [{split_type}] split train/val set")
    print(f"Data directory: {split_dir}")
    for phase in ["train", "val"]:
        csv_path = split_dir / f"{split_type}_{phase}.csv"
        save_path = feat_dir / f"{split_type}_1200_{phase}.pkl"
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if test_mode:
            df = df.sample(min(100, len(df)), random_state=42)
        print(f"----- Processing {split_type}_{phase} -----")
        X_full, smiles_list = extract_base_features(df, feat_dir, split_type, is_train=(phase=="train"))
        X_clean1200, y = process_1200_feat(X_full, df, feat_dir, split_type, is_train=(phase=="train"))
        save_data = {"X": X_clean1200, "SMILES": smiles_list, **({"y": y} if y is not None else {})}
        joblib.dump(save_data, save_path)
        check_data_consistency(save_path, csv_path, has_label=True)
        print(f"{split_type}_{phase} processing completed")
        print(f"   Feature shape: {X_clean1200.shape} | SMILES count: {len(smiles_list)}")
        print(f"   Label shape: {y.shape if y is not None else 'None'} | Save path: {save_path}")

# Process external test split
def process_single_split_test(split_type, test_mode=False):
    split_feat_dir = SPLIT_BASE_DIR / f"{split_type}_split" / "feature"
    split_feat_dir.mkdir(parents=True, exist_ok=True)
    EXTERNAL_FEAT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = EXTERNAL_FEAT_DIR / f"{split_type}_1200_external.pkl"
    if not check_train_deps(split_type):
        raise FileNotFoundError(f"[{split_type}] Missing training dependencies, please run --mode train --split {split_type} first")
    print(f"[{split_type}] Dependencies ready, processing external test set...")
    if not EXTERNAL_CSV.exists():
        raise FileNotFoundError(f"External test set not found: {EXTERNAL_CSV}")
    df = pd.read_csv(EXTERNAL_CSV)
    if "SMILES" not in df.columns or "pIC50" not in df.columns:
        raise ValueError("External test set must contain SMILES and pIC50 columns")
    if test_mode:
        df = df.sample(min(100, len(df)), random_state=42)
    print("=" * 50)
    print(f"Starting processing [{split_type}] split external test set")
    print(f"Data file: {EXTERNAL_CSV}")
    print(f"----- Processing {split_type}_external -----")
    X_full, smiles_list = extract_base_features(df, split_feat_dir, split_type, is_train=False)
    X_clean1200, y = process_1200_feat(X_full, df, split_feat_dir, split_type, is_train=False)
    joblib.dump({"X": X_clean1200, "y": y, "SMILES": smiles_list}, save_path)
    check_data_consistency(save_path, EXTERNAL_CSV, has_label=True)
    print(f"{split_type}_external processing completed")
    print(f"   Feature shape: {X_clean1200.shape} | SMILES count: {len(smiles_list)}")
    print(f"   Label shape: {y.shape} | Save path: {save_path}")

# Process prediction split (unlabeled)
def process_single_split_predict(split_type, predict_csv, save_pkl, test_mode=False):
    split_feat_dir = SPLIT_BASE_DIR / f"{split_type}_split" / "feature"
    split_feat_dir.mkdir(parents=True, exist_ok=True)
    predict_csv, save_pkl = Path(predict_csv), Path(save_pkl)
    if not check_train_deps(split_type):
        raise FileNotFoundError(f"[{split_type}] Missing training dependencies, please run --mode train --split {split_type} first")
    print(f"[{split_type}] Dependencies ready, processing unlabeled prediction set...")
    if not predict_csv.exists():
        raise FileNotFoundError(f"Prediction set not found: {predict_csv}")
    if "SMILES" not in pd.read_csv(predict_csv, nrows=1).columns:
        raise ValueError("Prediction set must contain SMILES column")
    df = pd.read_csv(predict_csv)
    if test_mode:
        df = df.sample(min(100, len(df)), random_state=42)
    save_pkl.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 50)
    print(f"Starting processing [{split_type}] split unlabeled prediction set")
    print(f"Data file: {predict_csv}")
    print(f"----- Processing {split_type}_predict -----")
    X_full, smiles_list = extract_base_features(df, split_feat_dir, split_type, is_train=False)
    X_clean1200, _ = process_1200_feat(X_full, df, split_feat_dir, split_type, is_train=False)
    joblib.dump({"X": X_clean1200, "SMILES": smiles_list}, save_pkl)
    check_data_consistency(save_pkl, predict_csv, has_label=False)
    print(f"{split_type}_predict processing completed")
    print(f"   Feature shape: {X_clean1200.shape} | SMILES count: {len(smiles_list)}")
    print(f"   Label shape: None | Save path: {save_pkl}")

# Batch process all splits
def batch_process(mode, path=None, save=None, test_mode=False):
    for split_type in BASE_SPLITS:
        if mode == "train":
            process_single_split_train(split_type, test_mode)
        elif mode == "test":
            process_single_split_test(split_type, test_mode)
        elif mode == "predict":
            save_pkl_single = save.parent / (save.name.replace("{split}", split_type) if "{split}" in save.name else f"{save.stem}_{split_type}{save.suffix}")
            process_single_split_predict(split_type, path, save_pkl_single, test_mode)

# Main entry with argument parsing and overwrite confirmation
def main():
    parser = argparse.ArgumentParser(description="1200D molecular feature extraction tool")
    parser.add_argument("--split", required=True, choices=BASE_SPLITS + ["all"], help="Data split: fingerprint/random/scaffold/all")
    parser.add_argument("--mode", required=True, choices=["train", "test", "predict"], help="Run mode: train/test/predict")
    parser.add_argument("--path", type=str, default=str(ROOT_DIR / "01_intermediate/fda_approvedrug_clean.csv"), help="Only for predict mode: prediction set CSV path")
    parser.add_argument("--save", type=str, default=str(SCRIPT_DIR.parent/ "results/summary/fda_drug/feature/{split}_ml_fda.pkl"), help="Only for predict mode: feature save path (.pkl)")
    parser.add_argument("--test-mode", action="store_true", default=False, help="Test mode, sample 100 records for quick validation")
    args = parser.parse_args()
    
    if args.mode == "predict":
        if args.path is None or args.save is None:
            parser.error("predict mode must specify both --path (prediction set CSV) and --save (feature save path)")
        args.path, args.save = Path(args.path), Path(args.save)
    
    need_process_splits = BASE_SPLITS if args.split == "all" else [args.split]
    all_exist_files = []
    for split_type in need_process_splits:
        if args.mode in ["test", "predict"] and not check_train_deps(split_type):
            raise FileNotFoundError(f"[{split_type}] split training extraction not completed, missing scaler/selector!\nPlease run first: python {sys.argv[0]} --split {split_type} --mode train")
        target_save_path = args.save if (args.mode == "predict" and args.split != "all") else (args.save.parent / f"{args.save.stem}_{split_type}{args.save.suffix}" if (args.mode == "predict" and args.split == "all") else None)
        all_exist_files.extend(get_all_exist_files(split_type, args.mode, target_save_path))
    
    if all_exist_files:
        all_exist_files = list(set(all_exist_files))
        mode_file_desc = "test set features" if args.mode == "test" else "prediction set features" if args.mode == "predict" else "train/val set features + dependencies"
        print(f"Detected {len(all_exist_files)} existing [{mode_file_desc}] files (covering target split)")
        while True:
            res = input(f"Delete these {mode_file_desc} files and re-extract? (y/n): ")
            if res.lower() == "n":
                print("User chose not to overwrite, exiting")
                sys.exit(0)
            elif res.lower() == "y":
                batch_delete_files(all_exist_files)
                print(f"Deleted all old {mode_file_desc} files, preparing to re-extract")
                break
            else:
                print("Invalid input, please enter y/n")
    
    try:
        if args.split == "all":
            batch_process(args.mode, args.path, args.save, args.test_mode)
        elif args.mode == "train":
            process_single_split_train(args.split, args.test_mode)
        elif args.mode == "test":
            process_single_split_test(args.split, args.test_mode)
        elif args.mode == "predict":
            save_path = Path(str(args.save).replace("{split}", args.split)) if "{split}" in str(args.save) else args.save
            process_single_split_predict(args.split, args.path, save_path, args.test_mode)
    except Exception as e:
        print(f"\nRuntime error: {type(e).__name__} - {str(e)}")
        sys.exit(1)
    print("=" * 50)
    print("All processing tasks completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()