"""
Train the LightGBM model and save it for subsequent SHAP analysis
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
from multiprocessing import Pool
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import r2_score
# ==================== root====================
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"/"shap"
MODELS_DIR = RESULTS_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(str(SCRIPT_DIR))  
import importlib
feature_extract = importlib.import_module('07_ml_feature_extract')

train_csv = DATA_DIR / "split" / "fingerprint_split" / "fingerprint_train.csv"
val_csv = DATA_DIR / "split" / "fingerprint_split" / "fingerprint_val.csv"

model_output = MODELS_DIR / "lightgbm_fp_physchem.pkl"
feature_output = MODELS_DIR / "shap_background.pkl"  
names_output = MODELS_DIR / "feature_names.json"

lgb_params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

n_processes = 8

# ==================== Generate feature names ====================
fp_names = [f'ECFP4_{i}' for i in range(1024)] + \
           [f'FCFP4_{i}' for i in range(1024)] + \
           [f'MACCS_{i}' for i in range(167)] + \
           ['OH', 'NH', 'COOH', 'AromRings', 'Ester']
phy_names = ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
             'NumRotatableBonds', 'RingCount', 'NumAromaticRings', 'FractionCSP3',
             'NHOHCount', 'MolMR', 'NumValenceElectrons', 'MaxPartialCharge',
             'MinPartialCharge', 'NumAliphaticRings', 'NumSaturatedRings',
             'NumHeteroatoms', 'HeavyAtomCount', 'NumSaturatedCarbocycles',
             'BalabanJ', 'Kappa1']
feature_names = fp_names + phy_names
print(f"feature dimension: {len(feature_names)}")
with open(names_output, 'w') as f:
    json.dump(feature_names, f)
print(f"feature names saved: {names_output}")

# ==================== Feature extraction function ====================
def extract_fp_physchem(smiles):
    mol = feature_extract.Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2241, dtype=np.float32)
    fp = feature_extract.fp_substruct_feat(mol).astype(np.float32)
    phy = feature_extract.phy_21(mol).astype(np.float32)
    return np.concatenate([fp, phy])

# ==================== Read the training set====================
print(f"Read the training set: {train_csv}")
df = pd.read_csv(train_csv)
smiles_list = df['SMILES'].tolist()
y = df['pIC50'].values
print(f"Training set sample count: {len(smiles_list)}")

# ==================== Parallel feature extraction ====================
print("Starting parallel feature extraction...")
with Pool(processes=n_processes) as pool:
    results = list(tqdm(pool.imap(extract_fp_physchem, smiles_list),
                        total=len(smiles_list), desc="Training set"))
X_full = np.array(results, dtype=np.float32)
print(f"Training set feature matrix shape: {X_full.shape}")

# ==================== Train LightGBM ====================
print("Training LightGBM model...")
model = lgb.LGBMRegressor(**lgb_params)
model.fit(X_full, y)

y_pred = model.predict(X_full)
r2_train = r2_score(y, y_pred)
print(f"Training set R² = {r2_train:.4f}")

# ==================== Validation set ====================
print(f"Reading validation set: {val_csv}")
df_val = pd.read_csv(val_csv)
smiles_list_val = df_val['SMILES'].tolist()
y_val = df_val['pIC50'].values

print("Extracting validation set features...")
with Pool(processes=n_processes) as pool:
    results_val = list(tqdm(pool.imap(extract_fp_physchem, smiles_list_val),
                            total=len(smiles_list_val), desc="Validation set"))
X_val = np.array(results_val, dtype=np.float32)

y_val_pred = model.predict(X_val)
r2_val = r2_score(y_val, y_val_pred)
print(f"Validation set R² = {r2_val:.4f}")

# ==================== Save the model====================
print(f"Saving model to: {model_output}")
joblib.dump(model, model_output)
print("Model saved successfully!")
shap_data = {
    'X_sample': X_full,      # Feature matrix for SHAP
    'y': y,                   # Corresponding labels
    'feature_names': feature_names,  # Feature names
    'r2_train': r2_train,     # Training set performance
    'r2_val': r2_val,         # Validation set performance
    'model_params': lgb_params  # Model parameters
}

print(f"Saving SHAP background data to: {feature_output}")
joblib.dump(shap_data, feature_output)
print("SHAP background data saved successfully!")

print("\n" + "="*50)
print("Training completed! Output files:")
print(f"  Model: {model_output}")
print(f"  SHAP data: {feature_output}")
print(f"  Feature names: {names_output}")
print("="*50)