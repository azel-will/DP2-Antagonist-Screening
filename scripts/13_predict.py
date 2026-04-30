import numpy as np, pandas as pd, joblib, torch, dgl, sys, argparse
from pathlib import Path
import importlib
from tqdm import tqdm
from scipy.stats import pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results"
SCALER_DIR = ROOT_DIR / "split"
BASE_SPLITS = ["fingerprint", "random", "scaffold"]

graph_module = importlib.import_module("10_graph_modeltrain")
simpleGCN, EnhancedGAT, EnhancedAFP = graph_module.simpleGCN, graph_module.EnhancedGAT, graph_module.EnhancedAFP

PIC50_MIN, PIC50_MAX = 3.0, 11.0
ACTIVITY_THRESHOLD, CONFIDENCE_POWER = 8.0, 2.5
HIGH_ACTIVE_WEIGHT_BOOST, UNCERTAINTY_WEIGHT = 1.4, 0.4
CORRELATION_PENALTY, DEVICE = 0.15, torch.device('cpu')

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Unlabeled data ensemble prediction with SMILES output")
    parser.add_argument("--split", required=True, choices=BASE_SPLITS)
    parser.add_argument("--predict-ml-feat", type=str, default=None, help="ML feature path (pkl)")
    parser.add_argument("--predict-graph-feat", type=str, default=None, help="Graph feature path (pkl)")
    parser.add_argument("--output", type=str, default=None, help="Output result path (csv)")
    args = parser.parse_args()
    
    # Set default paths after split is known
    if args.predict_ml_feat is None:
        args.predict_ml_feat = str(RESULT_DIR / f"summary/fda_drug/feature/{args.split}_ml_fda.pkl")
    if args.predict_graph_feat is None:
        args.predict_graph_feat = str(RESULT_DIR / "summary/fda_drug/feature/graphdata_fda.pkl")
    if args.output is None:
        args.output = str(RESULT_DIR / f"summary/fda_drug/{args.split}_fda_predict.csv")
    
    return args

# Load and calibrate label scaler for denormalization
def load_label_scaler(split):
    scaler_path = SCALER_DIR / f"{split}_split" / "feature" / f"graph_label_minmax_{split}.pkl"
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            scaler.data_min_ = np.array([PIC50_MIN])
            scaler.data_max_ = np.array([PIC50_MAX])
            scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
            scaler.min_ = -scaler.data_min_ * scaler.scale_
            print(f"Successfully loaded and calibrated label scaler: {scaler_path}")
            print(f"Scaler calibrated range: pIC50 [{PIC50_MIN}, {PIC50_MAX}]")
            return scaler
        except Exception as e:
            print(f"Failed to load scaler: {e}, will use normalized values")
            return None
    else:
        print(f"Scaler file not found: {scaler_path}, will use normalized values")
        return None

# Generate path configuration including scaler
def get_paths(split):
    model_base = RESULT_DIR / f"{split}_split" / "models"
    feat_base = ROOT_DIR / "split" / f"{split}_split" / "feature"
    label_scaler = load_label_scaler(split)
    return {
        "ensemble_config": model_base / f"{split}_ensemblemodel.pkl",
        "ml_train": feat_base / f"{split}_1200_train.pkl",
        "graph_train": feat_base / f"{split}_train_graphdata.pkl",
        "split_name": split,
        "model_base_dir": model_base,
        "label_scaler": label_scaler
    }

# Load data with optional labels and SMILES extraction
def load_data(ml_path, graph_path, require_labels=False):
    ml_data, graph_data = joblib.load(ml_path), joblib.load(graph_path)
    X_ml = ml_data['X']
    graphs = [item['graph'] for item in graph_data]
    smiles_list = [item['smiles'] for item in graph_data]
    assert len(X_ml) == len(graphs) == len(smiles_list), "Sample/graph/SMILES count mismatch"
    if require_labels:
        return X_ml, graphs, ml_data.get('y'), smiles_list
    else:
        return X_ml, graphs, smiles_list

# Load all base models and ensemble configuration
def load_models(paths):
    if not paths["ensemble_config"].exists():
        raise FileNotFoundError(f"Ensemble config missing: {paths['ensemble_config']}")
    config = joblib.load(paths["ensemble_config"])
    model_names, split_name, model_base = config["model_names"], paths["split_name"], paths["model_base_dir"]
    print(f"Loaded model list: {model_names}")
    models = {}
    for name in model_names:
        is_graph = name in ["AFP", "GCN", "GAT"]
        suffix = f"{name.lower()}_{split_name}.pt" if is_graph else f"{split_name}_{name.lower()}_model.pkl"
        model_path = model_base / suffix
        if not model_path.exists():
            raise FileNotFoundError(f"Model missing: {model_path}")
        model = torch.load(model_path, map_location=DEVICE) if is_graph else joblib.load(model_path)
        if is_graph:
            model.eval()
        models[name] = model
        print(f"Loaded: {name}")
    return models, config, model_names

# Get predictions from all models with denormalization for graph models
def get_predictions(ml_feat, graph_feat, models, model_names, label_scaler,data_type="unlabeled"):
    preds = {}
    graph_names = [n for n in model_names if n in ["AFP", "GCN", "GAT"]]
    ml_names = [n for n in model_names if n not in ["AFP", "GCN", "GAT"]]
    if graph_names and graph_feat:
        batch = dgl.batch(graph_feat).to(DEVICE)
        with torch.no_grad():
            for name in graph_names:
                pred_scaled = models[name](batch).detach().cpu().numpy().ravel()
                if label_scaler is not None:
                    pred_original = label_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                    pred_original = np.clip(pred_original, PIC50_MIN, PIC50_MAX)
                    preds[name] = pred_original
                else:
                    preds[name] = pred_scaled
    for name in tqdm(ml_names, desc="ML prediction"):
        preds[name] = models[name].predict(ml_feat)
    print(f"\nModel prediction statistics ({data_type} data, original pIC50 scale):")
    for name in model_names:
        pred_max, pred_min, pred_mean = np.max(preds[name]), np.min(preds[name]), np.mean(preds[name])
        print(f"  {name}: max={pred_max:.4f} | min={pred_min:.4f} | mean={pred_mean:.4f}")
    return preds

# Calculate dynamic weights based on model performance and correlation
def calculate_weights(preds_train, config, preds_unlabeled, model_names):
    n_samples = len(preds_unlabeled[model_names[0]])
    model_types, threshold = config["model_types"], ACTIVITY_THRESHOLD
    errors, variances = config["model_errors"], config["model_variances"]
    feat_imp, high_acc = config["feature_weights"], config["high_active_acc"]
    base_conf = {
        n: ((1-UNCERTAINTY_WEIGHT)/(errors.get(n, 1e-2)+1e-8) + 
            UNCERTAINTY_WEIGHT/(variances.get(n, 1e-2)+1e-8)) * feat_imp.get(n, 1/len(model_names))
        for n in model_names
    }
    base_conf = {n: v/sum(base_conf.values()) for n, v in base_conf.items()}
    corr_matrix = {}
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            c = pearsonr(preds_train[n1], preds_train[n2])[0] if n1 in preds_train and n2 in preds_train else 0.0
            corr_matrix[(n1,n2)] = corr_matrix[(n2,n1)] = 0.0 if np.isnan(c) else c
    weights = {n: np.zeros(n_samples) for n in model_names}
    for i in tqdm(range(n_samples), desc="Calculating weights"):
        vals = np.array([preds_unlabeled[n][i] for n in model_names])
        mean, dev, std = np.mean(vals), np.square(vals - np.mean(vals)), np.std(vals)
        w = np.array([base_conf[n] for n in model_names]) * (1/(dev+1e-8))**CONFIDENCE_POWER
        for j, name in enumerate(model_names):
            penalty = sum(corr_matrix.get((name, other), 0) * (1.2 if model_types[name] == model_types[other] else 0.8) 
                         for other in model_names if other != name)
            w[j] *= max(0.1, 1 - CORRELATION_PENALTY * penalty / (len(model_names)-1))
        if np.mean(vals) >= threshold:
            w *= np.array([high_acc[n] for n in model_names])
            w = w ** (CONFIDENCE_POWER * (0.5 if std < 0.3 else 0.3))
            w *= HIGH_ACTIVE_WEIGHT_BOOST
        weights_list = [w[idx] for idx in range(len(model_names))]
        total = sum(weights_list)
        for idx, name in enumerate(model_names):
            weights[name][i] = max(weights_list[idx] / total, 1e-8) if total > 0 else 1.0 / len(model_names)
    return weights

# Calculate weighted ensemble prediction with final clipping
def ensemble_predict(preds, weights):
    ensemble = sum(weights[n] * preds[n] for n in preds.keys())
    return np.clip(ensemble, PIC50_MIN, PIC50_MAX)

# Save prediction results with SMILES and model details
def save_results(preds, weights, ensemble, smiles_list, output):
    df = pd.DataFrame({
        "SMILES": smiles_list,
        "ensemble_pred": ensemble,
        "is_high_active": (ensemble >= ACTIVITY_THRESHOLD).astype(int)
    })
    for n in preds:
        df[f"{n}_pred"] = preds[n]
    for n in weights:
        df[f"{n}_weight"] = weights[n]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")
    print(f"\nResults saved (with SMILES): {output}")
    return df

# Main execution flow for ensemble prediction
def main():
    try:
        args = parse_args()
        print(f"\n===== Ensemble prediction | split: {args.split} =====")
        paths = get_paths(args.split)
        label_scaler = paths["label_scaler"]
        X_ml_u, X_graph_u, smiles_u = load_data(args.predict_ml_feat, args.predict_graph_feat)
        print(f"Unlabeled data: {len(X_ml_u)} samples | SMILES count: {len(smiles_u)}")
        models, config, model_names = load_models(paths)
        X_ml_t, X_graph_t, _, _ = load_data(paths["ml_train"], paths["graph_train"], require_labels=True)
        preds_train = get_predictions(X_ml_t, X_graph_t, models, model_names, label_scaler, data_type="training")
        preds_unlabeled = get_predictions(X_ml_u, X_graph_u, models, model_names, label_scaler, data_type="unlabeled")
        weights = calculate_weights(preds_train, config, preds_unlabeled, model_names)
        ensemble_result = ensemble_predict(preds_unlabeled, weights)
        ens_max, ens_min, ens_mean, ens_std = np.max(ensemble_result), np.min(ensemble_result), np.mean(ensemble_result), np.std(ensemble_result)
        high_active_count = sum(ensemble_result >= ACTIVITY_THRESHOLD)
        print("\nEnsemble prediction statistics (unlabeled data, original pIC50 scale):")
        print(f"  max={ens_max:.4f} | min={ens_min:.4f} | mean={ens_mean:.4f} | std={ens_std:.4f}")
        print(f"  High active count (≥{ACTIVITY_THRESHOLD}): {high_active_count}/{len(ensemble_result)} ({high_active_count/len(ensemble_result):.2%})")
        save_results(preds_unlabeled, weights, ensemble_result, smiles_u, args.output)
        print(f"\nCompleted! Result file contains SMILES, model predictions (original scale), weights, ensemble result")
    except Exception as e:
        print(f"\nFailed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()