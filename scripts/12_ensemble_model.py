import numpy as np, pandas as pd, joblib, torch, dgl, sys, argparse
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.base import clone
from tqdm import tqdm
import importlib

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results"
METRICS_DIR = RESULT_DIR / "summary" / "ensemble_result"
SCALER_DIR = ROOT_DIR / "split"
BASE_SPLITS = ["fingerprint", "random", "scaffold"]

graph_module = importlib.import_module("10_graph_modeltrain")
simpleGCN, EnhancedGAT, EnhancedAFP = graph_module.simpleGCN, graph_module.EnhancedGAT, graph_module.EnhancedAFP

ACTIVITY_THRESHOLD, TOP_RATIO, CONFIDENCE_POWER = 8.0, 0.1, 2.5
HIGH_ACTIVE_WEIGHT_BOOST, UNCERTAINTY_WEIGHT, CORRELATION_PENALTY = 1.4, 0.4, 0.15
HIGH_ACTIVE_TOLERANCE, KFOLD_SPLITS, RANDOM_SEED = 0.3, 5, 42
PIC50_MIN, PIC50_MAX = 3.0, 11.0
DEVICE = torch.device('cpu')

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic weighted ensemble model")
    parser.add_argument("--split", required=True, choices=BASE_SPLITS)
    parser.add_argument("--models", required=True, nargs="+",
                        choices=["afp", "gcn", "gat", "xgb", "lgb", "gb", "svm", "rf"])
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--external-ml-feat", type=str,
                        default=str(ROOT_DIR / "split/external_split/feature/{split}_1200_external.pkl"))
    parser.add_argument("--external-graph-feat", type=str,
                        default=str(ROOT_DIR / "split/external_split/feature/external_test_graphdata.pkl"))
    args = parser.parse_args()
    args.models = [m.upper() for m in args.models]
    args.external_ml_feat = args.external_ml_feat.format(split=args.split)
    args.external_graph_feat = args.external_graph_feat.format(split=args.split)
    return args

# Get file paths for all model components
def get_paths(split):
    feat_dir = SCALER_DIR / f"{split}_split" / "feature"
    model_dir = RESULT_DIR / f"{split}_split" / "models"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = SCALER_DIR / f"{split}_split" / "feature" / f"graph_label_minmax_{split}.pkl"
    return {
        "ml_train": feat_dir / f"{split}_1200_train.pkl",
        "ml_val": feat_dir / f"{split}_1200_val.pkl",
        "graph_train": feat_dir / f"{split}_train_graphdata.pkl",
        "graph_val": feat_dir / f"{split}_val_graphdata.pkl",
        "AFP": model_dir / f"afp_{split}.pt",
        "GCN": model_dir / f"gcn_{split}.pt",
        "GAT": model_dir / f"gat_{split}.pt",
        "XGB": model_dir / f"{split}_xgb_model.pkl",
        "LGB": model_dir / f"{split}_lgb_model.pkl",
        "GB": model_dir / f"{split}_gb_model.pkl",
        "SVM": model_dir / f"{split}_svm_model.pkl",
        "RF": model_dir / f"{split}_rf_model.pkl",
        "ensemble": model_dir / f"{split}_ensemblemodel.pkl",
        "metrics_summary": METRICS_DIR / f"{split}_ensemble_metrics.csv",
        "split_name": split,
        "label_scaler": scaler_path
    }

# Load and calibrate label scaler for denormalization
def load_label_scaler(scaler_path):
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
            print(f"Failed to load/calibrate scaler: {e}, will use normalized values")
            return None
    else:
        print(f"Scaler file not found: {scaler_path}, will use normalized values")
        return None

# Verify all required files exist
def check_paths(paths, mode, model_names):
    if mode == "train":
        for key in ["ml_train", "ml_val", "graph_train", "graph_val"]:
            if not paths[key].exists():
                raise FileNotFoundError(f"Feature missing: {paths[key]}")
    if not paths["ensemble"].exists() and mode == "test":
        raise FileNotFoundError(f"Ensemble model not trained: {paths['ensemble']}")
    for name in model_names:
        if not paths[name].exists():
            raise FileNotFoundError(f"Model missing: {paths[name]}")
    print("Path check passed")

# Load training or test data
def load_data(paths, mode):
    if mode == "train":
        ml_train = joblib.load(paths["ml_train"])
        ml_val = joblib.load(paths["ml_val"])
        graph_train = joblib.load(paths["graph_train"])
        graph_val = joblib.load(paths["graph_val"])
        y_train, y_val = ml_train['y'].ravel(), ml_val['y'].ravel()
        assert np.allclose(y_train, [i['label'] for i in graph_train], atol=1e-4)
        assert np.allclose(y_val, [i['label'] for i in graph_val], atol=1e-4)
        return (ml_train['X'], [i['graph'] for i in graph_train], y_train), \
               (ml_val['X'], [i['graph'] for i in graph_val], y_val)
    else:
        ml_test = joblib.load(paths["external_ml_feat"])
        graph_test = joblib.load(paths["external_graph_feat"])
        y_test = ml_test['y'].ravel()
        assert np.allclose(y_test, [i['label'] for i in graph_test], atol=1e-4)
        return ml_test['X'], [i['graph'] for i in graph_test], y_test

# Load all base models and label scaler
def load_models(paths, model_names):
    base_models = {}
    label_scaler = load_label_scaler(paths["label_scaler"])
    for name in model_names:
        try:
            if name in ["AFP", "GCN", "GAT"]:
                model = torch.load(paths[name], map_location=DEVICE)
                model.eval()
            else:
                model = joblib.load(paths[name])
            base_models[name] = model
            print(f"Loaded model: {name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {name}: {e}")
    return base_models, label_scaler

# Get predictions from all models with optional cross-validation
def get_predictions(ml_feat, graph_feat, models, model_names, label_scaler, is_train=False, y_true=None):
    preds = {}
    graph_models = [m for m in model_names if m in ["AFP", "GCN", "GAT"]]
    if graph_models and graph_feat:
        with torch.no_grad():
            batch = dgl.batch(graph_feat).to(DEVICE)
            for name in graph_models:
                pred_scaled = models[name](batch).detach().cpu().numpy().ravel()
                if label_scaler is not None:
                    pred_original = label_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                    pred_original = np.clip(pred_original, PIC50_MIN, PIC50_MAX)
                    preds[name] = pred_original
                    if name == graph_models[0] and len(pred_scaled) > 0:
                        print(f"\n{name} prediction denormalization example (first 5):")
                        print(f"  Normalized: {pred_scaled[:5]}")
                        print(f"  Original (clipped): {pred_original[:5]}")
    
    ml_models = [m for m in model_names if m in ["XGB", "LGB", "GB", "SVM", "RF"]]
    for name in tqdm(ml_models, desc="ML model predictions"):
        preds[name] = models[name].predict(ml_feat)
    
    errors, variances = {}, {}
    if is_train and y_true is not None:
        errors = {n: 0.0 for n in model_names}
        variances = {n: 0.0 for n in model_names}
        kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        
        for tr_idx, te_idx in tqdm(kf.split(ml_feat), desc="Cross-validation", total=KFOLD_SPLITS):
            for name in ml_models:
                if name == "LGB":
                    pred = models[name].predict(ml_feat[te_idx])
                else:
                    clf = clone(models[name])
                    clf.fit(ml_feat[tr_idx], y_true[tr_idx])
                    pred = clf.predict(ml_feat[te_idx])
                errors[name] += np.mean((pred - y_true[te_idx])**2)
                variances[name] += np.var(pred)
            
            for name in graph_models:
                graph_te = [graph_feat[i] for i in te_idx]
                if not graph_te:
                    continue
                with torch.no_grad():
                    pred_scaled = models[name](dgl.batch(graph_te).to(DEVICE)).detach().cpu().numpy().ravel()
                    if label_scaler is not None:
                        pred = label_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
                    else:
                        pred = pred_scaled
                errors[name] += np.mean((pred - y_true[te_idx])**2)
                variances[name] += np.var(pred)
        
        for name in model_names:
            errors[name] /= KFOLD_SPLITS
            variances[name] /= KFOLD_SPLITS
    
    return preds, errors, variances

# Calculate feature importance for weighting
def get_feature_importance(X_ml_train, models, model_names):
    feat_imp = {}
    ml_models = [m for m in model_names if m in ["XGB", "LGB", "GB", "SVM", "RF"]]
    graph_models = [m for m in model_names if m in ["AFP", "GCN", "GAT"]]
    
    for name in ml_models:
        if name in ["XGB", "RF", "GB"]:
            feat_imp[name] = np.mean(models[name].feature_importances_)
        elif name == "LGB":
            feat_imp[name] = np.mean(models[name].feature_importance(importance_type='gain'))
        else:
            feat_imp[name] = 1.0
    
    if graph_models and ml_models:
        ml_avg = np.mean([feat_imp[m] for m in ml_models])
        for name in graph_models:
            feat_imp[name] = ml_avg
    
    total = sum(feat_imp.values()) or len(feat_imp)
    return {k: v/total for k, v in feat_imp.items()}

# Calculate accuracy for high-activity compounds
def get_high_active_acc(preds, y_true):
    mask = y_true >= ACTIVITY_THRESHOLD
    acc = {}
    for name, pred in preds.items():
        if np.sum(mask) == 0:
            acc[name] = np.mean(np.abs(pred - y_true) <= HIGH_ACTIVE_TOLERANCE)
        else:
            acc[name] = np.mean(np.abs(pred[mask] - y_true[mask]) <= HIGH_ACTIVE_TOLERANCE)
    return acc

# Calculate dynamic weights for ensemble based on model performance and correlation
def calculate_weights(preds_train, y_train, preds_val, y_val, errors, variances, feat_imp, high_acc, model_names, is_test):
    n_samples = len(preds_val[model_names[0]])
    model_types = {"AFP": "graph", "GCN": "graph", "GAT": "graph", "XGB": "tree", "LGB": "tree", "GB": "tree", "RF": "tree", "SVM": "svm"}
    threshold = np.percentile(y_val, 90) if not is_test else ACTIVITY_THRESHOLD
    
    error_w = {n: 1/(errors.get(n, 1e-2)+1e-8) for n in model_names}
    var_w = {n: 1/(variances.get(n, 1e-2)+1e-8) for n in model_names}
    base_conf = {n: ((1-UNCERTAINTY_WEIGHT)*error_w[n] + UNCERTAINTY_WEIGHT*var_w[n]) * feat_imp.get(n, 1/len(model_names)) for n in model_names}
    base_conf = {n: v/sum(base_conf.values()) for n, v in base_conf.items()}
    
    corrs = {}
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            if n1 in preds_train and n2 in preds_train:
                corr = pearsonr(preds_train[n1], preds_train[n2])[0]
                corrs[(n1, n2)] = corrs[(n2, n1)] = 0.0 if np.isnan(corr) else corr
    
    weights = {n: np.zeros(n_samples) for n in model_names}
    for i in tqdm(range(n_samples), desc="Calculating dynamic weights"):
        pred_vals = np.array([preds_val[n][i] for n in model_names])
        mean = np.mean(pred_vals)
        dev = np.square(pred_vals - mean)
        std = np.std(pred_vals)
        
        sample_w = np.array([base_conf[n] for n in model_names]) * (1/(dev+1e-8))**CONFIDENCE_POWER
        
        for j, name in enumerate(model_names):
            penalty = sum(corrs.get((name, other), 0) * (1.2 if model_types[name] == model_types[other] else 0.8) 
                          for other in model_names if other != name)
            count = len(model_names) - 1
            if count > 0:
                sample_w[j] *= max(0.1, 1 - CORRELATION_PENALTY * penalty/count)
        
        is_high = (np.mean(pred_vals) >= threshold) or (y_val[i] >= ACTIVITY_THRESHOLD and not is_test)
        if is_high:
            sample_w *= np.array([high_acc[n] for n in model_names])
            power = 0.5 if std < 0.3 else 0.3
            sample_w = sample_w ** (CONFIDENCE_POWER * power)
            sample_w *= HIGH_ACTIVE_WEIGHT_BOOST
        
        sample_w = np.clip(sample_w, 1e-8, None)
        sample_w /= np.sum(sample_w)
        
        for idx, name in enumerate(model_names):
            weights[name][i] = sample_w[idx]
    
    return weights, threshold, model_types, base_conf

# Calculate final ensemble prediction with clipping
def ensemble_pred(preds, weights):
    model_names = list(preds.keys())
    if not model_names:
        raise ValueError("No base model predictions")
    ensemble = np.zeros_like(preds[model_names[0]])
    for name in model_names:
        ensemble += weights[name] * preds[name]
    return np.clip(ensemble, PIC50_MIN, PIC50_MAX)

# Calculate all evaluation metrics
def evaluate(y_true, y_pred):
    if len(y_true) == 0:
        return {"R2": 0.0, "MAE": 0.0, "RMSE": 0.0, "Spearman": 0.0, "Hit@top10%": 0.0, "EF@top10%": 0.0}
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman = spearmanr(y_true, y_pred)[0]
    spearman = 0.0 if np.isnan(spearman) else spearman
    
    top_n = max(int(round(len(y_true)*TOP_RATIO)), 1)
    top_idx = np.argsort(y_pred)[-top_n:]
    total_active = np.mean(y_true >= ACTIVITY_THRESHOLD)
    top_active = np.mean(y_true[top_idx] >= ACTIVITY_THRESHOLD)
    ef = top_active/total_active if total_active > 0 else 0.0
    
    return {"R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "Spearman": round(spearman, 4), "Hit@top10%": round(top_active, 4), "EF@top10%": round(ef, 4)}

# Run training mode with cross-validation
def run_train(args):
    print(f"\n===== Training ensemble model | split: {args.split} | models: {args.models} =====")
    paths = get_paths(args.split)
    check_paths(paths, "train", args.models)
    (X_ml_train, X_graph_train, y_train), (X_ml_val, X_graph_val, y_val) = load_data(paths, "train")
    models, label_scaler = load_models(paths, args.models)
    
    train_preds, errors, variances = get_predictions(X_ml_train, X_graph_train, models, args.models, label_scaler, True, y_train)
    val_preds, _, _ = get_predictions(X_ml_val, X_graph_val, models, args.models, label_scaler, False)
    feat_imp = get_feature_importance(X_ml_train, models, args.models)
    high_acc = get_high_active_acc(train_preds, y_train)
    weights, threshold, model_types, base_conf = calculate_weights(
        train_preds, y_train, val_preds, y_val, errors, variances, feat_imp, high_acc, args.models, False
    )
    
    ensemble_val_pred = ensemble_pred(val_preds, weights)
    base_metrics = {n: evaluate(y_val, val_preds[n]) for n in args.models}
    ensemble_metrics = evaluate(y_val, ensemble_val_pred)
    
    joblib.dump({
        "model_errors": errors, "model_variances": variances, "feature_weights": feat_imp,
        "high_active_acc": high_acc, "base_confidence": base_conf, "val_top_threshold": threshold,
        "model_types": model_types, "model_names": args.models,
        "params": {k: globals()[k] for k in ["CONFIDENCE_POWER", "HIGH_ACTIVE_WEIGHT_BOOST", "UNCERTAINTY_WEIGHT", 
                                             "CORRELATION_PENALTY", "ACTIVITY_THRESHOLD", "TOP_RATIO"]}
    }, paths["ensemble"])
    
    base_df = pd.DataFrame(base_metrics).T
    base_df["model_type"] = "base"
    base_df["avg_dynamic_weight"] = [round(np.mean(weights[n]), 4) for n in base_df.index]
    
    ensemble_df = pd.DataFrame([ensemble_metrics])
    ensemble_df["model_type"] = "ensemble"
    ensemble_df["avg_dynamic_weight"] = round(np.mean(list(weights.values())), 4)
    ensemble_df.index = ["DYNAMIC_ENSEMBLE"]
    
    all_df = pd.concat([base_df, ensemble_df])
    all_df["split_type"] = args.split
    all_df["mode"] = "train"
    all_df.to_csv(paths["metrics_summary"], index_label="model_name", encoding="utf-8")
    val_pred_dict = {'true_pIC50': y_val, 'ensemble_pred': ensemble_val_pred}
    for name in args.models:
        val_pred_dict[f'{name}_pred'] = val_preds[name]
    val_pred_df = pd.DataFrame(val_pred_dict).sort_values('true_pIC50').reset_index(drop=True)
    val_pred_df.to_csv(METRICS_DIR/ f"{args.split}_validation_predictions.csv", index=False)
    print(f"Saved validation predictions to {METRICS_DIR / f'{args.split}_validation_predictions.csv'}")
    print("\nBase model validation performance:")
    for n, m in base_metrics.items():
        print(f"{n}: {m}")
    print(f"\nEnsemble model validation performance: {ensemble_metrics}")
    print(f"\nAverage dynamic weights: {[f'{n}:{np.mean(weights[n]):.4f}' for n in args.models]}")
    print(f"\nEnsemble model saved to: {paths['ensemble']}")

# Run test mode on external test set
def run_test(args):
    print(f"\n===== Testing ensemble model | split: {args.split} | models: {args.models} =====")
    paths = get_paths(args.split)
    paths["external_ml_feat"] = args.external_ml_feat
    paths["external_graph_feat"] = args.external_graph_feat
    
    check_paths(paths, "test", args.models)
    config = joblib.load(paths["ensemble"])
    models, label_scaler = load_models(paths, args.models)
    
    assert set(args.models) == set(config["model_names"]), "Test models inconsistent with training"
    
    X_ml_test, X_graph_test, y_test = load_data(paths, "test")
    (X_ml_train, X_graph_train, y_train), _ = load_data(paths, "train")
    
    train_preds, _, _ = get_predictions(X_ml_train, X_graph_train, models, args.models, label_scaler, False)
    test_preds, _, _ = get_predictions(X_ml_test, X_graph_test, models, args.models, label_scaler, False)
    
    weights, _, _, _ = calculate_weights(
        train_preds, y_train, test_preds, y_test, config["model_errors"], config["model_variances"],
        config["feature_weights"], config["high_active_acc"], args.models, True
    )
    
    ensemble_test_pred = ensemble_pred(test_preds, weights)
    base_metrics = {n: evaluate(y_test, test_preds[n]) for n in args.models}
    ensemble_metrics = evaluate(y_test, ensemble_test_pred)
    
    base_df = pd.DataFrame(base_metrics).T
    base_df["model_type"] = "base"
    base_df["avg_dynamic_weight"] = [round(np.mean(weights[n]), 4) for n in base_df.index]
    
    ensemble_df = pd.DataFrame([ensemble_metrics])
    ensemble_df["model_type"] = "ensemble"
    ensemble_df["avg_dynamic_weight"] = round(np.mean(list(weights.values())), 4)
    ensemble_df.index = ["DYNAMIC_ENSEMBLE"]
    
    all_df = pd.concat([base_df, ensemble_df])
    all_df["split_type"] = args.split
    all_df["mode"] = "test"
    
    header = not paths["metrics_summary"].exists()
    all_df.to_csv(paths["metrics_summary"], mode="a", index_label="model_name", header=header, encoding="utf-8")
    test_pred_dict = {'true_pIC50': y_test, 'ensemble_pred': ensemble_test_pred}
    for name in args.models:
        test_pred_dict[f'{name}_pred'] = test_preds[name]
    test_pred_df = pd.DataFrame(test_pred_dict).sort_values('true_pIC50').reset_index(drop=True)
    test_pred_df.to_csv(METRICS_DIR/ f"{args.split}_external_test_predictions.csv", index=False)
    print(f"Saved external test predictions to {METRICS_DIR / f'{args.split}_external_test_predictions.csv'}")
    print("\nBase model test performance:")
    for n, m in base_metrics.items():
        print(f"{n}: {m}")
    print(f"\nEnsemble model test performance: {ensemble_metrics}")
    print(f"\nAverage dynamic weights: {[f'{n}:{np.mean(weights[n]):.4f}' for n in args.models]}")
    print(f"\nTest metrics appended to: {paths['metrics_summary']}")

if __name__ == "__main__":
    try:
        args = parse_args()
        run_train(args) if args.mode == "train" else run_test(args)
        print("\nExecution completed!")
    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)