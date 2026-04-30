import os, time, optuna, numpy as np, pandas as pd, joblib, lightgbm as lgb, xgboost as xgb
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.svm import NuSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results"
SPLIT_BASE_DIR = ROOT_DIR / "split"
BASE_SPLITS = ["fingerprint", "random", "scaffold"]
BASE_MODELS = ["lgb", "svm", "gb", "rf", "xgb"]
SUMMARY_RESULTS = []
ACTIVITY_THRESHOLD, TOP_RATIO = 8.0, 0.1
TRIALS_MAP = {"xgb": 150, "lgb": 150, "svm": 150, "gb": 150, "rf": 150}
RANDOM_SEED = 42

# Calculate hit rate and enrichment factor for top predictions
def calculate_hit_and_ef(y_true, y_pred, threshold=ACTIVITY_THRESHOLD, top_ratio=TOP_RATIO):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    n_total = len(y_true)
    total_active = np.mean(y_true >= threshold)
    top_n = max(1, int(round(n_total * top_ratio)))
    top_indices = np.argsort(y_pred)[-top_n:]
    top_active = np.mean(y_true[top_indices] >= threshold)
    return top_active, top_active / total_active if total_active > 0 else 0.0

# Calculate comprehensive regression metrics including Spearman correlation
def calculate_metrics(y_true, y_pred):
    spearman_corr, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Spearman": spearman_corr,
        "HitRate": calculate_hit_and_ef(y_true, y_pred)[0],
        "EF": calculate_hit_and_ef(y_true, y_pred)[1]
    }

# Load precomputed features for train and validation sets
def load_data(split):
    feat_dir = SPLIT_BASE_DIR / f"{split}_split" / "feature"
    train_path = feat_dir / f"{split}_1200_train.pkl"
    val_path = feat_dir / f"{split}_1200_val.pkl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"{split} split missing train/val feature files")
    data_train, data_val = joblib.load(train_path), joblib.load(val_path)
    X_train, y_train = data_train['X'], data_train['y'].ravel()
    X_val, y_val = data_val['X'], data_val['y'].ravel()
    print(f"{split} split - Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, y_train, X_val, y_val

# Create output directories for models and results
def create_dirs(split):
    model_dir = RESULT_DIR / f"{split}_split" / "models"
    result_dir = RESULT_DIR / "summary" / "ml"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, result_dir

# Save cross-validation fold results to CSV
def save_fold_result(fold_metrics, save_path):
    pd.DataFrame(fold_metrics).to_csv(save_path, index=False, encoding="utf-8")

# Record final model performance summary
def record_summary(split, model_name, final_metrics, model_path, fold_result_path):
    SUMMARY_RESULTS.append({
        "SplitType": split, "Model": model_name.upper(),
        "R2": round(final_metrics["R2"], 4), "MAE": round(final_metrics["MAE"], 4),
        "RMSE": round(final_metrics["RMSE"], 4), "Spearman": round(final_metrics["Spearman"], 4),
        "HitRate@10%": round(final_metrics["HitRate"], 4), "EF": round(final_metrics["EF"], 4),
        "ModelPath": str(model_path), "FoldResultPath": str(fold_result_path)
    })

# Save aggregated results from all experiments
def save_global_summary():
    if SUMMARY_RESULTS:
        summary_path = RESULT_DIR / "summary" / "ml" / f"all_model_results_summary_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(SUMMARY_RESULTS).to_csv(summary_path, index=False, encoding="utf-8")
        print(f"Global summary saved to: {summary_path}\n")

# Run Optuna hyperparameter optimization
def optuna_search(objective, trial_num):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=trial_num, show_progress_bar=True)
    return study.best_params.copy()

# Validate best parameters using 5-fold cross-validation
def best_param_cv_validate(X_train, y_train, fold_result_path, model_train_func):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_vl = X_train[train_idx], X_train[val_idx]
        y_tr, y_vl = y_train[train_idx], y_train[val_idx]
        model, y_pred = model_train_func(X_tr, y_tr, X_vl, y_vl)
        metrics = calculate_metrics(y_vl, y_pred)
        metrics["Fold"] = fold_idx
        fold_metrics.append(metrics)
    save_fold_result(fold_metrics, fold_result_path)
    print(f"5-fold CV results saved to: {fold_result_path}")

# Print formatted model performance metrics
def print_model_metrics(model_name, split, final_metrics):
    print("=" * 60)
    print(f"{model_name.upper()} Model - {split} Split Final Validation Metrics:")
    print(f"  R2: {final_metrics['R2']:.4f}, Spearman: {final_metrics['Spearman']:.4f}")
    print(f"  MAE: {final_metrics['MAE']:.4f}, RMSE: {final_metrics['RMSE']:.4f}")
    print(f"  Hit@top10%: {final_metrics['HitRate']:.4%}, EF: {final_metrics['EF']:.4f}")
    print("=" * 60)

# Generic training pipeline with hyperparameter search and validation
def train_model(split, model_name, model_class, param_space, train_func):
    X_train, y_train, X_val, y_val = load_data(split)
    model_dir, result_dir = create_dirs(split)
    param_path = model_dir / f"{split}_{model_name}_params.pkl"
    model_path = model_dir / f"{split}_{model_name}_model.pkl"
    fold_result_path = result_dir / f"{split}_{model_name}_fold_results.csv"
    def objective(trial):
        params = param_space(trial)
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        fold_rmse = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]
            model = model_class(**{k: v for k, v in params.items() if k != "early_stopping_rounds"})
            if "early_stopping_rounds" in params:
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
            else:
                model.fit(X_tr, y_tr)
            y_pred = model.predict(X_vl)
            fold_rmse.append(np.sqrt(mean_squared_error(y_vl, y_pred)))
        return np.mean(fold_rmse)
    if os.path.exists(param_path):
        print(f"Found {model_name.upper()} best params, loading: {param_path}")
        best_params = joblib.load(param_path)
    else:
        print(f"Starting {model_name.upper()} Optuna hyperparameter search...")
        best_params = optuna_search(objective, TRIALS_MAP[model_name])
        joblib.dump(best_params, param_path)
        print(f"{model_name.upper()} best params saved to: {param_path}")
    best_param_cv_validate(X_train, y_train, fold_result_path,
                          lambda X_tr, y_tr, X_vl, y_vl: train_func(best_params, X_tr, y_tr, X_vl, y_vl))
    best_model, y_val_pred = train_func(best_params, X_train, y_train, X_val, y_val)
    final_metrics = calculate_metrics(y_val, y_val_pred)
    print_model_metrics(model_name, split, final_metrics)
    joblib.dump(best_model, model_path)
    print(f"{model_name.upper()} model saved to: {model_path}")
    record_summary(split, model_name, final_metrics, model_path, fold_result_path)

# Train LightGBM model with custom parameter space
def train_lgb(split):
    def param_space(trial):
        return {
            "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
            "random_state": RANDOM_SEED, "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=10),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 5)
        }
    def train_func(params, X_tr, y_tr, X_vl, y_vl):
        lgb_train, lgb_val = lgb.Dataset(X_tr, label=y_tr), lgb.Dataset(X_vl, label=y_vl, reference=lgb.Dataset(X_tr, label=y_tr))
        model = lgb.train({**params, "verbosity": -1}, lgb_train, valid_sets=[lgb_val])
        return model, model.predict(X_vl)
    train_model(split, "lgb", lgb.LGBMRegressor, param_space, train_func)

# Train Support Vector Machine model
def train_svm(split):
    def param_space(trial):
        params = {
            "kernel": "rbf",
            "nu": trial.suggest_float("nu", 0.2, 0.8), 
            "C": trial.suggest_float("C", 0.01, 100, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "tol": trial.suggest_float("tol", 1e-5, 1e-3, log=True),
        }
        return params
    def train_func(params, X_tr, y_tr, X_vl, y_vl):
        model = NuSVR(**params)
        model.fit(X_tr, y_tr)
        return model, model.predict(X_vl)
    train_model(split, "svm", NuSVR, param_space, train_func)

# Train Gradient Boosting model
def train_gb(split):
    def param_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7, 0.9]),
            "random_state": RANDOM_SEED, "verbose": 0
        }
    def train_func(params, X_tr, y_tr, X_vl, y_vl):
        model = GradientBoostingRegressor(**params, validation_fraction=0.2, n_iter_no_change=50)
        model.fit(X_tr, y_tr)
        return model, model.predict(X_vl)
    train_model(split, "gb", GradientBoostingRegressor, param_space, train_func)

# Train Random Forest model
def train_rf(split):
    def param_space(trial):
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.7, 0.9]),
            "bootstrap": bootstrap,
            "oob_score": trial.suggest_categorical("oob_score", [False, True]) if bootstrap else False,
            "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": 0
        }
    def train_func(params, X_tr, y_tr, X_vl, y_vl):
        if not params.get("bootstrap", False):
            params["oob_score"] = False
        model = RandomForestRegressor(**params)
        model.fit(X_tr, y_tr)
        return model, model.predict(X_vl)
    train_model(split, "rf", RandomForestRegressor, param_space, train_func)

# Train XGBoost model
def train_xgb(split):
    def param_space(trial):
        return {
            "objective": "reg:squarederror", "eval_metric": "rmse", "booster": "gbtree",
            "random_state": RANDOM_SEED, "max_depth": trial.suggest_int("max_depth", 2, 14),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "early_stopping_rounds": 80
        }
    def train_func(params, X_tr, y_tr, X_vl, y_vl):
        model = xgb.XGBRegressor(**{k: v for k, v in params.items() if k != "early_stopping_rounds"})
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        return model, model.predict(X_vl)
    train_model(split, "xgb", xgb.XGBRegressor, param_space, train_func)

MODEL_TRAIN_MAP = {"lgb": train_lgb, "svm": train_svm, "gb": train_gb, "rf": train_rf, "xgb": train_xgb}

# Execute training pipeline for specified splits and models
def run_training(split_list, model_list):
    global SUMMARY_RESULTS
    SUMMARY_RESULTS = []
    for split in split_list:
        print("=" * 50 + f"\nProcessing [{split}] split\n" + "=" * 50)
        for model in model_list:
            print(f"Training [{model}] model")
            try:
                MODEL_TRAIN_MAP[model](split)
            except Exception as e:
                print(f"[{split}-{model}] training failed: {type(e).__name__} - {str(e)}\n")
    save_global_summary()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-model batch training tool (LGB/SVM/GB/RF/XGB)")
    parser.add_argument("--split", required=True, choices=BASE_SPLITS + ["all"])
    parser.add_argument("--models", required=True, choices=BASE_MODELS + ["all"])
    args = parser.parse_args()
    run_training(BASE_SPLITS if args.split == "all" else [args.split],
                BASE_MODELS if args.models == "all" else [args.models])

if __name__ == "__main__":
    main()