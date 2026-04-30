import os, torch, joblib, dgl, numpy as np, pandas as pd, argparse
from pathlib import Path
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import importlib

graph_train_module = importlib.import_module("10_graph_modeltrain")
simpleGCN, EnhancedGAT, EnhancedAFP = graph_train_module.simpleGCN, graph_train_module.EnhancedGAT, graph_train_module.EnhancedAFP

ROOT_DIR = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT_DIR / "data" / "split" / "external_split" / "feature"
SCALER_DIR = ROOT_DIR / "data" / "split"
SUMMARY_DIR = ROOT_DIR / "results" / "summary" / "external_test"
MODEL_ROOT = ROOT_DIR / "results"
THRESHOLD, TOP_PERCENT, BATCH_SIZE = 8.0, 0.1, 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_SPLITS = ["fingerprint", "random", "scaffold"]
BASE_MODEL_TYPES = ["ml", "graph", "all"]
ML_MODELS = ["SVM", "XGB", "RF", "GB", "LGB"]
GRAPH_MODELS = ["GCN", "GAT", "AFP"]
PIC50_MIN, PIC50_MAX = 3.0, 11.0
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
for split in BASE_SPLITS:
    (SUMMARY_DIR / split).mkdir(parents=True, exist_ok=True)
    (MODEL_ROOT / f"{split}_split" / "models").mkdir(parents=True, exist_ok=True)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Model external test set evaluation tool")
    parser.add_argument("--split", required=True, choices=BASE_SPLITS + ["all"])
    parser.add_argument("--mode", required=True, choices=BASE_MODEL_TYPES)
    args = parser.parse_args()
    return (BASE_SPLITS if args.split == "all" else [args.split]), args.mode

# Calculate hit rate and enrichment factor metrics
def calculate_metrics(y_true, y_pred, threshold=THRESHOLD, top_percent=TOP_PERCENT):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    n_total = len(y_true)
    total_active = np.mean(y_true >= threshold)
    top_n = max(1, int(round(n_total * top_percent)))
    top_indices = np.argsort(y_pred)[-top_n:]
    top_active = np.mean(y_true[top_indices] >= threshold)
    return top_active, top_active / total_active if total_active > 0 else 0.0

# ML model testing class
class MLTest:
    def __init__(self, split):
        self.split = split
        self.test_data_path = FEAT_DIR / f"{split}_1200_external.pkl"
        self.model_path_root = MODEL_ROOT / f"{split}_split" / "models"
        self.result_root = SUMMARY_DIR / split
        self.metrics_summary_path = self.result_root / f"{split}_ml_metrics_summary.csv"

    # Load test data from pickle file
    def load_test_data(self):
        test_data = joblib.load(self.test_data_path)
        X_test, y_test = test_data['X'], test_data['y'].ravel()
        smiles_list = test_data['SMILES']
        print(f"\n[ML-{self.split}] Test set loaded: {X_test.shape}, {y_test.shape}")
        return X_test, y_test, smiles_list

    # Evaluate single ML model and return metrics
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        r2, mae, rmse = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))
        try:
            spearman, _ = spearmanr(y_test, y_pred)
        except:
            spearman = np.nan
        hit_rate, ef = calculate_metrics(y_test, y_pred)
        pred_max, pred_min, pred_mean = np.max(y_pred), np.min(y_pred), np.mean(y_pred)
        
        print(f"\n{model_name} external test set metrics:")
        print(f"  R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | Spearman: {spearman:.4f}" if not np.isnan(spearman) else f"  R2: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | Spearman: N/A")
        print(f"  Hit@{int(TOP_PERCENT*100)}%: {hit_rate:.2%}" if not np.isnan(hit_rate) else f"  Hit@{int(TOP_PERCENT*100)}%: N/A")
        print(f"  EF@{int(TOP_PERCENT*100)}%: {ef:.4f}" if not np.isnan(ef) else f"  EF@{int(TOP_PERCENT*100)}%: N/A")
        print(f"  Prediction range: max={pred_max:.4f} | min={pred_min:.4f} | mean={pred_mean:.4f}")
        
        return y_pred, r2, mae, rmse, spearman, hit_rate, ef

    # Save detailed prediction results to CSV
    def save_single_result(self, y_test, y_pred, smiles_list, save_path, model_name):
        try:
            spearman, _ = spearmanr(y_test, y_pred)
        except:
            spearman = np.nan
        pd.DataFrame({
            "SMILES": smiles_list, "true_value": y_test, "predicted_value": y_pred, "error": y_pred - y_test,
            f"true_activity(>={THRESHOLD})": (y_test >= THRESHOLD).astype(int),
            f"predicted_activity(>={THRESHOLD})": (y_pred >= THRESHOLD).astype(int),
            "global_spearman": spearman
        }).to_csv(save_path, index=False)
        print(f"{model_name} predictions saved to: {save_path}")

    # Run evaluation for all ML models
    def run_ml_test(self):
        X_test, y_test, smiles_list = self.load_test_data()
        metrics_summary = []
        for model_name in ML_MODELS:
            model_path = self.model_path_root / f"{self.split}_{model_name.lower()}_model.pkl"
            result_path = self.result_root / f"ml_{model_name.lower()}_results.csv"
            if not model_path.exists():
                print(f"Warning: {model_name} model file not found, skipping")
                continue
            try:
                model = joblib.load(model_path)
                print(f"{model_name} model loaded successfully")
                y_pred, r2, mae, rmse, spearman, hit_rate, ef = self.evaluate_model(model, X_test, y_test, model_name)
                self.save_single_result(y_test, y_pred, smiles_list, result_path, model_name)
                metrics_summary.append({
                    "model_name": model_name, "R2": round(r2, 4), "MAE": round(mae, 4), "RMSE": round(rmse, 4),
                    "Spearman": round(spearman, 4) if not np.isnan(spearman) else np.nan,
                    f"Hit@10%": round(hit_rate, 4), f"EF@10%": round(ef, 4)
                })
            except Exception as e:
                print(f"Warning: {model_name} model evaluation failed: {e}")
                continue
        if metrics_summary:
            pd.DataFrame(metrics_summary).to_csv(self.metrics_summary_path, index=False)
            print(f"\n[ML-{self.split}] Metrics summary saved to: {self.metrics_summary_path}")

# Graph neural network testing class
class GraphTest:
    def __init__(self, split):
        self.split = split
        self.test_data_path = FEAT_DIR / f"external_test_graphdata.pkl"
        self.model_path_root = MODEL_ROOT / f"{split}_split" / "models"
        self.result_root = SUMMARY_DIR / split
        self.metrics_summary_path = self.result_root / f"{split}_graphmodels_metrics_summary.csv"
        self.label_scaler = self._load_label_scaler()

    # Load MinMaxScaler for label denormalization
    def _load_label_scaler(self):
        scaler_path =  SCALER_DIR / f"{split}_split" / "feature" / f"graph_label_minmax_{split}.pkl"
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                scaler.data_min_ = np.array([PIC50_MIN])
                scaler.data_max_ = np.array([PIC50_MAX])
                scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
                scaler.min_ = -scaler.data_min_ * scaler.scale_
                print(f"\nSuccessfully loaded and calibrated {self.split} label scaler: {scaler_path}")
                print(f"Scaler calibrated range: pIC50 [{PIC50_MIN}, {PIC50_MAX}]")
                return scaler
            except Exception as e:
                print(f"Failed to load/calibrate scaler: {e}, will evaluate with normalized values")
                return None

    # Load graph test data with DataLoader
    def load_test_data(self):
        test_data = joblib.load(self.test_data_path)
        test_g = [item['graph'] for item in test_data]
        test_y_original = np.array([item['label'] for item in test_data], dtype=np.float32).reshape(-1, 1)
        test_y = torch.tensor(test_y_original, dtype=torch.float32).view(-1, 1)
        smiles_list = [item['smiles'] for item in test_data]
        num_workers = os.cpu_count() - 1 if os.name != 'nt' else 0
        test_loader = GraphDataLoader(list(zip(test_g, test_y)), batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=num_workers, pin_memory=True, drop_last=False)
        print(f"\n[Graph-{self.split}] Test set loaded: {len(test_g)} samples")
        return test_loader, test_y_original, smiles_list

    # Test single graph model and return metrics
    def test_single_model(self, model_name, model_path, result_path, test_loader, test_y_original, smiles_list):
        try:
            model = torch.load(model_path, map_location=DEVICE, weights_only=False)
            model.to(DEVICE).eval()
            print(f"\n{model_name} model loaded successfully")
        except Exception as e:
            print(f"\n{model_name} model loading failed: {e}")
            return None
        
        preds_scaled, labs = [], []
        with torch.no_grad():
            for g, y in test_loader:
                g, y = g.to(DEVICE), y.to(DEVICE)
                pred = model(g).cpu()
                preds_scaled.append(pred)
                labs.append(y.cpu())
        
        preds_scaled = torch.cat(preds_scaled).numpy()
        labs_original = test_y_original
        
        if self.label_scaler is not None:
            preds_original = self.label_scaler.inverse_transform(preds_scaled)
            preds_original = np.clip(preds_original, PIC50_MIN, PIC50_MAX)
            print(f"\n{model_name} predictions denormalized and clipped:")
            print(f"  Normalized predictions range: [{preds_scaled.min():.4f}, {preds_scaled.max():.4f}]")
            print(f"  Original scale predictions range (clipped): [{preds_original.min():.4f}, {preds_original.max():.4f}]")
        else:
            preds_original = np.clip(preds_scaled, PIC50_MIN, PIC50_MAX)
        
        rmse = np.sqrt(mean_squared_error(labs_original, preds_original))
        mae = mean_absolute_error(labs_original, preds_original)
        r2 = r2_score(labs_original, preds_original)
        try:
            spearman, _ = spearmanr(labs_original.flatten(), preds_original.flatten())
        except:
            spearman = np.nan
        hit_rate, ef = calculate_metrics(labs_original, preds_original)
        pred_max, pred_min, pred_mean = np.max(preds_original), np.min(preds_original), np.mean(preds_original)
        
        print(f"{model_name} test results (original pIC50 scale):")
        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f} | Spearman: {spearman:.3f}" if not np.isnan(spearman) else f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f} | Spearman: N/A")
        print(f"Hit@{int(TOP_PERCENT*100)}%: {hit_rate:.2%}" if not np.isnan(hit_rate) else f"  Hit@{int(TOP_PERCENT*100)}%: N/A")
        print(f"EF@{int(TOP_PERCENT*100)}%: {ef:.2f}" if not np.isnan(ef) else f"  EF@{int(TOP_PERCENT*100)}%: N/A")
        print(f"Prediction range (original pIC50): max={pred_max:.4f} | min={pred_min:.4f} | mean={pred_mean:.4f}")
        
        pd.DataFrame({
            "SMILES": smiles_list,
            "true_value_original_pIC50": labs_original.flatten(),
            "predicted_value_original_pIC50": preds_original.flatten(),
            "predicted_value_normalized": preds_scaled.flatten(),
            f"true_activity(>={THRESHOLD})": (labs_original.flatten() >= THRESHOLD).astype(int),
            f"predicted_activity(>={THRESHOLD})": (preds_original.flatten() >= THRESHOLD).astype(int),
            "global_spearman": spearman
        }).to_csv(result_path, index=False)
        print(f"{model_name} predictions saved to: {result_path}")
        
        return {
            "model_name": model_name, "R2": round(r2, 3), "MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "Spearman": round(spearman, 3) if not np.isnan(spearman) else np.nan,
            f"Hit@{int(TOP_PERCENT*100)}%": round(hit_rate, 4), f"EF@{int(TOP_PERCENT*100)}%": round(ef, 4)
        }

    # Run evaluation for all graph models
    def run_graph_test(self):
        test_loader, test_y_original, test_smiles = self.load_test_data()
        metrics_summary = []
        for model_name in GRAPH_MODELS:
            model_path = self.model_path_root / f"{model_name.lower()}_{self.split}.pt"
            result_path = self.result_root / f"graph_{model_name.lower()}_results.csv"
            if not model_path.exists():
                print(f"Warning: {model_name} model file not found, skipping")
                continue
            model_metrics = self.test_single_model(model_name, model_path, result_path, test_loader, test_y_original, test_smiles)
            if model_metrics:
                metrics_summary.append(model_metrics)
        if metrics_summary:
            pd.DataFrame(metrics_summary).to_csv(self.metrics_summary_path, index=False)
            print(f"\n[Graph-{self.split}] Metrics summary saved to: {self.metrics_summary_path}")

# Main entry point for external test evaluation
def main():
    split_list, mode = parse_args()
    print(f"Starting external test set evaluation - splits: {split_list} | mode: {mode} | device: {DEVICE}")
    for split in split_list:
        print(f"\n{'='*60}\nProcessing [{split}] split\n{'='*60}")
        if mode in ["ml", "all"]:
            MLTest(split).run_ml_test()
        if mode in ["graph", "all"]:
            GraphTest(split).run_graph_test()
    print(f"\nAll test tasks completed! Results saved to -> {SUMMARY_DIR}")

if __name__ == "__main__":
    main()