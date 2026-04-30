import os, time
os.environ["DGLDETERMINISTIC"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import sys, torch, torch.nn as nn, numpy as np, pandas as pd, random, dgl, joblib, argparse
from pathlib import Path
from dgl.dataloading import GraphDataLoader
from dgllife.model import AttentiveFPPredictor, GATPredictor, GCNPredictor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "data"
RESULT_DIR = SCRIPT_DIR.parent / "results"
METRICS_DIR = RESULT_DIR / "summary" / "graph"
SCALER_DIR = ROOT_DIR / "split"
SPLIT_BASE_DIR = ROOT_DIR / "split"
BASE_SPLITS = ["fingerprint", "random", "scaffold"]
BASE_MODELS = ["afp", "gat", "gcn"]
for d in [RESULT_DIR, METRICS_DIR, SCALER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PIC50_MIN, PIC50_MAX = 3.0, 11.0
HIDDEN, DEPTH, DROPOUT, ATTENTION_HEADS, NUM_HEADS = 128, 3, 0.3, 6, 8
LR, BATCH_SIZE, EPOCHS, PATIENCE, WD, GRAD_CLIP = 1e-3, 64, 1500, 100, 1e-4, 3.0
ACTIVITY_THRESHOLD, TOP_RATIO = 8.0, 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SMILES_COL, LABEL_COL = 'SMILES', 'pIC50'
SUMMARY_RESULTS = []
torch.set_num_threads(1)
GLOBAL_SCALER = None

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# Save epoch metrics to CSV
def save_epoch_result(epoch_metrics, save_path):
    pd.DataFrame(epoch_metrics).to_csv(save_path, index=False, encoding="utf-8")

# Record final model performance summary
def record_summary(split, model_name, final_metrics, model_path, epoch_result_path):
    SUMMARY_RESULTS.append({
        "SplitType": split, "Model": model_name.upper(),
        "R2": round(final_metrics["R2"], 4), "MAE": round(final_metrics["MAE"], 4),
        "RMSE": round(final_metrics["RMSE"], 4), "Spearman": round(final_metrics["Spearman"], 4),
        "HitRate@10%": round(final_metrics["HitRate"], 4), "EF": round(final_metrics["EF"], 4),
        "ModelPath": str(model_path), "EpochResultPath": str(epoch_result_path)
    })

# Save global summary of all trained models
def save_global_summary(save_path):
    if SUMMARY_RESULTS:
        pd.DataFrame(SUMMARY_RESULTS).to_csv(save_path, index=False, encoding="utf-8")

# Load preprocessed graph data with fixed-range normalization
def load_preprocessed_data(split):
    global GLOBAL_SCALER
    feat_dir = SPLIT_BASE_DIR / f"{split}_split" / "feature"
    train_path = feat_dir / f"{split}_train_graphdata.pkl"
    val_path = feat_dir / f"{split}_val_graphdata.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training features not found: {train_path}")
    
    train_data = joblib.load(train_path)
    train_g = [item['graph'] for item in train_data]
    train_y_original = np.array([item['label'] for item in train_data], dtype=np.float32).reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.data_min_ = np.array([PIC50_MIN])
    scaler.data_max_ = np.array([PIC50_MAX])
    scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    
    train_y_scaled = np.clip(scaler.transform(train_y_original), 0.0, 1.0)
    scaler_save_path = SCALER_DIR / f"{split}_split" / "feature" / f"graph_label_minmax_{split}.pkl"
    joblib.dump(scaler, scaler_save_path)
    print(f"\nSaved {split} label scaler to: {scaler_save_path}")
    GLOBAL_SCALER = scaler
    train_y = torch.tensor(train_y_scaled, dtype=torch.float32).view(-1, 1)
    
    if not val_path.exists():
        raise FileNotFoundError(f"Validation features not found: {val_path}")
    val_data = joblib.load(val_path)
    val_g = [item['graph'] for item in val_data]
    val_y_original = np.array([item['label'] for item in val_data], dtype=np.float32).reshape(-1, 1)
    val_y_scaled = np.clip(scaler.transform(val_y_original), 0.0, 1.0)
    val_y = torch.tensor(val_y_scaled, dtype=torch.float32).view(-1, 1)
    
    print(f"\n[{split}] Loaded - Train: {len(train_g)} samples | Val: {len(val_g)} samples")
    print(f"Label normalization: fixed pIC50 range [{PIC50_MIN}, {PIC50_MAX}]")
    print(f"Train label range: original [{train_y_original.min():.2f}, {train_y_original.max():.2f}] -> normalized [{train_y_scaled.min():.2f}, {train_y_scaled.max():.2f}]")
    for i in range(min(3, len(train_data))):
        print(f"Sample {i+1}: {train_data[i]['smiles']} | original pIC50: {train_y_original[i][0]:.2f} | normalized: {train_y_scaled[i][0]:.2f}")
    return train_g, train_y, val_g, val_y, train_data, val_data

# Create DataLoader with deterministic seeding
def create_dataloader(graphs, labels, batch_size, shuffle, seed=42):
    if len(graphs) == 0:
        raise ValueError("Empty graph data")
    if len(graphs) != len(labels):
        raise ValueError(f"Graph count ({len(graphs)}) and label count ({len(labels)}) mismatch")
    gen = torch.Generator()
    gen.manual_seed(seed)
    def worker_init_fn(worker_id):
        set_seed(seed + worker_id)
    return GraphDataLoader(list(zip(graphs, labels)), batch_size=batch_size, shuffle=shuffle,
                           num_workers=0, pin_memory=True, drop_last=False, persistent_workers=False,
                           worker_init_fn=worker_init_fn, generator=gen)

# Calculate hit rate and enrichment factor
def calculate_hit_and_ef(y_true, y_pred, threshold=ACTIVITY_THRESHOLD, top_ratio=TOP_RATIO):
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    n_total = len(y_true)
    total_active = np.mean(y_true >= threshold)
    top_n = max(1, int(round(n_total * top_ratio)))
    top_indices = np.argsort(y_pred)[-top_n:]
    top_active = np.mean(y_true[top_indices] >= threshold)
    hit_rate = top_active
    ef = hit_rate / total_active if total_active > 0 else 0.0
    return hit_rate, ef

# Evaluate model and return metrics on original scale
def evaluate_base(loader, model, device):
    global GLOBAL_SCALER
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for g, y in loader:
            g, y = g.to(device), y.to(device)
            with autocast(enabled=device.type == 'cuda'):
                pred = model(g)
            preds.append(pred.cpu())
            labs.append(y.cpu())
    preds_scaled = torch.cat(preds).numpy()
    labs_scaled = torch.cat(labs).numpy()
    
    if GLOBAL_SCALER is not None:
        preds_original = np.clip(GLOBAL_SCALER.inverse_transform(preds_scaled), PIC50_MIN, PIC50_MAX)
        labs_original = np.clip(GLOBAL_SCALER.inverse_transform(labs_scaled), PIC50_MIN, PIC50_MAX)
    else:
        preds_original, labs_original = preds_scaled, labs_scaled
    
    rmse = np.sqrt(mean_squared_error(labs_original, preds_original))
    mae = mean_absolute_error(labs_original, preds_original)
    r2 = r2_score(labs_original, preds_original)
    spearman, _ = spearmanr(labs_original.flatten(), preds_original.flatten())
    hit_rate, ef = calculate_hit_and_ef(labs_original, preds_original)
    return rmse, mae, r2, spearman, hit_rate, ef

# AttentiveFP model with Sigmoid output constraint
class EnhancedAFP(nn.Module):
    def __init__(self, train_g, hidden=HIDDEN, depth=DEPTH, dropout=DROPOUT):
        super().__init__()
        if not train_g:
            raise RuntimeError("train_g not loaded")
        node_in_feat = train_g[0].ndata['h'].shape[1]
        edge_in_feat = train_g[0].edata['e'].shape[1]
        self.afp = AttentiveFPPredictor(
            node_feat_size=node_in_feat, edge_feat_size=edge_in_feat,
            num_layers=depth, graph_feat_size=hidden, dropout=dropout,
            n_tasks=1, num_timesteps=ATTENTION_HEADS
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g):
        return self.sigmoid(self.afp(g, g.ndata['h'].float(), g.edata['e'].float()))

# GAT model with Sigmoid output constraint
class EnhancedGAT(nn.Module):
    def __init__(self, train_g, hidden=HIDDEN, depth=DEPTH, dropout=DROPOUT):
        super().__init__()
        if not train_g:
            raise RuntimeError("train_g not loaded")
        node_in_feat = train_g[0].ndata['h'].shape[1]
        self.gat = GATPredictor(
            in_feats=node_in_feat, hidden_feats=[hidden] * depth,
            num_heads=[NUM_HEADS] * depth, feat_drops=[dropout] * depth,
            activations=[nn.ReLU()] * depth, classifier_hidden_feats=hidden, n_tasks=1
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g):
        return self.sigmoid(self.gat(g, g.ndata['h'].float()))

# GCN model with Sigmoid output constraint
class simpleGCN(nn.Module):
    def __init__(self, train_g, hidden=HIDDEN, depth=DEPTH, dropout=DROPOUT):
        super().__init__()
        if not train_g:
            raise RuntimeError("train_g not loaded")
        node_in_feat = train_g[0].ndata['h'].shape[1]
        self.gcn = GCNPredictor(
            in_feats=node_in_feat, hidden_feats=[hidden] * depth,
            dropout=[dropout] * depth, activation=[nn.ReLU()] * depth,
            classifier_hidden_feats=hidden, n_tasks=1
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, g):
        return self.sigmoid(self.gcn(g, g.ndata['h'].float()))

# Main training loop with early stopping
def train_model(split, model_name, model_class):
    global GLOBAL_SCALER
    set_seed(42)
    epoch_metrics = []
    model_save_path = RESULT_DIR / f"{split}_split" / "models" / f"{model_name}_{split}.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_result_path = METRICS_DIR / f"{model_name}_{split}_epochs.csv"
    
    train_g, train_y, val_g, val_y, _, _ = load_preprocessed_data(split)
    train_loader = create_dataloader(train_g, train_y, BATCH_SIZE, shuffle=True, seed=42)
    val_loader = create_dataloader(val_g, val_y, BATCH_SIZE, shuffle=False, seed=42)
    
    model = model_class(train_g).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=80, T_mult=2, eta_min=1e-6)
    criterion = nn.MSELoss()
    scaler = GradScaler() if DEVICE.type == 'cuda' else None
    
    best_val_rmse, best_val_r2, best_val_ef, patience_counter = float('inf'), -float('inf'), -float('inf'), 0
    header = ('|Epoch|Train Loss|Train RMSE|Train MAE|Train R2|Train Spearman|Train Hit@top10%|Train EF|'
              'Val RMSE|Val MAE| Val R2 |Val Spearman|Val Hit@top10%|Val EF| LR |')
    print(f"\n{'=' * 20} Training {model_name.upper()} model - {split} split {'=' * 20}")
    print(f"Model constraint: Sigmoid output -> normalized [0,1] -> denormalized pIC50 [{PIC50_MIN}, {PIC50_MAX}]")
    print('-' * len(header) + '\n' + header + '\n' + '-' * len(header))
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for g, y in train_loader:
            g, y = g.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=scaler is not None):
                pred = model(g)
                loss = criterion(pred, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            train_losses.append(loss.item())
            scheduler.step()
        
        avg_train_loss = np.mean(train_losses)
        train_rmse, train_mae, train_r2, train_spearman, train_hit, train_ef = evaluate_base(train_loader, model, DEVICE)
        val_rmse, val_mae, val_r2, val_spearman, val_hit, val_ef = evaluate_base(val_loader, model, DEVICE)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_metrics.append({
            "Epoch": epoch, "Train_Loss": round(avg_train_loss, 4), "Train_RMSE": round(train_rmse, 4),
            "Train_MAE": round(train_mae, 4), "Train_R2": round(train_r2, 4), "Train_Spearman": round(train_spearman, 4),
            "Train_HitRate@10%": round(train_hit, 4), "Train_EF": round(train_ef, 4),
            "Val_RMSE": round(val_rmse, 4), "Val_MAE": round(val_mae, 4), "Val_R2": round(val_r2, 4),
            "Val_Spearman": round(val_spearman, 4), "Val_HitRate@10%": round(val_hit, 4), "Val_EF": round(val_ef, 4),
            "LR": round(current_lr, 6)
        })
        
        print(f'|{epoch:5}|{avg_train_loss:10.3f}|{train_rmse:10.3f}|{train_mae:9.3f}|{train_r2:8.3f}|{train_spearman:14.3f}'
              f'|{train_hit:16.2%}|{train_ef:8.2f}|{val_rmse:8.3f}|{val_mae:7.3f}|{val_r2:8.3f}|{val_spearman:12.3f}'
              f'|{val_hit:14.2%}|{val_ef:6.2f}|{current_lr:4.1e}|')
        
        if (val_rmse < best_val_rmse) or (val_r2 > best_val_r2) or (val_ef > best_val_ef):
            best_val_rmse, best_val_r2, best_val_ef = val_rmse, val_r2, val_ef
            best_val_mae, best_val_spearman, best_val_hit = val_mae, val_spearman, val_hit
            patience_counter = 0
            torch.save(model, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break
    
    print('=' * len(header))
    best_metrics = {"RMSE": best_val_rmse, "MAE": best_val_mae, "R2": best_val_r2,
                    "Spearman": best_val_spearman, "HitRate": best_val_hit, "EF": best_val_ef}
    print(f'{model_name.upper()}-{split} best validation - RMSE: {best_val_rmse:.3f}, MAE: {best_val_mae:.3f}, '
          f'R2: {best_val_r2:.3f}, Spearman: {best_val_spearman:.3f}, Hit@top10%: {best_val_hit:.2%}, EF@top10%: {best_val_ef:.2f}')
    print(f'Best model saved to: {model_save_path}')
    save_epoch_result(epoch_metrics, epoch_result_path)
    record_summary(split, model_name, best_metrics, model_save_path, epoch_result_path)
    GLOBAL_SCALER = None

# Run training for all specified splits and models
def run_training(split_list, model_list):
    model_map = {"afp": EnhancedAFP, "gat": EnhancedGAT, "gcn": simpleGCN}
    for split in split_list:
        for model_name in model_list:
            try:
                train_model(split, model_name, model_map[model_name])
            except Exception as e:
                print(f"\nTraining {model_name}-{split} failed: {type(e).__name__} - {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    global_summary_path = METRICS_DIR / f"graph_model_best_summary_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.csv"
    save_global_summary(global_summary_path)
    print(f"\nAll specified models training completed!")
    print(f"Models saved to: {RESULT_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")
    print(f"Global summary: {global_summary_path}")
    print(f"Label scalers saved to: {SCALER_DIR}")
    print(f"Output constraint: Sigmoid -> normalized [0,1] -> denormalized pIC50 [{PIC50_MIN}, {PIC50_MAX}]")

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Graph model batch training tool (AFP/GAT/GCN) - output constraint + fixed range normalization")
    parser.add_argument("--split", required=True, choices=BASE_SPLITS + ["all"], help="Data split type")
    parser.add_argument("--models", required=True, choices=BASE_MODELS + ["all"], help="Model type to train")
    args = parser.parse_args()
    return (BASE_SPLITS if args.split == "all" else [args.split],
            BASE_MODELS if args.models == "all" else [args.models])

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    split_list, model_list = parse_args()
    print(f"Starting graph model batch training - splits: {split_list} | models: {model_list}")
    print(f"Training metrics will be saved to: {METRICS_DIR}")
    print(f"Label scalers will be saved to: {SCALER_DIR}")
    print(f"Fixed pIC50 normalization range: [{PIC50_MIN}, {PIC50_MAX}]")
    run_training(split_list, model_list)