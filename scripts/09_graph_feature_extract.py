import sys
import tqdm
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import dgl
import joblib
from rdkit import Chem
from dgllife.utils import (
    CanonicalAtomFeaturizer, 
    CanonicalBondFeaturizer,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "data"
SPLIT_BASE_DIR = ROOT_DIR / "split"
EXTERNAL_CSV = SPLIT_BASE_DIR / "external_split" / "external_test.csv"
SMILES_COL = 'SMILES'
LABEL_COL = 'pIC50'
ALLOWED_SPLITS = ["fingerprint", "scaffold", "random", "all"]
ALLOWED_MODES = ["train", "test", "predict"]
ALL_SPLITS = ["fingerprint", "scaffold", "random"]

def check_train_deps(split):
    feat_dir = SPLIT_BASE_DIR / f"{split}_split" / "feature"
    return (feat_dir / f"{split}_train_graphdata.pkl").exists()

def get_all_exist_files(split, mode, save_path=None):
    exist_files = []
    split_feat_dir = SPLIT_BASE_DIR / f"{split}_split" / "feature"
    external_feat_dir = SPLIT_BASE_DIR / "external_split" / "feature"
    if mode == "train":
        exist_files.extend([
            split_feat_dir / f"{split}_train_graphdata.pkl",
            split_feat_dir / f"{split}_val_graphdata.pkl"
        ])
    elif mode == "test":
        exist_files.append(external_feat_dir / f"external_test_graphdata.pkl")
    elif mode == "predict" and save_path:
        exist_files.append(Path(save_path))
    return [f for f in exist_files if f.exists()]

def batch_delete_files(file_list):
    for f in file_list:
        if f.exists():
            f.unlink()
    print(f"Deleted {len(file_list)} old files")

def parse_args():
    parser = argparse.ArgumentParser(description="Graph feature extraction tool (Canonical version)")
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument("--mode", required=True, choices=ALLOWED_MODES)
    parser.add_argument("--path", type=str, default=None, help="Input CSV path for predict mode only")
    parser.add_argument("--save", type=str, default=None, help="Save path for predict mode only")
    args = parser.parse_args()
    
    # Set default values only for predict mode
    if args.mode == "predict":
        if args.path is None:
            args.path = str(ROOT_DIR / "01_intermediate/fda_approvedrug_clean.csv")
        if args.save is None:
            args.save = str(SCRIPT_DIR.parent / "results/summary/fda_drug/feature/graphdata_fda.pkl")
    
    # Validate arguments based on mode
    if args.mode == "train" and (args.path is not None or args.save is not None):
        parser.error("--mode=train does not accept --path or --save")
    if args.mode == "test" and args.path is not None:
        parser.error("--mode=test does not accept --path")
    if args.mode == "predict" and (args.path is None or args.save is None):
        parser.error("--mode=predict requires both --path and --save")
    
    return args

class UniversalSMILES2Graph:
    """
    Universal feature extractor: generates both atom and bond features
    """
    def __init__(self):
        # Atom features: 74-dim (Canonical)
        self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
        print("Initializing universal feature extractor")
        print("  Atom features: 74-dim (CanonicalAtomFeaturizer)")
        
        # Bond features: 12-dim (Canonical)
        self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field='e')
        print("  Bond features: 12-dim (CanonicalBondFeaturizer)")
    
    def __call__(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                print(f"Invalid molecule: {smiles}")
                return None
            
            # Build graph manually to ensure edge features match edge count
            src, dst = [], []
            edge_feats = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Get bond features (undirected edge)
                bond_feat = self.bond_featurizer(mol)['e'][bond.GetIdx()]
                
                # Add two directed edges, sharing same bond feature
                src.extend([i, j])
                dst.extend([j, i])
                edge_feats.extend([bond_feat, bond_feat])
            
            # Create graph
            g = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
            
            # Add self-loops (required by GCN/GAT/AttentiveFP)
            g = dgl.add_self_loop(g)
            
            # Set node features
            g.ndata['h'] = self.atom_featurizer(mol)['h']
            
            # Set edge features: original edges + self-loop edges
            # Self-loops use zero vectors
            edge_feats = torch.stack(edge_feats)
            num_edges = g.num_edges()
            num_original_edges = edge_feats.shape[0]
            num_self_loops = num_edges - num_original_edges
            
            if num_self_loops > 0:
                self_loop_feats = torch.zeros(num_self_loops, edge_feats.shape[1])
                g.edata['e'] = torch.cat([edge_feats, self_loop_feats], dim=0)
            else:
                g.edata['e'] = edge_feats
            
            return g
            
        except Exception as e:
            print(f"Processing error: {smiles}, {e}")
            return None

def process_data(csv_path, save_path, converter, has_label=True):
    csv_path, save_path = Path(csv_path), Path(save_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if SMILES_COL not in df.columns:
        raise ValueError(f"SMILES column not found: {SMILES_COL}")
    if has_label and LABEL_COL not in df.columns:
        raise ValueError(f"Label column not found: {LABEL_COL}")
    
    data_list = []
    smiles_list = df[SMILES_COL].tolist()
    label_list = df[LABEL_COL].tolist() if has_label else [None] * len(smiles_list)
    
    pbar = tqdm.tqdm(zip(smiles_list, label_list), total=len(df), desc=f'Processing {csv_path.name}')
    for smi, y in pbar:
        g = converter(smi)
        if g is not None:
            data_dict = {'smiles': smi, 'graph': g}
            if has_label:
                data_dict['label'] = float(y)
            data_list.append(data_dict)
            pbar.set_postfix({"valid": len(data_list), "total": len(df)})
    
    if not data_list:
        raise ValueError("No valid features extracted")
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data_list, save_path)
    print(f"Saved: {save_path} | Valid: {len(data_list)}/{len(df)}")
    if len(data_list) > 0:
        sample_g = data_list[0]['graph']
        print(f"  Feature dims: atom {sample_g.ndata['h'].shape[1]}, bond {sample_g.edata['e'].shape[1]}")

def process_split_data(split, converter):
    if split not in ["fingerprint", "scaffold", "random"]:
        return
    
    split_dir = SPLIT_BASE_DIR / f"{split}_split"
    train_csv = split_dir / f"{split}_train.csv"
    train_feat = split_dir / "feature" / f"{split}_train_graphdata.pkl"
    val_csv = split_dir / f"{split}_val.csv"
    val_feat = split_dir / "feature" / f"{split}_val_graphdata.pkl"
    
    print(f"\nProcessing [{split}] split")
    process_data(train_csv, train_feat, converter, True)
    process_data(val_csv, val_feat, converter, True)

def process_bound_test(split, converter, save_path=None):
    print(f"\nProcessing [{split}] test set")
    if save_path is None:
        save_path = SPLIT_BASE_DIR / "external_split" / "feature" / f"external_test_graphdata.pkl"
    else:
        save_path = Path(save_path)
    
    process_data(EXTERNAL_CSV, save_path, converter, True)
    print(f"Saved: {save_path}")

def process_bound_predict(split, converter, csv_path, save_path):
    print(f"\nProcessing [{split}] prediction set")
    process_data(csv_path, save_path, converter, False)
    print(f"Saved: {save_path}")

def batch_process(mode, converter, path=None, save=None):
    print(f"\nBatch processing | mode={mode}")
    for split in ALL_SPLITS:
        if mode == "test":
            save_path = Path(save).with_stem(f"{Path(save).stem}_{split}") if save else None
            process_bound_test(split, converter, save_path)
        elif mode == "predict":
            save_path = Path(save).with_stem(f"{Path(save).stem}_{split}")
            process_bound_predict(split, converter, path, save_path)
    print("Batch processing complete")

def main():
    args = parse_args()
    print(f"Config | split={args.split} | mode={args.mode}")
    print("=" * 50)
    
    # Initialize universal feature extractor (one-time extraction)
    converter = UniversalSMILES2Graph()
    
    need_process_splits = ALL_SPLITS if args.split == "all" else [args.split]
    all_exist_files = []
    
    for split in need_process_splits:
        if args.mode in ["test", "predict"] and not check_train_deps(split):
            raise FileNotFoundError(f"[{split}] missing training features")
        target_save_path = None
        if args.mode == "predict":
            target_save_path = Path(args.save).with_stem(f"{Path(args.save).stem}_{split}") if args.split == "all" else Path(args.save)
        all_exist_files.extend(get_all_exist_files(split, args.mode, target_save_path))
    
    if all_exist_files:
        all_exist_files = list(set(all_exist_files))
        print(f"\nDetected {len(all_exist_files)} existing files")
        while True:
            res = input("Delete and re-extract? (y/n): ")
            if res.lower() == "n":
                sys.exit(0)
            elif res.lower() == "y":
                batch_delete_files(all_exist_files)
                break
            else:
                print("Invalid input")
    
    try:
        if args.mode == "train":
            if args.split == "all":
                for split in ALL_SPLITS:
                    process_split_data(split, converter)
            else:
                process_split_data(args.split, converter)
        else:
            if args.split == "all":
                batch_process(args.mode, converter, args.path, args.save)
            elif args.mode == "test":
                process_bound_test(args.split, converter, args.save)
            elif args.mode == "predict":
                process_bound_predict(args.split, converter, args.path, args.save)
    except Exception as e:
        print(f"Processing failed: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Task complete! ")
if __name__ == "__main__":
    main()