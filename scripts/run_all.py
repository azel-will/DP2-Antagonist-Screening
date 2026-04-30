#!/usr/bin/env python3
"""
Execute multiple Python scripts in sequence, and each script can have different parameters
Use a path relative to the current script for easy migration
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ---------- Path benchmark ----------
SCRIPT_DIR = Path(__file__).resolve().parent  

# Each script and its exclusive parameters
SCRIPTS_CONFIG = [
    # ("./01_fda_approvedrug_clean.py", []),
    # ("./02_chembl_clean.py", []),
    # ("./03_bindingdb_clean.py", []),
    # # ("./04_pubchem_clean.py", []),
    # ("./05_merged_clean.py", []),
    # ("./06_data_split.py", []),
    # ("./07_ml_feature_extract.py", ["--split", "all", "--mode", "train"]),
    # ("./07_ml_feature_extract.py", ["--split", "all", "--mode", "test"]),
    # ("./07_ml_feature_extract.py", ["--split", "all", "--mode", "predict"]),
    # ("./09_graph_feature_extract.py", ["--split", "all", "--mode", "train"]),
    # ("./09_graph_feature_extract.py", ["--split", "all", "--mode", "test"]),
    # ("./09_graph_feature_extract.py", ["--split", "all", "--mode", "predict"]),
    # ("./08_ml_modeltrain.py", ["--split", "all", "--model", "all"]),
    # ("./10_graph_modeltrain.py", ["--split", "all", "--model", "all"]),
    # ("./11_external_test.py", ["--split", "all", "--mode", "all"]),
    # ("./12_ensemble_model.py", ["--split", "fingerprint", "--model", "xgb", "svm", "gcn", "--mode", "train"]),
    # ("./12_ensemble_model.py", ["--split", "fingerprint", "--model", "xgb", "svm" ,"gcn", "--mode", "test"]),
    # ("./12_ensemble_model.py", ["--split", "scaffold", "--model", "lgb", "xgb", "gat", "--mode", "train"]),
    # ("./12_ensemble_model.py", ["--split", "scaffold", "--model", "lgb", "xgb", "gat", "--mode", "test"]),
    # ("./12_ensemble_model.py", ["--split", "random", "--model", "gb", "svm", "gcn", "--mode", "train"]),
    # ("./12_ensemble_model.py", ["--split", "random", "--model", "gb", "svm", "gcn", "--mode", "test"]),
    # ("./12_ensemble_model.py", ["--split", "fingerprint", "--model", "xgb svm gcn", "--mode", "train"]),
    # ("./12_ensemble_model.py", ["--split", "fingerprint", "--model", "xgb svm gcn", "--mode", "test"]),
    # ("./13_predict.py", ["--split", "fingerprint"]),
    # ("./13_predict.py", ["--split", "scaffold"]),
    # ("./13_predict.py", ["--split", "random"]),
]

DELAY = 30 


def run_script(script_relative_path, args):
    script = (SCRIPT_DIR / script_relative_path).resolve()
    
    if not script.exists():
        print(f"Error: File not found: {script}")
        print(f"  Relative path: {script_relative_path}")
        print(f"  Script directory: {SCRIPT_DIR}")
        return False
    
    if isinstance(args, str):
        args = [args] if args else []
    
    cmd = [sys.executable, str(script)] + args
    
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Start: {script.name}")
    print(f"Relative: {script_relative_path}")
    print(f"Full path: {script}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)
    
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    
    success = result.returncode == 0
    status = "Done" if success else "Failed"
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {status} | Time: {elapsed/60:.1f} min")
    
    return success


def main():
    print(f"Planned execution: {len(SCRIPTS_CONFIG)} scripts")
    print(f"Python: {sys.executable}")
    print(f"Script directory: {SCRIPT_DIR}")
    
    results = []
    for i, (script_rel, args) in enumerate(SCRIPTS_CONFIG, 1):
        print(f"\n{'='*70}")
        print(f"Progress: [{i}/{len(SCRIPTS_CONFIG)}]")
        
        success = run_script(script_rel, args)
        results.append((script_rel, success))
        
        if i < len(SCRIPTS_CONFIG) and DELAY > 0:
            print(f"\nWaiting {DELAY} seconds...")
            time.sleep(DELAY)
        
        if not success:
            print("\nExecution failed, stopping subsequent scripts")
            break
    
    print(f"\n{'='*70}")
    print("Execution Summary")
    print('='*70)
    for script_rel, ok in results:
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {script_rel}")
    
    success_count = sum(1 for _, ok in results if ok)
    print(f"\nTotal: {success_count}/{len(SCRIPTS_CONFIG)} successful")
    
    return 0 if success_count == len(SCRIPTS_CONFIG) else 1


if __name__ == "__main__":
    sys.exit(main())