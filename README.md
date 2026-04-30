# Download data
bindingdb_raw downloaded on September 13, 2025
pubchem_raw downloaded on September 14, 2025
chembl_raw downloaded on September 14, 2025，chembl_36
drugbank_fda downloaded on September 13, 2025，5.1.13

# DP2 Antagonist Screening: Dynamic Weighted Ensemble(ML&GNNs) 

## Overview
We developed a dynamic weighted ensemble integrating five ML models(SVM,XGB,GB,RF,LGB) and three GNNs(GAT,GCN,AFP) to screen FDA-approved drugs for DP2 antagonists.

## Getting Started

### Prerequisites
- **Python** 3.10 or higher
- **Conda** (recommended) or **pip**
- **Hardware**: CPU-only is sufficient; GPU is optional but will accelerate GNN training.

## Environment
We recommend using Conda:  
`conda env create -f environment.yml`  
Alternatively, use pip:  
`pip install -r requirements.txt`

### Data Preparation
Download the raw data from PubChem, ChEMBL, and BindingDB.

### Run the Pipeline
Step 0:Installation
Clone the repository:
    ```sh
    git clone git@github.com:azel-will/DP2-Antagonist-Screening.git
    cd DP2-Antagonist-Screening

Step 1:data clean
Preprocess using the scripts in `scripts/` following `00_env_check.py-06_data_split.py`.

Step 2: Generate features  (The result is saved in data/split/)
`python scripts/07_ml_feature_extract.py --split all --mode train`
`python scripts/09_graph_feature_extract.py --split all --mode train`
Feature data of all partitioned training sets and validation sets saved under /Nova/data can be extracted.
`python scripts/07_ml_feature_extract.py --split all --mode test`
`python scripts/09_graph_feature_extract.py --split all --mode test`
Feature data of all partitioned external test sets saved under /Nova/data can be extracted.
`python scripts/07_ml_feature_extract.py --split all --mode predict`
`python scripts/09_graph_feature_extract.py --split all --mode predict`
The features of the filtered data can be extracted and the path can be saved by default results/summary/fda_drug/feature，if you want to specify the path yourself, you can also use --path --save

Step 3: Train models (ML & GNNs)  (models save in results)
`python scripts/08_ml_modeltrain.py --split all --model alresultsl`  Train all models（ML） under all divisions
`python scripts/10_graph_modeltrain.py --split all --model all`Train all models（GNNs） under all divisions

Step 4:ensemble model （ensemble model save with models results）
`python scripts/12_ensemble_model.py --split fingerprint/scaffold/random --model  xgb svm gcn --mode train`Train ensemble models that can be freely combined under different divisions.
`python scripts/12_ensemble_model.py --split fingerprint/scaffold/random --model  xgb svm gcn --mode test`Test ensemble models that can be freely combined under different divisions.

Step 3: Virtual screening  (results save in /results/summary/fda_drug)
`python scripts/13_predict.py --split fingerprint/scaffold/random`Make predictions on the data.

Step 4: SHAP analysis  (results save in /results/shap)
`python scripts/14_shap_model.py`Train the shap model using the LGB model.

All outputs will be generated in `results/`.

python run_all.py，All the code can be run directly in sequence

A detailed list of contents can be viewed DIRECTORY LIST.txt
