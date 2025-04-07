# 🧬 Outreachy Contributions: Molecular Representation Learning with Ersilia & TDC

This repository documents my Outreachy contribution to the [Ersilia Model Hub](https://www.ersilia.io/model-hub), a platform providing AI models for drug discovery. The goal of this project is to explore molecular representation learning using the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) datasets, apply pretrained molecular featurizers from the Ersilia Model Hub, and build reproducible pipelines to train, evaluate, and compare ML models for drug discovery tasks.

---

## 📌 Project Objectives

- ✅ Understand and interact with the Ersilia Model Hub via SDK and CLI
- ✅ Apply molecular featurization to SMILES strings using pretrained models
- ✅ Train and evaluate ML classifiers using PyCaret and scikit-learn
- ✅ Automate dataset loading, featurization, model training, and evaluation
- ✅ Visualize and compare model performance across different molecular representations

---

##  Repository Structure

```bash
outreachy-contributions/
│
├── data/                            # Processed datasets and splits
│   ├── AMES/
│   ├── HIV/
│   └── .gitkeep
│
├── models/                          # Trained models
│   ├── ames/
│   └── .gitkeep
│
├── notebooks/                       # Interactive Jupyter notebooks
│   ├── AMES_Mutagenicity_Prediction.ipynb
│   ├── HIV_Inhibition_Prediction.ipynb
│   ├── figures/                     # Plots and Figures 
│   └── .gitkeep
│
├── scripts/                         # Core Python logic for the project
│   ├── eda_utils.py                 # EDA utilities (visuals, SMARTS, checks)
│   ├── featurise.py                 # Ersilia model featurization pipeline
│   ├── model_utils.py              # Preprocessing, training, evaluation
│   ├── tdc_dataset_download.py      # Dataset downloader for TDC
│   └── .gitkeep
│
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

---

## 📊 Datasets of Interest

Two binary classification datasets from TDC were used:

### 1. **AMES Mutagenicity (Primary Dataset)**
- Predict if a compound is **mutagenic** (1) or **non-mutagenic** (0)
- Input: SMILES strings
- Size: ~7,255 compounds
- [🔗 View](https://tdcommons.ai/single_pred_tasks/tox#ames-mutagenicity)

### 2. **HIV Inhibition (Secondary Dataset - to test scripts re-usability)**
- Predict if a compound **inhibits HIV replication** (1) or not (0)
- Input: SMILES strings
- Size: ~41,000 compounds
- [🔗 View](https://tdcommons.ai/single_pred_tasks/tox#hiv)

---

## 🧬 Molecular Representation Model Selected

Featurization converts SMILES into numeric vectors using pretrained models from the Ersilia Hub. Supported via both CLI and SDK with fallback logic and container readiness checks.

### ✅ Featurizer Used

| Featurizer ID | Description                             | Source                         |
|---------------|-----------------------------------------|--------------------------------|
| `eos5guo`     | Extended Reduced Graph (ErG) Descriptors| [GitHub](https://github.com/ersilia-os/eos5guo) |
| `eos4wt0`     | Morgan (Circular) Fingerprints          | [GitHub](https://github.com/ersilia-os/eos4wt0) |

---

## 🔬 Methodology Summary : 

### 🧬 AMES Mutagenicity as Primary Case Study

### 1. 📥 Fetch Dataset  
  - AMES dataset from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/)

    Downloaded using `TDCDatasetDownloader`, auto-split into train/valid/test.


### 2. 📊 Exploratory Data Analysis  

  - **Dataset Summary & Unique SMILES Check**

    Quick stats, missing values, SMILES uniqueness across splits and Target class distribution.

<img width="1064" alt="Screenshot 2025-04-07 at 1 22 32 PM" src="https://github.com/user-attachments/assets/f97d083a-d1c6-43cb-b838-c5eed62f955a" />

<img width="1064" alt="Screenshot 2025-04-07 at 1 22 10 PM" src="https://github.com/user-attachments/assets/059deb2c-d991-4208-8047-821f661ba0e4" />

<img width="1064" alt="Screenshot 2025-04-07 at 1 26 59 PM" src="https://github.com/user-attachments/assets/24b68373-0ccb-4811-bb76-402d5d5079f4" />

  
  - **RDKit Molecular Validity**

    Ensures SMILES strings represent valid molecules.

<img width="1073" alt="Screenshot 2025-04-07 at 1 28 52 PM" src="https://github.com/user-attachments/assets/2af26940-4509-47aa-b51e-2aa376449edf" />
 
  
  - **Descriptor Engineering (MW, LogP, TPSA, etc.)**

    Calculated physicochemical features using RDKit and correlation to target class (Y) across datasplits.
<img width="1063" alt="Screenshot 2025-04-07 at 1 30 42 PM" src="https://github.com/user-attachments/assets/f3495c45-af35-48fd-a481-1ca94845cc5b" />
  

  - **SMARTS Pattern Matching**

    Detected presence of key functional groups (e.g., amines, halogens).
<img width="1119" alt="Screenshot 2025-04-07 at 1 35 18 PM" src="https://github.com/user-attachments/assets/9b2d8c5a-6ff8-4f63-ba8e-d10bb375e07f" />

  
  - **Correlation Heatmaps, Boxplots, Histograms**

    Analyzed feature relationships and distribution shifts.
<img width="1062" alt="Screenshot 2025-04-07 at 1 36 38 PM" src="https://github.com/user-attachments/assets/65902b45-b871-4390-89c4-68cf57f464f5" />
<img width="1064" alt="Screenshot 2025-04-07 at 1 37 36 PM" src="https://github.com/user-attachments/assets/ee5c64d7-a105-4661-84fd-ee2f6f5f04b0" />
<img width="1065" alt="Screenshot 2025-04-07 at 2 07 09 PM" src="https://github.com/user-attachments/assets/64ce37a0-596e-4012-bcd1-f60edf465510" />
  
  - **Visualize Molecules by Class**

    Drew grid samples of molecules per label (mutagenic/non-mutagenic).
<img width="1060" alt="Screenshot 2025-04-07 at 1 38 40 PM" src="https://github.com/user-attachments/assets/82d437e1-e2ed-4c84-b2a1-a4fe5c913f51" />


### 3. 🧬 Molecular Featurization with Ersilia  

  - **ErG2D Descriptors – `eos5guo`**

    Pharmacophore-based 2D graph encoding of molecular structures.
  
  - **Morgan Fingerprints (Binary) – `eos4wt0`**

    Circular substructure presence encoding into 2048D binary vectors.
  
  - Featurized splits saved to `data/AMES/splits/`.


### 4. 🤖 Modeling Pipeline  

  - **Data Preprocessing with `ModelPreprocessor`**

    Auto-detects feature columns using prefix and splits data.
<img width="1065" alt="Screenshot 2025-04-07 at 2 12 03 PM" src="https://github.com/user-attachments/assets/34e60980-24ce-4e51-a4ec-d6212651ef13" />
<img width="1065" alt="Screenshot 2025-04-07 at 2 16 09 PM" src="https://github.com/user-attachments/assets/7e89494b-9d93-44d5-8797-c00533373973" />

  - **AutoML Exploration using PyCaret**

    Compared ML classifiers for framework selection in a single call.
<img width="1064" alt="Screenshot 2025-04-07 at 2 15 47 PM" src="https://github.com/user-attachments/assets/6934c082-cf7f-44d0-a236-e63b677d30a1" />
<img width="1064" alt="Screenshot 2025-04-07 at 2 16 48 PM" src="https://github.com/user-attachments/assets/3bf8d839-0389-4e89-af8f-3fb0f234f5d5" />
  
  - **Hyperparameter Tuning**
    - Auto `PyCaret` tuning on selected models
    - Manual `GridSearchCV` on selected models.
 <img width="1064" alt="Screenshot 2025-04-07 at 2 35 38 PM" src="https://github.com/user-attachments/assets/c84c8615-0ad4-4346-9f20-775bab1c4172" />
 
  - **Evaluation**
    
     Accuracy, F1, Precision, Recall, Kappa, AUC, MCC + Confusion Matrix + Feature Importance plots.
<img width="1064" alt="Screenshot 2025-04-07 at 3 18 47 PM" src="https://github.com/user-attachments/assets/836c104c-2155-4187-a95d-83ef5c5bd458" />
<img width="1067" alt="Screenshot 2025-04-07 at 2 21 54 PM" src="https://github.com/user-attachments/assets/e06686ed-6768-44f2-8ea0-d0370b994c87" />
<img width="1064" alt="Screenshot 2025-04-07 at 2 22 56 PM" src="https://github.com/user-attachments/assets/89c8b415-2afe-43aa-9b4e-948435a59555" />


### 5. ⚖️ Model Comparison  

  - Compared models trained using both `eos5guo` and `eos4wt0` featurizers.
  - Models:

    Random Forest, Extra Trees, LightGBM, XGBoost, Logistic Regression
  - Visual comparisons via bar charts.
<img width="1074" alt="Screenshot 2025-04-07 at 2 27 29 PM" src="https://github.com/user-attachments/assets/ce7b3729-5210-41e0-8a7b-aa2de7705de5" />


### 6. 🏆 Best Performing Model  

  - **Best model:** Extra Trees Classifier (ET) trained on Morgan Prints (`eos4wt0`)
  - Saved to `\models` folder


### 7. 🤖 Inference  


  - Use `ModelInference` class to predict on new SMILES.
  - Inference returns predictions + class probabilities.
<img width="1066" alt="Screenshot 2025-04-07 at 3 19 50 PM" src="https://github.com/user-attachments/assets/fb2d58df-bf12-4d48-9da5-aee599081787" />

---

## 📚 SCRIPTS DOCUMENTATION

### 1. 📥 `TDCDatasetDownloader` - `tdc_dataset_download.py`

A modular downloader class for fetching datasets and splits from the [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) for single prediction tasks.

#### Overview

The `TDCDatasetDownloader` handles:
- Validating dataset and category combinations
- Downloading raw `.csv` data
- Saving train/valid/test splits
- Relocating any matching `.tab` files into the dataset directory

##### Class: `TDCDatasetDownloader`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `__init__(self, category, name, save_dir=None)` | Initializes the downloader, validates the inputs, fetches the dataset and its splits, and organizes them into directories. | `category` (str): Category name from TDC (`'tox'`, `'hts'`, etc.)<br>`name` (str): Dataset name under the category<br>`save_dir` (str, optional): Target directory to save the data. Defaults to `../data` | `None` | Automatically creates folders and downloads data and splits. |
| `_validate_inputs(self)` | Ensures that the provided category and dataset name are valid according to the pre-defined mapping. | None | Raises `ValueError` if invalid category or dataset is provided. | 🔒 Internal method |
| `_download_data(self)` | Downloads the full dataset using the TDC API and saves it as a CSV. | None | None | Dataset saved as `{save_dir}/{name}/{name}.csv` |
| `_download_split(self)` | Retrieves training/validation/test splits and saves them into separate CSV files. | None | None | Saves as `{save_dir}/{name}/splits/*.csv` |
| `_relocate_tab_file(self)` | Searches for a `.tab` file (possibly raw format), moves it to the dataset folder if found, and optionally cleans up the empty source folder. | None | None | Searches in `../notebooks/data` and current working directory |
| `__main__` (CLI entry point) | Allows the script to be used as a standalone tool via command line. | `--category`, `--name`, `--save_dir` | Runs the downloader class | Use: `python tdc_dataset_download.py --category tox --name AMES` |

---
### 2. 📊 EDA Module – `eda_utils.py`
A modular utility for loading, visualizing, and analyzing molecular datasets used in machine learning workflows.

#### Overview

The module handles:

- Loading datasets and train/valid/test splits via `DatasetLoader`
- Performing detailed dataset inspections and visualizations with `EDAVisualizer`
- Computing molecular descriptors and SMARTS pattern presence
- Validating SMILES strings and checking data quality
- Comparing distributions across dataset splits
- Detecting substructure using `SMARTSPatternAnalyzer` with SMARTS patterns

##### Class: `DatasetLoader`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `__init__(self, dataset_name, data_dir='../data')` | Initializes loader with dataset path and placeholders for main/split data. | `dataset_name` (str): Name of the dataset<br>`data_dir` (str): Base directory (default: `'../data'`) | None | Sets up path references and internal state |
| `load_main(self)` | Loads the main `.csv` dataset file. | None | `pd.DataFrame` | Loads from `{data_dir}/{dataset_name}/{dataset_name}.csv` |
| `load_splits(self)` | Loads `train.csv`, `valid.csv`, and `test.csv` splits. | None | Tuple of `pd.DataFrame`s | Reads from `{data_dir}/{dataset_name}/splits/` |
| `load_all(self, print_shapes=True)` | Loads all datasets and optionally prints their shapes. | `print_shapes` (bool): Whether to print dataset shapes | Tuple of 4 `pd.DataFrame`s | Returns main, train, valid, test |


##### Class: `EDAVisualizer`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `show_dataset_info(...)` | Displays info (shape, missing values, preview) for datasets. | `loader_or_dfs`, `dataset_names` | None | Accepts loader or direct DataFrames |
| `plot_label_distribution(...)` | Visualizes target label distribution. | `df`, `target_col`, `title`, `labels` | None | Bar chart + stats printed |
| `smiles_to_mol(smiles)` | Converts SMILES to RDKit Mol object. | `smiles` (str) | `Mol` or `None` | Helper for molecule rendering |
| `draw_molecules(...)` | Draw molecules from SMILES list. | `smiles_list`, `legends`, `n_per_row`, `mols_per_row`, `sub_img_size` | Image | Uses `RDKit.Draw` |
| `draw_samples_by_class(...)` | Draws `n` molecules per class label. | `df`, `smiles_col`, `label_col`, `n` | Image + legend | Randomized but reproducible |
| `compare_label_distributions(...)` | Plots label distribution across datasets. | `dfs`, `df_names`, `target_col` | None | Stacked bar + % stats |
| `compare_numeric_distribution(...)` | Compares numeric column across datasets (hist). | `dfs`, `df_names`, `col`, `bins` | None | Uses Seaborn |
| `compare_boxplots(...)` | Plots boxplots side-by-side for a numeric column. | `dfs`, `df_names`, `col` | None | Boxplot per dataset |
| `compare_correlation_heatmaps(...)` | Plots correlation heatmaps for numeric cols. | `dfs`, `df_names`, `cols` | None | Subplot layout |
| `compare_unique_smiles(...)` | Bar chart of unique SMILES count across datasets. | `dfs`, `df_names`, `smiles_col`, `title` | None | Prints count summary |
| `add_molecular_descriptors(...)` | Adds RDKit descriptors like MolWt, LogP, etc. | `dfs`, `loader`, `smiles_col`, `names`, `inplace` | Optional list of modified `DataFrames` | Applies multiple descriptors |
| `check_smiles_length(...)` | Analyzes SMILES string lengths. | `dfs`, `loader`, `smiles_col`, `names` | None | Histograms + summary stats |
| `check_missing_duplicates(...)` | Prints missing/duplicate info. | `dfs`, `loader`, `smiles_col`, `names` | None | Row/SMILES-level checks |
| `check_molecular_validity(...)` | Checks if SMILES are valid molecules. | `dfs`, `loader`, `smiles_col`, `names` | None | Prints % invalid |
| `compare_smiles_length(...)` | Histogram comparing SMILES lengths. | `dfs`, `dataset_names`, `loader`, `smiles_col` | None | Groups by split label |


##### Class: `SMARTSPatternAnalyzer`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `__init__(self, smarts_dict=None)` | Compiles SMARTS patterns into RDKit Mol objects. | `smarts_dict` (dict) | None | Defaults to common chemical groups |
| `analyze(self, df, smiles_col='Drug')` | Flags presence of SMARTS substructures per compound. | `df`, `smiles_col` | Updated `DataFrame` | Adds `smarts_*` boolean columns |
| `summarize_patterns(self, df, label_col='Y')` | Aggregates and plots SMARTS hits by class. | `df`, `label_col` | None | Bar chart + class-wise counts |

---

### 3. 🧬 Molecular Featurisation Module – `featurise.py`

A robust utility for generating molecular descriptors using [Ersilia Model Hub](https://www.ersilia.io/) across dataset splits.

#### Overview

The module handles:

- Running Ersilia model fingerprinting (via SDK or CLI)
- Featurizing SMILES columns from train/valid/test splits
- Auto-detecting molecule formats (single or paired drugs)
- Managing temporary files and logging
- Merging target labels when applicable
- Returning or saving enriched datasets with molecular features


##### Class: `MolecularFeaturiser`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `__init__(self, dataset_name, model_id, use_python_api=True, auto_serve=False)` | Initializes the featurizer with dataset context, Ersilia model ID, and settings. | `dataset_name` (str), `model_id` (str), `use_python_api` (bool), `auto_serve` (bool) | None | Auto-switches between SDK and CLI if `auto_serve=True` |
| `run_ersilia_featurisation(self, input_path, output_path)` | Executes Ersilia fingerprinting on input CSV. | `input_path` (str), `output_path` (str) | None | Handles both SDK and CLI execution |
| `featurise_smiles_column(self, df, smiles_col, prefix)` | Featurizes a single SMILES column. | `df` (DataFrame), `smiles_col` (str), `prefix` (str) | `DataFrame` with features | Uses temporary files |
| `featurise_split(self, split, return_df=False)` | Featurizes a dataset split (`train`, `valid`, `test`). | `split` (str), `return_df` (bool) | Path or DataFrame | Merges features + labels if available |
| `featurise_all_splits(self)` | Featurizes all three splits. | None | None | Convenience wrapper for bulk run |
| `get_featurized_path(self, split)` | Returns path to featurized split file. | `split` (str) | `Path` | Based on dataset/model |
| `__del__(self)` | Safely shuts down Ersilia SDK server. | None | None | Only if SDK is active |

---

### 4. 🤖 Model Utilities Module – `model_utils.py`

A comprehensive module for model preprocessing, training, evaluation, and inference using PyCaret and sklearn-compatible estimators.

####  Overview

The module handles:

-  Loading and preprocessing featurized datasets via `ModelPreprocessor`
-  Automated and custom model training using `ModelTrainer`
-  Robust evaluation, tuning, visualization, and comparison across models
-  Inference on raw SMILES input using `ModelInference`

##### Class: `ModelPreprocessor`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `__init__(...)` | Initializes paths, verifies featurized files, and optionally runs featurization. | `dataset_name`, `model_id`, `feature_prefix`, `auto_featurise` | None | Auto-switches to `MolecularFeaturiser` if files are missing |
| `preprocess()` | Loads and returns `(X_train, y_train, X_valid, y_valid, X_test, y_test)` | None | Tuple of `DataFrames` | Also removes non-numeric columns |
| `_needs_featurisation()` | Checks whether required feature CSVs are missing | None | `bool` | Internal logic |
| `_detect_columns(...)` | Extracts feature/target columns by prefix | `DataFrame` | Feature cols, target col | Handles multiple prefixes |
| `_log_summary(...)` | Logs summary of shapes, types, NaNs, class balance | Internal | None | Info only |

##### Class: `ModelTrainer`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `run_pycaret()` | Uses PyCaret to compare and select the best model | None | Best model | Automatically calls `setup()` and `compare_models()` |
| `log_results()` | Displays top models from PyCaret leaderboard | None | None | Shows metrics summary |
| `evaluate_on_test()` | Runs `predict_model()` on test set | None | DataFrame | Uses PyCaret pipeline |
| `train_custom_model(...)` | Fits a custom sklearn model with metrics | `model_cls`, `**kwargs` | Trained model | Metrics stored in leaderboard |
| `train_selected_model(...)` | Trains a pre-initialized sklearn model | `model` | Trained model | Manual model fitting |
| `tune_model_with_pycaret(...)` | Tunes a model using `tune_model()` | `model`, `n_iter`, `choose_better` | Tuned model | PyCaret tuning |
| `evaluate_model(...)` | Prints metrics for model on any split | `dataset`, `external_data_path` | None | Supports external CSVs |
| `save_model(...)` | Saves model to `models/{dataset}/...` | `model`, `dataset_name`, `featurizer_id`, `framework` | None | Uses `joblib` |
| `load_model_from_file(...)` | Loads model + correct dataset context | `model_info`, `split`, `external_data_path` | Dict with model, features, etc. | Parses features |
| `plot_confusion_matrix(...)` | Heatmap of confusion matrix for split | `model`, `dataset`, `plot_title` | None | Seaborn heatmap |
| `plot_feature_importance(...)` | Top N feature importances or coefficients | `top_n`, `plot_title` | None | Auto-handles trees vs linear |
| `view_custom_leaderboard()` | Returns leaderboard of custom models | None | DataFrame | Sorted by F1 score |
| `compare_loaded_models_across_features(...)` | Compare multiple saved models across splits | `loaded_models`, `eval_split`, `plot_title` | DataFrame | Models can have diff features |
| `compare_models_on_metrics(...)` | Compare list of trained models side-by-side | `models`, `model_names`, `dataset`, `external_data_path` | DataFrame | Supports external data, custom names |


##### Class: `ModelInference`

| Function | Description | Parameters | Returns | Flags/Notes |
|----------|-------------|------------|---------|-------------|
| `__init__(...)` | Loads a model and infers dataset/featurizer from filename | `model_path` | None | Falls back to default AMES model |
| `predict(smiles_list)` | Featurizes input SMILES and returns predictions | `List[str]` | DataFrame with prediction and probability | Auto-aligns with trained model |
| `_parse_model_filename(...)` | Extracts dataset/featurizer from file | `model_path` | Tuple | Must follow `<dataset>_<featurizer>_<model>.pkl` |
| `_get_probability_column_name()` | Adjusts probability column name per dataset | None | str | E.g., "Mutagenic_Probability" |

---

## 🤝 Acknowledgments

Thanks to the [Ersilia Open Source Initiative](https://www.ersilia.io) and the [Outreachy program](https://www.outreachy.org/) for this opportunity.
