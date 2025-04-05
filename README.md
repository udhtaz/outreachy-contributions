# 🧬 Outreachy Contributions: Molecular Representation Learning using Ersilia

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
├── data/                    # Processed datasets and splits (AMES, HIV)
│   ├── AMES/
│   └── HIV/
│
├── models/                 # Trained model files (.pkl)
│   └── AMES/
│
├── scripts/                 # Python scripts for core logic
│   ├── tdc_dataset_download.py
│   ├── featurise.py
│   ├── eda_utils.py
│   ├── model_utils.py       # Full training, evaluation, and visualization logic
│
├── notebooks/              # Interactive Jupyter experiments
│   ├── AMES_Mutagenicity_Prediction.ipynb
│   └── HIV_Inhibition_Prediction.ipynb
│
└── README.md
```

---

## 📊 Datasets of Interest

Two binary classification datasets from TDC were used:

### 1. **AMES Mutagenicity**
- Predict if a compound is **mutagenic** (1) or **non-mutagenic** (0)
- Input: SMILES strings
- Size: ~7,255 compounds
- [🔗 View](https://tdcommons.ai/single_pred_tasks/tox#ames-mutagenicity)

### 2. **HIV Inhibition**
- Predict if a compound **inhibits HIV replication** (1) or not (0)
- Input: SMILES strings
- Size: ~41,000 compounds
- [🔗 View](https://tdcommons.ai/single_pred_tasks/tox#hiv)

---

## 🧬 Molecular Representation (Featurization)

Featurization converts SMILES into numeric vectors using pretrained models from the Ersilia Hub. Supported via both CLI and SDK with fallback logic and container readiness checks.

### ✅ Featurizer Used

| Featurizer ID | Description                             | Source                         |
|---------------|-----------------------------------------|--------------------------------|
| `eos5guo`     | Extended Reduced Graph (ErG) Descriptors| [GitHub](https://github.com/ersilia-os/eos5guo) |
| `eos4wt0`     | Morgan (Circular) Fingerprints          | [GitHub](https://github.com/ersilia-os/eos4wt0) |

---

## 🧠 Model Training & Evaluation

After featurization, models were trained using **ExtraTreesClassifier**, **RandomForest**, and **LightGBM**, and evaluated on standard metrics:

- Accuracy
- F1 Score
- Precision
- Recall
- AUC
- MCC
- Cohen's Kappa

A reusable `ModelTrainer` class handles all training, loading, and evaluation. Model comparison across featurizers is visualized with bar plots.

### 📉 Example: AMES Mutagenicity Evaluation Plot

<img src="notebooks/assets/ames_comparison_plot.png" alt="AMES Model Comparison" width="600"/>

---

## 🔄 Reproducibility

```bash
# 1. Download datasets
python scripts/tdc_dataset_download.py --task tox --dataset AMES

# 2. Run featurization
python scripts/featurise.py --dataset AMES --auto

# 3. Train and evaluate
# Inside a notebook or script
from scripts.model_utils import ModelTrainer
trainer = ModelTrainer(...)  # Pass featurized splits
trainer.train_with_pycaret()
trainer.compare_loaded_models_across_features(...)
```

---

##  What's Next

- [x] Load and preprocess HIV dataset
- [x] Train comparable models for HIV
- [ ] Deploy evaluation results as a static dashboard
- [ ] Package reusable pipeline CLI for external datasets

---

## 🤝 Acknowledgments

Thanks to the [Ersilia Open Source Initiative](https://www.ersilia.io) and the [Outreachy program](https://www.outreachy.org/) for this opportunity.