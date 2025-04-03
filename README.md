# 🧪 Outreachy Contributions: Molecular Representation Learning using Ersilia

This repository documents my Outreachy contribution to the [Ersilia Model Hub](https://www.ersilia.io/model-hub), a platform providing AI models for drug discovery. The goal of this project is to explore molecular representation learning by selecting datasets from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/), applying pretrained molecular featurizers from the Ersilia Model Hub, and building reproducible pipelines for drug discovery tasks.

---

## Project goals:
- Understand how to use and interact with the Ersilia Model Hub
- Demonstrate basic AI/ML knowledge
- Show your Python coding skills 
- Practice code documentation and end user documentation

---

## 🚀 Project Structure

```bash
outreachy-contributions/
│
├── data/                    # Datasets (e.g., AMES, HIV)
│   ├── AMES/
│   └── HIV/
│
├── scripts/                 # Python scripts for EDA, featurization, etc.
│   ├── tdc_dataset_download.py
│   ├── featurise.py
│   └── eda_utils.py
│
├── notebooks/              # Jupyter notebooks for experiments
│   ├── AMES_Mutagenicity_Prediction.ipynb
│   └── AMES_Mutagenicity_Prediction.ipynb
│
└── README.md               # Project documentation
```

---

## 📊 Dataset of Interest

For this project, I selected two datasets from TDC:

### 1. **AMES Mutagenicity Dataset**
- **Task**: Binary Classification (mutagenic or not)
- **Input**: SMILES strings
- **Output**: `1` (mutagenic) or `0` (non-mutagenic)
- **Size**: ~7,255 compounds
- [More info](https://tdcommons.ai/single_pred_tasks/tox#ames-mutagenicity)

### 2. **HIV Dataset**
- **Task**: Binary Classification (HIV inhibition or not)
- **Input**: SMILES strings of molecules
- **Output**: `1` (active) or `0` (inactive)
- **Size**: ~41,000 molecules
- [More info](https://tdcommons.ai/single_pred_tasks/tox#hiv)

Both datasets were downloaded using Python and stored in the `data/` directory with a clear structure (`train.csv`, `valid.csv`, `test.csv` under `splits/`).

✅ Dataset preprocessing, validation, and classification task confirmation are completed.

---

## 🧬 Molecular Featurization (Representation Learning)

After browsing the [Ersilia Model Hub](https://www.ersilia.io/model-hub), I tested multiple featurizers to encode SMILES strings into feature vectors:

### 🔍 Representation Models Used

- **✅ `eos5guo`**: ErG 2D Descriptors  
  - Uses pharmacophore-based Extended Reduced Graph descriptors.
  - Efficient (598MB) and successfully featurized both AMES and HIV datasets.
  - [GitHub](https://github.com/ersilia-os/eos5guo) | [DockerHub](https://hub.docker.com/r/ersiliaos/eos5guo)

📌 The featurization script supports both SDK and CLI modes and includes fallback logic, reproducibility, and error handling.

---

## 🧰 How to Run Featurization

```bash
# Activate environment
source .venv/bin/activate

# Install requirements
pip install pytdc pandas ersilia

# Run featurizer
python scripts/featurise.py --dataset AMES --auto
python scripts/featurise.py --dataset HIV --auto
```

---

## 📌 Next Steps

- Train an ML classifier on the featurized data.
- Evaluate model performance using accuracy, precision, recall, etc.
- Visualize metrics using plots and graphs.
- Update README with model evaluation results.
