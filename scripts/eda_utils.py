import os
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt

class DatasetLoader:
    """
    Loads main dataset and train/valid/test splits from a specified directory structure.
    """

    def __init__(self, dataset_name, data_dir='../data'):
        """
        Parameters:
        - dataset_name (str): Name of the dataset (e.g., 'AMES')
        - data_dir (str): Base directory containing the dataset folder (default: '../data')
        """
        self.dataset_name = dataset_name
        self.base_path = os.path.join(data_dir, dataset_name)

    def load_main(self):
        """
        Load the main dataset CSV.
        Returns:
        - pandas.DataFrame: Main dataset
        """
        path = os.path.join(self.base_path, f"{self.dataset_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Main dataset not found at {path}")
        return pd.read_csv(path)

    def load_splits(self):
        """
        Load train, validation, and test split CSVs.
        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, valid, test DataFrames
        """
        split_path = os.path.join(self.base_path, 'splits')
        train = pd.read_csv(os.path.join(split_path, 'train.csv'))
        valid = pd.read_csv(os.path.join(split_path, 'valid.csv'))
        test = pd.read_csv(os.path.join(split_path, 'test.csv'))
        return train, valid, test

    def load_all(self):
        """
        Load the full dataset and splits.
        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: main, train, valid, test
        """
        main = self.load_main()
        train, valid, test = self.load_splits()
        return main, train, valid, test


def show_dataset_info(df, name="Dataset"):
    print(f"📦 {name} Info")
    print("-" * 40)
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nPreview:")
    display(df.head())


def plot_label_distribution(df, target_col='Y', title='Target Distribution', labels=None):
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100

    print("🧮 Class Counts:")
    print(counts)
    print("\n📊 Class Percentages:")
    print(percentages.round(2))

    counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title(title)
    plt.xlabel('Class' if not labels else labels)
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def smiles_to_mol(smiles):
    """
    Convert a SMILES string to an RDKit Mol object.
    Returns None if the SMILES is invalid.
    """
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None

def draw_molecules(smiles_list, legends=None, n_per_row=5, mols_per_row=5, sub_img_size=(200, 200)):
    """
    Draw RDKit molecules from a list of SMILES strings.

    Parameters:
    - smiles_list (list): List of SMILES strings
    - legends (list): Optional list of legend strings
    - n_per_row (int): Molecules per row
    - sub_img_size (tuple): Size per molecule
    """
    mols = [smiles_to_mol(sm) for sm in smiles_list]
    return Draw.MolsToGridImage(
        mols, molsPerRow=n_per_row, subImgSize=sub_img_size,
        legends=legends if legends else ["" for _ in mols]
    )

def draw_samples_by_class(df, smiles_col='Drug', label_col='Y', n=4):
    """
    Show sample molecules for each class in the dataset.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - smiles_col (str): Column containing SMILES strings
    - label_col (str): Target column
    - n (int): Number of molecules to draw per class
    """
    unique_labels = df[label_col].unique()
    for label in unique_labels:
        subset = df[df[label_col] == label].sample(n=n, random_state=42)
        smiles = subset[smiles_col].tolist()
        print(f"🧪 {n} molecules for class {label}")
        display(draw_molecules(smiles, legends=[f"{label}_{i}" for i in range(len(smiles))]))
