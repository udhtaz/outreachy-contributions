import os
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem import rdMolDescriptors

import matplotlib.pyplot as plt
import seaborn as sns


###############################################################################
# 1) DatasetLoader
###############################################################################
class DatasetLoader:
    """
    Loads main dataset and train/valid/test splits from a specified directory structure.
    Also stores them internally so they can be referenced by name without
    having to manually pass them around.
    """

    def __init__(self, dataset_name, data_dir='../data'):
        """
        Parameters:
        - dataset_name (str): Name of the dataset (e.g., 'AMES')
        - data_dir (str): Base directory containing the dataset folder (default: '../data')
        """
        self.dataset_name = dataset_name
        self.base_path = os.path.join(data_dir, dataset_name)

        # Placeholders for DataFrames
        self.main_data = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def load_main(self):
        """
        Load the main dataset CSV.
        Returns:
        - pandas.DataFrame: Main dataset
        """
        path = os.path.join(self.base_path, f"{self.dataset_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Main dataset not found at {path}")
        self.main_data = pd.read_csv(path)
        return self.main_data

    def load_splits(self):
        """
        Load train, validation, and test split CSVs.
        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, valid, test DataFrames
        """
        split_path = os.path.join(self.base_path, 'splits')
        train_path = os.path.join(split_path, 'train.csv')
        valid_path = os.path.join(split_path, 'valid.csv')
        test_path  = os.path.join(split_path, 'test.csv')

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train split not found at {train_path}")
        if not os.path.exists(valid_path):
            raise FileNotFoundError(f"Valid split not found at {valid_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test split not found at {test_path}")

        self.train_data = pd.read_csv(train_path)
        self.valid_data = pd.read_csv(valid_path)
        self.test_data  = pd.read_csv(test_path)

        return self.train_data, self.valid_data, self.test_data

    def load_all(self, print_shapes=True):
        """
        Load the full dataset and splits. Optionally prints their shapes.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
          (main, train, valid, test)
        """
        main = self.load_main()
        train, valid, test = self.load_splits()

        if print_shapes:
            print(f"🧬 {self.dataset_name}_main  ➡️ {main.shape}")
            print(f"⚗️ {self.dataset_name}_train ➡️ {train.shape}")
            print(f"🔬 {self.dataset_name}_valid ➡️ {valid.shape}")
            print(f"🧪 {self.dataset_name}_test  ➡️ {test.shape}")

        return main, train, valid, test


###############################################################################
# 2) EDAVisualizer: Single Class With All EDA/Visualization Functions
###############################################################################
class EDAVisualizer:

    @staticmethod
    def _get_emoji_for_dataset(name: str) -> str:
        """
        Internal helper: returns the appropriate emoji for 
        'main', 'train', 'valid', 'test' based on the dataset name.
        """

        name_lower = name.lower()
        if 'main' in name_lower:
            return '🧬'
        elif 'train' in name_lower:
            return '⚗️'
        elif 'valid' in name_lower:
            return '🔬'
        elif 'test' in name_lower:
            return '🧪'
        else:
            return '📦'  # fallback if not matched

    @staticmethod
    def show_dataset_info(loader_or_dfs, dataset_names=None):
        """
        Print shape, columns, missing values, and head for one or more DataFrames.

        Usage:
        1) Pass a DatasetLoader instance (with data already loaded). By default, it shows
           info for main_data, train_data, valid_data, test_data.
           Example:
             loader = DatasetLoader('AMES')
             loader.load_all()
             EDAVisualizer.show_dataset_info(loader)
        2) Optionally pass a list or dict of DataFrames directly, or a subset of dataset names.

        Parameters:
        - loader_or_dfs: Can be:
            (a) a DatasetLoader instance, or
            (b) a dict or list of DataFrames
        - dataset_names (list or None): If using a DatasetLoader, specify which
          of ['main', 'train', 'valid', 'test'] you want to show. If None,
          shows all that exist.
        """
        # CASE 1: If user passed a DatasetLoader
        if hasattr(loader_or_dfs, 'main_data'):
            loader = loader_or_dfs
            all_datasets = {
                'main' : loader.main_data,
                'train': loader.train_data,
                'valid': loader.valid_data,
                'test' : loader.test_data
            }

            # If user doesn't pass dataset_names, show them all
            if not dataset_names:
                dataset_names = ['main','train','valid','test']

            for name in dataset_names:
                df = all_datasets.get(name, None)
                if df is None:
                    print(f"No data found for '{name}' in this loader.")
                else:
                    EDAVisualizer._print_dataset_info(
                        df, f"{loader.dataset_name}_{name}"
                    )

        else:
            # CASE 2: If user passed a dict or list of DataFrames
            if isinstance(loader_or_dfs, dict):
                # If it's a dict: e.g. {'train': train_df, 'test': test_df}
                for name, df in loader_or_dfs.items():
                    EDAVisualizer._print_dataset_info(df, name)

            elif isinstance(loader_or_dfs, list):
                # If it's a list, optionally we might have dataset_names in parallel
                if dataset_names and len(dataset_names) == len(loader_or_dfs):
                    for name, df in zip(dataset_names, loader_or_dfs):
                        EDAVisualizer._print_dataset_info(df, name)
                else:
                    # If no names, just enumerate
                    for idx, df in enumerate(loader_or_dfs):
                        EDAVisualizer._print_dataset_info(df, f"Dataset_{idx+1}")

            else:
                print("Invalid type for loader_or_dfs. Expecting DatasetLoader, dict, or list.")

    @staticmethod
    def _print_dataset_info(df, name="Dataset"):
        """
        Helper function that actually prints out the info for a single DataFrame.
        """
        emoji = EDAVisualizer._get_emoji_for_dataset(name)
        print(f"{emoji} {name} Info")
        print("-" * 40)
        print(f"Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nPreview:")
        display(df.head())
        print("\n")

    @staticmethod
    def plot_label_distribution(df, target_col='Y', title='Target Distribution', labels=None):
        """
        Print class counts and percentages, and plot a bar chart of the target distribution.
        """
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

    @staticmethod
    def smiles_to_mol(smiles):
        """
        Convert a SMILES string to an RDKit Mol object.
        Returns None if the SMILES is invalid.
        """
        try:
            return Chem.MolFromSmiles(smiles)
        except:
            return None

    @staticmethod
    def draw_molecules(smiles_list, legends=None, n_per_row=5, mols_per_row=5, sub_img_size=(200, 200)):
        """
        Draw RDKit molecules from a list of SMILES strings.
        """
        mols = [EDAVisualizer.smiles_to_mol(sm) for sm in smiles_list]
        return Draw.MolsToGridImage(
            mols, 
            molsPerRow=n_per_row, 
            subImgSize=sub_img_size,
            legends=legends if legends else ["" for _ in mols]
        )

    @staticmethod
    def draw_samples_by_class(df, smiles_col='Drug', label_col='Y', n=4):
        """
        Show sample molecules for each class in the dataset.
        """
        unique_labels = df[label_col].unique()
        for label in unique_labels:
            subset = df[df[label_col] == label].sample(n=n, random_state=42)
            smiles = subset[smiles_col].tolist()
            print(f"🧪 {n} molecules for class {label}")
            display(
                EDAVisualizer.draw_molecules(
                    smiles, 
                    legends=[f"{label}_{i}" for i in range(len(smiles))]
                )
            )

    @staticmethod
    def compare_label_distributions(dfs, df_names, target_col='Y'):
        """
        Compare label distributions across multiple DataFrames in a single bar plot.
        """
        combined = []
        for df, name in zip(dfs, df_names):
            temp = df[[target_col]].copy()
            temp['Dataset'] = name
            combined.append(temp)
        combined_df = pd.concat(combined)

        counts = combined_df.groupby(['Dataset', target_col]).size().unstack(fill_value=0)
        counts.plot(kind='bar', figsize=(8, 5))
        plt.title(f"Comparison of {target_col} distribution")
        plt.xlabel("Dataset")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.legend(title=target_col)
        plt.show()

        print("\nLabel distribution percentages:")
        print(counts.apply(lambda x: (x / x.sum()) * 100, axis=1).round(2))

    @staticmethod
    def compare_numeric_distribution(dfs, df_names, col, bins=30):
        """
        Compare a numeric column's distribution across multiple DataFrames (e.g., train vs. test).
        """
        plt.figure(figsize=(8, 6))
        for df, label in zip(dfs, df_names):
            sns.histplot(df[col].dropna(), kde=True, bins=bins, label=label, alpha=0.5)
        plt.title(f"Distribution of {col} across datasets")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend()
        plt.show()

    @staticmethod
    def compare_boxplots(dfs, df_names, col):
        """
        Compare boxplots of a numeric column across multiple DataFrames side-by-side.
        """
        combined = []
        for df, name in zip(dfs, df_names):
            temp = df[[col]].copy()
            temp['Dataset'] = name
            combined.append(temp)
        combined_df = pd.concat(combined)

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Dataset', y=col, data=combined_df)
        plt.title(f"Boxplot of {col} across multiple datasets")
        plt.show()

    @staticmethod
    def compare_correlation_heatmaps(dfs, df_names, cols=None):
        """
        Display side-by-side correlation heatmaps for each dataset in dfs.
        """
        n = len(dfs)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5), squeeze=False)

        for i, (df, name) in enumerate(zip(dfs, df_names)):
            subset = df.select_dtypes(include='number') if cols is None else df[cols]
            corr = subset.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0, i])
            axes[0, i].set_title(f"{name} Correlation Heatmap")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_unique_smiles(dfs, df_names, smiles_col='Drug', title='Unique Drug Counts'):
        """
        Compare and plot the count of unique SMILES (or compounds) in each dataset.
        Also prints the absolute counts for each dataset.
        """
        unique_counts = []
        for df in dfs:
            unique_smiles = df[smiles_col].nunique()
            unique_counts.append(unique_smiles)

        print("Unique SMILES counts per dataset:")
        for name, count in zip(df_names, unique_counts):
            print(f"{name}: {count}")

        plt.figure(figsize=(6,4))
        sns.barplot(x=df_names, y=unique_counts)
        plt.title(title)
        plt.xlabel("Dataset")
        plt.ylabel("Unique SMILES")
        plt.show()


    @staticmethod
    def add_molecular_descriptors(dfs=None, loader=None, smiles_col='Drug', names=None, inplace=True):
        """
        Compute multiple RDKit descriptors (MolWt, LogP, TPSA, HBD/HBA, Rotatable Bonds, Ring Counts)
        for one or more DataFrames.

        Parameters:
        - dfs (pd.DataFrame or list of pd.DataFrame): Optional list of DataFrames.
        - loader (DatasetLoader): If provided (and dfs is None), applies to all loaded datasets.
        - smiles_col (str): Column name containing SMILES strings (default: 'Drug').
        - names (list of str): Optional names for printing progress.
        - inplace (bool): If False, returns new DataFrames (only when dfs is passed).

        Returns:
        - List of DataFrames (only when inplace=False)
        """
        if dfs is None:
            if loader:
                dfs = [loader.main_data, loader.train_data, loader.valid_data, loader.test_data]
                names = ['Main', 'Train', 'Validation', 'Test']
            else:
                print("⚠️ Provide either dfs or loader.")
                return
        elif isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
            names = names or ['Dataset']
        elif isinstance(dfs, list):
            names = names or [f"Dataset_{i+1}" for i in range(len(dfs))]
        else:
            print("⚠️ Invalid type for dfs.")
            return

        processed_dfs = []

        for df, name in zip(dfs, names):
            print(f"🧬 Adding molecular descriptors to: {name}")

            if not inplace:
                df = df.copy()

            molwts = []
            logps = []
            tpsas = []
            hb_donors = []
            hb_acceptors = []
            rot_bonds = []
            ring_counts = []
            arom_rings = []

            for sm in df[smiles_col]:
                mol = Chem.MolFromSmiles(sm)
                if mol:
                    molwts.append(Descriptors.MolWt(mol))
                    logps.append(Descriptors.MolLogP(mol))
                    tpsas.append(Descriptors.TPSA(mol))
                    hb_donors.append(Descriptors.NumHDonors(mol))
                    hb_acceptors.append(Descriptors.NumHAcceptors(mol))
                    rot_bonds.append(Descriptors.NumRotatableBonds(mol))
                    ring_counts.append(rdMolDescriptors.CalcNumRings(mol))
                    arom_rings.append(rdMolDescriptors.CalcNumAromaticRings(mol))
                else:
                    molwts.append(None)
                    logps.append(None)
                    tpsas.append(None)
                    hb_donors.append(None)
                    hb_acceptors.append(None)
                    rot_bonds.append(None)
                    ring_counts.append(None)
                    arom_rings.append(None)

            df['MW'] = molwts
            df['LogP'] = logps
            df['TPSA'] = tpsas
            df['HBD'] = hb_donors
            df['HBA'] = hb_acceptors
            df['RotBonds'] = rot_bonds
            df['RingCount'] = ring_counts
            df['AromaticRings'] = arom_rings

            if not inplace:
                processed_dfs.append(df)

        if not inplace:
            return processed_dfs


    # @staticmethod
    # def check_smiles_length(df, smiles_col='Drug'):
    #     """
    #     Compute and visualize the distribution of SMILES string lengths.
    #     Adds a 'smiles_length' column to the DataFrame.

    #     Parameters:
    #     - df: pandas DataFrame containing SMILES strings
    #     - smiles_col: the column name containing SMILES (default 'Drug')
    #     """
    #     df['smiles_length'] = df[smiles_col].apply(len)

    #     df['smiles_length'].hist(bins=30)
    #     plt.title("SMILES String Length Distribution")
    #     plt.xlabel("Length")
    #     plt.ylabel("Count")
    #     plt.grid(axis='y', linestyle='--', alpha=0.7)
    #     plt.show()

    #     print("📏 SMILES Length Stats:")
    #     display(df['smiles_length'].describe())


    @staticmethod
    def check_smiles_length(dfs=None, loader=None, smiles_col='Drug', names=None):
        """
        Analyze SMILES length distribution across one or more datasets.

        Parameters:
        - dfs (pd.DataFrame or list of pd.DataFrame): Dataset(s) to analyze.
        - loader (DatasetLoader): If provided and dfs is None, defaults to main dataset only.
        - smiles_col (str): Column with SMILES strings.
        - names (list of str): Names for datasets (used for plots/prints).
        """
        if dfs is None:
            if loader:
                dfs = [loader.main_data]
                names = ['Main']
            else:
                print("⚠️ Provide either dfs or loader.")
                return
        elif isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
            names = names or ['Dataset']
        elif isinstance(dfs, list):
            names = names or [f"Dataset_{i+1}" for i in range(len(dfs))]
        else:
            print("⚠️ Invalid type for dfs.")
            return

        for df, name in zip(dfs, names):
            df['smiles_length'] = df[smiles_col].apply(len)
            print(f"📏 SMILES Length Stats for {name}:")
            display(df['smiles_length'].describe())

            df['smiles_length'].hist(bins=30)
            plt.title(f"SMILES Length Distribution - {name}")
            plt.xlabel("Length")
            plt.ylabel("Count")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()



    # @staticmethod
    # def check_missing_duplicates(df, smiles_col='Drug'):
    #     """
    #     Check for missing values and duplicates in the dataset.

    #     Parameters:
    #     - df: pandas DataFrame
    #     - smiles_col: column used to identify duplicate SMILES (default 'Drug')
    #     """
    #     print(f"🔁 Duplicate Rows: {df.duplicated().sum()}")
    #     print(f"🔁 Duplicate {smiles_col}: {df[smiles_col].duplicated().sum()}")
    #     print("🕳️ Null Values:")
    #     display(df.isnull().sum())


    @staticmethod
    def check_missing_duplicates(dfs=None, loader=None, smiles_col='Drug', names=None):
        """
        Check for missing values and duplicates in one or more datasets.

        Parameters:
        - dfs (pd.DataFrame or list of pd.DataFrame): Datasets to check.
        - loader (DatasetLoader): Optional fallback to use loader datasets.
        - smiles_col (str): Column for duplicate SMILES check.
        - names (list of str): Optional list of names for datasets.
        """
        if dfs is None:
            if loader is None:
                print("⚠️ Provide either dfs or a loader.")
                return
            dfs = [loader.main_data, loader.train_data, loader.valid_data, loader.test_data]
            names = ['Main', 'Train', 'Validation', 'Test']
        elif isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
            names = names or ['Dataset']
        elif isinstance(dfs, list):
            names = names or [f"Dataset_{i+1}" for i in range(len(dfs))]
        else:
            print("⚠️ Invalid input type for `dfs`.")
            return

        for df, name in zip(dfs, names):
            print(f"🔍 Duplicate & Missing Summary: {name}")
            print(f"🔁 Duplicate Rows: {df.duplicated().sum()}")
            print(f"🔁 Duplicate {smiles_col}: {df[smiles_col].duplicated().sum()}")
            print("🕳️ Null Values:")
            display(df.isnull().sum())
            print("\n")


    # @staticmethod
    # def check_molecular_validity(df, smiles_col='Drug'):
    #     """
    #     Check if SMILES strings are valid RDKit molecules.

    #     Parameters:
    #     - df: pandas DataFrame
    #     - smiles_col: column name with SMILES strings
    #     """
    #     df['is_valid_mol'] = df[smiles_col].apply(lambda x: EDAVisualizer.smiles_to_mol(x) is not None)
    #     print(df['is_valid_mol'].value_counts())
    #     print(f"❗ Invalid molecules: {(~df['is_valid_mol']).mean() * 100:.2f}%")

    @staticmethod
    def check_molecular_validity(dfs=None, loader=None, smiles_col='Drug', names=None):
        """
        Check if SMILES strings are valid RDKit molecules.

        Parameters:
        - dfs (pd.DataFrame or list of pd.DataFrame): One or more DataFrames to check.
        - loader (DatasetLoader): Used to fallback to main/train/valid/test splits if dfs is None.
        - smiles_col (str): Column name with SMILES strings (default: 'Drug')
        - names (list of str): Optional names for the datasets (for clearer printout)
        """
        # Case 1: Nothing passed — fallback to loader
        if dfs is None:
            if loader is None:
                print("⚠️ Please provide a DataFrame, list of DataFrames, or a loader.")
                return
            dfs = [loader.main_data, loader.train_data, loader.valid_data, loader.test_data]
            names = ['Main', 'Train', 'Validation', 'Test']
        elif isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
            names = names or ['Dataset']
        elif isinstance(dfs, list):
            names = names or [f"Dataset_{i+1}" for i in range(len(dfs))]
        else:
            print("⚠️ Invalid input type for `dfs`.")
            return

        for df, name in zip(dfs, names):
            df['is_valid_mol'] = df[smiles_col].apply(lambda x: EDAVisualizer.smiles_to_mol(x) is not None)
            validity_counts = df['is_valid_mol'].value_counts()
            print(f"🧪 Validity for {name}:")
            print(validity_counts)
            print(f"❗ Invalid molecules in {name}: {(~df['is_valid_mol']).mean() * 100:.2f}%\n")

        
    @staticmethod
    def compare_smiles_length(dfs=None, dataset_names=None, loader=None, smiles_col='Drug'):
        """
        Compare SMILES string length distributions across multiple DataFrames.

        Parameters:
        - dfs (list of pd.DataFrame): Custom list of datasets to compare (optional).
        - dataset_names (list of str): Optional names for the splits. If not provided, auto-assigned.
        - loader (DatasetLoader): If provided and dfs is None, will default to train/valid/test from loader.
        - smiles_col (str): Column containing SMILES strings (default: 'Drug')
        """
        # Default behavior: use loader’s splits if no custom dfs provided
        if dfs is None:
            if loader:
                dfs = [loader.train_data, loader.valid_data, loader.test_data]
                dataset_names = ['Train', 'Validation', 'Test']
            else:
                print("⚠️ Please provide either a list of DataFrames or a DatasetLoader instance.")
                return

        if dataset_names is None:
            dataset_names = [f"Dataset_{i+1}" for i in range(len(dfs))]

        # Add smiles_length and split labels
        for name, df in zip(dataset_names, dfs):
            df['smiles_length'] = df[smiles_col].apply(len)
            df['split'] = name

        combined_df = pd.concat(dfs)

        sns.histplot(
            data=combined_df,
            x='smiles_length',
            hue='split',
            bins=30,
            kde=True,
            palette='Set2'
        )
        plt.title("SMILES Length Distribution by Split")
        plt.xlabel("SMILES Length")
        plt.ylabel("Count")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()




###############################################################################
# 3) SMARTS pattern definitions and utility
###############################################################################

SMARTS_PATTERNS = {
    'Aromatic Amine': '[NX3;H2,H1;!$(NC=O)]',
    'Nitro Group': '[$([NX3](=O)=O)]',
    'Hydroxyl Group': '[OX2H]',
    'Carboxylic Acid': 'C(=O)[OH]',
    'Sulfonamide': 'S(=O)(=O)N',
    'Halogen': '[F,Cl,Br,I]',
    'Aromatic Ring': 'a1aaaaa1',
    'Alkene': 'C=C',
    'Alkyne': 'C#C'
}

class SMARTSPatternAnalyzer:
    """
    Detect and visualize presence of key SMARTS substructures in molecules.
    """

    def __init__(self, smarts_dict=None):
        self.patterns = smarts_dict or SMARTS_PATTERNS
        self.compiled = {name: Chem.MolFromSmarts(smarts) for name, smarts in self.patterns.items()}

    def analyze(self, df, smiles_col='Drug'):
        """
        Add boolean columns to DataFrame indicating presence of each SMARTS pattern.

        Parameters:
        - df: pd.DataFrame
        - smiles_col: column with SMILES strings

        Returns:
        - pd.DataFrame with new SMARTS flag columns
        """
        df = df.copy()
        for name, pattern in self.compiled.items():
            df[f'smarts_{name}'] = df[smiles_col].apply(
                lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(pattern) if Chem.MolFromSmiles(x) else False
            )
        return df
    

    def summarize_patterns(self, df, label_col='Y'):
        """
        Print and optionally plot SMARTS pattern frequency by class.
        """
        pattern_cols = [col for col in df.columns if col.startswith('smarts_')]
        print("✨ SMARTS Substructure Presence by Class\n")

        summary = {}
        for col in pattern_cols:
            counts = df.groupby(label_col)[col].sum()
            summary[col] = counts

        summary_df = pd.DataFrame(summary).T
        summary_df.columns = [f"Class {c}" for c in summary_df.columns]

        # ✨ Rename rows to remove 'smarts_' prefix for cleaner display
        summary_df.index = summary_df.index.str.replace('smarts_', '')

        # Display table
        display(summary_df.sort_values(by=summary_df.columns[0], ascending=False))

        # Plot
        summary_df.plot(kind='bar', figsize=(10, 6))
        plt.title("SMARTS Pattern Occurrence by Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
