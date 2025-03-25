import argparse
import os
import sys
import shutil
from tdc.single_pred import ADME, Tox, HTS, QM, Yields, Epitope, Develop, CRISPROutcome

# Mapping of category categories to their corresponding TDC classes and datasets
category_DATASET_MAPPING = {
    'adme': {
        'class': ADME,
        'datasets': [
            'Caco2_Wang', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma',
            'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'HydrationFreeEnergy_FreeSolv',
            'BBB_Martins', 'PPBR_AZ', 'VDss_Lombardo',
            'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP_1A2_Veith', 'CYP2C9_Veith',
            'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels',
            'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ'
        ]
    },
    'tox': {
        'class': Tox,
        'datasets': [
            'hERG', 'hERG_Karim', 'AMES', 'DILI', 'Skin Reaction', 'LD50_Zhu',
            'Carcinogens_Lagunin', 'ToxCast', 'Tox21', 'ClinTox'
        ]
    },
    'hts': {
        'class': HTS,
        'datasets': [
            'HIV', 'SARSCoV2_3CLPro_Diamond', 'SARSCoV2_Vitro_Touret',
            'Orexin1 Receptor', 'M1 Muscarinic Receptor Agonists', 'M1 Muscarinic Receptor Antagonists',
            'Potassium Ion Channel Kir2.1', 'KCNQ2 Potassium Channel', 'Cav3 T-type Calcium Channels',
            'Choline Transporter', 'Serine/Threonine Kinase 33', 'Tyrosyl-DNA Phosphodiesterase'
        ]
    },
    'qm': {
        'class': QM,
        'datasets': ['QM7b', 'QM8', 'QM9']
    },
    'yields': {
        'class': Yields,
        'datasets': ['Buchwald-Hartwig', 'USPTO']
    },
    'epitope': {
        'class': Epitope,
        'datasets': ['IEDB_Jespersen', 'PDB_Jespersen']
    },
    'develop': {
        'class': Develop,
        'datasets': ['TAP', 'SAbDab_Chen']
    },
    'crisproutcome': {
        'class': CRISPROutcome,
        'datasets': ['Leenay']
    }
}


class TDCDatasetDownloader:
    """
    A modular downloader class for fetching datasets and data splits
    from the Therapeutics Data Commons (TDC) single_pred categories.
    """

    def __init__(self, category, name, save_dir=None):
        self.category = category
        self.name = name
        self.save_dir = os.path.abspath(save_dir or os.path.join(os.path.dirname(__file__), '..', 'data'))

        self._validate_inputs()
        self.category_class = category_DATASET_MAPPING[category]['class']
        self.data = self.category_class(name=self.name)

        self.dataset_dir = os.path.join(self.save_dir, self.name)
        self.split_dir = os.path.join(self.dataset_dir, 'splits')
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.split_dir, exist_ok=True)

        self._download_data()
        self._download_split()
        self._relocate_tab_file()

    def _validate_inputs(self):
        if self.category not in category_DATASET_MAPPING:
            raise ValueError(f"category '{self.category}' is invalid. Choose from: {list(category_DATASET_MAPPING.keys())}")
        if self.name not in category_DATASET_MAPPING[self.category]['datasets']:
            raise ValueError(f"Dataset '{self.name}' is not available under category '{self.category}'.")

    def _download_data(self):
        df = self.data.get_data()
        save_path = os.path.join(self.dataset_dir, f"{self.name}.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ Dataset '{self.name}' saved to '{save_path}'")

    def _download_split(self):
        splits = self.data.get_split()
        for split_name, split_df in splits.items():
            split_path = os.path.join(self.split_dir, f"{split_name}.csv")
            split_df.to_csv(split_path, index=False)
            print(f"✅ {split_name} split saved to '{split_path}'")

    def _relocate_tab_file(self):
        tab_filename = f"{self.name.lower()}.tab"
        source_dirs = [
            os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'data'),
            os.getcwd()
        ]

        for src_dir in source_dirs:
            source_path = os.path.abspath(os.path.join(src_dir, tab_filename))
            if os.path.exists(source_path):
                target_path = os.path.join(self.dataset_dir, tab_filename)
                os.rename(source_path, target_path)
                # print(f"📦 Moved raw file from '{source_path}' to '{target_path}'")

                data_dir = os.path.dirname(source_path)
                try:
                    if len(os.listdir(data_dir)) == 0:
                        os.rmdir(data_dir)
                        # print(f"🧹 Removed empty folder: {data_dir}")
                except Exception as e:
                    print(f"⚠️ Could not remove folder: {data_dir} — {e}")
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TDC dataset and its splits.")
    parser.add_argument('--category', type=str, required=True, help="category category (e.g., 'tox', 'hts').")
    parser.add_argument('--name', type=str, required=True, help="Dataset name (e.g., 'AMES', 'HIV').")
    parser.add_argument('--save_dir', type=str, default='data', help="Directory to save dataset and splits.")
    args = parser.parse_args()

    downloader = TDCDatasetDownloader(args.category, args.name, args.save_dir)
