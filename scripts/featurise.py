import os
import argparse
import pandas as pd
import subprocess
import time
import logging
import tempfile
from pathlib import Path
from ersilia import ErsiliaModel


class MolecularFeaturiser:
    def __init__(self, dataset_name, model_id, use_python_api=True, auto_serve=False):
        self.dataset_name = dataset_name
        self.model_id = model_id
        self.use_python_api = use_python_api
        self.auto_serve = auto_serve

        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / dataset_name / "splits"

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.model = None

        if auto_serve:
            try:
                self.logger.info(f"[AUTO] Trying to serve model {model_id} via Python SDK...")
                self.model = ErsiliaModel(model=self.model_id)
                self.model.serve()
                self.use_python_api = True
                self.logger.info("✅ Model initialized using Python SDK.\n")
            except Exception as e:
                self.logger.warning("⚠️ Failed to initialize Python SDK. Switching to CLI mode...")
                self.logger.warning(f"🔍 Reason: {e}")
                self.use_python_api = False
                self.model = None
                self.logger.info("✅ CLI mode fallback activated.\n")
        elif self.use_python_api:
            self.logger.info(f"[API] Initializing Ersilia model: {self.model_id}")
            self.model = ErsiliaModel(model=self.model_id)
            self.model.serve()
            self.logger.info("✅ Model initialized Succesfully")

    def run_ersilia_featurisation(self, input_path, output_path):
        """
        Run Ersilia fingerprinting on a SMILES CSV file.
        """
        try:
            if self.use_python_api:
                self.model.run(input=input_path, output=output_path)
            else:
                command = [
                    "ersilia", "run",
                    "-i", input_path,
                    "-o", output_path
                ]
                subprocess.run(command, check=True)
        except Exception as e:
            self.logger.error(f"Featurisation error for {input_path}", exc_info=True)
            raise

    def featurise_smiles_column(self, df, smiles_col, prefix):
        """
        Featurise a given SMILES column from the dataframe.
        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".csv", delete=False) as temp_input:
            input_path = temp_input.name
            df[[smiles_col]].rename(columns={smiles_col: "SMILES"}).to_csv(input_path, index=False)

        temp_output = input_path.replace(".csv", "_output.csv")

        try:
            self.run_ersilia_featurisation(input_path, temp_output)
            features_df = pd.read_csv(temp_output)
            features_df.drop(columns=["input"], errors='ignore', inplace=True)
            features_df = features_df.add_prefix(f"{prefix}_feat_")
            return features_df
        finally:
            for f in [input_path, temp_output]:
                if os.path.exists(f):
                    os.remove(f)
                    self.logger.info(f"Deleted temporary file: {f}")

    def featurise_split(self, split, return_df=False):
        """
        Featurise a given split (train/valid/test) for the dataset.
        """
        input_file = self.data_dir / f"{split}.csv"
        output_file = self.data_dir / f"{split}_{self.model_id}_features.csv"

        if not input_file.exists():
            self.logger.warning(f"[!] {input_file} does not exist. Skipping.")
            return None

        self.logger.info(f"[+] Featurizing {input_file} -> {output_file}")

        start_time = time.time()
        try:
            df = pd.read_csv(input_file)
            df_final = df.copy()

            if "Drug1" in df.columns and "Drug2" in df.columns:
                feat1 = self.featurise_smiles_column(df, "Drug1", "Drug1")
                feat2 = self.featurise_smiles_column(df, "Drug2", "Drug2")
                df_final = pd.concat([df_final, feat1, feat2], axis=1)
                df_final.drop(columns=["Drug1", "Drug2"], inplace=True)
            elif any(col in df.columns for col in ["SMILES", "input", "Drug"]):
                smiles_col = next((col for col in ["SMILES", "input", "Drug"] if col in df.columns), None)
                feat = self.featurise_smiles_column(df, smiles_col, "Mol")
                df_final = pd.concat([df_final, feat], axis=1)
                df_final.drop(columns=[smiles_col], inplace=True)
            else:
                raise ValueError("No valid SMILES or Drug columns found for featurization.")

            # Try to merge labels from dataset-level CSV
            dataset_file = self.project_root / "data" / self.dataset_name / f"{self.dataset_name}.csv"
            if dataset_file.exists():
                self.logger.info(f"Attempting to merge labels from {dataset_file}")
                try:
                    label_df = pd.read_csv(dataset_file)
                    if "Drug" in label_df.columns:
                        df_final = df_final.merge(label_df[["Drug", "Y"]], left_on="input", right_on="Drug", how="inner")
                        df_final.drop(columns=["Drug"], inplace=True)
                except Exception as e:
                    self.logger.warning(f"Could not merge labels: {e}")

            if return_df:
                return df_final
            else:
                df_final.to_csv(output_file, index=False)
                elapsed_time = time.time() - start_time
                self.logger.info(f"Featurization completed in {elapsed_time:.2f} seconds for {split} split.")
                return output_file

        except Exception as e:
            self.logger.error(f"Featurization failed for {split} split.", exc_info=True)
            return None

    def featurise_all_splits(self):
        """
        Featurise all dataset splits.
        """
        for split in ["train", "valid", "test"]:
            self.featurise_split(split)

    def get_featurized_path(self, split: str) -> Path:
        """
        Returns the path to the featurized file for a given split.
        """
        return self.data_dir / f"{split}_{self.model_id}_features.csv"

    def __del__(self):
        if self.model and self.use_python_api:
            try:
                self.model.close()
                self.logger.info("Closed Ersilia model server.")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Featurise molecular datasets using Ersilia models.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. AMES, HIV)")
    parser.add_argument("--cli", action="store_true", help="Use CLI instead of Python API")
    parser.add_argument("--auto", action="store_true", help="Enable auto-serve fallback mode")
    args = parser.parse_args()

    featuriser = MolecularFeaturiser(
        dataset_name=args.dataset,
        model_id=args.model_id,
        use_python_api=not args.cli,
        auto_serve=args.auto
    )
    featuriser.featurise_all_splits()


if __name__ == "__main__":
    main()
