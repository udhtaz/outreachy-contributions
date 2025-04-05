import os
import time
import logging
import random

from collections import Counter
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt 

from pathlib import Path
from typing import List, Union
from scripts.featurise import MolecularFeaturiser
from collections import Counter

from pycaret.classification import (
    setup, compare_models, pull, predict_model,
    tune_model, save_model, load_model
)

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix
)

from sklearn.base import BaseEstimator


class ModelPreprocessor:
    """
    Loads and prepares featurized dataset splits for modeling.
    Automatically detects feature and target columns using one or more prefixes.
    Optionally triggers featurization only if the featurized files are missing.
    """

    def __init__(
        self,
        dataset_name: str,
        model_id: str,
        feature_prefix: Union[str, List[str]] = "Mol_feat_",
        auto_featurise: bool = True,
    ):
        self.dataset_name = dataset_name
        self.model_id = model_id
        self.feature_prefix = (
            [feature_prefix] if isinstance(feature_prefix, str) else feature_prefix
        )
        self.auto_featurise = auto_featurise

        # Logging setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = self.project_root / "data" / self.dataset_name / "splits"

        self.logger.info(f"🚦 Auto-featurise mode is {'ON' if self.auto_featurise else 'OFF'}")
        self.logger.info(f"📂 Checking for featurized files in: {self.data_dir.resolve()}")

        self._featuriser = None

        if self._needs_featurisation():
            if not auto_featurise:
                raise FileNotFoundError("Featurized files are missing and auto_featurise is False.")
            self.logger.info("📉 Featurized files missing — invoking featurisation.")
            self._ensure_featuriser().featurise_all_splits()
        else:
            self.logger.info("✅ All featurized files found. Skipping featurisation.")

    def _ensure_featuriser(self):
        if self._featuriser is None:
            self._featuriser = MolecularFeaturiser(self.dataset_name, self.model_id)
        return self._featuriser

    def _needs_featurisation(self):
        """
        Check whether featurized CSVs exist for all splits.
        """
        expected_files = [
            self.data_dir / f"{split}_{self.model_id}_features.csv" for split in ["train", "valid", "test"]
        ]
        for f in expected_files:
            if not f.exists():
                self.logger.warning(f"⚠️ Missing featurized file: {f.name}")
                return True
        return False

    def _load_split(self, split: str) -> pd.DataFrame:
        """
        Loads featurized CSV for the specified split without triggering model initialization.
        """
        path = self.data_dir / f"{split}_{self.model_id}_features.csv"
        self.logger.info(f"📄 Loading featurized split: {path.name}")
        return pd.read_csv(path)

    def _detect_columns(self, df: pd.DataFrame):
        """
        Detects feature columns using the given prefix(es), and identifies the target column.
        """
        feature_cols = [
            col for col in df.columns
            if any(col.startswith(prefix) for prefix in self.feature_prefix)
        ]
        if not feature_cols:
            raise ValueError(f"No columns found with prefix(es): {self.feature_prefix}")
        target_col = "Y" if "Y" in df.columns else df.columns[-1]
        return feature_cols, target_col

    def _log_summary(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        self.logger.info("✅ Preprocessing Complete\n")

        # Shape info
        self.logger.info("📊 Dataset Summary:")
        self.logger.info(f"   • X_train: {X_train.shape} =====> y_train: {y_train.shape}")
        self.logger.info(f"   • X_valid: {X_valid.shape} =====> y_valid: {y_valid.shape}")
        self.logger.info(f"   • X_test:  {X_test.shape} =====> y_test:  {y_test.shape}\n")

        # Target distribution (classification)
        self.logger.info("🏷️ Target Distribution (train):")
        dist = Counter(y_train)
        for label, count in dist.items():
            self.logger.info(f"   • {label} → {count}")


        # NaN check
        self.logger.info("🔍 Null Check:")
        for name, df in [("X_train", X_train), ("X_valid", X_valid), ("X_test", X_test)]:
            nulls = df.isnull().sum().sum()
            self.logger.info(f"   • {name}: {nulls} missing values")
        

        # Dtype balance
        self.logger.info("🧬 Feature Types (X_train):")
        dtype_counts = X_train.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            self.logger.info(f"   • {dtype} → {count} columns")


    def preprocess(self):
        """
        Loads all splits, extracts features and targets, and returns them in tuple form.
        """
        df_train = self._load_split("train")
        df_valid = self._load_split("valid")
        df_test = self._load_split("test")

        feature_cols, target_col = self._detect_columns(df_train)

        X_train = df_train[feature_cols]
        y_train = df_train[target_col]

        X_valid = df_valid[feature_cols]
        y_valid = df_valid[target_col]

        X_test = df_test[feature_cols]
        y_test = df_test[target_col]

        # 🧹 Drop non-numeric (e.g., object) columns'
        for split_name, df in zip(["X_train", "X_valid", "X_test"], [X_train, X_valid, X_test]):
            object_cols = df.select_dtypes(include="object").columns
            if len(object_cols) > 0:
                for col in object_cols:
                    self.logger.warning(f"🧹🗑️ Dropping column '{col}' from {split_name} because: (dtype=object)")
                df.drop(columns=object_cols, inplace=True)

        # Log summary after preprocessing
        self._log_summary(X_train, y_train, X_valid, y_valid, X_test, y_test)

        return X_train, y_train, X_valid, y_valid, X_test, y_test


class ModelTrainer:
    """
    A flexible trainer class for running classification models using PyCaret or custom sklearn-style models.

    Supports:
    - Automated comparison using PyCaret
    - Tuning with PyCaret's `tune_model`
    - Manual training with user-defined hyperparameters
    - Evaluation, confusion matrix, feature importance, and saving/loading models
    """

    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, experiment_name="default"):
        """
        Initialize the trainer with train, validation, and test sets.
        """
        self.experiment_name = experiment_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test

        self.best_model = None
        self.results = None
        self.custom_leaderboard = []

    def run_pycaret(self, target="Y"):
        """
        Run PyCaret setup and perform automatic model comparison.
        """
        print("⚙️ Initializing PyCaret...")
        combined_train = pd.concat([self.X_train, self.y_train], axis=1)

        setup(
            data=combined_train,
            target=target,
            session_id=42,
            experiment_name=self.experiment_name
        )

        print("⚖️ Comparing models...")
        self.best_model = compare_models()
        self.results = pull()

        print("✅ Best model selected:", self.best_model)
        return self.best_model

    def log_results(self):
        """
        Display the top models from PyCaret's leaderboard.
        """
        if self.results is not None:
            print("📋 PyCaret Leaderboard:")
            available_cols = self.results.columns.tolist()
            preferred_cols = ["Model", "Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "TT (Sec)"]
            display_cols = [col for col in preferred_cols if col in available_cols]
            display(self.results[display_cols].head())
        else:
            print("⚠️ No PyCaret results available. Run `run_pycaret()` first.")

    def evaluate_on_test(self):
        """
        Evaluate the best model on the test set using PyCaret's `predict_model`.
        """
        if self.best_model is None:
            raise RuntimeError("No model trained yet. Call run_pycaret() first.")

        combined_test = pd.concat([self.X_test, self.y_test], axis=1)
        predictions = predict_model(self.best_model, data=combined_test)
        return predictions


    def save_model(self, model: BaseEstimator = None, dataset_name: str = None, featurizer_id: str = None, framework: str = None):
        """
        Save a model to the models/{dataset}/{dataset}_{featurizer}_{framework}.pkl path.

        Parameters:
            model (BaseEstimator): The trained model to save. Defaults to self.best_model.
            dataset_name (str): Optional dataset name (e.g., "AMES"). If None, will infer from global variable `dataset`.
            featurizer_id (str): Optional featurizer ID. If None, will infer from global variable `model_id`.
            framework (str): Optional framework name. If None, will use model class name.
        """
        model = model or self.best_model
        if model is None:
            raise ValueError("No model available to save.")

        # Infer from global vars if not explicitly passed
        dataset_name = (dataset_name or globals().get("dataset", "unknown")).lower()
        featurizer_id = (featurizer_id or globals().get("model_id", "unknown")).lower()
        framework = (framework or model.__class__.__name__).lower()

        # Dynamically resolve project root based on notebook/script context
        notebook_root = Path.cwd()
        project_root = notebook_root.parent if notebook_root.name == "notebooks" else notebook_root
        model_dir = project_root / "models" / dataset_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Build filename and save path
        filename = f"{dataset_name}_{featurizer_id}_{framework}.pkl"
        filepath = model_dir / filename
        joblib.dump(model, filepath)

        print(f"💾 Model saved to: {filepath.resolve()}")


    def load(self, name="best_model"):
        """
        Load a model previously saved with PyCaret.
        """
        self.best_model = load_model(name)
        print(f"📦 Loaded model '{name}.pkl'")
        return self.best_model
    
    # def load_model_from_file(self, path: str) -> BaseEstimator:
    #     """
    #     Load a model from a specific .pkl file path using joblib.

    #     Parameters:
    #         path (str): Full path to the .pkl model file

    #     Returns:
    #         model (BaseEstimator): Loaded model instance
    #     """
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"❌ File not found: {path}")

    #     model = joblib.load(path)
    #     print(f"📦 Loaded model from: {path}")
    #     return model


    def load_model_from_file(self, model_info: dict, split: str = "valid", external_data_path: str = None):
        """
        Load a model and auto-attach its dataset context for evaluation.

        Parameters:
            model_info (dict): {
                "path": str,       # path to saved .pkl model
                "dataset": str,    # dataset name (e.g., "AMES")
                "model_id": str,   # featurizer used (e.g., "eos4wt0")
                "label": Optional[str]  # model label for reporting
            }
            split (str): Dataset split to use ("train", "valid", "test", or "external")
            external_data_path (str): Optional path to external CSV if split='external'

        Returns:
            dict: {
                "model": loaded model,
                "X": features from split,
                "y": labels from split,
                "label": model label,
                "dataset": dataset name,
                "model_id": model_id
            }
        """

        # === Extract info ===
        path = model_info["path"]
        dataset = model_info["dataset"]
        model_id = model_info["model_id"]
        label = model_info.get("label", Path(path).stem)

        # === Load model ===
        model = joblib.load(path)

        if split == "external":
            if not external_data_path:
                raise ValueError("External data path must be provided when split='external'")
            df = pd.read_csv(external_data_path)
            print(f"📄 Using EXTERNAL data from: {external_data_path}")
        else:
            assert split in ["train", "valid", "test"], f"Invalid split '{split}'"
            project_root = Path(__file__).resolve().parents[1] 
            split_file = project_root / "data" / dataset / "splits" / f"{split}_{model_id}_features.csv"
            if not split_file.exists():
                raise FileNotFoundError(f"Split file not found: {split_file}")
            df = pd.read_csv(split_file)
            print(f"📄 Using {split} data from: {split_file}")

        # === Extract features and target ===
        feature_cols = [col for col in df.columns if col.startswith("Mol_feat_")]
        X = df[feature_cols]
        y = df["Y"]

        print(f"📦 Loaded model from: {path}")

        return {
            "model": model,
            "X": X,
            "y": y,
            "label": label,
            "dataset": dataset,    
            "model_id": model_id    
        }


    def train_custom_model(self, model_cls, model_name=None, **kwargs):
        """
        Train any sklearn-style model with optional kwargs for hyperparameters.
        """
        model = model_cls(**kwargs)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_valid)
        proba = getattr(model, "predict_proba", lambda x: None)(self.X_valid)

        metrics = {
            "Model": model_name or model_cls.__name__,
            "Accuracy": accuracy_score(self.y_valid, preds),
            "F1": f1_score(self.y_valid, preds),
            "Precision": precision_score(self.y_valid, preds),
            "Recall": recall_score(self.y_valid, preds),
        }

        if proba is not None:
            try:
                metrics["AUC"] = roc_auc_score(self.y_valid, proba[:, 1])
            except:
                metrics["AUC"] = None

        self.custom_leaderboard.append(metrics)
        print(f"✅ Trained {metrics['Model']} | Accuracy: {metrics['Accuracy']:.3f}, F1: {metrics['F1']:.3f}")
        return model

    def view_custom_leaderboard(self):
        """
        Return a DataFrame summarizing all custom model metrics.
        """
        if not self.custom_leaderboard:
            print("ℹ️ No custom models trained yet.")
            return pd.DataFrame()
        return pd.DataFrame(self.custom_leaderboard).sort_values("F1", ascending=False)

    def tune_model_with_pycaret(self, model, n_iter=50, choose_better=True):
        """
        Tune a specific model using PyCaret's tune_model utility.
        """
        if not self.best_model:
            raise RuntimeError("Setup must be run before tuning.")
        print(f"🛠️ Tuning {model.__class__.__name__}...")
        tuned = tune_model(model, n_iter=n_iter, choose_better=choose_better)
        self.best_model = tuned
        return tuned

    def train_selected_model(self, model):
        """
        Train a provided sklearn-style model (already instantiated with custom parameters).
        """
        print(f"🔧 Training selected model: {model.__class__.__name__}")
        model.fit(self.X_train, self.y_train)
        self.best_model = model
        return model

    def evaluate_model(self, model=None):
        """
        Evaluate the given model (or the best model) on the validation set.
        """
        model = model or self.best_model
        start = time.time()

        val_preds = model.predict(self.X_valid)
        proba = getattr(model, "predict_proba", lambda x: None)(self.X_valid)
        val_proba = proba[:, 1] if proba is not None else None

        print("📊 Evaluation Metrics:")
        print(f"Accuracy: {accuracy_score(self.y_valid, val_preds):.4f}")
        if val_proba is not None:
            print(f"AUC: {roc_auc_score(self.y_valid, val_proba):.4f}")
        print(f"Recall: {recall_score(self.y_valid, val_preds):.4f}")
        print(f"Precision: {precision_score(self.y_valid, val_preds):.4f}")
        print(f"F1 Score: {f1_score(self.y_valid, val_preds):.4f}")
        print(f"Kappa: {cohen_kappa_score(self.y_valid, val_preds):.4f}")
        print(f"MCC: {matthews_corrcoef(self.y_valid, val_preds):.4f}")
        print(f"Time taken (seconds): {time.time() - start:.2f}")

    def plot_confusion_matrix(self, model=None):
        """
        Plot a confusion matrix for the given (or best) model on the validation set.
        """
        model = model or self.best_model
        preds = model.predict(self.X_valid)
        cm = confusion_matrix(self.y_valid, preds)
        sns.heatmap(cm, annot=True, fmt="g", cmap="Blues",
                    xticklabels=["False", "True"],
                    yticklabels=["False", "True"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("🌀 Confusion Matrix - Validation Set")
        plt.show()

    def plot_feature_importance(self, top_n=20):
        """
        Plot the top N feature importances (if supported by the model).
        """
        if not hasattr(self.best_model, "feature_importances_"):
            print("⚠️ Feature importances not available for this model.")
            return

        importances = self.best_model.feature_importances_
        features = self.X_train.columns

        importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
        top_features = importance_df.sort_values("Importance", ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_features, x="Importance", y="Feature")
        plt.title("🔥 Top Feature Importances")
        plt.tight_layout()
        plt.show()

    def compare_loaded_models_across_features(self, loaded_models: list, top_n_metrics: int = 7, use_preprocessor: bool = True):
        """
        Compare models that were trained on different feature sets and already loaded using `load_model_from_file`.

        Parameters:
            loaded_models (list): List of dictionaries returned by `load_model_from_file`
            top_n_metrics (int): Number of metrics to plot (sorted by average score)
            use_preprocessor (bool): Whether to clean and align the input X using ModelPreprocessor

        Returns:
            pd.DataFrame: Comparison table of metrics across models
        """

        all_metrics = []
        model_labels = []

        for model_dict in loaded_models:
            model = model_dict["model"]
            X = model_dict["X"]
            y = model_dict["y"]
            label = model_dict["label"]
            dataset = model_dict["dataset"]
            model_id = model_dict["model_id"]

            if use_preprocessor:
                try:
                    preprocessor = ModelPreprocessor(dataset, model_id)
                    X, y, _, _, _, _ = preprocessor.preprocess()
                except Exception as e:
                    print(f"⚠️ Failed to preprocess for {label}: {e}")

            # Align features with those used during training
            expected_features = getattr(model, 'feature_names_in_', X.columns)
            X = X.reindex(columns=expected_features, fill_value=0)

            # Evaluate
            preds = model.predict(X)
            proba = getattr(model, "predict_proba", lambda x: None)(X)
            auc_input = proba[:, 1] if proba is not None else None

            metrics = {
                "Accuracy": accuracy_score(y, preds),
                "F1 Score": f1_score(y, preds),
                "Precision": precision_score(y, preds),
                "Recall": recall_score(y, preds),
                "Kappa": cohen_kappa_score(y, preds),
                "MCC": matthews_corrcoef(y, preds),
            }

            if auc_input is not None:
                try:
                    metrics["AUC"] = roc_auc_score(y, auc_input)
                except:
                    metrics["AUC"] = None

            all_metrics.append(metrics)
            model_labels.append(label)

        # Convert to DataFrame
        comparison_df = pd.DataFrame(all_metrics, index=model_labels).T.reset_index()
        comparison_df.rename(columns={"index": "Metric"}, inplace=True)

        display(comparison_df)

        # === Plot
        melted = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
        avg_scores = melted.groupby("Metric")["Score"].mean().sort_values(ascending=False)
        top_metrics = avg_scores.head(top_n_metrics).index.tolist()

        melted_top = melted[melted["Metric"].isin(top_metrics)]

        plt.figure(figsize=(12, 6))  # wider to accommodate legend
        ax = sns.barplot(data=melted_top, x="Metric", y="Score", hue="Model")
        plt.title("📊 Cross-Featurizer Model Performance Comparison")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

        # ✅ Legend: top-right corner, vertical
        ax.legend(
            title="Model",
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            frameon=False
        )

        plt.tight_layout()
        plt.show()

        # return comparison_df


    def compare_models_on_metrics(self, models, model_names=None, dataset="valid", external_data_path=None, external_data_name=None, plot=True):
        """
        Compare multiple models side by side on selected dataset (validation, test, or external).

        Parameters:
            models (List): List of trained sklearn-compatible models
            model_names (List): Optional list of names to label each model
            dataset (str): "valid" (default), "test", or "external"
            external_data_path (str): Path to CSV file if using external data
            external_data_name (str): Optional name to show in titles if evaluating on external data
            plot (bool): Whether to generate a barplot

        Returns:
            pd.DataFrame: Comparison of metrics across the models
        """
        # === Select Dataset ===
        if dataset == "valid":
            X, y = self.X_valid, self.y_valid
            source = "Validation"
        elif dataset == "test":
            X, y = self.X_test, self.y_test
            source = "Test"
        elif dataset == "external":
            if not external_data_path:
                raise ValueError("Please provide path to external data when using dataset='external'")
            df = pd.read_csv(external_data_path)
            feature_cols = self.X_train.columns.tolist()
            X = df[feature_cols]
            y = df["Y"]
            source = external_data_name or "External"
        else:
            raise ValueError("Invalid dataset. Choose from 'valid', 'test', or 'external'.")

        # === Generate model names if not provided ===
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(models))]

        # === Evaluation logic ===
        def evaluate(model):
            preds = model.predict(X)
            proba = getattr(model, "predict_proba", lambda x: None)(X)
            auc = roc_auc_score(y, proba[:, 1]) if proba is not None else None

            return {
                "Accuracy": accuracy_score(y, preds),
                "F1 Score": f1_score(y, preds),
                "Precision": precision_score(y, preds),
                "Recall": recall_score(y, preds),
                "AUC": auc,
                "Kappa": cohen_kappa_score(y, preds),
                "MCC": matthews_corrcoef(y, preds),
            }

        # === Evaluate all models ===
        scores = [evaluate(m) for m in models]
        comparison_df = pd.DataFrame(scores, index=model_names).T.reset_index()
        comparison_df.rename(columns={"index": "Metric"}, inplace=True)

        # === Display table ===
        display(comparison_df)

        # === Plot if required ===
        if plot:
            melted_df = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
            plt.figure(figsize=(10, 6))
            # Randomly sample distinct colors from a large palette
            unique_models = melted_df["Model"].unique()
            color_count = len(unique_models)
            palette = random.sample(sns.color_palette("husl", 10), k=color_count)
            sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette=palette)
            # sns.barplot(data=melted_df, x="Metric", y="Score", hue="Model", palette=sns.color_palette("Set2"))
            plt.title(f"📊 Model Comparison on {source} Set")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()

        # return comparison_df
