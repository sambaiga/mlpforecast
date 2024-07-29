from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import optuna
from optuna import Trial

from mlpforecast.forecaster.common import PytorchForecast
from mlpforecast.forecaster.utils import get_latest_checkpoint
from mlpforecast.model.deterministic import MLPForecastModel
from mlpforecast.net.layers import ACTIVATIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLPF")


class MLPForecast(PytorchForecast):
    def __init__(
        self,
        hparams: dict,
        exp_name: str = "Tanesco",
        file_name: str = None,
        seed: int = 42,
        root_dir: str = "../",
        trial=None,
        metric: str = "val_mae",
        max_epochs: int = 10,
        wandb: bool = False,
        model_type: str = "MLPF",
        gradient_clip_val: float = 10.0,
        rich_progress_bar: bool = True,
    ):
        """
        MLP Forecasting class for managing training, evaluation, and prediction.

        Args:
            hparams (dict): Hyperparameters for the MLP model.
            exp_name (str, optional): Experiment name. Defaults to "Tanesco".
            file_name (str, optional): Name of the file for logging and saving checkpoints. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            root_dir (str, optional): Root directory for the project. Defaults to "../".
            trial (optuna.trial, optional): Optuna trial object for hyperparameter optimization. Defaults to None.
            metric (str, optional): Metric to monitor during training. Defaults to "val_mae".
            max_epochs (int, optional): Maximum number of epochs for training. Defaults to 10.
            wandb (bool, optional): Whether to use Weights and Biases for logging. Defaults to False.
            model_type (str, optional): Type of the model. Defaults to "MLPF".
            gradient_clip_val (float, optional): Value for gradient clipping. Defaults to 10.0.
            rich_progress_bar (bool, optional): Whether to use rich progress bar. Defaults to True.
        """
        super().__init__(
            file_name=file_name,
            seed=seed,
            root_dir=root_dir,
            trial=trial,
            metric=metric,
            max_epochs=max_epochs,
            wandb=wandb,
            model_type=model_type,
            gradient_clip_val=gradient_clip_val,
            rich_progress_bar=rich_progress_bar,
        )
        self.hparams = hparams
        self.model = MLPForecastModel(**hparams)

    def load_checkpoint(self):
        """
        Load the latest checkpoint for the model.

        This method retrieves the path of the latest checkpoint and loads the model from it.
        """
        path_best_model = get_latest_checkpoint(self.checkpoints)
        self.model = MLPForecastModel.load_from_checkpoint(path_best_model)
        self.model.eval()

    def get_search_params(self, trial: Trial) -> dict:
        """
        Define the search space for hyperparameter optimization using Optuna.

        Args:
            trial: An Optuna trial object to suggest parameters.

        Returns
        -------
            dict: A dictionary containing suggested hyperparameters.
        """
        params = {}

        # Define integer hyperparameters
        params["embedding_size"] = trial.suggest_int("embedding_size", 8, 64, step=2)
        params["hidden_size"] = trial.suggest_int("hidden_size", 8, 512, step=2)
        params["num_layers"] = trial.suggest_int("num_layers", 1, 5)
        params["expansion_factor"] = trial.suggest_int("expansion_factor", 1, 4)

        # Define categorical hyperparameters
        params["embedding_type"] = trial.suggest_categorical(
            "embedding_type", [None, "PosEmb", "RotaryEmb", "CombinedEmb"]
        )
        params["combination_type"] = trial.suggest_categorical("combination_type", ["addition-comb", "weighted-comb"])
        params["residual"] = trial.suggest_categorical("residual", [True, False])
        params["activation_function"] = trial.suggest_categorical("activation_function", ACTIVATIONS)
        params["out_activation_function"] = trial.suggest_categorical("out_activation_function", ACTIVATIONS)

        # Define float hyperparameters
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.9, step=0.05)
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 1.0, log=True)

        return params

    def auto_tune(self, train_df, val_df, num_trial=10, reduction_factor=3, patience=2):
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
        """
        self.train_df = train_df
        self.validation_df = val_df

        def print_callback(study, trial):
            logging.info(f"""Trial No: {trial.number}, Current value: {trial.value}, Current params: {trial.params}""")
            logging.info(f"""Best value: {study.best_value}, Best params: {study.best_trial.params}""")

        def objective(trial):
            params = self.get_search_params(trial)

            self.hparams.update(params)
            model = MLPForecast(
                self.hparams,
                exp_name=f"{self.exp_name}",
                seed=42,
                trial=trial,
                rich_progress_bar=True,
                file_name=trial.number,
            )

            val_cost = model.fit(self.train_df, self.validation_df)
            return val_cost

        study_name = f"{self.exp_name}_{self.model_type}"
        storage_name = f"sqlite:///{study_name}.db"
        base_pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=reduction_factor
        )
        pruner = optuna.pruners.PatientPruner(base_pruner, patience=patience, min_delta=0.0)
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name=self.exp_name,
            storage=storage_name,
            load_if_exists=True,
        )
        study.optimize(
            objective,
            n_trials=num_trial,  # Default to 100 trials if not specified
            callbacks=[print_callback],
        )
        self.hparams.update(study.best_trial.params)
        np.save(f"{self.results_path}/best_params.npy", study.best_trial.params)

    def predict(self, test_df=None, covariate_df=None, daily_feature=True):
        """
        Perform prediction on the test DataFrame and return a DataFrame with ground truth and forecasted values.

        Args:
            test_df (pd.DataFrame): The test DataFrame containing the input features for prediction.
            daily_feature (bool): Flag indicating whether daily features are used in the model. Default is True.

        Returns
        -------
            pd.DataFrame: A DataFrame containing the ground truth and forecasted values, indexed by timestamp.
        """
        if test_df is not None:
            test_df = pd.concat([self.train_df, test_df], axis=0)
        test_df = test_df.sort_values(by=self.model.data_pipeline.date_column)

        time_stamp, ground_truth = self.get_ground_truth(test_df=test_df, 
                                                         daily_feature=daily_feature)

        # Inverse transform predictions
        scaler = self.model.data_pipeline.data_pipeline.named_steps["scaling"]
        target_scaler = scaler.named_transformers_["target_scaler"]

        pred=self.perform_prediction(test_df=test_df)

        N, T, C = pred["pred"].size()
        pred["pred"] = target_scaler.inverse_transform(pred["pred"].numpy().reshape(N * T, C))
        pred["pred"] = pred["pred"].reshape(N, T, C)

        # Evaluate point forecast
        self.metrics = self.evaluate_point_forecast(ground_truth, pred["pred"], time_stamp)
        self.metrics["test-time"] = self.test_walltime
        self.metrics["Model"] = self.model_type.upper()

        # Assert that the prediction and ground truth shapes are the same
        assert (
            pred["pred"].shape == ground_truth.shape
        ), "Shape mismatch: pred['pred'] and ground_truth must have the same shape."

        # Create results DataFrame
        results_df = self.create_results_df(
            time_stamp,
            ground_truth,
            pred["pred"],
            self.model.data_pipeline.target_series,
            self.model.data_pipeline.date_column,
        )
        results_df["Model"] = self.model_type.upper()

        return results_df

