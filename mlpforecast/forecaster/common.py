from __future__ import annotations

import logging
from pathlib import Path
from timeit import default_timer

import pandas as pd
import torch

from mlpforecast.forecaster.utils import format_target, get_latest_checkpoint
from mlpforecast.metrics.deterministic import evaluate_point_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Forecaster")
import pytorch_lightning as pl
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

from mlpforecast.data.loader import TimeseriesDataModule


class PytorchForecast:
    """
    PytorchForecast class for setting up and managing the training process of a PyTorch model using PyTorch Lightning.

    Attributes:
        exp_name (str): The name of the experiment.
        file_name (str): The name of the file to save the logs and model checkpoints.
        seed (int): The seed for random number generation to ensure reproducibility.
        root_dir (str): The root directory for saving logs and checkpoints.
        trial (optuna.trial.Trial): Optuna trial for hyperparameter optimization.
        metric (str): The metric to monitor for early stopping and model checkpointing.
        max_epochs (int): The maximum number of training epochs.
        wandb (bool): Flag to use Wandb logger instead of TensorBoard logger.
        rich_progress_bar (bool): Flag to use rich progress bar for training visualization.
    """

    def __init__(
        self,
        exp_name="Tanesco",
        file_name=None,
        seed=42,
        root_dir="../",
        trial=None,
        metric="val_mae",
        max_epochs=10,
        wandb=False,
        model_type="MLPF",
        rich_progress_bar=False,
        gradient_clip_val=10,
    ):
        """
        Initializes the PytorchForecast class with the given parameters.

        Parameters:
            exp_name (str): The name of the experiment. Defaults to "Tanesco".
            file_name (str): The name of the file to save the logs and model checkpoints. Defaults to None.
            seed (int): The seed for random number generation. Defaults to 42.
            root_dir (str): The root directory for saving logs and checkpoints. Defaults to "../".
            trial (optuna.trial.Trial): Optuna trial for hyperparameter optimization. Defaults to None.
            metric (str): The metric to monitor for early stopping and model checkpointing. Defaults to "val_mae".
            max_epochs (int): The maximum number of training epochs. Defaults to 10.
            wandb (bool): Flag to use Wandb logger instead of TensorBoard logger. Defaults to False.
            rich_progress_bar (bool): Flag to use rich progress bar for training visualization. Defaults to True.
        """
        super().__init__()

        self.exp_name = exp_name
        self.file_name = file_name
        self.seed = seed
        self.root_dir = root_dir
        self.trial = trial
        self.metric = metric
        self.max_epochs = max_epochs
        self.wandb = wandb
        self.model_type = model_type
        self.rich_progress_bar = rich_progress_bar
        self.model_type = "default_model"  # Update with actual model type
        self.model = None
        self.datamodule = None
        self.gradient_clip_val = gradient_clip_val
        self._create_folder()

    def _create_folder(self):
        """
        Creates directories for storing the results, logs, and figures generated by the model.
        """
        self.results_path = Path(
            f"{self.root_dir}/results/{self.exp_name}/{self.model_type}/"
        )

        self.logs = Path(f"{self.root_dir}/logs/{self.exp_name}/{self.model_type}/")
        self.figures = Path(
            f"{self.root_dir}/figures/{self.exp_name}/{self.model_type}/"
        )
        self.figures.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        if self.file_name is not None:
            self.checkpoints = Path(
                f"{self.root_dir}/checkpoints/{self.exp_name}/{self.model_type}/{self.file_name}"
            )
        else:
            self.checkpoints = Path(
                f"{self.root_dir}/checkpoints/{self.exp_name}/{self.model_type}"
            )
        self.checkpoints.mkdir(parents=True, exist_ok=True)

    def _set_up_trainer(self):
        callback = []
        pl.seed_everything(self.seed, workers=True)
        if self.trial is not None:
            self.logger = True
            early_stopping = PyTorchLightningPruningCallback(
                self.trial, monitor=self.metric
            )
            callback.append(early_stopping)
        else:
            early_stopping = EarlyStopping(
                monitor=self.metric,
                min_delta=0.0,
                patience=int(self.max_epochs * 0.25),
                verbose=False,
                mode="min",
                check_on_train_epoch_end=True,
            )
            callback.append(early_stopping)
            if not self.wandb:
                self.logger = loggers.TensorBoardLogger(
                    save_dir=self.logs,
                    version=(self.file_name if self.file_name is not None else 0),
                )
            else:
                self.logger = loggers.WandbLogger(
                    save_dir=self.logs,
                    name=self.file_name,
                    project=self.exp_name,
                    log_model="all",
                )
                # self.logger.watch(self.model)
                # log gradients and model topology

            self.checkpoints.mkdir(parents=True, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.checkpoints,
                monitor=self.metric,
                mode="min",
                save_top_k=1,
                filename="{epoch:02d}",
            )
            callback.append(checkpoint_callback)
            lr_logger = LearningRateMonitor()
            callback.append(lr_logger)

        if self.rich_progress_bar:
            progress_bar = RichProgressBar(
                theme=RichProgressBarTheme(
                    description="green_yellow",
                    progress_bar="green1",
                    progress_bar_finished="green1",
                    progress_bar_pulse="#6206E0",
                    batch_progress="green_yellow",
                    time="grey82",
                    processing_speed="grey82",
                    metrics="grey82",
                )
            )
        else:
            progress_bar = TQDMProgressBar()

        callback.append(progress_bar)

        self.trainer = pl.Trainer(
            logger=self.logger,
            gradient_clip_val=self.gradient_clip_val,
            max_epochs=self.max_epochs,
            callbacks=callback,
            accelerator="auto",
            devices=1,
        )

    def fit(
        self,
        train_df,
        val_df=None,
        train_ratio=0.80,
        drop_last=False,
        num_worker=1,
        batch_size=64,
        pin_memory=True,
    ):
        if self.model is None:
            raise ValueError("Model instance is empty")
        self.model.checkpoint_path = self.checkpoints
        self.model.data_pipeline.fit(train_df.copy())

        if val_df is None and train_ratio > 0.0:
            train_df, val_df = (
                train_df.iloc[: int(train_ratio * len(train_df))],
                train_df.iloc[int(train_ratio * len(train_df)) :],
            )
            self.metric = f"val_{self.model.hparams['metric']}"
            val_feature, val_target = self.model.data_pipeline.transform(val_df)
        else:
            val_feature, val_target = None, None
            self.metric = f"train_{self.model.hparams['metric']}"

        train_feature, train_target = self.model.data_pipeline.transform(train_df)
        self.datamodule = TimeseriesDataModule(
            train_inputs=train_feature,
            train_targets=train_target,
            val_inputs=val_feature,
            val_targets=val_target,
            drop_last=drop_last,
            num_worker=num_worker,
            batch_size=batch_size,
            pin_memory=pin_memory,
        )

        self._set_up_trainer()
        start_time = default_timer()
        logger.info("""---------------Training started ---------------------------""")
        if self.trial is not None:
            self.trainer.fit(
                self.model,
                self.datamodule.train_dataloader(),
                self.datamodule.val_dataloader(),
            )

            self.train_walltime = default_timer() - start_time
            logging.info(f"training complete after {self.train_walltime / 60} minutes")
            return self.trainer.callback_metrics[self.metric].item()

        else:
            self.trainer.fit(
                self.model,
                self.datamodule.train_dataloader(),
                self.datamodule.val_dataloader(),
                ckpt_path=get_latest_checkpoint(self.checkpoints),
            )
            self.train_walltime = default_timer() - start_time
            logging.info(
                f"""training complete after {self.train_walltime / 60} minutes"""
            )
            return self.train_walltime
    def load_and_prepare_data(self, test_df, daily_feature):
        """Loads the checkpoint and prepares the ground truth data."""
        self.load_checkpoint()
        self.model.data_pipeline.daily_features = daily_feature

        # Prepare ground truth data
        ground_truth = test_df.iloc[self.model.data_pipeline.max_data_drop :].copy()
        ground_truth[self.model.data_pipeline.date_column] = pd.to_numeric(
            ground_truth[self.model.data_pipeline.date_column]
        )
        return ground_truth

    def perform_prediction(self, test_df):
        """Performs the model prediction."""
        features, _ = self.model.data_pipeline.transform(test_df.copy())
        features = torch.FloatTensor(features.copy())
        self.model.to(features.device)
        self.model.eval()
        return self.model.forecast(features)

    def create_results_df(
        self, time_stamp, ground_truth, predictions, target_series, date_column
    ):
        """Creates a DataFrame with the timestamp index and populates it with ground truth and forecasted values."""
        results_df = pd.DataFrame(index=pd.to_datetime(time_stamp.flatten(), unit="ns"))
        results_df.index.name = date_column

        for target_idx, target_name in enumerate(target_series):
            results_df[target_name] = ground_truth[:, :, target_idx].flatten()
            results_df[f"{target_name}_forecast"] = predictions[
                :, :, target_idx
            ].flatten()

        return results_df

    def evaluate_point_forecast(self, ground_truth, pred, time_stamp):
        """Evaluates the point forecast."""
        return evaluate_point_forecast(
            {
                "true": ground_truth,
                "loc": pred,
                "index": time_stamp,
                "targets": self.model.data_pipeline.target_series,
            }
        )

    def predict(self, test_df, daily_feature=True):
        """
        Perform prediction on the test DataFrame and return a DataFrame with ground truth and forecasted values.

        Args:
            test_df (pd.DataFrame): The test DataFrame containing the input features for prediction.
            daily_feature (bool): Flag indicating whether daily features are used in the model. Default is True.

        Returns:
            pd.DataFrame: A DataFrame containing the ground truth and forecasted values, indexed by timestamp.
        """
        ground_truth = self.load_and_prepare_data(test_df, daily_feature)
        time_stamp = ground_truth[[self.model.data_pipeline.date_column]].values
        ground_truth = ground_truth[self.model.data_pipeline.target_series].values

        time_stamp = format_target(
            time_stamp,
            self.model.hparams["input_window_size"],
            self.model.hparams["forecast_horizon"],
            daily_feature=self.model.data_pipeline.daily_features,
        )
        ground_truth = format_target(
            ground_truth,
            self.model.hparams["input_window_size"],
            self.model.hparams["forecast_horizon"],
            daily_feature=self.model.data_pipeline.daily_features,
        )

        start_time = default_timer()
        pred = self.perform_prediction(test_df)
        self.test_walltime = default_timer() - start_time

        # Inverse transform predictions
        scaler = self.model.data_pipeline.data_pipeline.named_steps["scaling"]
        target_scaler = scaler.named_transformers_["target_scaler"]

        N, T, C = pred["pred"].size()
        pred["pred"] = target_scaler.inverse_transform(
            pred["pred"].numpy().reshape(N * T, C)
        )
        pred["pred"] = pred["pred"].reshape(N, T, C)

        # Evaluate point forecast
        self.metrics = self.evaluate_point_forecast(
            ground_truth, pred["pred"], time_stamp
        )
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
