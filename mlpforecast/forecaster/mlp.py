from __future__ import annotations

import logging

from mlpforecast.forecaster.common import PytorchForecast
from mlpforecast.forecaster.utils import get_latest_checkpoint
from mlpforecast.model.deterministic import MLPForecastModel

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

        self.model = MLPForecastModel(**hparams)

    def load_checkpoint(self):
        """
        Load the latest checkpoint for the model.

        This method retrieves the path of the latest checkpoint and loads the model from it.
        """
        path_best_model = get_latest_checkpoint(self.checkpoints)
        self.model = MLPForecastModel.load_from_checkpoint(path_best_model)
        self.model.eval()

