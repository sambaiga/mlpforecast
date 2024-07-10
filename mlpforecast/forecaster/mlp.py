from __future__ import annotations
import torch
import pytorch_lightning as pl
import logging
import torchmetrics
from mlpforecast.model.point_forecast import MLPForecastModel
from mlpforecast.forecaster.common import PytorchForecast
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLPF")


class MLPForecast(PytorchForecast):
    def __init__(
        self,
        hparams:dict,
        exp_name="Tanesco",
        file_name: str = None,
        seed: int = 42,
        root_dir: str = "../",
        trial=None,
        metric="val_mae",
        max_epochs=10,
        wandb: bool = False,
        model_type: str = "MLPF",
        future_exogenous=True,
        scaler=None,
        target_scaler=None,
        gradient_clip_val=10,
        rich_progress_bar: bool = True,
    ):
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

        if hparams is None:
            hparams = {
                "data_pipeline":None,
                "target_series": ["NetLoad"],
                "unknown_features": [],
                "known_calender_features": [],
                "known_continuous_features": [],
                "embedding_size": 20,
                "embedding_type": None,
                "combination_type": "additional",
                "hidden_size": 256,
                "num_layers": 2,
                'expansion_factor':2,
                "residual":  False,
                "forecast_horizon": 48,
                "input_window_size": 96,
                "activation_function": "SiLU",
                "out_activation_function": "Identity",
                "dropout_rate": 0.25,
                "alpha": 0.25,
                "metric": "mae",
                "num_attention_heads": 4,
                "learning_rate":1e-3,
                "weight_decay":1e-6,
                "prob_decay_1":0.75,
                "prob_decay_2":0.9,
                "gamma":0.01,
                "max_epochs":10,
            }

        self.model = MLPForecastModel(**hparams)

    
