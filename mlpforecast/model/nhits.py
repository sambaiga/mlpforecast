import logging

import torch
from optuna import Trial
from mlpforecast.model.base_model import BaseForecastModel
from mlpforecast.net.nhits import NHITS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NHITS")

class NHITSForecastModel(BaseForecastModel):
    """
    MLP Forecast Model for time series point forecasting.

    Attributes:
        n_out (int): Number of output series.
        n_channels (int): Number of input channels.
        model (object): Model object.
        hparams (dict): Hyperparameters for the model.
    """

    def __init__(
        self,
        data_pipeline=None,
        target_series: list[str] | str = ["NetLoad"],
        unknown_features: list[str] = [],
        known_calendar_features: list[str] = [],
        known_continuous_features: list[str] = [],
        input_window_size: int = 96,
        forecast_horizon: int = 48,
        activation_function: str = "ReLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        stack_types: list = ["identity", "identity", "identity"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        n_pool_kernel_size: list = [2, 2, 1],
        n_freq_downsample: list = [4, 2, 1],
        pooling_mode: str = "MaxPool1d",
        interpolation_mode: str = "linear",
        decompose_forecast:bool=False,
        metric: str = "mae",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        prob_decay_1: float = 0.75,
        prob_decay_2: float = 0.9,
        gamma: float = 0.01,
        max_epochs: int = 10
    ):
        super().__init__(data_pipeline, metric)
        assert len(target_series) > 0, "target_series should not be empty."

        self.n_out = len(target_series)
        n_unknown = len(unknown_features) + self.n_out
        n_covariates = len(known_calendar_features) + len(known_continuous_features)
        self.n_channels = n_unknown + n_covariates
        self.model = NHITS(
                n_target_series=self.n_out,
                n_unknown_features=len(unknown_features),
                n_known_calendar_features=len(known_calendar_features),
                n_known_continuous_features=len(known_continuous_features),
                forecast_horizon=forecast_horizon,
                input_window_size=input_window_size,
                activation=activation_function,
                out_activation_function=out_activation_function,
                dropout_rate=dropout_rate,
                alpha=alpha,
                stack_types=stack_types,
                n_blocks = n_blocks,
                mlp_units = mlp_units,
                n_pool_kernel_size=n_pool_kernel_size,
                n_freq_downsample=n_freq_downsample,
                pooling_mode=pooling_mode,
                interpolation_mode=interpolation_mode,
                decompose_forecast=decompose_forecast)
        
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
        params["alpha"] = trial.suggest_float("alpha", 1e-3, 1.0, log=True)
        params["interpolation_mode"]=trial.suggest_categorical("interpolation_mode", ["linear", "nearest", "cubic"])
        params["activation_function"]= trial.suggest_categorical("activation",
                [
                    "ReLU",
                    "Softplus",
                    "Tanh",
                    "SELU",
                    "LeakyReLU",
                    "PReLU",
                    "Sigmoid",
                ]
            )

        params["mlp_units"]=trial.suggest_categorical("mlp_units",
                [
                    3 * [[16, 16]],
                    3 * [[32, 32]],
                    3 * [[64, 64]],
                    3 * [[128, 128]],
                    3 * [[256, 256]],
                    3 * [[512, 512]],
                ]

            )

        params["pooling_mode"]=trial.suggest_categorical(

                "pooling_mode", ["MaxPool1d", "AvgPool1d"]

            )

        params["dropout_rate"]=trial.suggest_float(

                "dropout_rate", 0.0, 0.9, step=0.1

            )
        params["n_pool_kernel_size"]=trial.suggest_categorical(
                "n_pool_kernel_size",
                [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]])

        params["n_freq_downsample"]=trial.suggest_categorical(
                "n_freq_downsample",
                [

                    [168, 24, 1],

                    [24, 12, 1],

                    [180, 60, 1],

                    [60, 8, 1],

                    [40, 20, 1],

                    [1, 1, 1],

                    [168, 24, 1],

                ],

            )

        return params