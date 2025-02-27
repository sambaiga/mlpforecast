import logging

import torch
from optuna.trial import Trial
from mlpforecast.model.base_model import BaseForecastModel
from mlpforecast.net.fedformer import FEDformer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEDformer")

class FeDformerForecastModel(BaseForecastModel):
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
        hidden_size: int = 64,
        activation_function: str = "gelu",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        version: str = "Fourier",
        modes: int = 64,
        mode_select: str = "random",
        n_head: int = 8,
        decoder_input_size_multiplier:float=0.5,
        conv_hidden_size: int = 32,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        MovingAvg_window: int = 25,
        metric: str = "mae",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        prob_decay_1: float = 0.75,
        prob_decay_2: float = 0.9,
        gamma: float = 0.01,
        max_epochs: int = 10,
        lambda_lasso:float=1e-3,
        gam_layer:bool=False,
    ):
        super().__init__(data_pipeline, metric)
        assert len(target_series) > 0, "target_series should not be empty."

        self.n_out = len(target_series)
        n_unknown = len(unknown_features) + self.n_out
        n_covariates = len(known_calendar_features) + len(known_continuous_features)
        self.n_channels = n_unknown + n_covariates
        self.model = FEDformer(
                n_target_series=self.n_out,
                n_unknown_features=len(unknown_features),
                n_known_calendar_features=len(known_calendar_features),
                n_known_continuous_features=len(known_continuous_features),
                hidden_size=hidden_size,
                forecast_horizon=forecast_horizon,
                input_window_size=input_window_size,
                activation=activation_function,
                out_activation_function=out_activation_function,
                dropout=dropout_rate,
                alpha=alpha,
                version=version,
                modes=modes,
                mode_select=mode_select,
                n_head= n_head,
                decoder_input_size_multiplier=decoder_input_size_multiplier,
                conv_hidden_size = conv_hidden_size,
                encoder_layers =  encoder_layers,
                decoder_layers = decoder_layers,
                MovingAvg_window = MovingAvg_window)
        
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
        params["encoder_layers"] = trial.suggest_int("encoder_layers", 1, 4)
        params["decoder_layers"] = trial.suggest_int("decoder_layers", 1, 4)
        params["activation_function"]= trial.suggest_categorical("activation",
                ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid']
            )
        # params["mode_select"]= trial.suggest_categorical("mode_select",["random", 'fixed'])
        params["hidden_size"] = trial.suggest_int("hidden_size", 32, 512, step=8)
        # params["n_head"] = trial.suggest_int("n_head", 4, 16, step=4)
        # params["modes"] = trial.suggest_int("modes", 16, 64, step=4)
        # params["MovingAvg_window"] = trial.suggest_int("MovingAvg_window", 4, 32)
        params["conv_hidden_size"] = trial.suggest_int("conv_hidden_size", 32, 512, step=8)
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.05)
        
        return params