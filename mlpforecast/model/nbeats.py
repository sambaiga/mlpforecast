import logging
import torch
from optuna import Trial
from mlpforecast.model.base_model import BaseForecastModel
from mlpforecast.net.nbeats import NBEATS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NBEATS")

class NBEATSForecastModel(BaseForecastModel):
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
        activation_function: str = "ReLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        n_harmonics: int = 2,
        n_polynomials: int = 2,
        decompose_forecast:bool=False,
        stack_types: list = ["identity", "trend", "seasonality"],
        n_blocks: list = [1, 1, 1],
        mlp_units: list = 3 * [[512, 512]],
        shared_weights: bool = False,
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
        self.model = NBEATS(
                n_target_series=self.n_out,
                n_unknown_features=len(unknown_features),
                n_known_calendar_features=len(known_calendar_features),
                n_known_continuous_features=len(known_continuous_features),
                forecast_horizon=forecast_horizon,
                input_window_size=input_window_size,
                activation=activation_function,
                out_activation_function=out_activation_function,
                dropout_rate=dropout_rate if dropout_rate==0.0 else 0.0,
                alpha=alpha,
                n_harmonics= n_harmonics,
                n_polynomials = n_polynomials,
                stack_types=stack_types,
                n_blocks=n_blocks,
                mlp_units=mlp_units,
                shared_weights=shared_weights,
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
        #params["activation_function"]= trial.suggest_categorical("activation",
        #       ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid']
        #    )

        params["n_harmonics"]=trial.suggest_int('n_harmonics', 1, 5)
        params["n_polynomials"]=trial.suggest_int('n_polynomials', 1, 5)
        params["shared_weights"]=trial.suggest_categorical('shared_weights',[False, True])
        # params["stack_types"]=trial.suggest_categorical(
        #         "stack_types", [['identity', 'trend', 'seasonality'], 
        #                         ['identity', 'trend'], 
        #                         ['identity', 'seasonality'],
        #                         ['trend', 'seasonality']])
        #mlp_unit = trial.suggest_categorical("mlp_units", [32, 64, 128, 256, 512])
        # params["mlp_units"] = [[mlp_unit]]*3

        # params["n_blocks"]=[trial.suggest_int('n_blocks', 1, 4)]*3

        return params