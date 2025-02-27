from mlpforecast.model.base_regressor import BaseRegressor
from sklearn.multioutput import MultiOutputRegressor
from  xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
supported_regressor=['CATBOOST', 'XGBOOST', 'LightGBM', 'LinearReg']

class CatBoostModel(BaseRegressor):
    """
    A regression model class using CatBoost as the underlying model.

    Inherits from:
        BaseRegressor: The base regressor class providing core methods.

    Args:
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        verbose (Optional[Union[int, bool]]): Controls verbosity of CatBoost. Defaults to 0.
        **kwargs: Additional keyword arguments to pass to the CatBoostRegressor.

    Attributes:
        model_name (str): Name of the model ("CATBOOST").
        model_params (dict): Parameters for the CatBoostRegressor.
        model (MultiOutputRegressor): A CatBoostRegressor wrapped for multi-output regression.
    """

    def __init__(
        self, 
        data_pipeline=None,
        model_params:dict={},
        seed: Optional[int] = None,
        verbose: Optional[Union[int, bool]] = 0):
        self.model_name = "CATBOOST"
        task_type="GPU" if torch.cuda.is_available() else "CPU"
        model_params["random_state"] = seed 
        model_params["task_type"] = task_type
        model_params["verbose"] = verbose 
        # Suppress writing CatBoost info files unless specified
        if "allow_writing_files" not in model_params:
            model_params["allow_writing_files"] = False
        self.model_params = model_params
        print(self.model_params)
        self.model_params.pop('seed', None)
        self.model = MultiOutputRegressor(CatBoostRegressor(**self.model_params))
        super().__init__(data_pipeline=data_pipeline)

    def get_search_params(self, trial):
        params = {}
        params["iterations"] = trial.suggest_int("iterations", 20, 1000)
        params["learning_rate"] = trial.suggest_float(

            "learning_rate", 1e-3, 1e-1, log=True)
        params["depth"] = trial.suggest_int("depth", 1, 12)
        params["min_data_in_leaf"] = trial.suggest_int(
            "min_data_in_leaf", 1, 100)
        return params



class XGBModel(BaseRegressor):
    """
    A regression model class using XGBoost as the underlying model.

    Inherits from:
        BaseRegressor: The base regressor class providing core methods.

    Args:
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the XGBRegressor.

    Attributes:
        model_name (str): Name of the model ('XGBOOST').
        model_params (dict): Parameters for the XGBRegressor.
        model (XGBRegressor): The XGBRegressor model instance.
    """

    def __init__(
        self, 
        data_pipeline=None,
        model_params:dict={},
        seed: Optional[int] = None):
        self.model_name = 'XGBOOST'
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model_params["random_state"] = seed 
        model_params["device"]=device
        self.model_params = model_params
        
        
        self.model = XGBRegressor(**self.model_params)
        super().__init__(data_pipeline=data_pipeline)

    def get_search_params(self, trial):

        # https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/

        params = {}

        params["iterations"] = trial.suggest_int("iterations", 20, 1000)

        params["objective"] = "reg:squarederror"

        params["learning_rate"] = trial.suggest_float(

            "learning_rate", 1e-3, 1e-1, log=True

        )

        params["subsample"] = trial.suggest_float("subsample", 0.05, 1.0)
        params["eta"] = trial.suggest_float("eta", 0.1, 1.0)
        params["gamma"] = trial.suggest_int("gamma", 0, 10)

        params["colsample_bytree"] = trial.suggest_float(

            "colsample_bytree", 0.05, 1.0

        )

        params["min_child_weight"] = trial.suggest_int(

            "min_child_weight", 1, 20

        )
        params["n_estimators"] = trial.suggest_int(
            "n_estimators", 10, 500)
        params["max_depth"] = trial.suggest_int("max_depth", 1, 10)
        
    
        return params


class LightGBMModel(BaseRegressor):
    """
    A regression model class using LightGBM as the underlying model.

    Inherits from:
        BaseRegressor: The base regressor class providing core methods.

    Args:
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the LGBMRegressor.

    Attributes:
        model_name (str): Name of the model ('LightGBM').
        model_params (dict): Parameters for the LGBMRegressor.
        model (MultiOutputRegressor): A LightGBM model wrapped for multi-output regression.
    """

    def __init__(
        self, 
        data_pipeline=None,
        seed: Optional[int] = None,
        model_params:dict={}):
        self.model_name = 'LightGBM'
        model_params["random_state"] = seed 
        self.model_params = model_params
        self.model = MultiOutputRegressor(LGBMRegressor(**self.model_params))
        super().__init__(data_pipeline=data_pipeline)

    def get_search_params(self, trial):

        # https://forecastegy.com/posts/how-to-use-optuna-to-tune-lightgbm-hyperparameters/

        params = {}

        params["iterations"] = trial.suggest_int("iterations", 20, 1000)

        params["objective"] = "regression"

        params["metric"] = "rmse"

        params["linear_tree"] = trial.suggest_categorical(

            "linear_tree", [True, False]

        )

        params["bagging_freq"] = 1

        params["learning_rate"] = trial.suggest_float(

            "learning_rate", 1e-3, 1e-1, log=True

        )

        params["num_leaves"] = trial.suggest_int("num_leaves", 2, 2**10)

        params["subsample"] = trial.suggest_float("subsample", 0.05, 1.0)

        params["colsample_bytree"] = trial.suggest_float(

            "colsample_bytree", 0.05, 1.0

        )

        params["min_data_in_leaf"] = (

            trial.suggest_int("min_data_in_leaf", 1, 100),

        )

        params["n_estimators"] = trial.suggest_int(

            "n_estimators", 10, 200, log=True

        )

        params["max_depth"] = trial.suggest_int("max_depth", 1, 10)

        return params



class LinearRegressionModel(BaseRegressor):
    """
    A regression model class using Linear Regression as the underlying model.

    Inherits from:
        BaseRegressor: The base regressor class providing core methods.

    Args:
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the LinearRegression model.

    Attributes:
        model_name (str): Name of the model ('LinearReg').
        model_params (dict): Parameters for the LinearRegression model.
        model (MultiOutputRegressor): A LinearRegression model wrapped for multi-output regression.
    """

    def __init__(
        self, 
        data_pipeline=None,
        model_params:dict={},
        seed: Optional[int] = None):
        self.model_name = 'LinearReg'
        self.model_params = model_params
        self.model = MultiOutputRegressor(LinearRegression(**self.model_params))
        super().__init__(data_pipeline=data_pipeline)