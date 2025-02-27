from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import optuna
import copy
from optuna import Trial
from mlpforecast.forecaster.utils import  get_latest_checkpoint
from mlpforecast.forecaster.common_regressor import BasicForecast
from mlpforecast.model.regressor_model import supported_regressor
from mlpforecast.model.regressor_model import (CatBoostModel, 
                                               XGBModel, 
                                               LinearRegressionModel, 
                                               LightGBMModel)
from mlpforecast.forecaster.utils import get_latest_checkpoint


    
class RegressorForecast(BasicForecast):
    """
    MLP Forecasting class for managing training, evaluation, and prediction.

    Attributes:
        hparams (dict): Hyperparameters for the MLP model.
        model (MLPForecastModel): PyTorch model.
        train_df (pd.DataFrame): Training DataFrame.
        validation_df (pd.DataFrame): Validation DataFrame.
    """
    
    def __init__(
        self,
        hparams:dict=None,
        exp_name: str = "Tanesco",
        file_name: str = None,
        seed: int = 42,
        root_dir: str = "../",
        trial=None,
        metric: str = "val_mae",
        model_type:str='default'
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
            exp_name=exp_name,
            file_name=file_name,
            seed=seed,
            root_dir=root_dir,
            trial=trial,
            metric=metric,
            model_type=model_type
        )
       
        self.model = None
        hparams.update({'seed':seed})
        self.hparams = hparams
       

    def objective_function(self, trial):
        pass
    
    def auto_tune(self, train_df, val_df, num_trial=10, reduction_factor=3, patience=2):
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            train_df (pd.DataFrame): Training DataFrame.
            val_df (pd.DataFrame): Validation DataFrame.
            num_trial (int, optional): Number of trials for hyperparameter optimization. Defaults to 10.
            reduction_factor (int, optional): Reduction factor for Hyperband pruner. Defaults to 3.
            patience (int, optional): Patience for the Patient pruner. Defaults to
        """
        self.train_df = train_df
        self.validation_df = val_df

        def print_callback(study, trial):
            logging.info(f"""Trial No: {trial.number}, Current value: {trial.value}, Current params: {trial.params}""")
            logging.info(f"""Best value: {study.best_value}, Best params: {study.best_trial.params}""")

        

        
        study_name = f"{self.exp_name}_{self.model_type}"
        file_path=f"{self.logs}/{study_name}.log"
        lock_obj = optuna.storages.journal.JournalFileOpenLock(file_path)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(file_path, lock_obj=lock_obj)
        )
        base_pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource="auto", reduction_factor=reduction_factor
        )
        pruner = optuna.pruners.PatientPruner(base_pruner, patience=patience, min_delta=0.0)
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(
            self.objective_function,
            n_trials=num_trial,  # Default to 100 trials if not specified
            callbacks=[print_callback],
        )
        self.hparams.update(study.best_trial.params)
        np.save(f"{self.results_path}/best_params.npy", study.best_trial.params)
        
class CatBoostForecast(RegressorForecast):
    def __init__(
        self,
        hparams:dict=None,
        exp_name: str = "Tanesco",
        file_name: str = None,
        seed: int = 42,
        root_dir: str = "../",
        trial=None,
        metric: str = "val_mae"
    ):
        super().__init__(
            hparams=hparams,
            exp_name=exp_name,
            file_name=file_name,
            seed=seed,
            root_dir=root_dir,
            trial=trial,
            metric=metric,
            model_type="CATBOOST"
            )
        
        params=copy.deepcopy(self.hparams)
        self.model=CatBoostModel(data_pipeline=params.pop('data_pipeline'),
                            model_params=params)
        
    
    def objective_function(self, trial):
            
        params = self.model.get_search_params(trial)

        self.hparams.update(params)
        model = CatBoostForecast(
                self.hparams,
                exp_name=f"{self.exp_name}",
                seed=42,
                trial=trial,
                file_name=trial.number,
            )

        val_cost = model.fit(self.train_df, self.validation_df)
        return val_cost
        
        
class XGBoostForecast(RegressorForecast):
    def __init__(
        self,
        hparams:dict=None,
        exp_name: str = "Tanesco",
        file_name: str = None,
        seed: int = 42,
        root_dir: str = "../",
        trial=None,
        metric: str = "val_mae"
    ):
        super().__init__(
            hparams=hparams,
            exp_name=exp_name,
            file_name=file_name,
            seed=seed,
            root_dir=root_dir,
            trial=trial,
            metric=metric,
            model_type="XGBOOST"
            )
        
        params=copy.deepcopy(self.hparams)
        self.model=XGBModel(data_pipeline=params.pop('data_pipeline'),
                            model_params=params)
    
    def objective_function(self, trial):
            
        params = self.model.get_search_params(trial)
        self.hparams.update(params)
        model = XGBoostForecast(
                self.hparams,
                exp_name=f"{self.exp_name}",
                seed=42,
                trial=trial,
                file_name=trial.number,
            )

        val_cost = model.fit(self.train_df, self.validation_df)
        return val_cost
        
        
class LightGBMForecast(RegressorForecast):
    def __init__(
        self,
        hparams:dict=None,
        exp_name: str = "Tanesco",
        file_name: str = None,
        seed: int = 42,
        root_dir: str = "../",
        trial=None,
        metric: str = "val_mae"
    ):
        super().__init__(
            hparams=hparams,
            exp_name=exp_name,
            file_name=file_name,
            seed=seed,
            root_dir=root_dir,
            trial=trial,
            metric=metric,
            model_type='LightGBM'
            )
        params=copy.deepcopy(self.hparams)
        params.pop('seed')
        self.model=LightGBMModel(data_pipeline=params.pop('data_pipeline'),
                            model_params=params)
      
    
    def objective_function(self, trial):
            
        params = self.model.get_search_params(trial)
        self.hparams.update(params)
        model = LightGBMForecast(
                self.hparams,
                exp_name=f"{self.exp_name}",
                seed=42,
                trial=trial,
                file_name=trial.number,
            )

        val_cost = model.fit(self.train_df, self.validation_df)
        return val_cost
        

class LinearRegForecast(RegressorForecast):
    def __init__(
        self,
        hparams:dict=None,
        exp_name: str = "Tanesco",
        file_name: str = None,
        seed: int = 42,
        root_dir: str = "../",
        trial=None,
        metric: str = "val_mae"
    ):
        super().__init__(
            hparams=hparams,
            exp_name=exp_name,
            file_name=file_name,
            seed=seed,
            root_dir=root_dir,
            trial=trial,
            metric=metric,
            model_type='LinearReg'
            )
     
        params=copy.deepcopy(self.hparams)
        self.model=LinearRegressionModel(data_pipeline=params.pop('data_pipeline'),
                            model_params={})
        


    




    





