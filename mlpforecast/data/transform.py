
import pandas as pd
import numpy as np
import torch
import pytimetk as tk
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler, FunctionTransformer
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from mlpforecast.data.processing import  combine_past_future_exogenous, fourier_series_t, compute_netload_ghi, get_n_sample_per_day
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class DatasetObjective(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):


    def __init__(self, target_series = ["NetLoad"],
                unknown_features = [],
                calender_variable=[],
                known_calender_features = [],
                known_continuous_features = [],
                add_ghi = False,  
                scaler = MinMaxScaler(),
                target_scaler = MinMaxScaler(), 
                lags=[1 * 48, 7 * 48],
                window=[1 * 48, 7 * 48],
                window_func=["mean"],
                period="30min",
                input_window_size=96,
                forecast_horizon=48,
                date_column="timestamp"):
        
        if isinstance(target_series, str):
            target_series = [target_series]

        elif not isinstance(target_series, list):
            raise ValueError(f"{target_series} should be a string or a list of strings.")

        n_samples = get_n_sample_per_day(period)
        self.scaler = scaler
        self.target_transformer = target_scaler
        self.installed_capacity=np.abs(self.target).max(0)
        self.numerical_features = unknown_features + known_continuous_features
        self.calender_variable=calender_variable
        self.date_column=date_column
        self.target_series = target_series
        self.unknown_features = unknown_features
        self.calender_variable=calender_variable
        self.known_calender_features = known_calender_features
        self.known_continuous_features = known_continuous_features
        self.input_window_size=input_window_size
        self.forecast_horizon=forecast_horizon


        steps = []
        transformers=[]
    
        if (lags is not None) and (len(lags) > 0):

            lags = [int(l * n_samples) for l in sorted(lags)]

            transformer_lags = FunctionTransformer(
                tk.augment_lags,
                kw_args={
                    "date_column": date_column,
                    "value_column": target_series,
                    "lags": lags,
                },
            )
            steps += [("lags_step", transformer_lags)]

            self.numerical_features+=[f"{value}_lag_{lag}" for value in target_series for lag in lags]

        if (window is not None) and (len(window) > 0):

            window = [int(l * n_samples) for l in sorted(window)]

            transformer_rolling = FunctionTransformer(
                tk.augment_rolling,
                kw_args={
                    "date_column": date_column,
                    "window_func": window_func,
                    "value_column": target_series,
                    "window": window,
                },
            )
            steps += [("rolling_step", transformer_rolling)]
            
            self.numerical_features+=[f"{value}_rolling_mean_{w}" for value in target_series for w in window]


        if len(steps)>0:
            steps += [("dropnan_step", FunctionTransformer(lambda x: x.dropna()))]
            transformers+= [("AR", Pipeline(steps=steps), target_series)]
            

        if len(self.numerical_features) > 0:
            input_scaler = StandardScaler() if input_scaler is None else input_scaler
            numeric_transformer = Pipeline(steps=[("scaler", input_scaler)])
            transformers += [("feat_scaler", numeric_transformer, self.numerical_features)]

        target_transformer = Pipeline(steps=[("target_scaler", target_scaler)])
        transformers += [("target_scaler", target_transformer, target_series)]
        self.data_pipeline = ColumnTransformer(
                                transformers=transformers,
                                remainder="passthrough",
                                verbose_feature_names_out=False,
                            )



    def fit(self, data, y=None):
        self.data_pipeline.fit(data)
        if len(self.calender_variable)>0:
            exog = data[self.calender_variable].values
            self.exog_periods=[len(np.unique(exog[:, l])) for l in range(exog.shape[-1])]
        
        return self
    
    def transform(self, data):

        data_transfomed=self.data_pipeline.transform(data.copy())
        self.data = self.data.dropna()
        data_transfomed = data_transfomed.sort_values(column=self.date_column)

        if self.calender_variable>0:
            exog = data_transfomed[self.calender_variable].astype(np.float32).values
            seasonalities =np.hstack([fourier_series_t(exog[:,i], self.exog_periods[i], 1) for i in range(len(exog_periods))])
            for i, col in enumerate(self.calender_variable):
                data_transfomed[f'{col}-sin']=seasonalities[:, i]
                data_transfomed[f'{col}-cosin']=seasonalities[:, i]+seasonalities[:, i+1]
                data_transfomed[f'{col}-cos']=seasonalities[:, i+1]
                i+=2

        
        features = data_transfomed[self.target_series+self.unknown_features].values.astype(np.float64)
        targets = data_transfomed[self.target_series].values.astype(np.float64)
        
        future_exogenous=data_transfomed[self.known_continuous_features+self.known_calender_features].values.astype(np.float64)
        features = np.concatenate([features, future_exogenous], 1).astype(np.float64)
        
        assert (
            len(features) > 0
        ), "Ensure you have at least one historical features to train the model."
        
        if len(future_exogenous)>0:
            future_exogenous = np.squeeze(
                np.lib.stride_tricks.sliding_window_view(
                    future_exogenous[self.input_window_size:],
                    (self.forecast_horizon, future_exogenous.shape[1]),
                ),
                axis=1,
            )
            future_exogenous = future_exogenous.reshape(
                future_exogenous.shape[0], self.forecast_horizon, -1
            )
        else:
            future_exogenous = None


        features = np.squeeze(
        np.lib.stride_tricks.sliding_window_view(
            features,
            window_shape=(self.input_window_size, features.shape[1]),
        ),
        axis=1)
        features = features.reshape(features.shape[0], self.input_window_size, -1)[
            :-self.forecast_horizon
        ]

        targets = np.squeeze(
        np.lib.stride_tricks.sliding_window_view(
            targets[self.input_window_size:],
            (self.forecast_horizon, targets.shape[1]),
        ),
        axis=1)
        features =combine_past_future_exogenous(features, future_exogenous)
        targets = self.targets.reshape(self.targets.shape[0], self.forecast_horizon, -1)
        return features, targets
        
   
    
    
   
    
    
