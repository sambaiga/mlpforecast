import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from mlpforecast.data.transform import DatasetObjective
from sklearn.preprocessing import MinMaxScaler
@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(0)
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2021-01-01', periods=100, freq='30T'),
        'NetLoad': np.random.rand(100),
        'Feature1': np.random.rand(100),
        'Calendar1': pd.date_range(start='2021-01-01', periods=100, freq='D').day
    })
    return data

def test_initialization():
    """Test initialization of the DatasetObjective class."""
    obj = DatasetObjective()
    assert obj.input_window_size == 96, "Incorrect default input_window_size"
    assert isinstance(obj.input_scaler, MinMaxScaler), "Default input_scaler should be MinMaxScaler"

def test_fit_method(sample_data):
    """Test the fit method."""
    obj = DatasetObjective(
        known_continuous_features=['Feature1'],
        calendar_variables=['Calendar1']
    )
    obj.fit(sample_data)
    assert hasattr(obj, 'data_pipeline'), "Data pipeline should be set after fitting"

def test_transform_method_not_fitted():
    """Ensure that transform raises an error if called before fit."""
    obj = DatasetObjective()
    with pytest.raises(NotFittedError):
        obj.transform(pd.DataFrame())

def test_transform_method(sample_data):
    """Test the transform method."""
    obj = DatasetObjective(
        known_continuous_features=['Feature1'],
        calendar_variables=['Calendar1']
    )
    obj.fit(sample_data)
    features, targets = obj.transform(sample_data)
    assert isinstance(features, np.ndarray), "Features should be a numpy array"
    assert isinstance(targets, np.ndarray), "Targets should be a numpy array"
    assert features.shape[0] == targets.shape[0], "Features and targets should have the same number of rows"

def test_fourier_feature_addition(sample_data):
    """Test addition of Fourier features to the dataset."""
    obj = DatasetObjective(
        calendar_variables=['Calendar1'],
        known_calendar_features=['Calendar1'],
        period='1D'  # daily data
    )
    obj.fit(sample_data)
    transformed_data = obj.transform(sample_data)
    assert 'Calendar1-sin' in transformed_data[0].columns, "Fourier sine features should be added"
    assert 'Calendar1-cos' in transformed_data[0].columns, "Fourier cosine features should be added"

# You may need to modify this test to match the specific outputs and implementations of your methods.

