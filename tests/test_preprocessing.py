import pytest
from mlpforecast.data.processing import _validate_target_series,get_n_sample_per_day, detect_missing_date
import pandas as pd
import numpy as np
def test_validate_target_series_with_string():
    result = _validate_target_series("NetLoad")
    assert result == ["NetLoad"], "Should return the string wrapped in a list"

def test_validate_target_series_with_list_of_strings():
    result = _validate_target_series(["NetLoad", "Temperature"])
    assert result == ["NetLoad", "Temperature"], "Should return the list as is"

'''def test_validate_target_series_with_empty_string():
    with pytest.raises(ValueError) as excinfo:
        _validate_target_series("")
    assert "should be a string or a list of strings" in str(excinfo.value)'''

def test_validate_target_series_with_invalid_input_type():
    with pytest.raises(ValueError) as excinfo:
        _validate_target_series(123)
    assert "should be a string or a list of strings" in str(excinfo.value)

def test_validate_target_series_with_list_of_invalid_types():
    with pytest.raises(ValueError) as excinfo:
        _validate_target_series([123, "NetLoad"])
    assert "should be a string or a list of strings" in str(excinfo.value)


def test_get_n_sample_per_day_with_valid_period():
    assert get_n_sample_per_day("30min") == 48, "For a 30-minute period, there should be 48 samples per day"

'''def test_get_n_sample_per_day_with_hour_period():
    assert get_n_sample_per_day("1hour") == 24, "For a 1-hour period, there should be 24 samples per day"
'''
def test_get_n_sample_per_day_with_invalid_format():
    with pytest.raises(ValueError):
        get_n_sample_per_day("hourly")  # This should raise an error because there are no digits to extract

def test_get_n_sample_per_day_with_large_period():
    assert get_n_sample_per_day("120min") == 12, "For a 120-minute period, there should be 12 samples per day"

def test_get_n_sample_per_day_with_single_digit():
    assert get_n_sample_per_day("5min") == 288, "For a 5-minute period, there should be 288 samples per day"

@pytest.fixture
def sample_dataset():
    """Fixture to generate a sample dataset with a datetime index."""
    rng = pd.date_range('2023-01-01', periods=4, freq='60T')
    return pd.DataFrame({
        'value': [1.0, 2.0, 4.0, 5.0]
    }, index=rng)

def test_detect_missing_date_no_missing(sample_dataset):
    """Test with no missing dates in the dataset."""
    result = detect_missing_date(sample_dataset, period=60)
    pd.testing.assert_frame_equal(result, sample_dataset)


def test_detect_missing_date_with_missing_dates(mocker, sample_dataset):
    """Test filling missing dates with NaN values."""
    # Simulate missing dates by dropping one row
    incomplete_dataset = sample_dataset.drop(sample_dataset.index[1])

    # Mocking pd.date_range to ensure the range is correctly generated
    #mock_date_range = mocker.patch('pandas.date_range', wraps=pd.date_range)

    result = detect_missing_date(incomplete_dataset, period=60)

    # Verify that pd.date_range was called with correct arguments
    #mock_date_range.assert_called_once_with(incomplete_dataset.index.min(),
    #                                        incomplete_dataset.index.max() + pd.Timedelta(minutes=30),
    #                                        freq="30T")

    # Expected DataFrame with NaN for the missing date
    expected_index = pd.date_range('2023-01-01', periods=4, freq='60T')
    expected_result = pd.DataFrame({
        'value': [1.0, np.nan, 4.0, 5.0]
    }, index=expected_index)

    pd.testing.assert_frame_equal(result, expected_result)