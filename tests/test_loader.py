import numpy as np
import torch
from mlpforecast.data.loader import TimeSeriesDataset, TimeseriesDataModule
def test_time_series_dataset_length():
    # Create sample data
    inputs = np.random.rand(100, 10)  # 100 samples, 10 features each
    targets = np.random.rand(100, 1)  # 100 target values

    # Initialize dataset
    dataset = TimeSeriesDataset(inputs, targets)

    # Check the length of the dataset
    assert len(dataset) == 100, "The dataset length should be equal to the number of samples."

def test_time_series_dataset_getitem():
    # Create sample data
    inputs = np.random.rand(10, 5)  # 10 samples, 5 features each
    targets = np.random.rand(10, 1)  # 10 target values

    # Initialize dataset
    dataset = TimeSeriesDataset(inputs, targets)

    # Check data retrieval from the dataset
    for i in range(len(dataset)):
        features, target = dataset[i]
        assert torch.all(torch.eq(features, torch.FloatTensor(inputs[i]))), "Features do not match."
        assert torch.all(torch.eq(target, torch.FloatTensor(targets[i]))), "Targets do not match."

def test_time_series_dataset_types():
    # Create sample data
    inputs = np.random.rand(50, 3)  # 50 samples, 3 features each
    targets = np.random.rand(50, 1)  # 50 target values

    # Initialize dataset
    dataset = TimeSeriesDataset(inputs, targets)

    # Check data types
    assert isinstance(dataset.inputs, torch.FloatTensor), "Inputs should be torch.FloatTensor."
    assert isinstance(dataset.targets, torch.FloatTensor), "Targets should be torch.FloatTensor."

# Tests for TimeseriesDataModule
def test_datamodule_initialization():
    train_inputs = np.random.randn(100, 10)
    train_targets = np.random.randn(100, 1)
    module = TimeseriesDataModule(train_inputs, train_targets)
    assert isinstance(module.train_dataset, TimeSeriesDataset), "Train dataset should be a TimeSeriesDataset instance"
    assert module.val_dataset is None, "Validation dataset should be None when not provided"

def test_datamodule_with_validation_data():
    train_inputs = np.random.randn(100, 10)
    train_targets = np.random.randn(100, 1)
    val_inputs = np.random.randn(50, 10)
    val_targets = np.random.randn(50, 1)
    module = TimeseriesDataModule(train_inputs, train_targets, val_inputs, val_targets)
    assert isinstance(module.val_dataset, TimeSeriesDataset), "Validation dataset should be a TimeSeriesDataset instance"

def test_train_dataloader():
    train_inputs = np.random.randn(100, 10)
    train_targets = np.random.randn(100, 1)
    module = TimeseriesDataModule(train_inputs, train_targets)
    train_loader = module.train_dataloader()
    assert isinstance(train_loader, torch.utils.data.DataLoader), "Should return a DataLoader instance"
    assert train_loader.batch_size == 64, "Default batch size should be 64"
    assert train_loader.drop_last, "Should drop the last incomplete batch"

def test_val_dataloader():
    train_inputs = np.random.randn(100, 10)
    train_targets = np.random.randn(100, 1)
    val_inputs = np.random.randn(50, 10)
    val_targets = np.random.randn(50, 1)
    module = TimeseriesDataModule(train_inputs, train_targets, val_inputs, val_targets)
    val_loader = module.val_dataloader()
    assert isinstance(val_loader, torch.utils.data.DataLoader), "Should return a DataLoader instance"
    assert val_loader.batch_size == 64, "Default batch size should be 64"
    assert val_loader.drop_last == module.drop_last, "drop_last should be consistent with the module configuration"
