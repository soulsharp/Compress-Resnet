import pytest
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T 

from data.load_data import build_train_dataset, build_eval_dataset
from data.load_data import build_train_dataloader, build_eval_dataloader

@pytest.fixture
def dummy_dataset():
    class DummyDataset(Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx): 
            return torch.randn(3, 32, 32), torch.tensor(idx % 2)
    return DummyDataset()

@pytest.fixture
def get_cfg():
    return {
    "train_data_path": "/tmp",
    "test_data_path": "/tmp",
    "mean": [0.4914, 0.4822, 0.4465],
    "std": [0.247, 0.243, 0.261], 
}

def test_build_train_dataset_for_transform_types(get_cfg, mocker):
    cfg = get_cfg
    mock_cifar = mocker.patch("data.load_data.CIFAR10", return_value=None)

    value = build_train_dataset(cfg)
    mock_cifar.assert_called_once()
    _, kwargs = mock_cifar.call_args
    transform = kwargs["transform"]

    assert isinstance(transform.transforms[0], T.RandomCrop)
    assert isinstance(transform.transforms[1], T.RandomHorizontalFlip)
    assert isinstance(transform.transforms[2], T.ToTensor)
    assert isinstance(transform.transforms[3], T.Normalize)
    assert kwargs["train"] == True
    assert value == None

def test_build_eval_dataset_for_transform_types(get_cfg, mocker):
    cfg = get_cfg
    mock_cifar = mocker.patch("data.load_data.CIFAR10", return_value=None)

    value = build_eval_dataset(cfg)
    mock_cifar.assert_called_once()
    _, kwargs = mock_cifar.call_args
    transform = kwargs["transform"]
    
    assert isinstance(transform.transforms[0], T.ToTensor)
    assert isinstance(transform.transforms[1], T.Normalize)
    assert kwargs["train"] == False
    assert value == None

@pytest.mark.parametrize("cfg", [
    {"num_workers": 0, "pin_memory": False},   
    {"train_batch_size": 4, "pin_memory": False}, 
    {"train_batch_size": 4, "num_workers": 0},
])
def test_missing_config_raises(dummy_dataset, cfg):
    with pytest.raises(AssertionError) as exc_info:
        build_train_dataloader(dummy_dataset, cfg)
    assert "Missing key in config" in str(exc_info.value)

@pytest.mark.parametrize("batch_size, num_workers, pin_memory", [
    (1, 0, False),
    (4, 0, True),
    (8, 2, False),
])
def test_train_dataloader_properties(dummy_dataset, batch_size, num_workers, pin_memory):
    cfg = {"train_batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    loader = build_train_dataloader(dummy_dataset, cfg)
    
    # Checks DataLoader config
    assert loader.batch_size == batch_size
    assert loader.num_workers == num_workers
    assert loader.pin_memory == pin_memory
    
    # Checks whether dataset is assigned correctly
    assert loader.dataset == dummy_dataset

@pytest.mark.parametrize("batch_size, num_workers, pin_memory", [
    (1, 0, False),
    (4, 0, True),
    (8, 2, False),
])
def test_eval_dataloader_properties(dummy_dataset, batch_size, num_workers, pin_memory):
    cfg = {"val_batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    loader = build_eval_dataloader(dummy_dataset, cfg)
    
    # Checks DataLoader config
    assert loader.batch_size == batch_size
    assert loader.num_workers == num_workers
    assert loader.pin_memory == pin_memory
    
    # Checks whether dataset is assigned correctly
    assert loader.dataset == dummy_dataset

@pytest.mark.parametrize("batch_size", [1, 4, 5])
def test_train_dataloader_shapes(dummy_dataset, batch_size):
    cfg = {"train_batch_size": batch_size, "num_workers": 0, "pin_memory": False}
    loader = build_train_dataloader(dummy_dataset, cfg)

    # Computes expected number of batches
    expected_batches = len(dummy_dataset) // batch_size

    batches = list(loader)
    assert len(batches) == expected_batches

    # Checks shapes of first batch
    images, labels = batches[0]
    assert images.shape[0] == batch_size       
    assert images.shape[1:] == (3, 32, 32)     
    assert labels.shape[0] == batch_size  