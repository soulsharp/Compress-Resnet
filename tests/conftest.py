import torch
import pytest

from torch.utils.data import Dataset

@pytest.fixture
def create_fake_batch():
    images = torch.randn((2, 3, 224, 224), dtype=torch.float32)
    labels = torch.randint(0, 10, size=(2,))
    return images, labels

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

@pytest.fixture
def get_fake_logits_and_labels():
    rand_tensor = torch.randn(4, 10)
    labels = torch.argmax(rand_tensor, dim=1)
    return rand_tensor, labels

