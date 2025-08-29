import torch
import pytest

from benchmarks.benchmark import run_benchmark
from model.resnet import resnet50


@pytest.mark.parametrize("k", [1, 3])
def test_run_benchmark(dummy_dataset, k):
    model = resnet50()
    test_loader = torch.utils.data.DataLoader(
        dummy_dataset, batch_size=1, shuffle=False
    )
    acc, avg_time = run_benchmark(model, test_loader, k)

    assert isinstance(acc, float)
    assert isinstance(avg_time, float)
