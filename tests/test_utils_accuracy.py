import torch
import pytest
from utils.utils import AverageMeter, load_yaml, get_topk_accuracy


@pytest.mark.parametrize(
    "updates",
    [
        [(3, 5)],
        [(2, 1), (4, 2)],
        [(-4, 3), (4.0, 4)],
    ],
)
def test_average_meter_updates_and_reset(updates):
    metric = AverageMeter()

    total_sum = 0
    total_count = 0
    last_val = None

    for val, count in updates:
        metric.update(val, count)
        last_val = val
        total_sum += val * count
        total_count += count

    # Checs cumulative values
    assert metric.val == last_val
    assert metric.sum == total_sum
    assert metric.count == total_count
    assert metric.avg == pytest.approx(total_sum / total_count)

    # Checks whether reset works
    metric.reset()
    assert metric.val == 0
    assert metric.sum == 0
    assert metric.count == 0
    assert metric.avg == 0


def test_load_yaml_success(mocker):
    # Simulates a YAML file with valid content
    fake_content = "key: value\nnumber: 42"
    mock_file = mocker.mock_open(read_data=fake_content)
    mocker.patch("builtins.open", mock_file)

    result = load_yaml("dummy_path.yaml")
    assert isinstance(result, dict)
    assert result["key"] == "value"
    assert result["number"] == 42
    mock_file.assert_called_once_with("dummy_path.yaml", "r")


def test_load_yaml_file_not_found(mocker):
    # Simulates FileNotFoundError
    mocker.patch("builtins.open", side_effect=FileNotFoundError)

    result = load_yaml("nonexistent.yaml")
    assert result is None


def test_load_yaml_invalid_yaml(mocker):
    # Simulates invalid YAML content
    bad_yaml = "key: value: another"
    mock_file = mocker.mock_open(read_data=bad_yaml)
    mocker.patch("builtins.open", mock_file)

    result = load_yaml("bad.yaml")
    assert result is None


def get_correct_topk_labels(k):
    labels = [1, 2, 3, 4, 5]
    logits = torch.randn(5, 10)
    rearranged_logits = torch.sort(logits, dim=1, descending=True)


@pytest.mark.parametrize(
    "logits, labels, k, expected",
    [
        # Single sample, top-1 correct
        (torch.tensor([[0.1, 0.9, 0.0]]), torch.tensor([1]), 1, 1.0),
        # Single sample, top-1 wrong, but top-2 correct
        (torch.tensor([[0.5, 0.3, 0.2]]), torch.tensor([1]), 2, 1.0),
        # Single sample, top-1 wrong and top-2 wrong
        (torch.tensor([[0.5, 0.3, 0.2]]), torch.tensor([2]), 1, 0.0),
        # Batch: all correct
        (torch.tensor([[0.9, 0.1], [0.1, 0.9]]), torch.tensor([0, 1]), 1, 1.0),
        # Batch: all wrong
        (torch.tensor([[0.1, 0.9], [0.9, 0.1]]), torch.tensor([0, 0]), 1, 0.5),
    ],
)
def test_topk_accuracy_correctness(logits, labels, k, expected):
    acc = get_topk_accuracy(logits, labels, k)
    assert abs(acc - expected) < 1e-6


# When k=num_classes, topk_accuracy = 1.0
def test_k_equals_num_classes():
    logits = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    labels = torch.tensor([0, 1])
    k = logits.size(1)
    assert get_topk_accuracy(logits, labels, k) == 1.0


# Invalid input tests
@pytest.mark.parametrize(
    "logits, labels, k, error",
    [
        # Logits not 2D
        (torch.tensor([0.1, 0.9]), torch.tensor([1]), 1, AssertionError),
        # Labels not 1D
        (torch.tensor([[0.1, 0.9]]), torch.tensor([[1]]), 1, AssertionError),
        # k is zero
        (torch.tensor([[0.1, 0.9]]), torch.tensor([1]), 0, AssertionError),
        # k is negative
        (torch.tensor([[0.1, 0.9]]), torch.tensor([1]), -1, AssertionError),
        # k not int
        (torch.tensor([[0.1, 0.9]]), torch.tensor([1]), 1.5, AssertionError),
    ],
)
def test_topk_accuracy_invalid_inputs(logits, labels, k, error):
    with pytest.raises(error):
        get_topk_accuracy(logits, labels, k)
