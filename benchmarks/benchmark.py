import time

import torch

from utils.utils import AverageMeter, get_topk_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_benchmark(model, test_loader, k):
    """
    Evaluates a trained classification model on a test dataset using Top-K accuracy.
    Also computes the average inference time per sample.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test/validation dataset.
        k (int): The 'k' in Top-K accuracy (e.g., 1 for Top-1, 5 for Top-5).

    Returns:
        tuple: (avg_topk_accuracy: float, avg_time_per_sample: float)
    """
    device = next(model.parameters()).device
    model_name = model.__class__.__name__
    accuracy = AverageMeter()
    total_samples = 0
    start_time = time.time()

    if model_name == "Resnet50Module":
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            _, acc = model((images, targets))
            # acc = get_topk_accuracy(outputs, targets, k)
            accuracy.update(acc, n=images.size(0))
            total_samples += images.size(0)

    # when using the plain resnet model from model/resnet.py
    else:
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            acc = get_topk_accuracy(outputs, targets, k)
            accuracy.update(acc, n=images.size(0))
            total_samples += images.size(0)

    total_time = time.time() - start_time
    avg_time_per_sample = total_time / total_samples

    return accuracy.avg, avg_time_per_sample
