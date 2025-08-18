import torch
import argparse

from .prepare_dataset import prepare_cifar10_dataset
from utils.utils import AverageMeter, build_dataloader, get_topk_accuracy, load_yaml
from compress.export import load_resnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad
def run_benchmark(model, cfg_path, k):
    """
    Evaluates a trained classification model on a test dataset using Top-K accuracy.
    The default dataset on which the benchmarking is performed is CIFAR-10.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        cfg_path (str): Path to the YAML configuration file containing evaluation and dataset settings.
        k (int): The 'k' in Top-K accuracy (e.g., 1 for Top-1, 5 for Top-5).

    Returns:
        float : Number representing the average Top-K accuracy over the entire test set.
    """
    model.eval()
    cfg = load_yaml(cfg_path)
    dataset_cfg = None
    eval_cfg = None

    if cfg is not None:
        eval_cfg = cfg["eval"]
        dataset_cfg = cfg["dataset_eval_params"]
    
    test_dataset = prepare_cifar10_dataset(dataset_cfg)
    test_loader = build_dataloader(test_dataset, eval_cfg)

    accuracy = AverageMeter()

    for (images, targets) in test_loader:
        x = images.to(device)
        y = targets.to(device)
        outputs = model(x)
        acc = get_topk_accuracy(outputs, y, k)
        accuracy.update(acc, 1)

    return accuracy.avg

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Arguments for model benchmarking')
    parser.add_argument('--config', default='config/config.yaml', type=str, 
                        help="path containing evaluation config")
    parser.add_argument("--k", default=1, type=int, help="K in Top-K accuracy")
    model = load_resnet()
    args = parser.parse_args()
    report_acc = run_benchmark(model, args.config, args.k)
    print(f"Observed Accuracy : {report_acc:.3f}")




    
