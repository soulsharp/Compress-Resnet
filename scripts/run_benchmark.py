import argparse

from model.resnet_pl import Resnet50Module
from data.load_data import prepare_dataloader
from benchmarks.benchmark import run_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for model benchmarking")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        type=str,
        help="Path containing evaluation config",
    )
    parser.add_argument("--k", default=1, type=int, help="K in Top-K accuracy")
    parser.add_argument(
        "--num_classes", default=10, type=int, help="Num_classes in the dataset"
    )
    parser.add_argument(
        "--pretrained_weights_path",
        default="model/weights/resnet50.pt",
        type=str,
        help="Path containing pretrained Resnet50 Model weights",
    )

    args = parser.parse_args()
    model = Resnet50Module(args.num_classes, args.config, args.pretrained_weights_path)
    test_loader = prepare_dataloader(False)
    report_acc = run_benchmark(model, test_loader, args.k)
    print(f"Observed Accuracy : {report_acc:.3f}")
