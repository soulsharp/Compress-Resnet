import argparse
from tools.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for model retraining/ training"
    )
    parser.add_argument(
        "--logger",
        default="tb",
        type=str,
        options=["wandb, tb"],
        help="Logger to log training/ testing metrics",
    )

    parser.add_argument(
        "--log_path", default="logs/tb_logs", type=str, help="Path to store logs"
    )
    parser.add_argument(
        "--checkpoint_path",
        default="checkpoints",
        type=str,
        help="Path to store checkpoints",
    )
    parser.add_argument(
        "--config_path",
        default="config/config.yaml",
        type=str,
        help="Path where config lives",
    )
    parser.add_argument(
        "--num_classes", default=10, type=int, help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--pretrained_path",
        default="model/weights/resnet50.pt",
        type=str,
        help="Path where pretrained weights are stored",
    )
    args = parser.parse_args()
    train(args)
