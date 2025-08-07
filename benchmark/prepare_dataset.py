from torchvision.transforms import InterpolationMode
import torchvision
from torchvision import transforms as T


def build_eval_transforms(cfg):
    """
    Builds a torchvision transform pipeline for evaluation (of the CIFAR-10 dataset) 
    based on configuration settings.

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - "mean" (list of float): Per-channel means for normalization.
            - "std" (list of float): Per-channel standard deviations for normalization.
            - "im_size" (int): Final image size after cropping/resizing.
            - "interpolation_mode" (str): Interpolation method name for resizing.
              Must be one of the keys in torchvision.transforms.InterpolationMode (e.g., "BICUBIC", "BILINEAR").

    Returns:
        torchvision.transforms.Compose: Composed transform pipeline for eval
            - Evaluation: Resize(Resize + CenterCrop) → ToTensor → Normalize
    """

    normalize = T.Normalize(mean=cfg["mean"], std=cfg["std"])
    transforms = None

    transforms = T.Compose([
        T.Resize(
            (cfg["precrop_size"]),
            interpolation=InterpolationMode[cfg["interpolation_mode"]]
        ),
        T.CenterCrop(cfg["im_size"]),
        T.ToTensor(),
        normalize
        ])
    
    return transforms

def prepare_cifar10_dataset(cfg):
    """
    Prepares the CIFAR-10 test dataset with configurable transforms.

    Args:
        cfg (dict): Configuration dictionary.
            Required keys include:
                - "mean": List of 3 floats for channel-wise mean normalization.
                - "std": List of 3 floats for channel-wise std normalization.
                - "im_size": Integer target image size.
                - "precrop_size": Integer pre-crop image size.
                - "interpolation_mode": String name of the interpolation mode (e.g., "BICUBIC").
                
    Returns:
        - dataset_test (torchvision.datasets.CIFAR10): Test dataset with deterministic transforms.
    """

    transform_test = build_eval_transforms(cfg)

    dataset_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    return dataset_test