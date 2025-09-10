import numpy as np
import torch
from torch import nn
from model.resnet import resnet50
from utils.utils import count_parameters
import math

def compute_layer_importance_heuristic(layers):
    importance_list = []
    for conv_layer in layers:
        assert isinstance(conv_layer, nn.Conv2d), "This method only prunes Conv layers at the moment"
        num_parameters = count_parameters(conv_layer, inspect_layer=True)
        importance_list.append(num_parameters)
    
    total = np.sum(importance_list, axis=0)
    importance_list = importance_list / total

    return importance_list

def collect_convolution_layers(model:nn.Module):
    pass


if __name__ == "__main__":
    model = resnet50(pretrained=True)
    prune_conv_modules = []
    prune_conv_modules_name = []
    for name, module in model.named_modules():
        if "conv" in name and not name.endswith("conv3"):
            prune_conv_modules.append(module)
            prune_conv_modules_name.append(name)
    
    importance_list = compute_layer_importance_heuristic(prune_conv_modules)
    # print(prune_conv_modules_name)
    print()
    assert math.isclose(np.sum(importance_list), 1.0, abs_tol=0.00001), "importance scores must sum to 1" 
    assert len(prune_conv_modules) == len(importance_list), "Num modules to be pruned must match the importance list"
    print(importance_list * 100)