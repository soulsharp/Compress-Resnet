import torch
from torch.nn.utils.prune import L1Unstructured

class PruneResnet50:
    def __init__(self, prune_amount=0.5, type_pruning="L1Unstructured"):
        self.__init__()
        self.prune_fraction = prune_amount
